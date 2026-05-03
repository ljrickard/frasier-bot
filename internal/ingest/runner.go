package ingest

import (
	"context"
	"fmt"
	"log"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"frasier-bot/internal/database"
	"frasier-bot/internal/embeddings"
	"frasier-bot/internal/models"
	"frasier-bot/internal/scraper"
)

const numWorkers = 5

type job struct {
	episode scraper.EpisodeInfo
}

// Runner orchestrates the entire data ingestion pipeline
type Runner struct {
	DB         *database.DB
	Scraper    *scraper.Scraper
	Embeddings *embeddings.Service
	Logger     *log.Logger
}

func (r *Runner) Run(ctx context.Context) error {
	// --- Wipe & Migrate ---
	r.Logger.Println("Wiping existing database schema.")
	if err := r.DB.WipeDatabase(ctx); err != nil {
		return fmt.Errorf("failed to wipe database: %w", err)
	}
	r.Logger.Println("Database wiped.")

	if err := r.DB.RunMigrations(ctx); err != nil {
		return fmt.Errorf("failed to run migrations: %w", err)
	}
	r.Logger.Println("Migrations completed successfully.")

	// --- Create Show ---
	show := &models.Show{
		Title:       "Frasier",
		Description: "Frasier TV Show Transcripts",
	}
	if err := r.DB.GetOrCreateShow(ctx, show); err != nil {
		return fmt.Errorf("failed to get or create show: %w", err)
	}
	r.Logger.Printf("Using show id=%d title=%q", show.ID, show.Title)

	if scraper.RootURL == "" {
		return fmt.Errorf("scraper.RootURL is not set")
	}

	episodes, err := r.Scraper.DiscoverEpisodes()
	if err != nil {
		return fmt.Errorf("failed to discover episodes: %w", err)
	}

	if len(episodes) == 0 {
		return fmt.Errorf("no episodes discovered")
	}

	seasonSet := make(map[int]bool)
	for _, ep := range episodes {
		seasonSet[ep.Season] = true
	}
	if len(seasonSet) != 11 {
		return fmt.Errorf("FATAL: Expected 11 seasons but found %d", len(seasonSet))
	}
	r.Logger.Printf("Found 11 Seasons with %d total episodes", len(episodes))

	sort.Slice(episodes, func(i, j int) bool {
		if episodes[i].Season != episodes[j].Season {
			return episodes[i].Season < episodes[j].Season
		}
		return episodes[i].Episode < episodes[j].Episode
	})

	var toIngest []scraper.EpisodeInfo
	totalSkipped := 0
	for _, ep := range episodes {
		exists, err := r.DB.HasParentChunkForURL(ctx, ep.URL)
		if err != nil {
			r.Logger.Printf("WARN: failed to check existing transcripts for %s: %v", ep.URL, err)
			continue
		}
		if exists {
			totalSkipped++
			continue
		}
		toIngest = append(toIngest, ep)
	}

	r.Logger.Printf("To ingest: %d, already ingested: %d", len(toIngest), totalSkipped)

	if len(toIngest) == 0 {
		r.Logger.Println("Nothing to ingest. Done.")
		return nil
	}

	jobs := make(chan job, len(toIngest))
	for _, ep := range toIngest {
		jobs <- job{episode: ep}
	}
	close(jobs)

	var totalIngested atomic.Int64
	var wg sync.WaitGroup

	for w := 1; w <= numWorkers; w++ {
		wg.Add(1)
		go r.worker(ctx, w, show.ID, jobs, &wg, &totalIngested)
	}

	wg.Wait()

	r.Logger.Printf("Done. Ingested %d episodes, skipped %d.", totalIngested.Load(), totalSkipped)
	return nil
}

func (r *Runner) worker(ctx context.Context, id int, showID int64, jobs <-chan job, wg *sync.WaitGroup, totalIngested *atomic.Int64) {
	defer wg.Done()

	prefix := fmt.Sprintf("[Worker %d]", id)
	r.Logger.Printf("%s Started", prefix)

	for j := range jobs {
		ep := j.episode
		seasonEp := fmt.Sprintf("S%02dE%02d", ep.Season, ep.Episode)

		time.Sleep(500 * time.Millisecond)

		result, err := r.Scraper.ScrapeTranscript(ep.URL)
		if err != nil {
			r.Logger.Printf("%s WARN: failed to scrape %s: %v", prefix, seasonEp, err)
			continue
		}

		saved := 0
		for i, pc := range result.Chunks {
			parent := &models.ParentChunk{
				ShowID:       showID,
				Content:      pc.ParentContent,
				Season:       ep.Season,
				Episode:      ep.Episode,
				EpisodeTitle: ep.EpisodeTitle,
				URL:          ep.URL,
			}
			if err := r.DB.CreateParentChunk(ctx, parent); err != nil {
				r.Logger.Printf("%s WARN: failed to save parent chunk %d for %s: %v", prefix, i+1, seasonEp, err)
				continue
			}

			for k, child := range pc.Children {
				// Injecting our clean embeddings service call
				embedding, err := r.Embeddings.GenerateEmbedding(ctx, child)
				if err != nil {
					r.Logger.Printf("%s WARN: embedding failed for %s chunk %d.%d: %v", prefix, seasonEp, i+1, k+1, err)
					continue
				}

				c := &models.Chunk{
					ShowID:       showID,
					Content:      child,
					Embedding:    embedding,
					Season:       ep.Season,
					Episode:      ep.Episode,
					EpisodeTitle: ep.EpisodeTitle,
					ParentID:     &parent.ID,
				}

				if err := r.DB.CreateChunk(ctx, c); err != nil {
					r.Logger.Printf("%s WARN: failed to save %s chunk %d.%d: %v", prefix, seasonEp, i+1, k+1, err)
					continue
				}
				saved++
			}
		}

		r.Logger.Printf("%s Ingested %s: %s (%d chunks)", prefix, seasonEp, ep.EpisodeTitle, saved)
		totalIngested.Add(1)
	}

	r.Logger.Printf("%s Stopped", prefix)
}
