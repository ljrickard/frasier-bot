package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"omnicorp-analyst/internal/database"
	"omnicorp-analyst/internal/embeddings"
	"omnicorp-analyst/internal/models"
	"omnicorp-analyst/internal/scraper"
)

const numWorkers = 5

type job struct {
	episode scraper.EpisodeInfo
}

func main() {
	ctx := context.Background()

	logger := log.New(os.Stdout, "", log.Ldate|log.Ltime|log.Lshortfile)

	logger.Println("Connecting to database...")
	db, err := database.New(ctx)
	if err != nil {
		logger.Fatalf("Failed to connect to database: %v", err)
	}
	defer db.Close()
	logger.Println("Connected to database successfully.")

	if err := db.RunMigrations(ctx); err != nil {
		logger.Fatalf("Failed to run migrations: %v", err)
	}
	logger.Println("Migrations completed successfully.")

	if err := embeddings.Preflight(); err != nil {
		logger.Fatalf("Embedding service preflight check failed: %v", err)
	}
	logger.Println("Embedding service configured correctly.")

	show := &models.Company{
		Name:        "Frasier",
		Ticker:      "FRASIER",
		Description: "Frasier TV Show Transcripts",
	}
	if err := db.GetOrCreateCompany(ctx, show); err != nil {
		logger.Fatalf("Failed to get or create show: %v", err)
	}
	logger.Printf("Using company id=%d name=%q", show.ID, show.Name)

	if scraper.RootURL == "" {
		logger.Fatalf("scraper.RootURL is not set")
	}

	sc := scraper.New(logger)

	episodes, err := sc.DiscoverEpisodes()
	if err != nil {
		logger.Fatalf("Failed to discover episodes: %v", err)
	}

	if len(episodes) == 0 {
		logger.Fatalf("No episodes discovered. Check the root URL: %s", scraper.RootURL)
	}

	sort.Slice(episodes, func(i, j int) bool {
		if episodes[i].Season != episodes[j].Season {
			return episodes[i].Season < episodes[j].Season
		}
		return episodes[i].Episode < episodes[j].Episode
	})

	var toIngest []scraper.EpisodeInfo
	totalSkipped := 0
	for _, ep := range episodes {
		exists, err := db.HasArticlesForSource(ctx, ep.URL)
		if err != nil {
			logger.Printf("WARN: failed to check existing articles for %s: %v", ep.URL, err)
			continue
		}
		if exists {
			totalSkipped++
			continue
		}
		toIngest = append(toIngest, ep)
	}

	logger.Printf("To ingest: %d, already ingested: %d", len(toIngest), totalSkipped)

	if len(toIngest) == 0 {
		logger.Println("Nothing to ingest. Done.")
		return
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
		go worker(ctx, w, logger, db, sc, show.ID, jobs, &wg, &totalIngested)
	}

	wg.Wait()

	logger.Printf("Done. Ingested %d episodes, skipped %d.", totalIngested.Load(), totalSkipped)
}

func worker(ctx context.Context, id int, logger *log.Logger, db *database.DB, sc *scraper.Scraper, companyID int64, jobs <-chan job, wg *sync.WaitGroup, totalIngested *atomic.Int64) {
	defer wg.Done()

	prefix := fmt.Sprintf("[Worker %d]", id)
	logger.Printf("%s Started", prefix)

	for j := range jobs {
		ep := j.episode
		seasonEp := fmt.Sprintf("S%02dE%02d", ep.Season, ep.Episode)

		time.Sleep(500 * time.Millisecond)

		result, err := sc.ScrapeTranscript(ep.URL)
		if err != nil {
			logger.Printf("%s WARN: failed to scrape %s: %v", prefix, seasonEp, err)
			continue
		}

		saved := 0
		for i, pc := range result.Chunks {
			parent := &models.ParentChunk{
				Content:      pc.ParentContent,
				Season:       ep.Season,
				Episode:      ep.Episode,
				EpisodeTitle: ep.EpisodeTitle,
				URL:          ep.URL,
			}
			if err := db.CreateParentChunk(ctx, parent); err != nil {
				logger.Printf("%s WARN: failed to save parent chunk %d for %s: %v", prefix, i+1, seasonEp, err)
				continue
			}

			for k, child := range pc.Children {
				partTitle := fmt.Sprintf("%s: %s (Part %d.%d)", seasonEp, ep.EpisodeTitle, i+1, k+1)

				embedding, err := embeddings.GenerateEmbedding(ctx, child)
				if err != nil {
					logger.Printf("%s WARN: embedding failed for %s chunk %d.%d: %v", prefix, seasonEp, i+1, k+1, err)
					continue
				}

				a := &models.Article{
					CompanyID:    companyID,
					Title:        partTitle,
					Content:      child,
					Source:       ep.URL,
					Embedding:    embedding,
					Season:       ep.Season,
					Episode:      ep.Episode,
					EpisodeTitle: ep.EpisodeTitle,
					ParentID:     &parent.ID,
				}

				if err := db.CreateArticle(ctx, a); err != nil {
					logger.Printf("%s WARN: failed to save %s chunk %d.%d: %v", prefix, seasonEp, i+1, k+1, err)
					continue
				}
				saved++
			}
		}

		logger.Printf("%s Ingested %s: %s (%d chunks)", prefix, seasonEp, ep.EpisodeTitle, saved)
		totalIngested.Add(1)
	}

	logger.Printf("%s Stopped", prefix)
}
