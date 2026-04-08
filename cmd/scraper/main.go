package main

import (
	"context"
	"fmt"
	"log"
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

	log.Println("Connecting to database...")
	db, err := database.New(ctx)
	if err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}
	defer db.Close()
	log.Println("Connected to database successfully.")

	if err := db.RunMigrations(ctx); err != nil {
		log.Fatalf("Failed to run migrations: %v", err)
	}
	log.Println("Migrations completed successfully.")

	log.Println("Checking embedding service configuration...")
	if err := embeddings.Preflight(); err != nil {
		log.Fatalf("Embedding service preflight check failed: %v", err)
	}
	log.Println("Embedding service configured correctly.")

	// Get or create the Frasier company/show entry
	show := &models.Company{
		Name:        "Frasier",
		Ticker:      "FRASIER",
		Description: "Frasier TV Show Transcripts",
	}
	if err := db.GetOrCreateCompany(ctx, show); err != nil {
		log.Fatalf("Failed to get or create show: %v", err)
	}
	log.Printf("Using company id=%d name=%q", show.ID, show.Name)

	// Discover all episodes
	log.Printf("Discovering episodes from %s", scraper.RootURL)
	if scraper.RootURL == "" {
		log.Fatalf("scraper.RootURL is not set")
	}
	episodes, err := scraper.DiscoverEpisodes()
	if err != nil {
		log.Fatalf("Failed to discover episodes: %v", err)
	}
	log.Printf("Discovered %d episodes", len(episodes))

	if len(episodes) == 0 {
		log.Fatalf("No episodes discovered. Check the root URL: %s", scraper.RootURL)
	}

	// Sort by season then episode
	sort.Slice(episodes, func(i, j int) bool {
		if episodes[i].Season != episodes[j].Season {
			return episodes[i].Season < episodes[j].Season
		}
		return episodes[i].Episode < episodes[j].Episode
	})

	// Filter out already-ingested episodes
	var toIngest []scraper.EpisodeInfo
	totalSkipped := 0
	for _, ep := range episodes {
		seasonEp := fmt.Sprintf("S%02dE%02d", ep.Season, ep.Episode)
		exists, err := db.HasArticlesForSource(ctx, ep.URL)
		if err != nil {
			log.Printf("Warning: failed to check existing articles for %s: %v", ep.URL, err)
			continue
		}
		if exists {
			log.Printf("Skipping %s: %s (already ingested)", seasonEp, ep.EpisodeTitle)
			totalSkipped++
			continue
		}
		toIngest = append(toIngest, ep)
	}

	log.Printf("Episodes to ingest: %d, already skipped: %d", len(toIngest), totalSkipped)

	if len(toIngest) == 0 {
		log.Println("Nothing to ingest. Done.")
		return
	}

	// Create jobs channel
	jobs := make(chan job, len(toIngest))
	for _, ep := range toIngest {
		jobs <- job{episode: ep}
	}
	close(jobs)

	// Counters
	var totalIngested atomic.Int64

	// Worker pool
	var wg sync.WaitGroup
	for w := 1; w <= numWorkers; w++ {
		wg.Add(1)
		go worker(ctx, w, db, show.ID, jobs, &wg, &totalIngested)
	}

	wg.Wait()

	log.Printf("Done. Ingested %d episodes, skipped %d already-ingested episodes.", totalIngested.Load(), totalSkipped)
}

func worker(ctx context.Context, id int, db *database.DB, companyID int64, jobs <-chan job, wg *sync.WaitGroup, totalIngested *atomic.Int64) {
	defer wg.Done()

	for j := range jobs {
		ep := j.episode
		seasonEp := fmt.Sprintf("S%02dE%02d", ep.Season, ep.Episode)

		log.Printf("[Worker %d] Scraping %s: %s from %s", id, seasonEp, ep.EpisodeTitle, ep.URL)

		// Be polite to the host
		time.Sleep(500 * time.Millisecond)

		result, err := scraper.ScrapeTranscript(ep.URL)
		if err != nil {
			log.Printf("[Worker %d] Warning: failed to scrape %s: %v", id, seasonEp, err)
			continue
		}

		saved := 0
		for i, pc := range result.Chunks {
			// Save parent chunk first
			parent := &models.ParentChunk{
				Content:      pc.ParentContent,
				Season:       ep.Season,
				Episode:      ep.Episode,
				EpisodeTitle: ep.EpisodeTitle,
				URL:          ep.URL,
			}
			if err := db.CreateParentChunk(ctx, parent); err != nil {
				log.Printf("[Worker %d] Warning: failed to save parent chunk %d for %s: %v", id, i+1, seasonEp, err)
				continue
			}

			// Save each child snippet with embedding
			for k, child := range pc.Children {
				partTitle := fmt.Sprintf("%s: %s (Part %d.%d)", seasonEp, ep.EpisodeTitle, i+1, k+1)

				embedding, err := embeddings.GenerateEmbedding(ctx, child)
				if err != nil {
					log.Printf("[Worker %d] Warning: failed to generate embedding for %s chunk %d.%d: %v", id, seasonEp, i+1, k+1, err)
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
					log.Printf("[Worker %d] Warning: failed to save %s chunk %d.%d: %v", id, seasonEp, i+1, k+1, err)
					continue
				}
				saved++
			}
		}

		fmt.Printf("[Worker %d] Ingested S%02dE%02d: %s (%d chunks)\n", id, ep.Season, ep.Episode, ep.EpisodeTitle, saved)
		totalIngested.Add(1)
	}
}
