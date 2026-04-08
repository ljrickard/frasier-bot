package main

import (
	"context"
	"fmt"
	"log"
	"sort"

	"omnicorp-analyst/internal/database"
	"omnicorp-analyst/internal/embeddings"
	"omnicorp-analyst/internal/models"
	"omnicorp-analyst/internal/scraper"
)

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

	totalIngested := 0
	totalSkipped := 0

	for _, ep := range episodes {
		seasonEp := fmt.Sprintf("S%02dE%02d", ep.Season, ep.Episode)

		// Check if we already have articles for this URL
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

		log.Printf("Scraping %s: %s from %s", seasonEp, ep.EpisodeTitle, ep.URL)
		result, err := scraper.ScrapeTranscript(ep.URL)
		if err != nil {
			log.Printf("Warning: failed to scrape %s: %v", seasonEp, err)
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
				log.Printf("Warning: failed to save parent chunk %d for %s: %v", i+1, seasonEp, err)
				continue
			}

			// Save each child snippet with embedding
			for j, child := range pc.Children {
				partTitle := fmt.Sprintf("%s: %s (Part %d.%d)", seasonEp, ep.EpisodeTitle, i+1, j+1)

				embedding, err := embeddings.GenerateEmbedding(ctx, child)
				if err != nil {
					log.Printf("Warning: failed to generate embedding for %s chunk %d.%d: %v", seasonEp, i+1, j+1, err)
					continue
				}

				a := &models.Article{
					CompanyID:    show.ID,
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
					log.Printf("Warning: failed to save %s chunk %d.%d: %v", seasonEp, i+1, j+1, err)
					continue
				}
				saved++
			}
		}

		fmt.Printf("Ingested S%02dE%02d: %s (%d chunks)\n", ep.Season, ep.Episode, ep.EpisodeTitle, saved)
		totalIngested++
	}

	log.Printf("Done. Ingested %d episodes, skipped %d already-ingested episodes.", totalIngested, totalSkipped)
}
