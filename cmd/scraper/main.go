package main

import (
	"context"
	"fmt"
	"log"

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

	// Scrape a single episode
	url := "https://www.kacl780.net/frasier/transcripts/season_1/episode_1/the_good_son.html"
	// https://www.kacl780.net/frasier/transcripts/season_1/episode_1/the_good_son.html
	seasonEp := "S01E01"

	log.Printf("Scraping transcript from %s", url)
	result, err := scraper.ScrapeTranscript(url)
	if err != nil {
		log.Fatalf("Failed to scrape transcript: %v", err)
	}
	log.Printf("Extracted title: %q with %d chunks", result.Title, len(result.Chunks))

	saved := 0
	for i, chunk := range result.Chunks {
		partTitle := fmt.Sprintf("%s - %s - Part %d", seasonEp, result.Title, i+1)

		log.Printf("Generating embedding for chunk %d/%d: %q", i+1, len(result.Chunks), partTitle)
		embedding, err := embeddings.GenerateEmbedding(ctx, chunk)
		if err != nil {
			log.Fatalf("Failed to generate embedding for chunk %d: %v", i+1, err)
		}

		a := &models.Article{
			CompanyID: show.ID,
			Title:     partTitle,
			Content:   chunk,
			Source:    url,
			Embedding: embedding,
		}

		log.Printf("Saving chunk %d/%d: %q", i+1, len(result.Chunks), partTitle)
		if err := db.CreateArticle(ctx, a); err != nil {
			log.Printf("Warning: failed to save chunk %d: %v", i+1, err)
			continue
		}
		saved++
		log.Printf("Saved article id=%d title=%q", a.ID, a.Title)
	}

	log.Printf("Done. Saved %d/%d chunks to the database.", saved, len(result.Chunks))
}
