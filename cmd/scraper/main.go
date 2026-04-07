package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"omnicorp-analyst/internal/database"
	"omnicorp-analyst/internal/models"
	"omnicorp-analyst/internal/scraper"
)

func main() {
	ctx := context.Background()

	if len(os.Args) < 3 {
		log.Fatal("Usage: scraper <company_id> <url>")
	}

	companyIDStr := os.Args[1]
	url := os.Args[2]

	var companyID int64
	if _, err := fmt.Sscanf(companyIDStr, "%d", &companyID); err != nil {
		log.Fatalf("Invalid company_id %q: %v", companyIDStr, err)
	}

	db, err := database.New(ctx)
	if err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}
	defer db.Close()

	log.Printf("Scraping articles from %s for company_id=%d", url, companyID)

	articles, err := scraper.ScrapeArticles(url)
	if err != nil {
		log.Fatalf("Failed to scrape articles: %v", err)
	}

	log.Printf("Found %d articles", len(articles))

	inserted := 0
	for i := range articles {
		articles[i].CompanyID = companyID
		a := &models.Article{
			CompanyID: articles[i].CompanyID,
			Title:     articles[i].Title,
			Content:   articles[i].Content,
			Source:    articles[i].Source,
		}

		if err := db.InsertArticle(ctx, a); err != nil {
			log.Printf("Warning: failed to insert article %q: %v", a.Title, err)
			continue
		}
		inserted++
		log.Printf("Inserted article id=%d title=%q", a.ID, a.Title)
	}

	log.Printf("Done. Inserted %d/%d articles.", inserted, len(articles))
}
