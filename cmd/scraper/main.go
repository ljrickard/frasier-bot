package main

import (
	"context"
	"log"

	"omnicorp-analyst/internal/database"
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

	defaultCompany := &models.Company{
		Name:   "BBC Sport",
		Ticker: "BBC",
	}
	if err := db.GetOrCreateCompany(ctx, defaultCompany); err != nil {
		log.Fatalf("Failed to get or create default company: %v", err)
	}
	log.Printf("Using company id=%d name=%q", defaultCompany.ID, defaultCompany.Name)

	url := "https://www.bbc.co.uk/sport/golf/articles/cn89zp1e38no"
	log.Printf("Scraping articles from %s", url)

	articles, err := scraper.ScrapeArticles(url)
	if err != nil {
		log.Fatalf("Failed to scrape articles: %v", err)
	}
	log.Printf("Found %d articles.", len(articles))

	saved := 0
	for i := range articles {
		a := &models.Article{
			CompanyID: defaultCompany.ID,
			Title:     articles[i].Title,
			Content:   articles[i].Content,
			Source:    articles[i].Source,
		}

		log.Printf("Saving article %d/%d: %q", i+1, len(articles), a.Title)
		if err := db.CreateArticle(ctx, a); err != nil {
			log.Printf("Warning: failed to save article %q: %v", a.Title, err)
			continue
		}
		saved++
		log.Printf("Saved article id=%d title=%q", a.ID, a.Title)
	}

	log.Printf("Done. Saved %d/%d articles to the database.", saved, len(articles))
}
