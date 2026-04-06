package scraper

import (
	"fmt"
	"strings"

	"github.com/gocolly/colly/v2"
	"omnicorp-analyst/internal/models"
)

func ScrapeArticles(url string) ([]models.Article, error) {
	var articles []models.Article

	c := colly.NewCollector()

	c.OnHTML("article", func(e *colly.HTMLElement) {
		title := strings.TrimSpace(e.ChildText("h1, h2, h3"))
		if title == "" {
			return
		}

		var paragraphs []string
		e.ForEach("p", func(_ int, p *colly.HTMLElement) {
			text := strings.TrimSpace(p.Text)
			if text != "" {
				paragraphs = append(paragraphs, text)
			}
		})

		content := strings.Join(paragraphs, "\n\n")

		articles = append(articles, models.Article{
			Title:   title,
			Content: content,
			Source:  url,
		})
	})

	err := c.Visit(url)
	if err != nil {
		return nil, fmt.Errorf("failed to scrape %s: %w", url, err)
	}

	return articles, nil
}
