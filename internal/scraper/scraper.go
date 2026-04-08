package scraper

import (
	"fmt"
	"strings"

	"github.com/gocolly/colly/v2"
)

// TranscriptResult holds the extracted title and text chunks.
type TranscriptResult struct {
	Title  string
	Chunks []string
}

// estimateTokens gives a rough token count (1 token ≈ 4 chars for English).
func estimateTokens(s string) int {
	return len(s) / 4
}

// chunkText splits text into chunks of roughly maxTokens tokens with overlap.
func chunkText(text string, maxTokens, overlapTokens int) []string {
	words := strings.Fields(text)
	if len(words) == 0 {
		return nil
	}

	// Estimate words-per-token ratio: ~0.75 words per token (i.e. ~1.33 tokens per word)
	maxWords := int(float64(maxTokens) * 0.75)
	overlapWords := int(float64(overlapTokens) * 0.75)

	if maxWords < 1 {
		maxWords = 1
	}
	if overlapWords >= maxWords {
		overlapWords = maxWords / 2
	}

	var chunks []string
	start := 0
	for start < len(words) {
		end := start + maxWords
		if end > len(words) {
			end = len(words)
		}
		chunk := strings.Join(words[start:end], " ")
		chunks = append(chunks, chunk)

		// Advance by (maxWords - overlapWords) so the next chunk overlaps
		start += maxWords - overlapWords
	}

	return chunks
}

// ScrapeTranscript fetches a Frasier transcript page, extracts the title
// and transcript body, and splits the body into overlapping chunks.
func ScrapeTranscript(url string) (*TranscriptResult, error) {
	var title string
	var paragraphs []string

	c := colly.NewCollector()

	c.UserAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"

	c.OnRequest(func(r *colly.Request) {
		r.Headers.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8")
		r.Headers.Set("Accept-Language", "en-US,en;q=0.9")
	})

	// Extract the episode title from the <h1> or <title> tag
	c.OnHTML("title", func(e *colly.HTMLElement) {
		if title == "" {
			title = strings.TrimSpace(e.Text)
		}
	})

	c.OnHTML("h1", func(e *colly.HTMLElement) {
		t := strings.TrimSpace(e.Text)
		if t != "" {
			title = t
		}
	})

	// Extract transcript text from the main content area
	c.OnHTML("div.content, div#content, div.main, main, div.transcript, body", func(e *colly.HTMLElement) {
		// Only process the first match that yields content
		if len(paragraphs) > 0 {
			return
		}
		e.ForEach("p", func(_ int, p *colly.HTMLElement) {
			text := strings.TrimSpace(p.Text)
			if text != "" {
				paragraphs = append(paragraphs, text)
			}
		})
	})

	err := c.Visit(url)
	if err != nil {
		return nil, fmt.Errorf("failed to scrape %s: %w", url, err)
	}

	if len(paragraphs) == 0 {
		return nil, fmt.Errorf("no transcript text found at %s", url)
	}

	fullText := strings.Join(paragraphs, "\n\n")

	// Chunk: 1500 tokens max, 200 token overlap
	chunks := chunkText(fullText, 1500, 200)

	return &TranscriptResult{
		Title:  title,
		Chunks: chunks,
	}, nil
}
