package scraper

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/gocolly/colly/v2"
)

const RootURL = "https://www.kacl780.net/frasier/transcripts/"

// EpisodeInfo holds metadata about a discovered episode.
type EpisodeInfo struct {
	Season       int
	Episode      int
	EpisodeTitle string
	URL          string
}

// TranscriptResult holds the extracted title and text chunks.
type TranscriptResult struct {
	Title  string
	Chunks []string
}

// cleanTranscript removes navigation/header/footer lines and finds where
// the real transcript data begins.
func cleanTranscript(raw string) string {
	lines := strings.Split(raw, "\n")

	// Filter out nav/header/footer lines
	skipKeywords := []string{"Home", "About", "Transcripts", "Seasons", "KACL780.NET"}
	var filtered []string
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}
		skip := false
		for _, kw := range skipKeywords {
			if strings.Contains(trimmed, kw) {
				skip = true
				break
			}
		}
		if !skip {
			filtered = append(filtered, trimmed)
		}
	}

	// Find where the real data begins
	startIdx := 0
	for i, line := range filtered {
		if strings.HasPrefix(line, "Transcript {") || strings.HasPrefix(line, "Act One") {
			startIdx = i
			break
		}
	}

	if startIdx < len(filtered) {
		filtered = filtered[startIdx:]
	}

	return strings.Join(filtered, "\n")
}

// chunkByWords splits text into chunks of roughly maxWords words.
func chunkByWords(text string, maxWords int) []string {
	words := strings.Fields(text)
	if len(words) == 0 {
		return nil
	}

	var chunks []string
	for start := 0; start < len(words); start += maxWords {
		end := start + maxWords
		if end > len(words) {
			end = len(words)
		}
		chunk := strings.Join(words[start:end], " ")
		chunks = append(chunks, chunk)
	}

	return chunks
}

// ScrapeTranscript fetches a Frasier transcript page, extracts the body
// inner text, cleans it, and splits it into word-based chunks.
func ScrapeTranscript(url string) (*TranscriptResult, error) {
	var bodyText string

	c := colly.NewCollector()

	c.UserAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"

	c.OnRequest(func(r *colly.Request) {
		r.Headers.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8")
		r.Headers.Set("Accept-Language", "en-US,en;q=0.9")
	})

	// Select the <body> element and get its inner text
	c.OnHTML("body", func(e *colly.HTMLElement) {
		bodyText = e.Text
	})

	err := c.Visit(url)
	if err != nil {
		return nil, fmt.Errorf("failed to scrape %s: %w", url, err)
	}

	if bodyText == "" {
		return nil, fmt.Errorf("no body text found at %s", url)
	}

	// Clean the raw text
	text := cleanTranscript(bodyText)

	fmt.Printf("Successfully scraped %d characters of text\n", len(text))

	if text == "" {
		return nil, fmt.Errorf("no transcript text remaining after cleaning from %s", url)
	}

	// Split into 1,000-word chunks
	chunks := chunkByWords(text, 1000)

	return &TranscriptResult{
		Chunks: chunks,
	}, nil
}

// DiscoverEpisodes crawls the root transcripts page, finds all season pages,
// then finds all episode links within each season page.
func DiscoverEpisodes() ([]EpisodeInfo, error) {
	var episodes []EpisodeInfo

	seasonRe := regexp.MustCompile(`/season_(\d+)/?$`)
	episodeRe := regexp.MustCompile(`/season_(\d+)/episode_(\d+)/([^/]+)\.html$`)

	seasonCollector := colly.NewCollector()
	seasonCollector.UserAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

	episodeCollector := colly.NewCollector()
	episodeCollector.UserAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

	// On the main transcripts page, find season links
	seasonCollector.OnHTML("a[href]", func(e *colly.HTMLElement) {
		href := e.Request.AbsoluteURL(e.Attr("href"))
		if seasonRe.MatchString(href) {
			episodeCollector.Visit(href)
		}
	})

	// On each season page, find episode links
	episodeCollector.OnHTML("a[href]", func(e *colly.HTMLElement) {
		href := e.Request.AbsoluteURL(e.Attr("href"))
		matches := episodeRe.FindStringSubmatch(href)
		if matches == nil {
			return
		}

		season, _ := strconv.Atoi(matches[1])
		episode, _ := strconv.Atoi(matches[2])
		rawTitle := matches[3]

		title := formatTitle(rawTitle)

		episodes = append(episodes, EpisodeInfo{
			Season:       season,
			Episode:      episode,
			EpisodeTitle: title,
			URL:          href,
		})
	})

	err := seasonCollector.Visit(RootURL)
	if err != nil {
		return nil, fmt.Errorf("failed to visit root URL %s: %w", RootURL, err)
	}

	seasonCollector.Wait()
	episodeCollector.Wait()

	return episodes, nil
}

// formatTitle converts a filename slug like "the_good_son" to "The Good Son".
func formatTitle(slug string) string {
	slug = strings.ReplaceAll(slug, "_", " ")
	words := strings.Fields(slug)
	for i, w := range words {
		if len(w) > 0 {
			words[i] = strings.ToUpper(w[:1]) + w[1:]
		}
	}
	return strings.Join(words, " ")
}
