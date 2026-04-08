package scraper

import (
	"bytes"
	"fmt"
	"log"
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

// ParentChildChunk holds a parent chunk and its child snippets.
type ParentChildChunk struct {
	ParentContent string
	Children      []string
}

// TranscriptResult holds the extracted title and parent-child chunks.
type TranscriptResult struct {
	Title  string
	Chunks []ParentChildChunk
}

// Scraper holds configuration and a logger.
type Scraper struct {
	logger *log.Logger
}

// New creates a new Scraper with the given logger.
func New(logger *log.Logger) *Scraper {
	return &Scraper{logger: logger}
}

// toValidUTF8 strips invalid UTF-8 sequences and NUL bytes from a string.
func toValidUTF8(s string) string {
	s = strings.ToValidUTF8(s, "")
	b := bytes.ReplaceAll([]byte(s), []byte{0x00}, []byte{})
	return string(b)
}

// cleanTranscript removes navigation/header/footer lines and finds where
// the real transcript data begins.
func cleanTranscript(raw string) string {
	lines := strings.Split(raw, "\n")

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

// chunkParentChild splits text into ~1500-word parent chunks, then splits
// each parent into five ~300-word child snippets.
func chunkParentChild(text string) []ParentChildChunk {
	parents := chunkByWords(text, 1500)

	var result []ParentChildChunk
	for _, parent := range parents {
		children := chunkByWords(parent, 300)
		result = append(result, ParentChildChunk{
			ParentContent: parent,
			Children:      children,
		})
	}

	return result
}

// ScrapeTranscript fetches a Frasier transcript page, extracts the body
// inner text, cleans it, sanitizes it, and splits it into parent-child chunks.
func (s *Scraper) ScrapeTranscript(url string) (*TranscriptResult, error) {
	var bodyText string

	c := colly.NewCollector()
	c.UserAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"

	c.OnRequest(func(r *colly.Request) {
		r.Headers.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8")
		r.Headers.Set("Accept-Language", "en-US,en;q=0.9")
	})

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

	// Sanitize: strip invalid UTF-8 and NUL bytes
	bodyText = toValidUTF8(bodyText)

	// Clean the raw text
	text := cleanTranscript(bodyText)

	if text == "" {
		return nil, fmt.Errorf("no transcript text remaining after cleaning from %s", url)
	}

	// Split into parent-child chunks (1500-word parents, 300-word children)
	chunks := chunkParentChild(text)

	return &TranscriptResult{
		Chunks: chunks,
	}, nil
}

// isAllowedLink returns true only if the link is safe to follow.
// It rejects geocities, archive.org, wstub, and any other external links.
func isAllowedLink(rawHref string) bool {
	lower := strings.ToLower(rawHref)

	// Block known bad domains
	if strings.Contains(lower, "geocities") ||
		strings.Contains(lower, "archive.org") ||
		strings.Contains(lower, "wstub") {
		return false
	}

	// Allow relative links
	if strings.HasPrefix(rawHref, "./") || strings.HasPrefix(rawHref, "../") {
		return true
	}
	if strings.HasPrefix(rawHref, "episode_") || strings.HasPrefix(rawHref, "season_") {
		return true
	}
	if strings.HasPrefix(rawHref, "/") && !strings.HasPrefix(rawHref, "//") {
		return true
	}

	// Allow kacl780.net absolute links
	if strings.Contains(lower, "kacl780.net") {
		return true
	}

	// Block everything else (external links)
	if strings.Contains(rawHref, "://") || strings.HasPrefix(rawHref, "//") {
		return false
	}

	// Bare relative filenames are ok
	return true
}

// DiscoverEpisodes crawls the root transcripts page, finds all season pages,
// then finds all episode links within each season page.
func (s *Scraper) DiscoverEpisodes() ([]EpisodeInfo, error) {
	var episodes []EpisodeInfo
	var seasonURLs []string

	seasonRe := regexp.MustCompile(`/season_(\d+)/?$`)

	rootCollector := colly.NewCollector()
	rootCollector.UserAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

	rootCollector.OnHTML("a[href]", func(e *colly.HTMLElement) {
		rawHref := e.Attr("href")
		if !isAllowedLink(rawHref) {
			return
		}
		href := e.Request.AbsoluteURL(rawHref)
		if !seasonRe.MatchString(href) {
			return
		}
		for _, u := range seasonURLs {
			if u == href {
				return
			}
		}
		seasonURLs = append(seasonURLs, href)
	})

	s.logger.Printf("Discovering episodes from %s", RootURL)
	err := rootCollector.Visit(RootURL)
	if err != nil {
		return nil, fmt.Errorf("failed to visit root URL %s: %w", RootURL, err)
	}
	rootCollector.Wait()

	episodeRe := regexp.MustCompile(`/season_(\d+)/episode_(\d+)/([^/]+)\.html$`)

	for _, seasonURL := range seasonURLs {
		seasonCollector := colly.NewCollector()
		seasonCollector.UserAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

		seasonCollector.OnHTML("a[href]", func(e *colly.HTMLElement) {
			rawHref := e.Attr("href")
			if !isAllowedLink(rawHref) {
				return
			}
			href := e.Request.AbsoluteURL(rawHref)

			if !strings.HasSuffix(href, ".html") {
				return
			}
			if strings.HasSuffix(href, "index.html") {
				return
			}

			matches := episodeRe.FindStringSubmatch(href)
			if matches == nil {
				return
			}

			season, _ := strconv.Atoi(matches[1])
			episode, _ := strconv.Atoi(matches[2])
			rawTitle := matches[3]
			title := formatTitle(rawTitle)

			for _, ep := range episodes {
				if ep.URL == href {
					return
				}
			}

			episodes = append(episodes, EpisodeInfo{
				Season:       season,
				Episode:      episode,
				EpisodeTitle: title,
				URL:          href,
			})
		})

		err := seasonCollector.Visit(seasonURL)
		if err != nil {
			s.logger.Printf("WARN: Failed to visit season page %s: %v", seasonURL, err)
			continue
		}
		seasonCollector.Wait()
	}

	s.logger.Printf("Found %d seasons and %d episodes total", len(seasonURLs), len(episodes))
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
