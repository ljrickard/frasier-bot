package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"omnicorp-analyst/internal/ai"
	"omnicorp-analyst/internal/database"
	"omnicorp-analyst/internal/embeddings"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <search query>\n", os.Args[0])
		os.Exit(1)
	}

	query := strings.Join(os.Args[1:], " ")
	ctx := context.Background()

	// Preflight check
	if err := embeddings.Preflight(); err != nil {
		log.Fatalf("Embedding service preflight check failed: %v", err)
	}

	// Connect to database
	db, err := database.New(ctx)
	if err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}
	defer db.Close()

	// Generate query embedding
	log.Printf("Generating embedding for query: %q", query)
	queryEmbedding, err := embeddings.GenerateQueryEmbedding(ctx, query)
	if err != nil {
		log.Fatalf("Failed to generate query embedding: %v", err)
	}

	// Search articles (children)
	results, err := db.SearchArticles(ctx, queryEmbedding, 5)
	if err != nil {
		log.Fatalf("Failed to search articles: %v", err)
	}

	if len(results) == 0 {
		fmt.Println("No results found.")
		return
	}

	// Collect unique parent IDs
	parentIDSet := make(map[int64]bool)
	var parentIDs []int64
	for _, r := range results {
		if r.ParentID != nil && !parentIDSet[*r.ParentID] {
			parentIDSet[*r.ParentID] = true
			parentIDs = append(parentIDs, *r.ParentID)
		}
	}

	// Fetch parent chunks
	var parentResults []database.SearchResult
	if len(parentIDs) > 0 {
		parents, err := db.GetParentChunksByIDs(ctx, parentIDs)
		if err != nil {
			log.Fatalf("Failed to fetch parent chunks: %v", err)
		}
		for _, p := range parents {
			parentResults = append(parentResults, database.SearchResult{
				Title:   fmt.Sprintf("S%02dE%02d: %s", p.Season, p.Episode, p.EpisodeTitle),
				URL:     p.URL,
				Content: p.Content,
			})
		}
	}

	// Fall back to child content if no parents found
	searchResultsForAI := parentResults
	if len(searchResultsForAI) == 0 {
		searchResultsForAI = results
	}

	// Display results
	fmt.Println()
	fmt.Println(strings.Repeat("=", 80))
	fmt.Printf("  Search Results for: %q\n", query)
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println()

	for i, r := range results {
		fmt.Printf("  %d. %s\n", i+1, r.Title)
		fmt.Printf("     URL:        %s\n", r.URL)
		fmt.Printf("     Similarity: %.4f\n", r.Similarity)
		if r.ParentID != nil {
			fmt.Printf("     Parent ID:  %d\n", *r.ParentID)
		}
		fmt.Println()
	}

	fmt.Println(strings.Repeat("=", 80))
	fmt.Printf("  %d child result(s), %d unique parent(s) fetched\n", len(results), len(parentIDs))
	fmt.Println(strings.Repeat("=", 80))

	fmt.Println()

	// Generate AI answer using parent content
	log.Println("Generating AI answer from parent chunks...")
	answer, err := ai.GenerateAnswer(ctx, query, searchResultsForAI)
	if err != nil {
		log.Fatalf("Failed to generate answer: %v", err)
	}

	fmt.Println()
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("  AI-Generated Answer")
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println()
	fmt.Println(answer)
	fmt.Println()
	fmt.Println(strings.Repeat("=", 80))
}
