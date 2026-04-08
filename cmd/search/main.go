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

	// Search articles
	results, err := db.SearchArticles(ctx, queryEmbedding, 3)
	if err != nil {
		log.Fatalf("Failed to search articles: %v", err)
	}

	if len(results) == 0 {
		fmt.Println("No results found.")
		return
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
		fmt.Println()
	}

	fmt.Println(strings.Repeat("=", 80))
	fmt.Printf("  %d result(s) returned\n", len(results))
	fmt.Println(strings.Repeat("=", 80))

	fmt.Println()

	// Generate AI answer using retrieved articles
	log.Println("Generating AI answer...")
	answer, err := ai.GenerateAnswer(ctx, query, results)
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
