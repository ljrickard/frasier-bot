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

	// Step 1: Dynamic Top-K classification
	log.Printf("Classifying query: %q", query)
	classification, err := ai.ClassifyQuery(ctx, query)
	if err != nil {
		log.Printf("Warning: failed to classify query, defaulting to GENERAL: %v", err)
		classification = "GENERAL"
	}

	topK := 10
	if classification == "SPECIFIC" {
		topK = 3
	}
	log.Printf("Query classified as %s, using Top-K = %d", classification, topK)

	// Step 2a: Generate Vanilla Answer (no context)
	log.Println("Generating Vanilla AI answer (no database)...")
	vanillaAnswer, err := ai.GenerateVanillaAnswer(ctx, query)
	if err != nil {
		log.Printf("Warning: failed to generate vanilla answer: %v", err)
		vanillaAnswer = "(Vanilla answer unavailable)"
	}

	// Step 2b: RAG search
	log.Printf("Generating embedding for query: %q", query)
	queryEmbedding, err := embeddings.GenerateQueryEmbedding(ctx, query)
	if err != nil {
		log.Fatalf("Failed to generate query embedding: %v", err)
	}

	// Search articles (children)
	results, err := db.SearchArticles(ctx, queryEmbedding, topK)
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

	// Generate RAG answer
	log.Println("Generating RAG AI answer from parent chunks...")
	ragAnswer, err := ai.GenerateAnswer(ctx, query, searchResultsForAI)
	if err != nil {
		log.Fatalf("Failed to generate RAG answer: %v", err)
	}

	// Step 3: Evaluation
	log.Println("Evaluating answers...")
	evaluation, err := ai.EvaluateAnswers(ctx, query, vanillaAnswer, ragAnswer)
	if err != nil {
		log.Printf("Warning: failed to evaluate answers: %v", err)
		evaluation = "(Evaluation unavailable)"
	}

	// Display search results
	fmt.Println()
	fmt.Println(strings.Repeat("=", 80))
	fmt.Printf("  Search Results for: %q  [%s, Top-K=%d]\n", query, classification, topK)
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

	// Display Vanilla Answer
	fmt.Println()
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("  === VANILLA AI (No Database) ===")
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println()
	fmt.Println(vanillaAnswer)

	// Display RAG Answer
	fmt.Println()
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("  === RAG AI (Frasier Database) ===")
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println()
	fmt.Println(ragAnswer)

	// Display Evaluation
	fmt.Println()
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("  === EVALUATION ===")
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println()
	fmt.Println(evaluation)
	fmt.Println()
	fmt.Println(strings.Repeat("=", 80))
}
