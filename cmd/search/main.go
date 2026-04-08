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

	logger := log.New(os.Stdout, "", log.Ldate|log.Ltime|log.Lshortfile)

	if err := embeddings.Preflight(); err != nil {
		logger.Fatalf("Embedding service preflight check failed: %v", err)
	}

	db, err := database.New(ctx)
	if err != nil {
		logger.Fatalf("Failed to connect to database: %v", err)
	}
	defer db.Close()

	// Step 1: Dynamic Top-K classification
	logger.Printf("Classifying query: %q", query)
	classification, err := ai.ClassifyQuery(ctx, query)
	if err != nil {
		logger.Printf("WARN: failed to classify query, defaulting to GENERAL: %v", err)
		classification = "GENERAL"
	}

	topK := 10
	if classification == "SPECIFIC" {
		topK = 3
	}
	logger.Printf("Query classified as %s, Top-K = %d", classification, topK)

	// Step 2a: Generate Vanilla Answer (no context)
	logger.Println("Generating Vanilla AI answer...")
	vanillaAnswer, err := ai.GenerateVanillaAnswer(ctx, query)
	if err != nil {
		logger.Printf("WARN: failed to generate vanilla answer: %v", err)
		vanillaAnswer = "(Vanilla answer unavailable)"
	}

	// Step 2b: RAG search
	logger.Println("Generating query embedding...")
	queryEmbedding, err := embeddings.GenerateQueryEmbedding(ctx, query)
	if err != nil {
		logger.Fatalf("Failed to generate query embedding: %v", err)
	}

	results, err := db.SearchArticles(ctx, queryEmbedding, topK)
	if err != nil {
		logger.Fatalf("Failed to search articles: %v", err)
	}

	if len(results) == 0 {
		logger.Println("No results found.")
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
			logger.Fatalf("Failed to fetch parent chunks: %v", err)
		}
		for _, p := range parents {
			parentResults = append(parentResults, database.SearchResult{
				Title:   fmt.Sprintf("S%02dE%02d: %s", p.Season, p.Episode, p.EpisodeTitle),
				URL:     p.URL,
				Content: p.Content,
			})
		}
	}

	searchResultsForAI := parentResults
	if len(searchResultsForAI) == 0 {
		searchResultsForAI = results
	}

	// Generate RAG answer
	logger.Println("Generating RAG AI answer...")
	ragAnswer, err := ai.GenerateAnswer(ctx, query, searchResultsForAI)
	if err != nil {
		logger.Fatalf("Failed to generate RAG answer: %v", err)
	}

	// Step 3: Evaluation
	logger.Println("Evaluating answers...")
	evaluation, err := ai.EvaluateAnswers(ctx, query, vanillaAnswer, ragAnswer)
	if err != nil {
		logger.Printf("WARN: failed to evaluate answers: %v", err)
		evaluation = "(Evaluation unavailable)"
	}

	// Display results
	sep := strings.Repeat("=", 80)

	fmt.Println()
	fmt.Println(sep)
	fmt.Printf("  Search Results for: %q  [%s, Top-K=%d]\n", query, classification, topK)
	fmt.Println(sep)
	fmt.Println()

	for i, r := range results {
		fmt.Printf("  %d. %s (similarity: %.4f)\n", i+1, r.Title, r.Similarity)
	}

	fmt.Printf("\n  %d child result(s), %d unique parent(s)\n", len(results), len(parentIDs))

	fmt.Println()
	fmt.Println(sep)
	fmt.Println("  === VANILLA AI (No Database) ===")
	fmt.Println(sep)
	fmt.Println()
	fmt.Println(vanillaAnswer)

	fmt.Println()
	fmt.Println(sep)
	fmt.Println("  === RAG AI (Frasier Database) ===")
	fmt.Println(sep)
	fmt.Println()
	fmt.Println(ragAnswer)

	fmt.Println()
	fmt.Println(sep)
	fmt.Println("  === EVALUATION ===")
	fmt.Println(sep)
	fmt.Println()
	fmt.Println(evaluation)
	fmt.Println()
	fmt.Println(sep)
}
