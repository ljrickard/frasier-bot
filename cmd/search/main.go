package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"omnicorp-analyst/internal/ai"
	"omnicorp-analyst/internal/database"
	"omnicorp-analyst/internal/embeddings"
	"omnicorp-analyst/internal/ui"
)

const maxHistory = 10

func main() {
	ctx := context.Background()

	logger := log.New(os.Stderr, "", log.Ldate|log.Ltime|log.Lshortfile)

	// Check for --compare flag
	compareMode := false
	for _, arg := range os.Args[1:] {
		if arg == "--compare" {
			compareMode = true
		}
	}

	if err := embeddings.Preflight(); err != nil {
		logger.Fatalf("Embedding service preflight check failed: %v", err)
	}

	db, err := database.New(ctx)
	if err != nil {
		logger.Fatalf("Failed to connect to database: %v", err)
	}
	defer db.Close()

	sep := strings.Repeat("=", 80)

	fmt.Println()
	fmt.Println(sep)
	fmt.Println("  🎧 Frasier Chat Session Started")
	fmt.Println("  Ask me anything about Frasier! Type 'exit' or 'quit' to end.")
	if compareMode {
		fmt.Println("  [Compare Mode ON: Vanilla AI + Evaluation enabled]")
	}
	fmt.Println(sep)
	fmt.Println()

	var chatHistory []string
	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Print("You: ")
		if !scanner.Scan() {
			break
		}

		query := strings.TrimSpace(scanner.Text())
		if query == "" {
			continue
		}

		lower := strings.ToLower(query)
		if lower == "exit" || lower == "quit" {
			fmt.Println()
			fmt.Println("Goodbye! Thanks for chatting about Frasier.")
			break
		}

		// Start spinner for the entire processing pipeline
		spin := ui.NewSpinner("Analyzing query...")
		spin.Start()

		// Step 1: Reformulate query using chat history
		reformulated, err := ai.ReformulateQuery(ctx, query, chatHistory)
		if err != nil {
			logger.Printf("WARN: failed to reformulate query: %v", err)
			reformulated = query
		}

		// Step 2: Classify the reformulated query for Top-K
		spin.UpdateMessage("Classifying query...")
		classification, err := ai.ClassifyQuery(ctx, reformulated)
		if err != nil {
			logger.Printf("WARN: failed to classify query, defaulting to GENERAL: %v", err)
			classification = "GENERAL"
		}

		fetchK := 50
		perEpisodeLimit := 3
		finalK := 20
		if classification == "SPECIFIC" {
			fetchK = 30
			finalK = 10
		}

		// Step 3: Generate embedding for the reformulated query
		spin.UpdateMessage("Searching transcripts...")
		queryEmbedding, err := embeddings.GenerateQueryEmbedding(ctx, reformulated)
		if err != nil {
			spin.Stop()
			logger.Printf("WARN: failed to generate query embedding: %v", err)
			fmt.Println("\nSorry, I had trouble processing that query. Please try again.\n")
			continue
		}

		// Step 4: Search articles with diversity capping (children) — fetch wide
		results, err := db.SearchArticlesDiverse(ctx, queryEmbedding, fetchK, perEpisodeLimit)
		if err != nil {
			spin.Stop()
			logger.Printf("WARN: failed to search articles: %v", err)
			fmt.Println("\nSorry, I had trouble searching the database. Please try again.\n")
			continue
		}

		if len(results) == 0 {
			spin.Stop()
			fmt.Println("\nI couldn't find any relevant transcripts for that question.\n")
			continue
		}

		// Step 5: Collect unique parent IDs and fetch parent chunks
		parentIDSet := make(map[int64]bool)
		var parentIDs []int64
		for _, r := range results {
			if r.ParentID != nil && !parentIDSet[*r.ParentID] {
				parentIDSet[*r.ParentID] = true
				parentIDs = append(parentIDs, *r.ParentID)
			}
		}

		var parentResults []database.SearchResult
		if len(parentIDs) > 0 {
			parents, err := db.GetParentChunksByIDs(ctx, parentIDs)
			if err != nil {
				logger.Printf("WARN: failed to fetch parent chunks: %v", err)
			} else {
				for _, p := range parents {
					enrichedContent := fmt.Sprintf("[S%02dE%02d] %s", p.Season, p.Episode, p.Content)
					parentResults = append(parentResults, database.SearchResult{
						Title:   fmt.Sprintf("S%02dE%02d: %s", p.Season, p.Episode, p.EpisodeTitle),
						URL:     p.URL,
						Content: enrichedContent,
					})
				}
			}
		}

		searchResultsForAI := parentResults
		if len(searchResultsForAI) == 0 {
			searchResultsForAI = results
		}

		// Step 5b: Rerank chunks using LLM scoring
		spin.UpdateMessage("Reranking results...")
		preRerankCount := len(searchResultsForAI)
		reranked, err := ai.RerankChunks(ctx, reformulated, searchResultsForAI, finalK)
		if err != nil {
			logger.Printf("WARN: reranker failed, using original order: %v", err)
		} else {
			searchResultsForAI = reranked
		}

		// Step 6: Generate RAG answer (use original query for natural response)
		spin.UpdateMessage("Consulting the Crane brothers...")
		ragAnswer, err := ai.GenerateAnswer(ctx, query, searchResultsForAI)
		if err != nil {
			spin.Stop()
			logger.Printf("WARN: failed to generate RAG answer: %v", err)
			fmt.Println("\nSorry, I had trouble generating an answer. Please try again.\n")
			continue
		}

		spin.Stop()

		// Print debug info after spinner is cleared
		if reformulated != query {
			fmt.Printf("  \033[36mDEBUG: Reformulated -> %q\033[0m\n", reformulated)
		}
		fmt.Printf("  \033[36mDEBUG: Switchboard -> [%s, Fetch=%d, Final=%d, PerEpisode=%d]\033[0m\n", classification, fetchK, finalK, perEpisodeLimit)
		fmt.Printf("  \033[36mDEBUG: Reranker kept %d out of %d chunks\033[0m\n", len(searchResultsForAI), preRerankCount)

		// Count unique episodes
		uniqueEpisodes := make(map[string]bool)
		for _, r := range searchResultsForAI {
			uniqueEpisodes[r.Title] = true
		}
		fmt.Printf("  \033[36mDEBUG: Context -> %d chunks from %d unique episodes\033[0m\n", len(searchResultsForAI), len(uniqueEpisodes))

		// Display RAG answer
		fmt.Println()
		fmt.Println(sep)
		fmt.Printf("  === RAG AI (Frasier Database) ===\n")
		fmt.Println(sep)
		fmt.Println()
		fmt.Println(ragAnswer)
		fmt.Println()

		// Step 7: Compare mode (optional)
		if compareMode {
			vanillaAnswer, err := ai.GenerateVanillaAnswer(ctx, query)
			if err != nil {
				logger.Printf("WARN: failed to generate vanilla answer: %v", err)
				vanillaAnswer = "(Vanilla answer unavailable)"
			}

			fmt.Println(sep)
			fmt.Println("  === VANILLA AI (No Database) ===")
			fmt.Println(sep)
			fmt.Println()
			fmt.Println(vanillaAnswer)
			fmt.Println()

			evaluation, err := ai.EvaluateAnswers(ctx, query, vanillaAnswer, ragAnswer)
			if err != nil {
				logger.Printf("WARN: failed to evaluate answers: %v", err)
				evaluation = "(Evaluation unavailable)"
			}

			fmt.Println(sep)
			fmt.Println("  === EVALUATION ===")
			fmt.Println(sep)
			fmt.Println()
			fmt.Println(evaluation)
			fmt.Println()
		}

		fmt.Println(sep)
		fmt.Println()

		// Step 8: Update chat history
		chatHistory = append(chatHistory, fmt.Sprintf("User: %s", query))
		chatHistory = append(chatHistory, fmt.Sprintf("Assistant: %s", ragAnswer))

		// Keep only the last maxHistory entries
		if len(chatHistory) > maxHistory*2 {
			chatHistory = chatHistory[len(chatHistory)-maxHistory*2:]
		}
	}
}
