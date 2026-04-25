package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"frasier-bot/internal/ai"
	"frasier-bot/internal/config"
	"frasier-bot/internal/database"
	"frasier-bot/internal/embeddings"
	"frasier-bot/internal/models"
	"frasier-bot/internal/ui"
)

const maxHistory = 10

func startChatLoop(ctx context.Context, db *database.DB, cfg *config.RAGConfig, logger *log.Logger) {
	sep := strings.Repeat("=", 80)
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

		spin := ui.NewSpinner("Analyzing query...")
		spin.Start()

		// Step 1: Reformulate query using chat history
		var reformulated string
		var err error
		if cfg.UseExpansion {
			reformulated, err = ai.ReformulateQuery(ctx, query, chatHistory)
			if err != nil {
				logger.Printf("WARN: failed to reformulate query: %v", err)
				reformulated = query
			}
		} else {
			reformulated = query
		}

		// Step 2: Classify the reformulated query for Top-K
		var classification string
		fetchK := 50
		perEpisodeLimit := 3
		finalK := 20

		if cfg.UseSwitchboard {
			spin.UpdateMessage("Classifying query...")
			classification, err = ai.ClassifyQuery(ctx, reformulated)
			if err != nil {
				logger.Printf("WARN: failed to classify query, defaulting to GENERAL: %v", err)
				classification = "GENERAL"
			}
			if classification == "SPECIFIC" {
				fetchK = 30
				finalK = 10
			}
		} else {
			classification = "OFF"
			fetchK = 5
			finalK = 5
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

		// Step 4: Search chunks (children) — fetch wide
		var results []models.SearchResult
		if cfg.UseDiversity {
			results, err = db.SearchChunksDiverse(ctx, queryEmbedding, fetchK, perEpisodeLimit)
		} else {
			results, err = db.SearchChunks(ctx, queryEmbedding, fetchK)
		}
		if err != nil {
			spin.Stop()
			logger.Printf("WARN: failed to search chunks: %v", err)
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

		var parentResults []models.SearchResult
		if len(parentIDs) > 0 {
			parents, err := db.GetParentChunksByIDs(ctx, parentIDs)
			if err != nil {
				logger.Printf("WARN: failed to fetch parent chunks: %v", err)
			} else {
				for _, p := range parents {
					content := p.Content
					if cfg.UseMetadata {
						content = fmt.Sprintf("[S%02dE%02d] %s", p.Season, p.Episode, content)
					}
					parentResults = append(parentResults, models.SearchResult{
						Title:   fmt.Sprintf("S%02dE%02d: %s", p.Season, p.Episode, p.EpisodeTitle),
						URL:     p.URL,
						Content: content,
					})
				}
			}
		}

		searchResultsForAI := parentResults
		if len(searchResultsForAI) == 0 {
			searchResultsForAI = results
		}

		// Step 5b: Rerank chunks using LLM scoring
		preRerankCount := len(searchResultsForAI)
		if cfg.UseReranker {
			spin.UpdateMessage("Reranking results...")
			reranked, err := ai.RerankChunks(ctx, reformulated, searchResultsForAI, finalK)
			if err != nil {
				logger.Printf("WARN: reranker failed, using original order: %v", err)
			} else {
				searchResultsForAI = reranked
			}
		} else if len(searchResultsForAI) > finalK {
			searchResultsForAI = searchResultsForAI[:finalK]
		}

		// Step 6: Generate RAG answer
		spin.UpdateMessage("Consulting the Crane brothers...")
		ragAnswer, err := ai.GenerateAnswer(ctx, query, searchResultsForAI, cfg.UsePersona)
		if err != nil {
			spin.Stop()
			logger.Printf("WARN: failed to generate RAG answer: %v", err)
			fmt.Println("\nSorry, I had trouble generating an answer. Please try again.\n")
			continue
		}

		// ==============================================================
		// Step 7: LIVE RAG EVALUATION
		// ==============================================================
		var contextStrings []string
		for _, r := range searchResultsForAI {
			contextStrings = append(contextStrings, r.Content)
		}
		spin.UpdateMessage("Evaluating answer quality...")
		scores, evalErr := EvaluateInteraction(query, contextStrings, ragAnswer)
		spin.Stop()
		// ==============================================================

		// Print Debug Info
		if cfg.Debug {
			if reformulated != query {
				fmt.Printf("  \033[36mDEBUG: Reformulated -> %q\033[0m\n", reformulated)
			}
			fmt.Printf("  \033[36mDEBUG: Switchboard -> [%s, Fetch=%d, Final=%d, PerEpisode=%d]\033[0m\n", classification, fetchK, finalK, perEpisodeLimit)

			if cfg.UseReranker {
				fmt.Printf("  \033[36mDEBUG: Reranker kept %d out of %d chunks\033[0m\n", len(searchResultsForAI), preRerankCount)
			} else {
				fmt.Printf("  \033[36mDEBUG: Reranker -> [DISABLED] Bypassed\033[0m\n")
			}

			uniqueEpisodes := make(map[string]bool)
			for _, r := range searchResultsForAI {
				uniqueEpisodes[r.Title] = true
			}
			fmt.Printf("  \033[36mDEBUG: Context -> %d chunks from %d unique episodes\033[0m\n", len(searchResultsForAI), len(uniqueEpisodes))
		}

		// Print the RAG Answer
		fmt.Println()
		fmt.Println(sep)
		fmt.Printf("  === RAG AI (Frasier Database) ===\n")
		fmt.Println(sep)
		fmt.Println()
		fmt.Println(ragAnswer)
		fmt.Println()

		// Print the Live Evaluation Scores
		if evalErr != nil {
			// Print a warning but don't crash if the evaluation server is down
			logger.Printf("  \033[33mWARN: Live evaluation skipped or failed: %v\033[0m\n", evalErr)
		} else if len(scores) > 0 {
			fmt.Println(sep)
			fmt.Println("  📊 EVALUATION SCORES")
			fmt.Println(sep)
			for k, v := range scores {
				// Format floats to 4 decimal places for a cleaner look
				if floatVal, ok := v.(float64); ok {
					fmt.Printf("  • \033[32m%-15s\033[0m : %.4f\n", k, floatVal)
				} else {
					fmt.Printf("  • \033[32m%-15s\033[0m : %v\n", k, v)
				}
			}
			fmt.Println()
		}

		// Step 8: Update chat history
		chatHistory = append(chatHistory, fmt.Sprintf("User: %s", query))
		chatHistory = append(chatHistory, fmt.Sprintf("Assistant: %s", ragAnswer))

		if len(chatHistory) > maxHistory*2 {
			chatHistory = chatHistory[len(chatHistory)-maxHistory*2:]
		}
	}
}
