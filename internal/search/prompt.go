package search

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"frasier-bot/internal/config"
	"frasier-bot/internal/database"
	"frasier-bot/internal/ui"
)

// StartChatLoop handles the terminal UI, reading input, and printing the results cleanly.
func StartChatLoop(ctx context.Context, db *database.DB, cfg *config.RAGConfig, logger *log.Logger) {
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

		// Hook the spinner's update method into our decoupled RAG pipeline
		statusCb := func(msg string) {
			spin.UpdateMessage(msg)
		}

		// Execute the Core Pipeline
		res, err := RunRAGPipeline(ctx, db, cfg, logger, query, chatHistory, statusCb)
		spin.Stop()

		// Error Handling
		if err != nil {
			if err.Error() == "no relevant transcripts found" {
				fmt.Print("\nI couldn't find any relevant transcripts for that question.\n")
			} else {
				fmt.Print("\nSorry, I had trouble processing that query. Please try again.\n")
			}
			continue
		}

		// Print Debug Info
		if cfg.Debug {
			if res.Reformulated != query {
				fmt.Printf("  \033[36mDEBUG: Reformulated -> %q\033[0m\n", res.Reformulated)
			}
			fmt.Printf("  \033[36mDEBUG: Switchboard -> [%s, Fetch=%d, Final=%d, PerEpisode=%d]\033[0m\n", res.Classification, res.FetchK, res.FinalK, res.PerEpisodeLimit)

			if cfg.UseReranker {
				fmt.Printf("  \033[36mDEBUG: Reranker kept %d out of %d chunks\033[0m\n", len(res.Contexts), res.PreRerankCount)
			} else {
				fmt.Printf("  \033[36mDEBUG: Reranker -> [DISABLED] Bypassed\033[0m\n")
			}

			uniqueEpisodes := make(map[string]bool)
			for _, r := range res.Contexts {
				uniqueEpisodes[r.Title] = true
			}
			fmt.Printf("  \033[36mDEBUG: Context -> %d chunks from %d unique episodes\033[0m\n", len(res.Contexts), len(uniqueEpisodes))
		}

		// Print the RAG Answer
		fmt.Println()
		fmt.Println(sep)
		fmt.Printf("  === RAG AI (Frasier Database) ===\n")
		fmt.Println(sep)
		fmt.Println()
		fmt.Println(res.Answer)
		fmt.Println()

		// Print the Live Evaluation Scores
		if res.EvalErr != nil {
			logger.Printf("  \033[33mWARN: Live evaluation skipped or failed: %v\033[0m\n", res.EvalErr)
		} else if len(res.Scores) > 0 {
			fmt.Println(sep)
			fmt.Println("  📊 EVALUATION SCORES")
			fmt.Println(sep)
			for k, v := range res.Scores {
				if floatVal, ok := v.(float64); ok {
					fmt.Printf("  • \033[32m%-15s\033[0m : %.4f\n", k, floatVal)
				} else {
					fmt.Printf("  • \033[32m%-15s\033[0m : %v\n", k, v)
				}
			}
			fmt.Println()
		}

		// Update chat history
		chatHistory = append(chatHistory, fmt.Sprintf("User: %s", query))
		chatHistory = append(chatHistory, fmt.Sprintf("Assistant: %s", res.Answer))

		if len(chatHistory) > maxHistory*2 {
			chatHistory = chatHistory[len(chatHistory)-maxHistory*2:]
		}
	}
}
