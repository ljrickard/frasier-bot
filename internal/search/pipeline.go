package search

import (
	"context"
	"fmt"
	"frasier-bot/internal/ai"
	"frasier-bot/internal/config"
	"frasier-bot/internal/database"
	"frasier-bot/internal/embeddings"
	"frasier-bot/internal/models"
	"log"
	"strings"
)

const maxHistory = 10

type RAGResult struct {
	Answer         string
	Scores         map[string]any
	EvalErr        error
	Contexts       []models.SearchResult
	Reformulated   string
	Classification string
	FetchK         int
	FinalK         int
	EpisodeLimit   int
	PreRerankCount int
}

func RunRAGPipeline(
	ctx context.Context,
	db *database.DB,
	cfg *config.RAGConfig,
	logger *log.Logger,
	query string,
	chatHistory []string,
	statusCallback func(string),
) (RAGResult, error) {

	var res RAGResult
	var searchResultsForAI []models.SearchResult

	updateStatus := func(msg string) {
		if statusCallback != nil {
			statusCallback(msg)
		}
	}

	// Step 1: Handle Retrieval if RAG is enabled
	if cfg.UseRAG {
		updateStatus("Analyzing query...")
		res.Reformulated = query
		if cfg.UseExpansion {
			ref, err := ai.ExpandQuery(ctx, query, chatHistory)
			if err == nil {
				res.Reformulated = ref
			}
		}

		// Step 2: Switchboard Logic
		res.FetchK = 50
		res.EpisodeLimit = 3
		res.FinalK = 12 // Reduced from 20 to prevent 503 timeouts

		if cfg.UseQueryClassification {
			updateStatus("Classifying query...")
			classification, err := ai.ClassifyQuery(ctx, res.Reformulated)
			if err != nil {
				classification = "GENERAL"
			}
			res.Classification = classification
			if classification == "SPECIFIC" {
				res.FetchK = 30
				res.FinalK = 8
			}
		} else {
			res.Classification = "OFF"
			res.FetchK = 10
			res.FinalK = 5
		}

		// Step 3: Embeddings
		updateStatus("Searching transcripts...")
		queryEmbedding, err := embeddings.GenerateQueryEmbedding(ctx, res.Reformulated)
		if err != nil {
			return res, fmt.Errorf("embedding error: %w", err)
		}

		// Step 4: DB Search
		if cfg.UseEpisodeLimit {
			searchResultsForAI, err = db.SearchChunksWithEpisodeLimit(ctx, queryEmbedding, res.FetchK, res.EpisodeLimit)
		} else {
			searchResultsForAI, err = db.SearchChunks(ctx, queryEmbedding, res.FetchK)
		}
		if err != nil || len(searchResultsForAI) == 0 {
			return res, fmt.Errorf("no relevant transcripts found")
		}

		// Step 5: Reranking
		res.PreRerankCount = len(searchResultsForAI)
		if cfg.UseReranker {
			updateStatus(fmt.Sprintf("Reranking results via %s...", cfg.RerankerBackend))
			reranked, err := ai.RerankChunks(ctx, cfg.RerankerBackend, res.Reformulated, searchResultsForAI, res.FinalK)
			if err == nil {
				searchResultsForAI = reranked
			} else {
				logger.Printf("WARN: Reranking failed, falling back to original search order: %v", err)
				if len(searchResultsForAI) > res.FinalK {
					searchResultsForAI = searchResultsForAI[:res.FinalK]
				}
			}
		} else if len(searchResultsForAI) > res.FinalK {
			searchResultsForAI = searchResultsForAI[:res.FinalK]
		}
		res.Contexts = searchResultsForAI

	} else {
		updateStatus("Bypassing RAG (Vanilla AI mode)...")
		res.Classification = "VANILLA"
	}

	// Step 6: Final Augmentation & Generation
	// Use a strings.Builder to build the prompt context properly
	var contextBuilder strings.Builder
	var contextStrings []string

	for i, c := range searchResultsForAI {
		contextBuilder.WriteString(fmt.Sprintf("Chunk %d:\n", i+1))
		if cfg.UseMetadata {
			contextBuilder.WriteString(fmt.Sprintf("Episode: %s [S%02dE%02d]\n", c.Title, c.Season, c.Episode))
		}
		contextBuilder.WriteString(fmt.Sprintf("Content: %s\n\n", c.Content))
		contextStrings = append(contextStrings, c.Content)
	}

	updateStatus("Consulting the Crane brothers...")
	// We pass the searchResultsForAI to the AI
	ragAnswer, err := ai.GenerateAnswer(ctx, query, searchResultsForAI, cfg.UsePersona)
	if err != nil {
		return res, fmt.Errorf("generation error: %w", err)
	}
	res.Answer = ragAnswer

	if cfg.UseEval {
		// Step 7: LIVE RAG EVALUATION
		updateStatus("Evaluating answer quality...")

		// THE REAL FIX: Only inject dummy context if the user EXPLICITLY requested a baseline run.
		if !cfg.UseRAG {
			contextStrings = []string{"[NO DATABASE CONTEXT PROVIDED FOR THIS BASELINE RUN]"}
		} else if len(contextStrings) == 0 {
			// If RAG is ON but we have no contexts, that is a real error we should not mask!
			logger.Printf("WARN: RAG is enabled but no context chunks were generated.")
			return res, fmt.Errorf("evaluation failed: no context available for active RAG run")
		}

		scores, evalErr := EvaluateInteraction(query, contextStrings, ragAnswer)
		res.Scores = scores
		res.EvalErr = evalErr
	}
	return res, nil
}
