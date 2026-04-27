package search

import (
	"context"
	"fmt"
	"frasier-bot/internal/ai"
	"frasier-bot/internal/config"
	"frasier-bot/internal/database"
	"frasier-bot/internal/embeddings"
	"frasier-bot/internal/models"
	"log/slog"
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

func RunRAGPipeline(ctx context.Context, db *database.DB, cfg *config.RAGConfig, aiSvc *ai.Service, query string) (RAGResult, error) {
	var res RAGResult
	var searchResultsForAI []models.SearchResult

	// Step 1: Handle Retrieval if RAG is enabled
	if cfg.UseRAG {
		slog.Debug("Analyzing query...")
		res.Reformulated = query
		if cfg.UseQueryExpansion {
			ref, err := aiSvc.ExpandQuery(ctx, query)
			if err == nil {
				res.Reformulated = ref
			}
		}

		// Step 2: Switchboard Logic
		res.FetchK = 50
		res.EpisodeLimit = 3
		res.FinalK = 12 // Reduced from 20 to prevent 503 timeouts

		if cfg.UseQueryClassification {
			slog.Debug("Classifying query...")
			classification, err := aiSvc.ClassifyQuery(ctx, res.Reformulated)
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
		slog.Debug("Searching transcripts...")
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
			slog.Debug(fmt.Sprintf("Reranking results via %s...", cfg.RerankerBackend))
			reranked, err := aiSvc.RerankChunks(ctx, cfg.RerankerBackend, res.Reformulated, searchResultsForAI, res.FinalK)
			if err == nil {
				searchResultsForAI = reranked
			} else {
				slog.Warn("Reranking failed, falling back to original search order", "error", err)
				if len(searchResultsForAI) > res.FinalK {
					searchResultsForAI = searchResultsForAI[:res.FinalK]
				}
			}
		} else if len(searchResultsForAI) > res.FinalK {
			searchResultsForAI = searchResultsForAI[:res.FinalK]
		}
		res.Contexts = searchResultsForAI

	} else {
		slog.Debug("Bypassing RAG (Vanilla AI mode)...")
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

	slog.Debug("Consulting the Crane brothers...")
	// We pass the searchResultsForAI to the AI
	ragAnswer, err := aiSvc.GenerateAnswer(ctx, query, searchResultsForAI, cfg.UsePersona)
	if err != nil {
		return res, fmt.Errorf("generation error: %w", err)
	}
	res.Answer = ragAnswer

	return res, nil
}
