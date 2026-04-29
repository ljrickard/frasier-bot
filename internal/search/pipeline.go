package search

import (
	"context"
	"fmt"
	"frasier-bot/internal/ai"
	"frasier-bot/internal/config"
	"frasier-bot/internal/database"
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

	slog.Info("🚀 [Step 0/6] RAG Pipeline Initiated", "query", query)

	if cfg.UseRAG {
		// Step 1: Query Expansion
		res.Reformulated = query
		if cfg.UseQueryExpansion {
			slog.Info("🔍 [Step 1/6] Expanding Query...")
			ref, err := aiSvc.ExpandQuery(ctx, query)
			if err == nil {
				res.Reformulated = ref
				slog.Info("✅ [Step 1/6] Query Expanded", "reformulated", res.Reformulated)
			}
		} else {
			slog.Info("⏩ [Step 1/6] Query Expansion Skipped")
		}

		// Step 2: Classification (Switchboard)

		// 1. Establish the baseline (This is either the default 50/12, or your JSON overrides)
		res.FetchK = cfg.FetchK
		res.FinalK = cfg.FinalK
		res.EpisodeLimit = 3

		// 2. Classify the intent
		if cfg.UseQueryClassification {
			slog.Info("🧠 [Step 2/6] Classifying Query Intent...")
			classification, err := aiSvc.ClassifyQuery(ctx, res.Reformulated)
			if err != nil {
				classification = "GENERAL"
			}
			res.Classification = classification
			slog.Info("✅ [Step 2/6] Query Classified", "intent", res.Classification)

			// 3. Proportional Scaling for Specific Queries
			if classification == "SPECIFIC" {
				originalFetch := res.FetchK
				originalFinal := res.FinalK

				// Apply the configurable scale ratios!
				res.FetchK = int(float64(cfg.FetchK) * cfg.SpecificScaleFetch)
				res.FinalK = int(float64(cfg.FinalK) * cfg.SpecificScaleFinal)

				// Safety rails: ensure we never scale down to 0
				if res.FetchK < 1 {
					res.FetchK = 1
				}
				if res.FinalK < 1 {
					res.FinalK = 1
				}

				slog.Debug("📉 Dynamically scaled K values for SPECIFIC intent",
					"scale_fetch", cfg.SpecificScaleFetch,
					"scale_final", cfg.SpecificScaleFinal,
					"original_fetch", originalFetch,
					"new_fetch", res.FetchK,
					"original_final", originalFinal,
					"new_final", res.FinalK,
				)
			}
		} else {
			res.Classification = "OFF"
			res.FetchK = 10
			res.FinalK = 5
			slog.Info("⏩ [Step 2/6] Query Classification Skipped (Using hardcoded OFF defaults)")
		}

		// 4. Absolute Visibility: Log the exact variables about to be used
		slog.Info("⚙️ [Step 2/6] Active Configuration Locked",
			"intent", res.Classification,
			"fetch_k", res.FetchK,
			"final_k", res.FinalK,
			"episode_limit", res.EpisodeLimit,
		)

		// Step 3: Embeddings
		slog.Info("🧮 [Step 3/6] Generating Embeddings via Vertex AI...")
		queryEmbedding, err := aiSvc.EmbedQuery(ctx, res.Reformulated)
		if err != nil {
			return res, fmt.Errorf("embedding error: %w", err)
		}

		// Step 4: Vector Search
		slog.Info("🔎 [Step 4/6] Executing Vector Search in pgvector...", "fetch_limit", res.FetchK)
		if cfg.UseEpisodeLimit {
			searchResultsForAI, err = db.SearchChunksWithEpisodeLimit(ctx, queryEmbedding, res.FetchK, res.EpisodeLimit)
		} else {
			searchResultsForAI, err = db.SearchChunks(ctx, queryEmbedding, res.FetchK)
		}
		if err != nil || len(searchResultsForAI) == 0 {
			return res, fmt.Errorf("no relevant transcripts found")
		}
		slog.Info("✅ [Step 4/6] Transcripts Retrieved", "chunks_found", len(searchResultsForAI))

		// Step 5: Reranking
		res.PreRerankCount = len(searchResultsForAI)
		if cfg.UseReranker {
			slog.Info("⚖️ [Step 5/6] Reranking Results via Cross-Encoder...", "backend", cfg.RerankerBackend)
			reranked, err := aiSvc.RerankChunks(ctx, cfg.RerankerBackend, res.Reformulated, searchResultsForAI, res.FinalK)
			if err == nil {
				searchResultsForAI = reranked
				slog.Info("✅ [Step 5/6] Reranking Complete")
			} else {
				slog.Warn("⚠️ [Step 5/6] Reranking Failed, falling back to vector similarity", "error", err)
				if len(searchResultsForAI) > res.FinalK {
					searchResultsForAI = searchResultsForAI[:res.FinalK]
				}
			}
		} else {
			slog.Info("⏩ [Step 5/6] Reranking Skipped")
			if len(searchResultsForAI) > res.FinalK {
				searchResultsForAI = searchResultsForAI[:res.FinalK]
			}
		}
		res.Contexts = searchResultsForAI

	} else {
		slog.Info("⏩ [Steps 1-5 Skipped] Bypassing RAG (Vanilla AI mode)")
		res.Classification = "VANILLA"
	}

	// Build context
	var contextBuilder strings.Builder
	for i, c := range searchResultsForAI {
		contextBuilder.WriteString(fmt.Sprintf("Chunk %d:\n", i+1))
		if cfg.UseMetadata {
			contextBuilder.WriteString(fmt.Sprintf("Episode: %s [S%02dE%02d]\n", c.Title, c.Season, c.Episode))
		}
		contextBuilder.WriteString(fmt.Sprintf("Content: %s\n\n", c.Content))
	}

	// Step 6: Generation
	slog.Info("🤖 [Step 6/6] Generating Final LLM Answer via Gemini...")
	ragAnswer, err := aiSvc.GenerateAnswer(ctx, query, searchResultsForAI, cfg.UsePersona)
	if err != nil {
		return res, fmt.Errorf("generation error: %w", err)
	}
	res.Answer = ragAnswer
	slog.Info("🏁 Pipeline Complete", "answer_length", len(res.Answer))

	return res, nil
}
