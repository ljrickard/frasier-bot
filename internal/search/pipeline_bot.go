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

	"go.opentelemetry.io/otel"
)

const maxHistory = 10

type RAGResult struct {
	Answer         string
	Scores         map[string]any
	EvalErr        error
	Contexts       []models.SearchResult // The top K sent to the LLM
	RawContexts    []models.SearchResult // The full 50 from pgvector
	Reformulated   string
	Classification string
	FetchK         int
	FinalK         int
	EpisodeLimit   int
	PreRerankCount int
}

func RunRAGPipeline(ctx context.Context, db *database.DB, cfg *config.RAGConfig, aiSvc *ai.Service, query string) (RAGResult, error) {
	// 1. Grab the tracer and start the parent span
	tracer := otel.Tracer("frasier-rag-pipeline")
	ctx, span := tracer.Start(ctx, "Search.RunRAGPipeline")
	defer span.End()

	var res RAGResult
	var searchResultsForAI []models.SearchResult

	slog.Info("🚀 [Step 0/6] RAG Pipeline Initiated", "query", query)

	if cfg.UseRAG {
		// Step 1: Query Expansion
		res.Reformulated = query
		if cfg.UseQueryExpansion {
			slog.Info("🔍 [Step 1/6] Expanding Query...")

			ctx, expSpan := tracer.Start(ctx, "AI.ExpandQuery")
			ref, err := aiSvc.ExpandQuery(ctx, query)
			expSpan.End()

			if err == nil {
				res.Reformulated = ref
				slog.Info("✅ [Step 1/6] Query Expanded", "reformulated", res.Reformulated)
			}
		} else {
			slog.Info("⏩ [Step 1/6] Query Expansion Skipped")
		}

		// Step 2: Classification (Switchboard)
		res.FetchK = cfg.FetchK
		res.FinalK = cfg.FinalK
		res.EpisodeLimit = 3

		if cfg.UseQueryClassification {
			slog.Info("🧠 [Step 2/6] Classifying Query Intent...")

			ctx, classSpan := tracer.Start(ctx, "AI.ClassifyQuery")
			classification, err := aiSvc.ClassifyQuery(ctx, res.Reformulated)
			classSpan.End()

			if err != nil {
				classification = "GENERAL"
			}
			res.Classification = classification
			slog.Info("✅ [Step 2/6] Query Classified", "intent", res.Classification)

			// Proportional Scaling for Specific Queries
			if classification == "SPECIFIC" {
				originalFetch := res.FetchK
				originalFinal := res.FinalK

				res.FetchK = int(float64(cfg.FetchK) * cfg.SpecificScaleFetch)
				res.FinalK = int(float64(cfg.FinalK) * cfg.SpecificScaleFinal)

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

		slog.Info("⚙️ [Step 2/6] Active Configuration Locked",
			"intent", res.Classification,
			"fetch_k", res.FetchK,
			"final_k", res.FinalK,
			"episode_limit", res.EpisodeLimit,
		)

		// Step 3: Embeddings
		slog.Info("🧮 [Step 3/6] Generating Embeddings via Vertex AI...")

		ctx, embedSpan := tracer.Start(ctx, "VertexAI.EmbedQuery")
		queryEmbedding, err := aiSvc.EmbedQuery(ctx, res.Reformulated)
		embedSpan.End()

		if err != nil {
			return res, fmt.Errorf("embedding error: %w", err)
		}

		// Step 4: Vector Search
		slog.Info("🔎 [Step 4/6] Executing Vector Search in pgvector...", "fetch_limit", res.FetchK)

		ctx, dbSpan := tracer.Start(ctx, "Postgres.pgvectorSearch")
		if cfg.UseEpisodeLimit {
			searchResultsForAI, err = db.SearchChunksWithEpisodeLimit(ctx, queryEmbedding, res.FetchK, res.EpisodeLimit)
		} else {
			searchResultsForAI, err = db.SearchChunks(ctx, queryEmbedding, res.FetchK)
		}
		dbSpan.End()

		if err != nil || len(searchResultsForAI) == 0 {
			return res, fmt.Errorf("no relevant transcripts found")
		}
		slog.Info("✅ [Step 4/6] Transcripts Retrieved", "chunks_found", len(searchResultsForAI))

		res.RawContexts = make([]models.SearchResult, len(searchResultsForAI))
		copy(res.RawContexts, searchResultsForAI)

		// Step 5: Reranking
		res.PreRerankCount = len(searchResultsForAI)
		if cfg.UseReranker {
			slog.Info("⚖️ [Step 5/6] Reranking Results via Cross-Encoder...", "backend", cfg.RerankerBackend)

			ctx, rankSpan := tracer.Start(ctx, "CrossEncoder.Rerank")
			reranked, err := aiSvc.RerankChunks(ctx, cfg.RerankerBackend, res.Reformulated, searchResultsForAI, res.FinalK)
			rankSpan.End()

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

	// Build context string
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

	ctx, genSpan := tracer.Start(ctx, "Gemini.GenerateFinalAnswer")
	ragAnswer, err := aiSvc.GenerateAnswer(ctx, query, searchResultsForAI, cfg.UsePersona)
	genSpan.End()

	if err != nil {
		return res, fmt.Errorf("generation error: %w", err)
	}
	res.Answer = ragAnswer
	slog.Info("🏁 Pipeline Complete", "answer_length", len(res.Answer))

	return res, nil
}
