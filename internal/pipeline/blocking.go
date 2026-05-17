package pipeline

import (
	"context"
	"fmt"
	"frasier-bot/internal/ai"
	"frasier-bot/internal/config"
	"frasier-bot/internal/database"
	"frasier-bot/internal/models"
	"frasier-bot/tracing"
	"log/slog"

	"go.opentelemetry.io/otel"
)

type RAGResult struct {
	Answer         string
	Scores         map[string]any
	EvalErr        error
	Contexts       []models.SearchResult
	RawContexts    []models.SearchResult
	Reformulated   string
	Classification string
	FetchK         int
	FinalK         int
	EpisodeLimit   int
	PreRerankCount int
}

func RunRAGPipeline(ctx context.Context, db *database.DB, cfg *config.RAGConfig, aiSvc *ai.Service, query string) (RAGResult, error) {
	tracer := otel.Tracer("frasier-rag-pipeline")
	ctx, span := tracer.Start(ctx, "Pipeline.RunRAGPipeline")
	defer span.End()

	traceID := tracing.GetTraceID(ctx)
	var res RAGResult
	var searchResultsForAI []models.SearchResult

	slog.Info("🚀 [Pipeline Sync] RAG Pipeline Initiated", "query", query, "trace_id", traceID)

	if cfg.UseRAG {
		res.Reformulated = query
		if cfg.UseQueryExpansion {
			slog.Info("🔍 [Step 1/6] Expanding Query...", "trace_id", traceID)
			ctx, expSpan := tracer.Start(ctx, "AI.ExpandQuery")
			ref, err := aiSvc.ExpandQuery(ctx, query)
			expSpan.End()
			if err == nil {
				res.Reformulated = ref
			}
		}

		res.FetchK = cfg.FetchK
		res.FinalK = cfg.FinalK
		res.EpisodeLimit = 3

		if cfg.UseQueryClassification {
			slog.Info("🧠 [Step 2/6] Classifying Query Intent...", "trace_id", traceID)
			ctx, classSpan := tracer.Start(ctx, "AI.ClassifyQuery")
			classification, err := aiSvc.ClassifyQuery(ctx, res.Reformulated)
			classSpan.End()

			if err != nil {
				classification = "GENERAL"
			}
			res.Classification = classification

			if classification == "SPECIFIC" {
				res.FetchK = int(float64(cfg.FetchK) * cfg.SpecificScaleFetch)
				res.FinalK = int(float64(cfg.FinalK) * cfg.SpecificScaleFinal)
			}
		} else {
			res.Classification = "OFF"
		}

		slog.Info("🧮 [Step 3/6] Generating Embeddings...", "trace_id", traceID)
		ctx, embedSpan := tracer.Start(ctx, "VertexAI.EmbedQuery")
		queryEmbedding, err := aiSvc.EmbedQuery(ctx, res.Reformulated)
		embedSpan.End()
		if err != nil {
			return res, fmt.Errorf("embedding error: %w", err)
		}

		slog.Info("🔎 [Step 4/6] Executing Vector Search...", "fetch_limit", res.FetchK, "trace_id", traceID)
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

		res.RawContexts = make([]models.SearchResult, len(searchResultsForAI))
		copy(res.RawContexts, searchResultsForAI)

		if cfg.UseReranker {
			slog.Info("⚖️ [Step 5/6] Reranking Results...", "backend", cfg.RerankerBackend, "trace_id", traceID)
			ctx, rankSpan := tracer.Start(ctx, "CrossEncoder.Rerank")
			reranked, err := aiSvc.RerankChunks(ctx, cfg.RerankerBackend, res.Reformulated, searchResultsForAI, res.FinalK)
			rankSpan.End()
			if err == nil {
				searchResultsForAI = reranked
			} else if len(searchResultsForAI) > res.FinalK {
				searchResultsForAI = searchResultsForAI[:res.FinalK]
			}
		} else if len(searchResultsForAI) > res.FinalK {
			searchResultsForAI = searchResultsForAI[:res.FinalK]
		}
		res.Contexts = searchResultsForAI
	}

	slog.Info("🤖 [Step 6/6] Generating Final LLM Answer...", "trace_id", traceID)
	ctx, genSpan := tracer.Start(ctx, "Gemini.GenerateFinalAnswer")
	ragAnswer, err := aiSvc.GenerateAnswer(ctx, query, searchResultsForAI, cfg.UsePersona)
	genSpan.End()
	if err != nil {
		return res, fmt.Errorf("generation error: %w", err)
	}

	res.Answer = ragAnswer
	slog.Info("🏁 [Pipeline Sync] Complete", "trace_id", traceID)
	return res, nil
}
