package pipeline

import (
	"context"
	"encoding/json"
	"fmt"
	"frasier-bot/internal/ai"
	"frasier-bot/internal/config"
	"frasier-bot/internal/database"
	"frasier-bot/internal/models"
	"frasier-bot/tracing"
	"log/slog"

	"go.opentelemetry.io/otel"
)

func RunRAGStreamPipeline(ctx context.Context, db *database.DB, cfg *config.RAGConfig, aiSvc *ai.Service, query string) (<-chan string, error) {
	tracer := otel.Tracer("frasier-rag-pipeline")
	traceID := tracing.GetTraceID(ctx)
	outCh := make(chan string, 100)

	go func() {
		defer close(outCh)

		// Call the tracking packaging helper function systematically using the trace name
		tracing.SendStatusMessage(outCh, "Pipeline.RunRAGStreamPipeline")

		var res RAGResult
		var searchResultsForAI []models.SearchResult
		var err error

		slog.Info("🚀 [Pipeline Stream] RAG Process Started", "query", query, "trace_id", traceID)

		if cfg.UseRAG {
			res.Reformulated = query
			if cfg.UseQueryExpansion {
				tracing.SendStatusMessage(outCh, "AI.ExpandQuery")
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
				tracing.SendStatusMessage(outCh, "AI.ClassifyQuery")
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
			}

			tracing.SendStatusMessage(outCh, "VertexAI.EmbedQuery")
			ctx, embedSpan := tracer.Start(ctx, "VertexAI.EmbedQuery")
			queryEmbedding, err := aiSvc.EmbedQuery(ctx, res.Reformulated)
			embedSpan.End()
			if err != nil {
				outCh <- fmt.Sprintf("event: error\ndata: {\"message\": \"embedding failure: %v\"}\n\n", err)
				return
			}

			tracing.SendStatusMessage(outCh, "Postgres.pgvectorSearch")
			ctx, dbSpan := tracer.Start(ctx, "Postgres.pgvectorSearch")
			if cfg.UseEpisodeLimit {
				searchResultsForAI, err = db.SearchChunksWithEpisodeLimit(ctx, queryEmbedding, res.FetchK, res.EpisodeLimit)
			} else {
				searchResultsForAI, err = db.SearchChunks(ctx, queryEmbedding, res.FetchK)
			}
			dbSpan.End()
			if err != nil || len(searchResultsForAI) == 0 {
				outCh <- "event: error\ndata: {\"message\": \"no relevant transcripts found\"}\n\n"
				return
			}

			res.RawContexts = make([]models.SearchResult, len(searchResultsForAI))
			copy(res.RawContexts, searchResultsForAI)

			if cfg.UseReranker {
				tracing.SendStatusMessage(outCh, "CrossEncoder.Rerank")
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

		metaPayload, _ := json.Marshal(map[string]any{
			"reformulated": res.Reformulated,
			"contexts":     res.Contexts,
			"raw_contexts": res.RawContexts,
		})
		outCh <- fmt.Sprintf("event: metadata\ndata: %s\n\n", string(metaPayload))

		tracing.SendStatusMessage(outCh, "Gemini.GenerateFinalAnswerStream")
		slog.Info("🤖 [Pipeline Stream] Spawning generation channel...", "trace_id", traceID)

		ctx, genSpan := tracer.Start(ctx, "Gemini.GenerateFinalAnswerStream")
		streamChan, err := aiSvc.GenerateAnswerStream(ctx, query, searchResultsForAI, cfg.UsePersona)
		genSpan.End()
		if err != nil {
			outCh <- fmt.Sprintf("event: error\ndata: {\"message\": \"generation channel dropped: %v\"}\n\n", err)
			return
		}

		// Fixed w reference leak: Tokens stream down the unified channel wrapped in an SSE protocol layer
		for chunk := range streamChan {
			if chunk.Err != nil {
				outCh <- fmt.Sprintf("event: error\ndata: {\"message\": \"stream token read drop: %v\"}\n\n", chunk.Err)
				return
			}
			outCh <- fmt.Sprintf("data: %s\n\n", chunk.Text)
		}

		outCh <- "data: [DONE]\n\n"
	}()

	return outCh, nil
}
