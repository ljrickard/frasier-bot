package tracing

import (
	"context"
	"fmt"

	"go.opentelemetry.io/otel/trace"
)

// GetTraceID extracts the string representation of the current Trace ID
func GetTraceID(ctx context.Context) string {
	spanCtx := trace.SpanContextFromContext(ctx)
	if spanCtx.HasTraceID() {
		return spanCtx.TraceID().String()
	}
	return ""
}

func SendStatusMessage(ch chan<- string, spanName string) {
	messages := map[string]string{
		"Pipeline.RunRAGStreamPipeline":    "RAG Pipeline Initiated",
		"AI.ExpandQuery":                   "Expanding Query via Gemini",
		"AI.ClassifyQuery":                 "Classifying Query Intent",
		"VertexAI.EmbedQuery":              "Generating Query Embeddings",
		"Postgres.pgvectorSearch":          "Executing pgvector Search",
		"CrossEncoder.Rerank":              "Reranking Results via Cross-Encoder",
		"Gemini.GenerateFinalAnswerStream": "Initializing text generation stream",
	}

	msg, exists := messages[spanName]
	if !exists {
		msg = spanName // Fallback cleanly to the tracer span name if not mapped
	}
	ch <- fmt.Sprintf("event: status\ndata: %s\n\n", msg)
}
