package ai

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"frasier-bot/internal/crossencoder"
	"frasier-bot/internal/llm"
	"frasier-bot/internal/models"
	"frasier-bot/tracing"
)

const (
	defaultTemperature = float32(0.0)
)

type Reranker interface {
	Rerank(ctx context.Context, query string, passages []string) ([]crossencoder.RerankResponse, error)
}

type Service struct {
	LLM      llm.Client
	Reranker Reranker
}

func NewService(llm llm.Client, reranker Reranker) *Service {
	return &Service{
		LLM:      llm,
		Reranker: reranker,
	}
}

func (s *Service) GenerateAnswer(ctx context.Context, query string, chunks []models.SearchResult, usePersona bool) (string, error) {
	traceID := tracing.GetTraceID(ctx)
	var prompt string
	temperature := defaultTemperature

	slog.Debug("🤖 [Service] Constructing context-grounded prompt compilation pipeline", "use_persona", usePersona, "chunks_count", len(chunks), "trace_id", traceID)

	if len(chunks) == 0 {
		if usePersona {
			prompt = fmt.Sprintf(promptPersonaVanilla, query)
			temperature = 0.2
		} else {
			prompt = fmt.Sprintf(promptStandardVanilla, query)
		}
	} else {
		var contextBuilder strings.Builder
		for i, c := range chunks {
			contextBuilder.WriteString(fmt.Sprintf("Chunk %d:\n", i+1))
			contextBuilder.WriteString(fmt.Sprintf("Episode: %s [S%02dE%02d]\n", c.Title, c.Season, c.Episode))
			contextBuilder.WriteString(fmt.Sprintf("URL: %s\n", c.URL))
			contextBuilder.WriteString(fmt.Sprintf("Content: %s\n", c.Content))
			contextBuilder.WriteString(fmt.Sprintf("Similarity: %.4f\n\n", c.Similarity))
		}

		if usePersona {
			prompt = fmt.Sprintf(promptPersonaRAG, contextBuilder.String(), query)
			temperature = 0.2
		} else {
			prompt = fmt.Sprintf(promptStandardRAG, contextBuilder.String(), query)
		}
	}

	answer, err := s.LLM.GenerateText(ctx, prompt, temperature)
	if err != nil {
		return "", fmt.Errorf("failed to generate answer: %w", err)
	}

	return answer, nil
}

func (s *Service) ClassifyQuery(ctx context.Context, query string) (string, error) {
	traceID := tracing.GetTraceID(ctx)
	slog.Debug("🧠 [Service] Dispatching classification request to intent switchboard", "trace_id", traceID)

	prompt := fmt.Sprintf(promptClassify, query)

	response, err := s.LLM.GenerateText(ctx, prompt, defaultTemperature)
	if err != nil {
		return "", fmt.Errorf("failed to classify query: %w", err)
	}

	result := strings.ToUpper(strings.TrimSpace(response))
	if strings.Contains(result, "SPECIFIC") {
		return "SPECIFIC", nil
	}
	return "GENERAL", nil
}

func (s *Service) ExpandQuery(ctx context.Context, query string) (string, error) {
	traceID := tracing.GetTraceID(ctx)
	slog.Debug("🔍 [Service] Transforming query parameters via reformulation rules", "trace_id", traceID)

	prompt := fmt.Sprintf(promptReformulate, query)

	response, err := s.LLM.GenerateText(ctx, prompt, defaultTemperature)
	if err != nil {
		return "", fmt.Errorf("failed to reformulate query: %w", err)
	}

	result := strings.TrimSpace(response)
	if result == "" {
		return query, nil
	}
	return result, nil
}

func (s *Service) EmbedQuery(ctx context.Context, query string) ([]float32, error) {
	return s.LLM.EmbedText(ctx, query)
}
