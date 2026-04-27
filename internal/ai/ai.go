package ai

import (
	"context"
	"fmt"
	"strings"

	"frasier-bot/internal/crossencoder"
	"frasier-bot/internal/gemini"
	"frasier-bot/internal/models"
)

// Service encapsulates external clients
type Service struct {
	LLM     *gemini.Client
	Encoder *crossencoder.Client // Added Cross-Encoder
}

func NewService(llm *gemini.Client, encoder *crossencoder.Client) *Service {
	return &Service{
		LLM:     llm,
		Encoder: encoder,
	}
}

func (s *Service) GenerateAnswer(ctx context.Context, query string, chunks []models.SearchResult, usePersona bool) (string, error) {
	var prompt string

	if len(chunks) == 0 {
		if usePersona {
			prompt = fmt.Sprintf(promptPersonaVanilla, query)
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
		} else {
			prompt = fmt.Sprintf(promptStandardRAG, contextBuilder.String(), query)
		}
	}

	// The wrapper handles retries, extraction, and temperature internally now!
	answer, err := s.LLM.GenerateText(ctx, prompt)
	if err != nil {
		return "", fmt.Errorf("failed to generate answer: %w", err)
	}

	return answer, nil
}

func (s *Service) ClassifyQuery(ctx context.Context, query string) (string, error) {
	prompt := fmt.Sprintf(promptClassify, query)

	response, err := s.LLM.GenerateText(ctx, prompt)
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
	prompt := fmt.Sprintf(promptReformulate, query)

	response, err := s.LLM.GenerateText(ctx, prompt)
	if err != nil {
		return "", fmt.Errorf("failed to reformulate query: %w", err)
	}

	result := strings.TrimSpace(response)
	if result == "" {
		return query, nil
	}
	return result, nil
}
