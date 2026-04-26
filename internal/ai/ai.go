package ai

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"frasier-bot/internal/ai/gemini"
	"frasier-bot/internal/models"

	"google.golang.org/genai"
)

func init() {
	// Redirect default logger (used by SDKs) to stderr to keep stdout clean
	log.SetOutput(os.Stderr)
}

// GenerateAnswer dynamically switches between strict RAG and Vanilla depending on context length
func GenerateAnswer(ctx context.Context, query string, chunks []models.SearchResult, usePersona bool) (string, error) {
	client, err := gemini.GetClient(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to create genai client: %w", err)
	}

	var prompt string

	if len(chunks) == 0 {
		// VANILLA BASELINE: If we have no context chunks, be a helpful expert using internal knowledge.
		if usePersona {
			prompt = fmt.Sprintf(promptPersonaVanilla, query)
		} else {
			prompt = fmt.Sprintf(promptStandardVanilla, query)
		}
	} else {
		// RAG MODE: If we have chunks, build the context and enforce strict adherence.
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

	temperature := float32(0.2)
	resp, err := gemini.CallWithRetry(ctx, func() (*genai.GenerateContentResponse, error) {
		return client.Models.GenerateContent(ctx, gemini.GeminiModel, genai.Text(prompt), &genai.GenerateContentConfig{
			Temperature: &temperature,
		})
	})
	if err != nil {
		return "", fmt.Errorf("failed to generate content: %w", err)
	}

	answer := extractText(resp)
	if answer == "" {
		return "", fmt.Errorf("empty response from model")
	}

	return answer, nil
}

func ClassifyQuery(ctx context.Context, query string) (string, error) {
	client, err := gemini.GetClient(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to create genai client: %w", err)
	}

	prompt := fmt.Sprintf(promptClassify, query)
	temperature := float32(0.0)

	resp, err := gemini.CallWithRetry(ctx, func() (*genai.GenerateContentResponse, error) {
		return client.Models.GenerateContent(ctx, gemini.GeminiModel, genai.Text(prompt), &genai.GenerateContentConfig{
			Temperature: &temperature,
		})
	})
	if err != nil {
		return "", fmt.Errorf("failed to classify query: %w", err)
	}

	result := strings.TrimSpace(extractText(resp))
	result = strings.ToUpper(result)

	if strings.Contains(result, "SPECIFIC") {
		return "SPECIFIC", nil
	}
	return "GENERAL", nil
}

func ExpandQuery(ctx context.Context, query string, history []string) (string, error) {
	historyText := "(No prior conversation)"
	if len(history) > 0 {
		var hb strings.Builder
		for _, h := range history {
			hb.WriteString(h)
			hb.WriteString("\n")
		}
		historyText = hb.String()
	}

	client, err := gemini.GetClient(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to create genai client: %w", err)
	}

	prompt := fmt.Sprintf(promptReformulate, historyText, query)
	temperature := float32(0.0)

	resp, err := gemini.CallWithRetry(ctx, func() (*genai.GenerateContentResponse, error) {
		return client.Models.GenerateContent(ctx, gemini.GeminiModel, genai.Text(prompt), &genai.GenerateContentConfig{
			Temperature: &temperature,
		})
	})
	if err != nil {
		return "", fmt.Errorf("failed to reformulate query: %w", err)
	}

	result := strings.TrimSpace(extractText(resp))
	if result == "" {
		return query, nil
	}
	return result, nil
}
