package gemini

import (
	"context"
	"fmt"
	"log/slog"
	"math/rand"
	"strings"
	"time"

	"google.golang.org/genai"
)

type RetryConfig struct {
	MaxRetries int
	BaseDelay  time.Duration
}

type Config struct {
	ProjectID      string
	Location       string
	Model          string
	EmbeddingModel string // Add this!
	Retry          RetryConfig
}

type Client struct {
	rawClient      *genai.Client
	modelName      string
	embeddingModel string
	retryCfg       RetryConfig
}

func NewClient(ctx context.Context, cfg Config) (*Client, error) {
	c, err := genai.NewClient(ctx, &genai.ClientConfig{
		Project:  cfg.ProjectID,
		Location: cfg.Location,
		Backend:  genai.BackendVertexAI,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to initialize genai client: %w", err)
	}

	if cfg.Retry.MaxRetries > 0 && cfg.Retry.BaseDelay == 0 {
		cfg.Retry.BaseDelay = 2 * time.Second
	}

	return &Client{
		rawClient:      c,
		modelName:      cfg.Model,
		embeddingModel: cfg.EmbeddingModel,
		retryCfg:       cfg.Retry,
	}, nil
}

// GenerateText (Unchanged logic, just using the new generic executeWithRetry)
func (c *Client) GenerateText(ctx context.Context, prompt string) (string, error) {
	temperature := float32(0.2)
	resp, err := executeWithRetry(ctx, c.retryCfg, func() (*genai.GenerateContentResponse, error) {
		return c.rawClient.Models.GenerateContent(ctx, c.modelName, genai.Text(prompt), &genai.GenerateContentConfig{
			Temperature: &temperature,
		})
	})

	if err != nil {
		return "", fmt.Errorf("failed to generate content: %w", err)
	}
	return c.extractText(resp), nil
}

// EmbedText - The new capability!
func (c *Client) EmbedText(ctx context.Context, text string) ([]float32, error) {
	if c.embeddingModel == "" {
		return nil, fmt.Errorf("embedding model is not configured")
	}

	resp, err := executeWithRetry(ctx, c.retryCfg, func() (*genai.EmbedContentResponse, error) {
		return c.rawClient.Models.EmbedContent(ctx, c.embeddingModel, genai.Text(text), nil)
	})

	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}
	if resp == nil || len(resp.Embeddings) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}

	return resp.Embeddings[0].Values, nil
}

// executeWithRetry [T any] handles exponential backoff for BOTH text and embeddings
func executeWithRetry[T any](ctx context.Context, cfg RetryConfig, fn func() (T, error)) (T, error) {
	var zero T

	for attempt := 0; attempt <= cfg.MaxRetries; attempt++ {
		resp, err := fn()
		if err == nil {
			return resp, nil
		}

		errStr := err.Error()
		is429 := strings.Contains(errStr, "429") || strings.Contains(errStr, "RESOURCE_EXHAUSTED")

		if !is429 || attempt == cfg.MaxRetries {
			return zero, err
		}

		delay := cfg.BaseDelay * (1 << uint(attempt))
		wait := delay + time.Duration(rand.Int63n(int64(delay)/4))
		slog.Warn("⚠️ Rate limited (429)", "retry", attempt+1, "wait", wait)

		select {
		case <-ctx.Done():
			return zero, ctx.Err()
		case <-time.After(wait):
		}
	}
	return zero, fmt.Errorf("max retries exceeded")
}

func (c *Client) extractText(resp *genai.GenerateContentResponse) string {
	if resp == nil || len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
		return ""
	}
	var parts []string
	for _, part := range resp.Candidates[0].Content.Parts {
		if part.Text != "" {
			parts = append(parts, part.Text)
		}
	}
	return strings.Join(parts, "\n")
}
