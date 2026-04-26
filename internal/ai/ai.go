package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"sort"
	"strings"
	"time"

	"frasier-bot/internal/models"

	"google.golang.org/genai"
)

const geminiModel = "gemini-2.5-flash"

func init() {
	// Redirect default logger (used by SDKs) to stderr to keep stdout clean
	log.SetOutput(os.Stderr)
}

func suppressSDKWarnings(f func()) {
	original := log.Writer()
	log.SetOutput(io.Discard)
	defer log.SetOutput(original)
	f()
}

func callWithRetry(ctx context.Context, fn func() (*genai.GenerateContentResponse, error)) (*genai.GenerateContentResponse, error) {
	maxRetries := 12
	baseDelay := 2 * time.Second

	for attempt := 0; attempt <= maxRetries; attempt++ {
		resp, err := fn()
		if err == nil {
			return resp, nil
		}

		errStr := err.Error()
		is429 := strings.Contains(errStr, "429") ||
			strings.Contains(errStr, "RESOURCE_EXHAUSTED") ||
			strings.Contains(errStr, "resource exhausted") ||
			strings.Contains(errStr, "Resource has been exhausted")

		if !is429 || attempt == maxRetries {
			return nil, err
		}

		delay := baseDelay * (1 << uint(attempt))
		jitter := time.Duration(rand.Int63n(int64(delay) / 4))
		wait := delay + jitter

		log.Printf("Rate limited (429) | err=[%v], retry %d/%d in %v...", errStr, attempt+1, maxRetries, wait)

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(wait):
		}
	}

	return nil, fmt.Errorf("max retries exceeded")
}

func getClient(ctx context.Context) (*genai.Client, error) {
	project := os.Getenv("GOOGLE_CLOUD_PROJECT")
	if project == "" {
		return nil, fmt.Errorf("GOOGLE_CLOUD_PROJECT environment variable is not set")
	}

	location := os.Getenv("GOOGLE_CLOUD_LOCATION")
	if location == "" {
		location = "us-central1"
	}

	var client *genai.Client
	var clientErr error
	// fmt.Printf("\n!!! RUNTIME PROJECT ID: %q !!!\n\n", project)
	suppressSDKWarnings(func() {
		client, clientErr = genai.NewClient(ctx, &genai.ClientConfig{
			Project:  project,
			Location: location,
			Backend:  genai.BackendVertexAI,
		})
	})
	return client, clientErr
}

// GenerateAnswer dynamically switches between strict RAG and Vanilla depending on context length
func GenerateAnswer(ctx context.Context, query string, chunks []models.SearchResult, usePersona bool) (string, error) {
	client, err := getClient(ctx)
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
	resp, err := callWithRetry(ctx, func() (*genai.GenerateContentResponse, error) {
		return client.Models.GenerateContent(ctx, geminiModel, genai.Text(prompt), &genai.GenerateContentConfig{
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
	client, err := getClient(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to create genai client: %w", err)
	}

	prompt := fmt.Sprintf(promptClassify, query)
	temperature := float32(0.0)

	resp, err := callWithRetry(ctx, func() (*genai.GenerateContentResponse, error) {
		return client.Models.GenerateContent(ctx, geminiModel, genai.Text(prompt), &genai.GenerateContentConfig{
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

func ReformulateQuery(ctx context.Context, query string, history []string) (string, error) {
	historyText := "(No prior conversation)"
	if len(history) > 0 {
		var hb strings.Builder
		for _, h := range history {
			hb.WriteString(h)
			hb.WriteString("\n")
		}
		historyText = hb.String()
	}

	client, err := getClient(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to create genai client: %w", err)
	}

	prompt := fmt.Sprintf(promptReformulate, historyText, query)
	temperature := float32(0.0)

	resp, err := callWithRetry(ctx, func() (*genai.GenerateContentResponse, error) {
		return client.Models.GenerateContent(ctx, geminiModel, genai.Text(prompt), &genai.GenerateContentConfig{
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

// GenerateVanillaAnswer is a legacy wrapper. You can probably remove this entirely
// since GenerateAnswer now handles Vanilla generation beautifully when len(chunks) == 0.
func GenerateVanillaAnswer(ctx context.Context, query string) (string, error) {
	client, err := getClient(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to create genai client: %w", err)
	}

	prompt := fmt.Sprintf(promptStandardVanilla, query)
	temperature := float32(0.2)

	resp, err := callWithRetry(ctx, func() (*genai.GenerateContentResponse, error) {
		return client.Models.GenerateContent(ctx, geminiModel, genai.Text(prompt), &genai.GenerateContentConfig{
			Temperature: &temperature,
		})
	})
	if err != nil {
		return "", fmt.Errorf("failed to generate vanilla answer: %w", err)
	}

	answer := extractText(resp)
	if answer == "" {
		return "", fmt.Errorf("empty response from model")
	}
	return answer, nil
}

func EvaluateAnswers(ctx context.Context, query, vanillaAnswer, ragAnswer string) (string, error) {
	client, err := getClient(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to create genai client: %w", err)
	}

	prompt := fmt.Sprintf(promptEvaluateCompare, query, vanillaAnswer, ragAnswer)
	temperature := float32(0.2)

	resp, err := callWithRetry(ctx, func() (*genai.GenerateContentResponse, error) {
		return client.Models.GenerateContent(ctx, geminiModel, genai.Text(prompt), &genai.GenerateContentConfig{
			Temperature: &temperature,
		})
	})
	if err != nil {
		return "", fmt.Errorf("failed to evaluate answers: %w", err)
	}

	result := extractText(resp)
	if result == "" {
		return "", fmt.Errorf("empty response from model")
	}
	return result, nil
}

type RerankChunk struct {
	Index    int
	Title    string
	URL      string
	Content  string
	ParentID *int64
	Score    float64
}

func RerankChunks(ctx context.Context, query string, chunks []models.SearchResult, topN int) ([]models.SearchResult, error) {
	if len(chunks) <= topN {
		return chunks, nil
	}

	client, err := getClient(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to create genai client: %w", err)
	}

	var chunkList strings.Builder
	for i, c := range chunks {
		content := c.Content
		if len(content) > 500 {
			content = content[:500] + "..."
		}
		chunkList.WriteString(fmt.Sprintf("Chunk %d:\n%s\n\n", i, content))
	}

	prompt := fmt.Sprintf(promptRerank, query, chunkList.String())
	temperature := float32(0.0)

	resp, err := callWithRetry(ctx, func() (*genai.GenerateContentResponse, error) {
		return client.Models.GenerateContent(ctx, geminiModel, genai.Text(prompt), &genai.GenerateContentConfig{
			Temperature: &temperature,
		})
	})
	if err != nil {
		return nil, fmt.Errorf("failed to rerank chunks: %w", err)
	}

	result := strings.TrimSpace(extractText(resp))

	type scoreEntry struct {
		ID    int     `json:"id"`
		Score float64 `json:"score"`
	}

	result = strings.TrimPrefix(result, "```json")
	result = strings.TrimPrefix(result, "```")
	result = strings.TrimSuffix(result, "```")
	result = strings.TrimSpace(result)

	var scores []scoreEntry
	if err := json.Unmarshal([]byte(result), &scores); err != nil {
		log.Printf("WARN: reranker JSON parse failed, using original order: %v", err)
		if len(chunks) > topN {
			return chunks[:topN], nil
		}
		return chunks, nil
	}

	// ==========================================
	// L6 KNOWLEDGE DISTILLATION: DATA COLLECTION
	// ==========================================
	file, err := os.OpenFile("frasier_reranker_dataset.jsonl", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Printf("WARN: Failed to open dataset file for logging: %v", err)
	} else {
		encoder := json.NewEncoder(file)
		for _, s := range scores {
			// Ensure the ID from Gemini actually maps to a real chunk
			if s.ID >= 0 && s.ID < len(chunks) {
				row := struct {
					Query   string  `json:"query"`
					Passage string  `json:"passage"`
					Score   float64 `json:"score"`
				}{
					Query:   query,
					Passage: chunks[s.ID].Content,
					Score:   s.Score,
				}

				if err := encoder.Encode(row); err != nil {
					log.Printf("WARN: Failed to write training row: %v", err)
				}
			}
		}
		file.Close()
	}
	// ==========================================

	sort.Slice(scores, func(i, j int) bool {
		return scores[i].Score > scores[j].Score
	})

	var reranked []models.SearchResult
	for _, s := range scores {
		if s.ID >= 0 && s.ID < len(chunks) {
			reranked = append(reranked, chunks[s.ID])
		}
		if len(reranked) >= topN {
			break
		}
	}

	if len(reranked) < topN {
		seen := make(map[int]bool)
		for _, s := range scores {
			seen[s.ID] = true
		}
		for i, c := range chunks {
			if !seen[i] {
				reranked = append(reranked, c)
			}
			if len(reranked) >= topN {
				break
			}
		}
	}

	return reranked, nil
}

func extractText(resp *genai.GenerateContentResponse) string {
	if resp == nil || len(resp.Candidates) == 0 {
		return ""
	}

	candidate := resp.Candidates[0]
	if candidate.Content == nil || len(candidate.Content.Parts) == 0 {
		return ""
	}

	var parts []string
	for _, part := range candidate.Content.Parts {
		if part.Text != "" {
			parts = append(parts, part.Text)
		}
	}

	return strings.Join(parts, "\n")
}
