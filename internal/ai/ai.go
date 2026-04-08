package ai

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"google.golang.org/genai"
	"omnicorp-analyst/internal/database"
)

const (
	geminiModel = "gemini-2.5-flash"
)

// GenerateAnswer takes a user query and a slice of search results,
// constructs an augmented prompt, and sends it to Gemini for generation.
func GenerateAnswer(ctx context.Context, query string, articles []database.SearchResult) (string, error) {
	project := os.Getenv("GOOGLE_CLOUD_PROJECT")
	if project == "" {
		return "", fmt.Errorf("GOOGLE_CLOUD_PROJECT environment variable is not set")
	}

	location := os.Getenv("GOOGLE_CLOUD_LOCATION")
	if location == "" {
		location = "europe-west2"
	}

	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		Project:  project,
		Location: location,
		Backend:  genai.BackendVertexAI,
	})
	if err != nil {
		return "", fmt.Errorf("failed to create genai client: %w", err)
	}

	// Augmentation: build the prompt
	var contextBuilder strings.Builder
	for i, a := range articles {
		contextBuilder.WriteString(fmt.Sprintf("Article %d:\n", i+1))
		contextBuilder.WriteString(fmt.Sprintf("Title: %s\n", a.Title))
		contextBuilder.WriteString(fmt.Sprintf("URL: %s\n", a.URL))
		contextBuilder.WriteString(fmt.Sprintf("Similarity: %.4f\n", a.Similarity))
		contextBuilder.WriteString("\n")
	}

	prompt := fmt.Sprintf(`You are a helpful Omnicorp news analyst. Answer the user's question using ONLY the provided context below. Do not make up information. If the context does not contain enough information to answer the question, say so.

Context:
%s
Question: %s`, contextBuilder.String(), query)

	log.Printf("Sending prompt to %s...", geminiModel)

	temperature := float32(0.2)

	// Generation: send to Gemini
	resp, err := client.Models.GenerateContent(ctx, geminiModel, genai.Text(prompt), &genai.GenerateContentConfig{
		Temperature: &temperature,
	})
	if err != nil {
		return "", fmt.Errorf("failed to generate content: %w", err)
	}

	// Extract text from response
	answer := extractText(resp)
	if answer == "" {
		return "", fmt.Errorf("empty response from model")
	}

	return answer, nil
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
