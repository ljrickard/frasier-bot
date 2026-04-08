package ai

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"cloud.google.com/go/vertexai/genai"
	"omnicorp-analyst/internal/database"
)

const (
	defaultLocation = "europe-west2"
	geminiModel     = "gemini-1.5-flash"
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
		location = defaultLocation
	}

	client, err := genai.NewClient(ctx, project, location)
	if err != nil {
		return "", fmt.Errorf("failed to create vertex ai client: %w", err)
	}
	defer client.Close()

	model := client.GenerativeModel(geminiModel)
	model.SetTemperature(0.2)

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

	// Generation: send to Gemini
	resp, err := model.GenerateContent(ctx, genai.Text(prompt))
	if err != nil {
		return "", fmt.Errorf("failed to generate content: %w", err)
	}

	// Extract text from response
	answer, err := extractText(resp)
	if err != nil {
		return "", fmt.Errorf("failed to extract text from response: %w", err)
	}

	return answer, nil
}

func extractText(resp *genai.GenerateContentResponse) (string, error) {
	if resp == nil || len(resp.Candidates) == 0 {
		return "", fmt.Errorf("empty response from model")
	}

	candidate := resp.Candidates[0]
	if candidate.Content == nil || len(candidate.Content.Parts) == 0 {
		return "", fmt.Errorf("no content parts in response")
	}

	var parts []string
	for _, part := range candidate.Content.Parts {
		if text, ok := part.(genai.Text); ok {
			parts = append(parts, string(text))
		}
	}

	if len(parts) == 0 {
		return "", fmt.Errorf("no text parts found in response")
	}

	return strings.Join(parts, "\n"), nil
}
