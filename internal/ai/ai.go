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

func init() {
	// Redirect default logger (used by SDKs) to stderr to keep stdout clean
	log.SetOutput(os.Stderr)
}

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
		contextBuilder.WriteString(fmt.Sprintf("Content: %s\n", a.Content))
		contextBuilder.WriteString(fmt.Sprintf("Similarity: %.4f\n", a.Similarity))
		contextBuilder.WriteString("\n")
	}

	prompt := fmt.Sprintf(`You are a helpful Omnicorp news analyst. Answer the user's question using ONLY the provided context below. Do not make up information. If the context does not contain enough information to answer the question, say so.

Context:
%s
Question: %s`, contextBuilder.String(), query)

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

// ClassifyQuery asks Gemini to classify a query as SPECIFIC or GENERAL.
func ClassifyQuery(ctx context.Context, query string) (string, error) {
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

	prompt := fmt.Sprintf(`Classify this query as 'SPECIFIC' (asking for a name, date, or quote) or 'GENERAL' (asking for a summary or theme). Respond with only one word.

Query: %s`, query)

	temperature := float32(0.0)

	resp, err := client.Models.GenerateContent(ctx, geminiModel, genai.Text(prompt), &genai.GenerateContentConfig{
		Temperature: &temperature,
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

// ReformulateQuery rewrites a user query using chat history to make it standalone.
func ReformulateQuery(ctx context.Context, query string, history []string) (string, error) {
	// If no history, the query is already standalone
	if len(history) == 0 {
		return query, nil
	}

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

	var historyBuilder strings.Builder
	for _, h := range history {
		historyBuilder.WriteString(h)
		historyBuilder.WriteString("\n")
	}

	prompt := fmt.Sprintf(`Given the conversation history below, rewrite the latest user question into a standalone search query that can be understood without the history. If it is already standalone, return it unchanged. Respond with ONLY the rewritten query, nothing else.

Conversation History:
%s
Latest Question: %s`, historyBuilder.String(), query)

	temperature := float32(0.0)

	resp, err := client.Models.GenerateContent(ctx, geminiModel, genai.Text(prompt), &genai.GenerateContentConfig{
		Temperature: &temperature,
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

// GenerateVanillaAnswer asks Gemini the question with no RAG context.
func GenerateVanillaAnswer(ctx context.Context, query string) (string, error) {
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

	prompt := fmt.Sprintf(`You are a helpful assistant who is knowledgeable about the TV show Frasier. Answer the following question to the best of your ability.

Question: %s`, query)

	temperature := float32(0.2)

	resp, err := client.Models.GenerateContent(ctx, geminiModel, genai.Text(prompt), &genai.GenerateContentConfig{
		Temperature: &temperature,
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

// EvaluateAnswers asks Gemini to compare the vanilla and RAG answers.
func EvaluateAnswers(ctx context.Context, query, vanillaAnswer, ragAnswer string) (string, error) {
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

	prompt := fmt.Sprintf(`You are an evaluator. A user asked a question about the TV show Frasier. Two AI systems answered: one with no database context ("Vanilla AI") and one with actual transcript data ("RAG AI").

Question: %s

Vanilla AI Answer:
%s

RAG AI Answer:
%s

Based on the RAG AI's context-grounded answer, did the Vanilla AI get anything wrong? Write a brief footnote (2-3 sentences max) comparing accuracy.`, query, vanillaAnswer, ragAnswer)

	temperature := float32(0.2)

	resp, err := client.Models.GenerateContent(ctx, geminiModel, genai.Text(prompt), &genai.GenerateContentConfig{
		Temperature: &temperature,
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
