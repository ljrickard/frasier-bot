package ai

import (
	"context"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"strings"
	"time"

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

// suppressSDKWarnings temporarily discards default log output during f(),
// then restores it. This silences noisy SDK warnings like
// "Warning: The user provided project/location will take precedence..."
func suppressSDKWarnings(f func()) {
	original := log.Writer()
	log.SetOutput(io.Discard)
	defer log.SetOutput(original)
	f()
}

// callWithRetry wraps a Gemini API call with exponential backoff + jitter.
// It retries up to 5 times on 429 / Resource Exhausted errors.
func callWithRetry(ctx context.Context, fn func() (*genai.GenerateContentResponse, error)) (*genai.GenerateContentResponse, error) {
	maxRetries := 5
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

		// Exponential backoff: 2s, 4s, 8s, 16s, 32s + jitter
		delay := baseDelay * (1 << uint(attempt))
		jitter := time.Duration(rand.Int63n(int64(delay) / 4))
		wait := delay + jitter

		log.Printf("Rate limited (429), retry %d/%d in %v...", attempt+1, maxRetries, wait)

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(wait):
		}
	}

	return nil, fmt.Errorf("max retries exceeded")
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
		location = "us-central1"
	}

	var client *genai.Client
	var clientErr error
	suppressSDKWarnings(func() {
		client, clientErr = genai.NewClient(ctx, &genai.ClientConfig{
			Project:  project,
			Location: location,
			Backend:  genai.BackendVertexAI,
		})
	})
	if clientErr != nil {
		return "", fmt.Errorf("failed to create genai client: %w", clientErr)
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

	prompt := fmt.Sprintf(`You are the ultimate Frasier expert with the wit and vocabulary of the Crane brothers. You must remain strictly factual based on the provided context — never invent information — but present your answers with sophisticated humor and the eloquent flair worthy of a Crane.

Guidelines:
- Pay strict attention to the [SxxExx] metadata to determine chronological order. Season 1 is the oldest; Season 11 is the most recent.
- When citing episodes, tuck the references naturally into parentheses at the end of sentences, e.g. "Niles finally declared his love (S07E24)" rather than leading with the code.
- For minor or fleeting romantic interests (e.g. Poppy, Marjorie), feel free to characterize them as "brief dalliances" or "passing encounters" to distinguish them from significant relationships.
- When discussing Niles and Daphne, recognize their relationship as the definitive romantic arc of the series — a slow burn worthy of the finest literature.
- Use natural, flowing prose. Avoid bullet-point lists unless the user explicitly asks for a list.
- If the context does not contain enough information to answer, say so with appropriate Crane-like regret.

Context:
%s
Question: %s`, contextBuilder.String(), query)

	temperature := float32(0.2)

	// Generation: send to Gemini
	resp, err := callWithRetry(ctx, func() (*genai.GenerateContentResponse, error) {
		return client.Models.GenerateContent(ctx, geminiModel, genai.Text(prompt), &genai.GenerateContentConfig{
			Temperature: &temperature,
		})
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
		location = "us-central1"
	}

	var client *genai.Client
	var clientErr error
	suppressSDKWarnings(func() {
		client, clientErr = genai.NewClient(ctx, &genai.ClientConfig{
			Project:  project,
			Location: location,
			Backend:  genai.BackendVertexAI,
		})
	})
	if clientErr != nil {
		return "", fmt.Errorf("failed to create genai client: %w", clientErr)
	}

	prompt := fmt.Sprintf(`Classify this query as 'SPECIFIC' or 'GENERAL'.

SPECIFIC: asking for a single name, exact date, or a direct quote from one scene.
GENERAL: asking for a summary, theme, character history, relationship arc, or anything spanning multiple episodes.

IMPORTANT: Questions about character history, "how many", "who did they date", "list of", "all the times", or any question that could span multiple episodes or seasons MUST be classified as GENERAL to ensure we capture the entire 11-season timeline.

Respond with only one word: SPECIFIC or GENERAL.

Query: %s`, query)

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

// ReformulateQuery rewrites and expands a user query using chat history to make
// it standalone and optimised for vector search against the transcript database.
func ReformulateQuery(ctx context.Context, query string, history []string) (string, error) {
	// Even with no history, we still want to expand the query for better search
	historyText := "(No prior conversation)"
	if len(history) > 0 {
		var hb strings.Builder
		for _, h := range history {
			hb.WriteString(h)
			hb.WriteString("\n")
		}
		historyText = hb.String()
	}

	project := os.Getenv("GOOGLE_CLOUD_PROJECT")
	if project == "" {
		return "", fmt.Errorf("GOOGLE_CLOUD_PROJECT environment variable is not set")
	}

	location := os.Getenv("GOOGLE_CLOUD_LOCATION")
	if location == "" {
		location = "us-central1"
	}

	var client *genai.Client
	var clientErr error
	suppressSDKWarnings(func() {
		client, clientErr = genai.NewClient(ctx, &genai.ClientConfig{
			Project:  project,
			Location: location,
			Backend:  genai.BackendVertexAI,
		})
	})
	if clientErr != nil {
		return "", fmt.Errorf("failed to create genai client: %w", clientErr)
	}

	prompt := fmt.Sprintf(`You are a search query optimizer for a Frasier TV show transcript database. Your goal is to turn the user's question into the best possible vector search terms.

Rules:
1. If there is conversation history, rewrite the question to be standalone (resolve pronouns like "he", "she", "they" using context).
2. Expand narrow words into broader search terms to cover the full 11-season history:
   - "lovers", "dating", "relationships" → expand to include "marriage, wives, ex-wives, husband, romantic interests, significant others, girlfriend, boyfriend, dating, affair"
   - "jobs", "career" → expand to include "work, profession, employment, fired, hired, promotion, radio show, private practice"
   - "fights", "arguments" → expand to include "conflict, disagreement, feud, rivalry, confrontation, tension"
   - "family" → expand to include "father, brother, son, wife, ex-wife, mother, children"
3. Always include character names if the question implies specific characters.
4. Respond with ONLY the rewritten/expanded query, nothing else.

Conversation History:
%s
Latest Question: %s`, historyText, query)

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

// GenerateVanillaAnswer asks Gemini the question with no RAG context.
func GenerateVanillaAnswer(ctx context.Context, query string) (string, error) {
	project := os.Getenv("GOOGLE_CLOUD_PROJECT")
	if project == "" {
		return "", fmt.Errorf("GOOGLE_CLOUD_PROJECT environment variable is not set")
	}

	location := os.Getenv("GOOGLE_CLOUD_LOCATION")
	if location == "" {
		location = "us-central1"
	}

	var client *genai.Client
	var clientErr error
	suppressSDKWarnings(func() {
		client, clientErr = genai.NewClient(ctx, &genai.ClientConfig{
			Project:  project,
			Location: location,
			Backend:  genai.BackendVertexAI,
		})
	})
	if clientErr != nil {
		return "", fmt.Errorf("failed to create genai client: %w", clientErr)
	}

	prompt := fmt.Sprintf(`You are a helpful assistant who is knowledgeable about the TV show Frasier. Answer the following question to the best of your ability.

Question: %s`, query)

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

// EvaluateAnswers asks Gemini to compare the vanilla and RAG answers.
func EvaluateAnswers(ctx context.Context, query, vanillaAnswer, ragAnswer string) (string, error) {
	project := os.Getenv("GOOGLE_CLOUD_PROJECT")
	if project == "" {
		return "", fmt.Errorf("GOOGLE_CLOUD_PROJECT environment variable is not set")
	}

	location := os.Getenv("GOOGLE_CLOUD_LOCATION")
	if location == "" {
		location = "us-central1"
	}

	var client *genai.Client
	var clientErr error
	suppressSDKWarnings(func() {
		client, clientErr = genai.NewClient(ctx, &genai.ClientConfig{
			Project:  project,
			Location: location,
			Backend:  genai.BackendVertexAI,
		})
	})
	if clientErr != nil {
		return "", fmt.Errorf("failed to create genai client: %w", clientErr)
	}

	prompt := fmt.Sprintf(`You are an evaluator. A user asked a question about the TV show Frasier. Two AI systems answered: one with no database context ("Vanilla AI") and one with actual transcript data ("RAG AI").

Question: %s

Vanilla AI Answer:
%s

RAG AI Answer:
%s

Based on the RAG AI's context-grounded answer, did the Vanilla AI get anything wrong? Write a brief footnote (2-3 sentences max) comparing accuracy.`, query, vanillaAnswer, ragAnswer)

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
