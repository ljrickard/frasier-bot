package gemini

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
)

const GeminiModel = "gemini-2.5-flash"

func CallWithRetry(ctx context.Context, fn func() (*genai.GenerateContentResponse, error)) (*genai.GenerateContentResponse, error) {
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

func GetClient(ctx context.Context) (*genai.Client, error) {
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
	suppressSDKWarnings(func() {
		client, clientErr = genai.NewClient(ctx, &genai.ClientConfig{
			Project:  project,
			Location: location,
			Backend:  genai.BackendVertexAI,
		})
	})
	return client, clientErr
}

func suppressSDKWarnings(f func()) {
	original := log.Writer()
	log.SetOutput(io.Discard)
	defer log.SetOutput(original)
	f()
}
