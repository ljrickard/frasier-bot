package llm

import "context"

// StreamResult decouples provider-specific chunk variants into a clean abstraction
type StreamResult struct {
	Text string
	Err  error
}

// Client defines the core capabilities required across the RAG system boundary
type Client interface {
	GenerateText(ctx context.Context, prompt string, temperature float32) (string, error)
	GenerateTextStream(ctx context.Context, prompt string, temperature float32) (<-chan StreamResult, error)
	EmbedText(ctx context.Context, text string) ([]float32, error)
}
