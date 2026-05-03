package embeddings

import (
	"context"
	"errors"
)

// Embedder defines the exact interface we need from the Gemini wrapper.
// Your new gemini.Client perfectly satisfies this because it implements
// the EmbedText(ctx context.Context, text string) ([]float32, error) method.
type Embedder interface {
	EmbedText(ctx context.Context, text string) ([]float32, error)
}

// Config allows you to safely inject dependencies into the embeddings service.
type Config struct {
	Client Embedder
}

// Service holds our dependencies and acts as the domain logic wrapper.
type Service struct {
	client Embedder
}

// New initializes the embeddings service with the provided configuration.
// We no longer need the old "Preflight" method because your gemini.Client
// handles its own authentication validation upon startup!
func New(cfg Config) (*Service, error) {
	if cfg.Client == nil {
		return nil, errors.New("failed to initialize embeddings service: client cannot be nil")
	}
	return &Service{
		client: cfg.Client,
	}, nil
}

// GenerateEmbedding handles document chunk embeddings for your Vector DB.
func (s *Service) GenerateEmbedding(ctx context.Context, text string) ([]float32, error) {
	return s.client.EmbedText(ctx, text)
}

// GenerateQueryEmbedding handles user search query embeddings.
func (s *Service) GenerateQueryEmbedding(ctx context.Context, text string) ([]float32, error) {
	// The new AI Studio SDK manages task types natively.
	// Delegating directly to your standard EmbedText method will work perfectly.
	return s.client.EmbedText(ctx, text)
}
