package embeddings

import (
	"context"
	"errors"
)

type Embedder interface {
	EmbedText(ctx context.Context, text string) ([]float32, error)
}

type Config struct {
	Client Embedder
}

type Service struct {
	client Embedder
}

func New(cfg Config) (*Service, error) {
	if cfg.Client == nil {
		return nil, errors.New("failed to initialize embeddings service: client cannot be nil")
	}
	return &Service{
		client: cfg.Client,
	}, nil
}

func (s *Service) GenerateEmbedding(ctx context.Context, text string) ([]float32, error) {
	return s.client.EmbedText(ctx, text)
}

func (s *Service) GenerateQueryEmbedding(ctx context.Context, text string) ([]float32, error) {
	return s.client.EmbedText(ctx, text)
}
