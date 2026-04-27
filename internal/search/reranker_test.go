package search

import (
	"context"
	"log"
	"os"
	"testing"
	"time"

	"frasier-bot/internal/config"
	"frasier-bot/internal/database"
)

func Test_Reranker(t *testing.T) {
	// 1. Setup Environment
	ctx := context.Background()
	logger := log.New(os.Stdout, "", log.LstdFlags)

	db, err := database.New(ctx)
	if err != nil {
		t.Fatalf("Failed to connect to database: %v", err)
	}
	defer db.Close()

	questions := []string{
		"Who was Frasier married to?",
		"Who was the gangster that Niles hired?",
		"Why did Frasier and Martin fight so much?",
	}

	configs := []struct {
		Name string
		Cfg  *config.RAGConfig
	}{
		{
			Name: "1_raranking_using_gemini",
			Cfg: &config.RAGConfig{
				UseRAG: true, UseQueryExpansion: true, UseQueryClassification: true, UseReranker: true,
				UseEpisodeLimit: true, UseMetadata: true, UsePersona: true, RerankerBackend: "gemini", UseEval: true,
			},
		},
		{
			Name: "2_raranking_using_local_cross_embedded_model",
			Cfg: &config.RAGConfig{
				UseRAG: true, UseQueryExpansion: true, UseQueryClassification: true, UseReranker: true,
				UseEpisodeLimit: true, UseMetadata: true, UsePersona: true, RerankerBackend: "local", UseEval: true,
			},
		},
	}

	for _, conf := range configs {
		t.Run(conf.Name, func(t *testing.T) {
			var totalFaithfulness float64
			var totalRelevancy float64

			for _, q := range questions {
				res, err := RunRAGPipeline(ctx, db, conf.Cfg, logger, q, nil, nil)
				if err != nil {
					t.Fatalf("Failed on question '%s': %v", q, err)
				}

				// Safely extract both scores
				fScore := 0.0
				rScore := 0.0
				if val, ok := res.Scores["faithfulness"].(float64); ok {
					fScore = val
				}
				if val, ok := res.Scores["answer_relevancy"].(float64); ok {
					rScore = val
				}

				totalFaithfulness += fScore
				totalRelevancy += rScore

				t.Logf("Q: %-45s | Faith: %.4f | Relevancy: %.4f", q, fScore, rScore)

				// DELIBERATE PAUSE: Prevents Google Vertex AI 429 Rate Limits
				time.Sleep(10 * time.Second)
			}

			// Calculate the aggregate scores for this specific configuration
			avgFaithfulness := totalFaithfulness / float64(len(questions))
			avgRelevancy := totalRelevancy / float64(len(questions))

			t.Logf("=== Config: %s | AVG Faith: %.4f | AVG Rel: %.4f ===", conf.Name, avgFaithfulness, avgRelevancy)
		})
	}
}
