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

func TestFrasierRAG(t *testing.T) {
	// 1. Setup Environment
	ctx := context.Background()
	logger := log.New(os.Stdout, "", log.LstdFlags)

	db, err := database.New(ctx)
	if err != nil {
		t.Fatalf("Failed to connect to database: %v", err)
	}
	defer db.Close()

	// 2. Define the Test Questions
	// We use a mix of general knowledge (wives) and deep trivia (gangsters)
	questions := []string{
		"Who was Frasier married to?",
		"Who was the gangster that Niles hired?",
		"Why did Frasier and Martin fight so much?",
	}

	// 3. The Scientific Progression Matrix
	// We start at zero-shot and add one major component at a time to measure the lift.
	configs := []struct {
		Name string
		Cfg  *config.RAGConfig
	}{
		{
			Name: "1_Vanilla_Baseline_(No_Database)",
			// Pure LLM knowledge. Expect Faithfulness = 0.00, High Relevancy.
			Cfg: &config.RAGConfig{
				UseRAG: false, UsePersona: false,
			},
		},
		{
			Name: "2_Standard_RAG_(Basic_Search)",
			// Turns on the DB, but leaves advanced reasoning off.
			Cfg: &config.RAGConfig{
				UseRAG: true, UseExpansion: false, UseSwitchboard: false,
				UseReranker: false, UseDiversity: false, UseMetadata: false, UsePersona: false,
			},
		},
		{
			Name: "3_Advanced_RAG_(Switchboard_+_Expansion)",
			// Adds query expansion and dynamic context sizing.
			Cfg: &config.RAGConfig{
				UseRAG: true, UseExpansion: true, UseSwitchboard: true,
				UseReranker: false, UseDiversity: false, UseMetadata: false, UsePersona: false,
			},
		},
		{
			Name: "4_Production_Candidate_(Added_Reranker)",
			// Turns on all the heavy ML lifting to find the perfect facts.
			Cfg: &config.RAGConfig{
				UseRAG: true, UseExpansion: true, UseSwitchboard: true,
				UseReranker: true, UseDiversity: true, UseMetadata: true, UsePersona: false,
			},
		},
		{
			Name: "5_Brand_Voice_(Production_+_Persona)",
			// Production candidate, but adds the Persona to measure the Relevancy drop.
			Cfg: &config.RAGConfig{
				UseRAG: true, UseExpansion: true, UseSwitchboard: true,
				UseReranker: true, UseDiversity: true, UseMetadata: true, UsePersona: true,
			},
		},
	}

	// 4. Run the Table Test Matrix
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
