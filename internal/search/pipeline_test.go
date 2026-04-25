package search

import (
	"context"
	"frasier-bot/internal/config"
	"frasier-bot/internal/database"
	"log"
	"net/http" // ADD THIS
	"os"
	"testing"
)

func TestFrasierRAG(t *testing.T) {
	// 1. PRE-FLIGHT HEALTH CHECK
	resp, err := http.Get("http://127.0.0.1:8000/health")
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("🚨 Python evaluation server is NOT reachable. Please run 'python eval_server.py' first.\nError: %v", err)
	}
	resp.Body.Close()

	// 2. SETUP
	ctx := context.Background()
	logger := log.New(os.Stderr, "", log.Ldate|log.Ltime|log.Lshortfile)

	db, err := database.New(ctx)
	if err != nil {
		t.Fatalf("Failed to connect to database: %v", err)
	}
	defer db.Close()

	// 2. Define the Golden Dataset
	questions := []string{
		"Who was Niles married to?",                 // Simple Fact
		"Why did Frasier and Martin fight so much?", // Thematic / Broad
		"What is the name of the dog?",              // Entity extraction
	}

	// 3. Define the Configurations you actually care about
	configs := []struct {
		Name string
		Cfg  *config.RAGConfig
	}{
		{
			Name: "Baseline (Vanilla RAG)",
			Cfg:  &config.RAGConfig{UsePersona: false, UseReranker: false, UseExpansion: false},
		},
		{
			Name: "Production Candidate (Reranker + Switchboard)",
			Cfg:  &config.RAGConfig{UsePersona: false, UseReranker: true, UseSwitchboard: true},
		},
		{
			Name: "Persona Mode (Checking Answer Relevancy Drop)",
			Cfg:  &config.RAGConfig{UsePersona: true, UseReranker: true, UseSwitchboard: true},
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
				fScore := res.Scores["faithfulness"].(float64)
				rScore := res.Scores["answer_relevancy"].(float64)

				totalFaithfulness += fScore
				totalRelevancy += rScore // Don't forget to declare this var at the top of the test!

				t.Logf("Q: %-45s | Faith: %.2f | Relevancy: %.2f", q, fScore, rScore)
			}

			// Calculate the aggregate score for this specific configuration
			// Calculate the aggregate scores for this specific configuration
			avgFaithfulness := totalFaithfulness / float64(len(questions))
			avgRelevancy := totalRelevancy / float64(len(questions))

			// Print the final summary line
			t.Logf("=== Config: %s | AVG Faithfulness: %.2f | AVG Relevancy: %.2f ===", conf.Name, avgFaithfulness, avgRelevancy)
		})
	}
}
