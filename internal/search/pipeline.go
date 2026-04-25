package search

import (
	"context"
	"fmt"
	"frasier-bot/internal/ai"
	"frasier-bot/internal/config"
	"frasier-bot/internal/database"
	"frasier-bot/internal/embeddings"
	"frasier-bot/internal/models"
	"log"
)

const maxHistory = 10

// RAGResult holds all the data and metadata from a single pipeline execution.
// This makes it incredibly easy to assert against in your Go table tests.
type RAGResult struct {
	Answer          string
	Scores          map[string]interface{}
	EvalErr         error
	Contexts        []models.SearchResult
	Reformulated    string
	Classification  string
	FetchK          int
	FinalK          int
	PerEpisodeLimit int
	PreRerankCount  int
}

// RunRAGPipeline executes the core RAG logic. It is completely decoupled from the CLI UI.
// You can call this directly from your test files by passing a JSON struct and setting statusCallback to nil.
func RunRAGPipeline(
	ctx context.Context,
	db *database.DB,
	cfg *config.RAGConfig,
	logger *log.Logger,
	query string,
	chatHistory []string,
	statusCallback func(string),
) (RAGResult, error) {

	var res RAGResult
	var searchResultsForAI []models.SearchResult

	// Helper to safely call the status callback if it exists
	updateStatus := func(msg string) {
		if statusCallback != nil {
			statusCallback(msg)
		}
	}

	if !cfg.NoRAGMode {

		// Step 1: Reformulate query using chat history
		updateStatus("Analyzing query...")
		res.Reformulated = query
		if cfg.UseExpansion {
			ref, err := ai.ReformulateQuery(ctx, query, chatHistory)
			if err != nil {
				logger.Printf("WARN: failed to reformulate query: %v", err)
			} else {
				res.Reformulated = ref
			}
		}

		// Step 2: Classify the reformulated query for Top-K
		res.FetchK = 50
		res.PerEpisodeLimit = 3
		res.FinalK = 20

		if cfg.UseSwitchboard {
			updateStatus("Classifying query...")
			classification, err := ai.ClassifyQuery(ctx, res.Reformulated)
			if err != nil {
				logger.Printf("WARN: failed to classify query, defaulting to GENERAL: %v", err)
				classification = "GENERAL"
			}
			res.Classification = classification
			if classification == "SPECIFIC" {
				res.FetchK = 30
				res.FinalK = 10
			}
		} else {
			res.Classification = "OFF"
			res.FetchK = 5
			res.FinalK = 5
		}

		// Step 3: Generate embedding for the reformulated query
		updateStatus("Searching transcripts...")
		queryEmbedding, err := embeddings.GenerateQueryEmbedding(ctx, res.Reformulated)
		if err != nil {
			logger.Printf("WARN: failed to generate query embedding: %v", err)
			return res, fmt.Errorf("embedding error")
		}

		// Step 4: Search chunks (children) — fetch wide
		var results []models.SearchResult
		if cfg.UseDiversity {
			results, err = db.SearchChunksDiverse(ctx, queryEmbedding, res.FetchK, res.PerEpisodeLimit)
		} else {
			results, err = db.SearchChunks(ctx, queryEmbedding, res.FetchK)
		}
		if err != nil {
			logger.Printf("WARN: failed to search chunks: %v", err)
			return res, fmt.Errorf("search error")
		}
		if len(results) == 0 {
			return res, fmt.Errorf("no relevant transcripts found")
		}

		// Step 5: Collect unique parent IDs and fetch parent chunks
		parentIDSet := make(map[int64]bool)
		var parentIDs []int64
		for _, r := range results {
			if r.ParentID != nil && !parentIDSet[*r.ParentID] {
				parentIDSet[*r.ParentID] = true
				parentIDs = append(parentIDs, *r.ParentID)
			}
		}

		var parentResults []models.SearchResult
		if len(parentIDs) > 0 {
			parents, err := db.GetParentChunksByIDs(ctx, parentIDs)
			if err != nil {
				logger.Printf("WARN: failed to fetch parent chunks: %v", err)
			} else {
				for _, p := range parents {
					content := p.Content
					if cfg.UseMetadata {
						content = fmt.Sprintf("[S%02dE%02d] %s", p.Season, p.Episode, content)
					}
					parentResults = append(parentResults, models.SearchResult{
						Title:   fmt.Sprintf("S%02dE%02d: %s", p.Season, p.Episode, p.EpisodeTitle),
						URL:     p.URL,
						Content: content,
					})
				}
			}
		}

		searchResultsForAI := parentResults
		if len(searchResultsForAI) == 0 {
			searchResultsForAI = results
		}

		// Step 5b: Rerank chunks using LLM scoring
		res.PreRerankCount = len(searchResultsForAI)
		if cfg.UseReranker {
			updateStatus("Reranking results...")
			reranked, err := ai.RerankChunks(ctx, res.Reformulated, searchResultsForAI, res.FinalK)
			if err != nil {
				logger.Printf("WARN: reranker failed, using original order: %v", err)
			} else {
				searchResultsForAI = reranked
			}
		} else if len(searchResultsForAI) > res.FinalK {
			searchResultsForAI = searchResultsForAI[:res.FinalK]
		}
		res.Contexts = searchResultsForAI

	} else {
		updateStatus("Bypassing RAG (Vanilla AI mode)...")
		res.Classification = "VANILLA"
	}

	// Step 6: Generate RAG answer
	updateStatus("Consulting the Crane brothers...")
	ragAnswer, err := ai.GenerateAnswer(ctx, query, searchResultsForAI, cfg.UsePersona)
	if err != nil {
		logger.Printf("WARN: failed to generate RAG answer: %v", err)
		return res, fmt.Errorf("generation error")
	}
	res.Answer = ragAnswer

	// Step 7: LIVE RAG EVALUATION
	var contextStrings []string
	for _, r := range searchResultsForAI {
		contextStrings = append(contextStrings, r.Content)
	}
	updateStatus("Evaluating answer quality...")
	scores, evalErr := EvaluateInteraction(query, contextStrings, ragAnswer)

	res.Scores = scores
	res.EvalErr = evalErr

	return res, nil
}
