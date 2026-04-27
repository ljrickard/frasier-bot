package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sort"
	"strings"

	"frasier-bot/internal/models"
)

type RerankChunk struct {
	Index    int
	Title    string
	URL      string
	Content  string
	ParentID *int64
	Score    float64
}

// RerankChunks acts as a router for different reranking backends.
func (s *Service) RerankChunks(ctx context.Context, backend string, query string, chunks []models.SearchResult, topN int) ([]models.SearchResult, error) {
	if len(chunks) <= topN {
		return chunks, nil
	}

	switch strings.ToLower(backend) {
	case "local":
		return s.rerankWithLocal(ctx, query, chunks, topN)
	case "gemini":
		return s.rerankWithGemini(ctx, query, chunks, topN)
	default:
		log.Printf("WARN: Unknown reranker backend '%s', defaulting to gemini", backend)
		return s.rerankWithGemini(ctx, query, chunks, topN)
	}
}

// ==========================================
// LOCAL CROSS-ENCODER IMPLEMENTATION
// ==========================================

type localRerankReq struct {
	Query    string   `json:"query"`
	Passages []string `json:"passages"`
}

type localRerankResp struct {
	Index int     `json:"index"`
	Score float64 `json:"score"`
}

func (s *Service) rerankWithLocal(ctx context.Context, query string, chunks []models.SearchResult, topN int) ([]models.SearchResult, error) {
	// 1. Prepare passages
	passages := make([]string, len(chunks))
	for i, c := range chunks {
		passages[i] = c.Content
	}

	// 2. Call the new injected client
	scores, err := s.Encoder.Rerank(ctx, query, passages)
	if err != nil {
		return nil, fmt.Errorf("cross-encoder service failed: %w", err)
	}

	// 3. Sort descending
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].Score > scores[j].Score
	})

	// 4. Build the final Top-N slice
	var reranked []models.SearchResult
	for _, scoreItem := range scores {
		if scoreItem.Index >= 0 && scoreItem.Index < len(chunks) {
			reranked = append(reranked, chunks[scoreItem.Index])
		}
		if len(reranked) >= topN {
			break
		}
	}

	return reranked, nil
}

func (s *Service) rerankWithGemini(ctx context.Context, query string, chunks []models.SearchResult, topN int) ([]models.SearchResult, error) {
	var chunkList strings.Builder
	for i, c := range chunks {
		content := c.Content
		if len(content) > 500 {
			content = content[:500] + "..."
		}
		chunkList.WriteString(fmt.Sprintf("Chunk %d:\n%s\n\n", i, content))
	}

	prompt := fmt.Sprintf(promptRerank, query, chunkList.String())

	// Use the wrapper!
	response, err := s.LLM.GenerateText(ctx, prompt)
	if err != nil {
		return nil, fmt.Errorf("failed to rerank chunks: %w", err)
	}

	result := strings.TrimSpace(response)
	result = strings.TrimPrefix(result, "```json")
	result = strings.TrimPrefix(result, "```")
	result = strings.TrimSuffix(result, "```")
	result = strings.TrimSpace(result)

	type scoreEntry struct {
		ID    int     `json:"id"`
		Score float64 `json:"score"`
	}

	var scores []scoreEntry
	if err := json.Unmarshal([]byte(result), &scores); err != nil {
		log.Printf("WARN: reranker JSON parse failed, using original order: %v", err)
		if len(chunks) > topN {
			return chunks[:topN], nil
		}
		return chunks, nil
	}

	sort.Slice(scores, func(i, j int) bool {
		return scores[i].Score > scores[j].Score
	})

	var reranked []models.SearchResult
	for _, s := range scores {
		if s.ID >= 0 && s.ID < len(chunks) {
			reranked = append(reranked, chunks[s.ID])
		}
		if len(reranked) >= topN {
			break
		}
	}

	if len(reranked) < topN {
		seen := make(map[int]bool)
		for _, s := range scores {
			seen[s.ID] = true
		}
		for i, c := range chunks {
			if !seen[i] {
				reranked = append(reranked, c)
			}
			if len(reranked) >= topN {
				break
			}
		}
	}

	return reranked, nil
}
