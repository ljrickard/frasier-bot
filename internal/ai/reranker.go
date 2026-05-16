package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"sort"
	"strings"

	"frasier-bot/internal/models"
	"frasier-bot/tracing"
)

type RerankChunk struct {
	Index    int
	Title    string
	URL      string
	Content  string
	ParentID *int64
	Score    float64
}

func (s *Service) RerankChunks(ctx context.Context, backend string, query string, chunks []models.SearchResult, topN int) ([]models.SearchResult, error) {
	traceID := tracing.GetTraceID(ctx)
	if len(chunks) <= topN {
		return chunks, nil
	}

	switch strings.ToLower(backend) {
	case "local":
		return s.rerankWithLocal(ctx, query, chunks, topN)
	case "gemini":
		return s.rerankWithGemini(ctx, query, chunks, topN)
	default:
		slog.Warn("⚠️ Unknown reranker backend matching payload router, defaulting to gemini", "backend", backend, "trace_id", traceID)
		return s.rerankWithGemini(ctx, query, chunks, topN)
	}
}

type localRerankReq struct {
	Query    string   `json:"query"`
	Passages []string `json:"passages"`
}

type localRerankResp struct {
	Index int     `json:"index"`
	Score float64 `json:"score"`
}

func (s *Service) rerankWithLocal(ctx context.Context, query string, chunks []models.SearchResult, topN int) ([]models.SearchResult, error) {
	traceID := tracing.GetTraceID(ctx)
	slog.Debug("⚖️ [Reranker] Calling local cross-encoder model matrix calculations", "passages_count", len(chunks), "trace_id", traceID)

	passages := make([]string, len(chunks))
	for i, c := range chunks {
		passages[i] = c.Content
	}

	scores, err := s.Reranker.Rerank(ctx, query, passages)
	if err != nil {
		return nil, fmt.Errorf("reranker backend service failed: %w", err)
	}

	sort.Slice(scores, func(i, j int) bool {
		return scores[i].Score > scores[j].Score
	})

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
	traceID := tracing.GetTraceID(ctx)
	slog.Debug("⚖️ [Reranker] Inbound request routing to Gemini-fallback zero-shot relevance list grading", "passages_count", len(chunks), "trace_id", traceID)

	var chunkList strings.Builder
	for i, c := range chunks {
		content := c.Content
		if len(content) > 500 {
			content = content[:500] + "..."
		}
		chunkList.WriteString(fmt.Sprintf("Chunk %d:\n%s\n\n", i, content))
	}

	prompt := fmt.Sprintf(promptRerank, query, chunkList.String())

	response, err := s.LLM.GenerateText(ctx, prompt, defaultTemperature)
	if err != nil {
		return nil, fmt.Errorf("failed to rerank chunks: %w", err)
	}

	result := strings.TrimSpace(response)
	startIdx := strings.Index(result, "```json")
	if startIdx != -1 {
		result = result[startIdx+7:]
		endIdx := strings.Index(result, "```")
		if endIdx != -1 {
			result = result[:endIdx]
		}
	} else {
		startIdx = strings.Index(result, "```")
		if startIdx != -1 {
			result = result[startIdx+3:]
			endIdx := strings.Index(result, "```")
			if endIdx != -1 {
				result = result[:endIdx]
			}
		}
	}
	result = strings.TrimSpace(result)

	type scoreEntry struct {
		ID    int     `json:"id"`
		Score float64 `json:"score"`
	}

	var scores []scoreEntry
	if err := json.Unmarshal([]byte(result), &scores); err != nil {
		slog.Warn("⚠️ Reranker JSON block parse failed, defaulting structurally to raw vector similarity ranks", "trace_id", traceID, "error", err)
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
