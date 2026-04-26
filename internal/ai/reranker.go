package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"frasier-bot/internal/ai/gemini"
	"frasier-bot/internal/models"
	"log"
	"net/http"
	"os"
	"sort"
	"strings"

	"google.golang.org/genai"
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
func RerankChunks(ctx context.Context, backend string, query string, chunks []models.SearchResult, topN int) ([]models.SearchResult, error) {
	if len(chunks) <= topN {
		return chunks, nil
	}

	switch strings.ToLower(backend) {
	case "local":
		return rerankWithLocal(ctx, query, chunks, topN)
	case "gemini":
		return rerankWithGemini(ctx, query, chunks, topN)
	default:
		log.Printf("WARN: Unknown reranker backend '%s', defaulting to gemini", backend)
		return rerankWithGemini(ctx, query, chunks, topN)
	}
}

// ==========================================
// LOCAL CROSS-ENCODER IMPLEMENTATION
// ==========================================

// Request payload for the local Python server
type localRerankReq struct {
	Query    string   `json:"query"`
	Passages []string `json:"passages"`
}

// Response payload from the local Python server
type localRerankResp struct {
	Index int     `json:"index"`
	Score float64 `json:"score"`
}

func rerankWithLocal(ctx context.Context, query string, chunks []models.SearchResult, topN int) ([]models.SearchResult, error) {
	// 1. Build the payload
	reqBody := localRerankReq{
		Query:    query,
		Passages: make([]string, len(chunks)),
	}
	for i, c := range chunks {
		reqBody.Passages[i] = c.Content
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal local reranker request: %w", err)
	}

	// 2. Make the HTTP Request
	req, err := http.NewRequestWithContext(ctx, "POST", "http://localhost:8001/rerank", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create local reranker request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("local reranker request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("local reranker returned status %d", resp.StatusCode)
	}

	// 3. Decode the response
	var scores []localRerankResp
	if err := json.NewDecoder(resp.Body).Decode(&scores); err != nil {
		return nil, fmt.Errorf("failed to decode local reranker response: %w", err)
	}

	// 4. Sort descending (higher cross-encoder logits are better)
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].Score > scores[j].Score
	})

	// 5. Build the final Top-N slice
	var reranked []models.SearchResult
	for _, s := range scores {
		if s.Index >= 0 && s.Index < len(chunks) {
			reranked = append(reranked, chunks[s.Index])
		}
		if len(reranked) >= topN {
			break
		}
	}

	return reranked, nil
}

func rerankWithGemini(ctx context.Context, query string, chunks []models.SearchResult, topN int) ([]models.SearchResult, error) {
	if len(chunks) <= topN {
		return chunks, nil
	}

	client, err := gemini.GetClient(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to create genai client: %w", err)
	}

	var chunkList strings.Builder
	for i, c := range chunks {
		content := c.Content
		if len(content) > 500 {
			content = content[:500] + "..."
		}
		chunkList.WriteString(fmt.Sprintf("Chunk %d:\n%s\n\n", i, content))
	}

	prompt := fmt.Sprintf(promptRerank, query, chunkList.String())
	temperature := float32(0.0)

	resp, err := gemini.CallWithRetry(ctx, func() (*genai.GenerateContentResponse, error) {
		return client.Models.GenerateContent(ctx, gemini.GeminiModel, genai.Text(prompt), &genai.GenerateContentConfig{
			Temperature: &temperature,
		})
	})
	if err != nil {
		return nil, fmt.Errorf("failed to rerank chunks: %w", err)
	}

	result := strings.TrimSpace(extractText(resp))

	type scoreEntry struct {
		ID    int     `json:"id"`
		Score float64 `json:"score"`
	}

	result = strings.TrimPrefix(result, "```json")
	result = strings.TrimPrefix(result, "```")
	result = strings.TrimSuffix(result, "```")
	result = strings.TrimSpace(result)

	var scores []scoreEntry
	if err := json.Unmarshal([]byte(result), &scores); err != nil {
		log.Printf("WARN: reranker JSON parse failed, using original order: %v", err)
		if len(chunks) > topN {
			return chunks[:topN], nil
		}
		return chunks, nil
	}

	// ==========================================
	// L6 KNOWLEDGE DISTILLATION: DATA COLLECTION
	// ==========================================
	file, err := os.OpenFile("frasier_reranker_dataset.jsonl", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Printf("WARN: Failed to open dataset file for logging: %v", err)
	} else {
		encoder := json.NewEncoder(file)
		for _, s := range scores {
			// Ensure the ID from Gemini actually maps to a real chunk
			if s.ID >= 0 && s.ID < len(chunks) {
				row := struct {
					Query   string  `json:"query"`
					Passage string  `json:"passage"`
					Score   float64 `json:"score"`
				}{
					Query:   query,
					Passage: chunks[s.ID].Content,
					Score:   s.Score,
				}

				if err := encoder.Encode(row); err != nil {
					log.Printf("WARN: Failed to write training row: %v", err)
				}
			}
		}
		file.Close()
	}
	// ==========================================

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
