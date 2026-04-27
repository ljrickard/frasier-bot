package crossencoder

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

type Client struct {
	baseURL    string
	httpClient *http.Client
}

type RerankRequest struct {
	Query    string   `json:"query"`
	Passages []string `json:"passages"`
}

type RerankResponse struct {
	Index int     `json:"index"`
	Score float64 `json:"score"`
}

func NewClient(baseURL string) *Client {
	return &Client{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 15 * time.Second, // Prevent indefinite hangs!
		},
	}
}

func (c *Client) Rerank(ctx context.Context, query string, passages []string) ([]RerankResponse, error) {
	reqBody := RerankRequest{
		Query:    query,
		Passages: passages,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("cross-encoder returned status %d", resp.StatusCode)
	}

	var scores []RerankResponse
	if err := json.NewDecoder(resp.Body).Decode(&scores); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return scores, nil
}
