package search

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// EvalPayload matches the JSON structure required by your evaluation server
type EvalPayload struct {
	Question    string   `json:"question"`
	Contexts    []string `json:"contexts"`
	Answer      string   `json:"answer"`
	GroundTruth string   `json:"ground_truth"`
}

// EvalResponse matches the incoming {"status": "success", "scores": ...}
type EvalResponse struct {
	Status string         `json:"status"`
	Scores map[string]any `json:"scores"` // Flexible map to catch any scoring metrics returned
}

// EvaluateInteraction sends the RAG data to the local evaluation server and returns the scores.
func EvaluateInteraction(question string, contexts []string, answer string) (map[string]any, error) {
	payload := EvalPayload{
		Question:    question,
		Contexts:    contexts,
		Answer:      answer,
		GroundTruth: "", // Hardcoded to empty string per your request
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal payload: %w", err)
	}

	// 30-second timeout to ensure the chat doesn't hang forever if the Python server is slow
	client := &http.Client{Timeout: 600 * time.Second}
	resp, err := client.Post("http://127.0.0.1:8000/evaluate", "application/json", bytes.NewBuffer(body))
	if err != nil {
		return nil, fmt.Errorf("failed to connect to evaluation server: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("evaluation server returned status: %s", resp.Status)
	}

	var evalResp EvalResponse
	if err := json.NewDecoder(resp.Body).Decode(&evalResp); err != nil {
		return nil, fmt.Errorf("failed to decode evaluation response: %w", err)
	}

	if evalResp.Status != "success" {
		return nil, fmt.Errorf("evaluation failed with status: %s", evalResp.Status)
	}

	return evalResp.Scores, nil
}
