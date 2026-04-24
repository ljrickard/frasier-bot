package models

import (
	"time"
)

// Show represents a TV series (e.g., "Frasier")
type Show struct {
	ID          int64     `json:"id"`
	Title       string    `json:"title"`
	Description string    `json:"description,omitempty"`
	CreatedAt   time.Time `json:"created_at_utc"`
	UpdatedAt   time.Time `json:"updated_at_utc"`
}

// ParentChunk represents a full transcript of a scene or episode.
type ParentChunk struct {
	ID           int64  `json:"id"`
	ShowID       int64  `json:"show_id"`
	Content      string `json:"content"`
	Season       int    `json:"season"`
	Episode      int    `json:"episode"`
	EpisodeTitle string `json:"episode_title,omitempty"`
	URL          string `json:"url,omitempty"`
}

// Chunk represents a small, embeddable segment of dialogue.
type Chunk struct {
	ID           int64                  `json:"id"`
	ShowID       int64                  `json:"show_id"`
	ParentID     *int64                 `json:"parent_id,omitempty"`
	Content      string                 `json:"content"`
	Embedding    []float32              `json:"embedding,omitempty"`
	Season       int                    `json:"season"`
	Episode      int                    `json:"episode"`
	EpisodeTitle string                 `json:"episode_title,omitempty"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"` // For storing JSONB data like characters speaking
	CreatedAt    time.Time              `json:"created_at_utc"`
	UpdatedAt    time.Time              `json:"updated_at_utc"`
}

// SearchResult holds a single semantic search result returned by pgvector.
type SearchResult struct {
	Title      string  `json:"title"` // Corresponds to episode_title
	URL        string  `json:"url,omitempty"`
	Content    string  `json:"content"`
	ParentID   *int64  `json:"parent_id,omitempty"`
	Similarity float64 `json:"similarity"`
	Season     int     `json:"season"`
	Episode    int     `json:"episode"`
}
