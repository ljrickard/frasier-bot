package models

import "time"

type Company struct {
	ID          int64     `json:"id"`
	Name        string    `json:"name"`
	Ticker      string    `json:"ticker"`
	Sector      string    `json:"sector,omitempty"`
	Description string    `json:"description,omitempty"`
	CreatedAt   time.Time `json:"created_at_utc"`
	UpdatedAt   time.Time `json:"updated_at_utc"`
}

type ParentChunk struct {
	ID           int64  `json:"id"`
	Content      string `json:"content"`
	Season       int    `json:"season"`
	Episode      int    `json:"episode"`
	EpisodeTitle string `json:"episode_title"`
	URL          string `json:"url"`
}

type Article struct {
	ID               int64      `json:"id"`
	CompanyID        int64      `json:"company_id"`
	Title            string     `json:"title"`
	Content          string     `json:"content,omitempty"`
	Source           string     `json:"source,omitempty"`
	ParentID         *int64     `json:"parent_id,omitempty"`
	Season           int        `json:"season,omitempty"`
	Episode          int        `json:"episode,omitempty"`
	EpisodeTitle     string     `json:"episode_title,omitempty"`
	Metadata         *string    `json:"metadata,omitempty"`
	PublishedAt      *time.Time `json:"published_at,omitempty"`
	PublishedAtLocal *string    `json:"published_at_local,omitempty"`
	CreatedAt        time.Time  `json:"created_at_utc"`
	UpdatedAt        time.Time  `json:"updated_at_utc"`
	Embedding        []float32  `json:"embedding,omitempty"`
}
