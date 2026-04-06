package models

import "time"

type Company struct {
	ID          int64     `json:"id"`
	Name        string    `json:"name"`
	Ticker      string    `json:"ticker"`
	Sector      string    `json:"sector,omitempty"`
	Description string    `json:"description,omitempty"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
}

type Article struct {
	ID          int64      `json:"id"`
	CompanyID   int64      `json:"company_id"`
	Title       string     `json:"title"`
	Content     string     `json:"content,omitempty"`
	Source      string     `json:"source,omitempty"`
	PublishedAt *time.Time `json:"published_at,omitempty"`
	CreatedAt   time.Time  `json:"created_at"`
	UpdatedAt   time.Time  `json:"updated_at"`
}
