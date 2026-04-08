package database

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/pgvector/pgvector-go"
	"omnicorp-analyst/internal/models"
)

func (db *DB) CreateArticle(ctx context.Context, article *models.Article) error {
	if article.PublishedAt != nil {
		local := article.PublishedAt.Format(time.RFC3339)
		article.PublishedAtLocal = &local
	}

	var embedding *pgvector.Vector
	if len(article.Embedding) > 0 {
		v := pgvector.NewVector(article.Embedding)
		embedding = &v
	}

	query := `
		INSERT INTO articles (company_id, title, content, source, published_at, published_at_local, embedding)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
		RETURNING id, created_at_utc, updated_at_utc`

	err := db.Pool.QueryRow(ctx, query,
		article.CompanyID,
		article.Title,
		article.Content,
		article.Source,
		article.PublishedAt,
		article.PublishedAtLocal,
		embedding,
	).Scan(&article.ID, &article.CreatedAt, &article.UpdatedAt)
	if err != nil {
		return fmt.Errorf("failed to create article: %w", err)
	}

	return nil
}

func (db *DB) InsertArticle(ctx context.Context, article *models.Article) error {
	if article.PublishedAt != nil {
		local := article.PublishedAt.Format(time.RFC3339)
		article.PublishedAtLocal = &local
	}

	var embedding *pgvector.Vector
	if len(article.Embedding) > 0 {
		v := pgvector.NewVector(article.Embedding)
		embedding = &v
	}

	query := `
		INSERT INTO articles (company_id, title, content, source, published_at, published_at_local, embedding)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
		ON CONFLICT DO NOTHING
		RETURNING id, created_at_utc, updated_at_utc`

	err := db.Pool.QueryRow(ctx, query,
		article.CompanyID,
		article.Title,
		article.Content,
		article.Source,
		article.PublishedAt,
		article.PublishedAtLocal,
		embedding,
	).Scan(&article.ID, &article.CreatedAt, &article.UpdatedAt)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			// Row was skipped due to ON CONFLICT DO NOTHING
			return nil
		}
		return fmt.Errorf("failed to insert article: %w", err)
	}

	return nil
}

func (db *DB) GetArticleByID(ctx context.Context, id int64) (*models.Article, error) {
	query := `
		SELECT id, company_id, title, content, source, published_at, published_at_local, created_at_utc, updated_at_utc
		FROM articles
		WHERE id = $1`

	article := &models.Article{}
	err := db.Pool.QueryRow(ctx, query, id).Scan(
		&article.ID,
		&article.CompanyID,
		&article.Title,
		&article.Content,
		&article.Source,
		&article.PublishedAt,
		&article.PublishedAtLocal,
		&article.CreatedAt,
		&article.UpdatedAt,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to get article by id: %w", err)
	}

	return article, nil
}

func (db *DB) ListArticlesByCompany(ctx context.Context, companyID int64) ([]*models.Article, error) {
	query := `
		SELECT id, company_id, title, content, source, published_at, published_at_local, created_at_utc, updated_at_utc
		FROM articles
		WHERE company_id = $1
		ORDER BY published_at DESC`

	rows, err := db.Pool.Query(ctx, query, companyID)
	if err != nil {
		return nil, fmt.Errorf("failed to list articles: %w", err)
	}
	defer rows.Close()

	var articles []*models.Article
	for rows.Next() {
		article := &models.Article{}
		err := rows.Scan(
			&article.ID,
			&article.CompanyID,
			&article.Title,
			&article.Content,
			&article.Source,
			&article.PublishedAt,
			&article.PublishedAtLocal,
			&article.CreatedAt,
			&article.UpdatedAt,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan article row: %w", err)
		}
		articles = append(articles, article)
	}

	return articles, nil
}

func (db *DB) DeleteArticle(ctx context.Context, id int64) error {
	query := `DELETE FROM articles WHERE id = $1`

	_, err := db.Pool.Exec(ctx, query, id)
	if err != nil {
		return fmt.Errorf("failed to delete article: %w", err)
	}

	return nil
}

// SearchResult holds a single semantic search result.
type SearchResult struct {
	Title      string
	URL        string
	Similarity float64
}

func (db *DB) SearchArticles(ctx context.Context, queryEmbedding []float32, limit int) ([]SearchResult, error) {
	vec := pgvector.NewVector(queryEmbedding)

	query := `
		SELECT title, source, 1 - (embedding <=> $1) AS similarity
		FROM articles
		ORDER BY embedding <=> $1
		LIMIT $2`

	rows, err := db.Pool.Query(ctx, query, vec, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to search articles: %w", err)
	}
	defer rows.Close()

	var results []SearchResult
	for rows.Next() {
		var r SearchResult
		if err := rows.Scan(&r.Title, &r.URL, &r.Similarity); err != nil {
			return nil, fmt.Errorf("failed to scan search result: %w", err)
		}
		results = append(results, r)
	}

	return results, nil
}
