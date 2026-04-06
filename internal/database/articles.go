package database

import (
	"context"
	"fmt"

	"omnicorp-analyst/internal/models"
)

func (db *DB) CreateArticle(ctx context.Context, article *models.Article) error {
	if article.PublishedAt != nil {
		utc := article.PublishedAt.UTC()
		article.PublishedAt = &utc
	}

	query := `
		INSERT INTO articles (company_id, title, content, source, published_at)
		VALUES ($1, $2, $3, $4, $5)
		RETURNING id, created_at, updated_at`

	err := db.Pool.QueryRow(ctx, query,
		article.CompanyID,
		article.Title,
		article.Content,
		article.Source,
		article.PublishedAt,
	).Scan(&article.ID, &article.CreatedAt, &article.UpdatedAt)
	if err != nil {
		return fmt.Errorf("failed to create article: %w", err)
	}

	return nil
}

func (db *DB) GetArticleByID(ctx context.Context, id int64) (*models.Article, error) {
	query := `
		SELECT id, company_id, title, content, source, published_at, created_at, updated_at
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
		SELECT id, company_id, title, content, source, published_at, created_at, updated_at
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
