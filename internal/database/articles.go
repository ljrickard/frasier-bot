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
		INSERT INTO articles (company_id, title, content, source, published_at, published_at_local, embedding, season, episode, episode_title, metadata, parent_id)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
		RETURNING id, created_at_utc, updated_at_utc`

	err := db.Pool.QueryRow(ctx, query,
		article.CompanyID,
		article.Title,
		article.Content,
		article.Source,
		article.PublishedAt,
		article.PublishedAtLocal,
		embedding,
		article.Season,
		article.Episode,
		article.EpisodeTitle,
		article.Metadata,
		article.ParentID,
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
	Content    string
	ParentID   *int64
	Similarity float64
}

func (db *DB) SearchArticles(ctx context.Context, queryEmbedding []float32, limit int) ([]SearchResult, error) {
	vec := pgvector.NewVector(queryEmbedding)

	query := `
		SELECT title, source, content, parent_id, 1 - (embedding <=> $1) AS similarity
		FROM articles
		WHERE embedding IS NOT NULL
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
		if err := rows.Scan(&r.Title, &r.URL, &r.Content, &r.ParentID, &r.Similarity); err != nil {
			return nil, fmt.Errorf("failed to scan search result: %w", err)
		}
		results = append(results, r)
	}

	return results, nil
}

// SearchArticlesDiverse retrieves the top-K semantically similar articles but
// limits results to at most perEpisodeLimit chunks per (season, episode) pair,
// ensuring the context window spans a diverse range of episodes rather than
// being dominated by a single highly-matching episode.
func (db *DB) SearchArticlesDiverse(ctx context.Context, queryEmbedding []float32, limit int, perEpisodeLimit int) ([]SearchResult, error) {
	vec := pgvector.NewVector(queryEmbedding)

	// Fetch a larger candidate pool so we have enough to fill `limit` slots
	// after per-episode capping. A multiplier of 5x is generous but bounded.
	candidateLimit := limit * 5

	query := `
		SELECT title, source, content, parent_id, season, episode, 1 - (embedding <=> $1) AS similarity
		FROM articles
		WHERE embedding IS NOT NULL
		ORDER BY embedding <=> $1
		LIMIT $2`

	rows, err := db.Pool.Query(ctx, query, vec, candidateLimit)
	if err != nil {
		return nil, fmt.Errorf("failed to search articles (diverse): %w", err)
	}
	defer rows.Close()

	type candidateRow struct {
		SearchResult
		season  int
		episode int
	}

	episodeCounts := make(map[[2]int]int)
	var results []SearchResult

	for rows.Next() {
		var c candidateRow
		if err := rows.Scan(&c.Title, &c.URL, &c.Content, &c.ParentID, &c.season, &c.episode, &c.Similarity); err != nil {
			return nil, fmt.Errorf("failed to scan search result (diverse): %w", err)
		}

		key := [2]int{c.season, c.episode}
		if episodeCounts[key] < perEpisodeLimit {
			episodeCounts[key]++
			results = append(results, c.SearchResult)
		}

		if len(results) >= limit {
			break
		}
	}

	return results, nil
}

// HasArticlesForSource checks if any articles already exist for a given source URL.
func (db *DB) HasArticlesForSource(ctx context.Context, source string) (bool, error) {
	var exists bool
	query := `SELECT EXISTS(SELECT 1 FROM articles WHERE source = $1)`
	err := db.Pool.QueryRow(ctx, query, source).Scan(&exists)
	if err != nil {
		return false, fmt.Errorf("failed to check articles for source: %w", err)
	}
	return exists, nil
}

// CreateParentChunk inserts a parent chunk and returns its ID.
func (db *DB) CreateParentChunk(ctx context.Context, chunk *models.ParentChunk) error {
	query := `
		INSERT INTO parent_chunks (content, season, episode, episode_title, url)
		VALUES ($1, $2, $3, $4, $5)
		RETURNING id`

	err := db.Pool.QueryRow(ctx, query,
		chunk.Content,
		chunk.Season,
		chunk.Episode,
		chunk.EpisodeTitle,
		chunk.URL,
	).Scan(&chunk.ID)
	if err != nil {
		return fmt.Errorf("failed to create parent chunk: %w", err)
	}

	return nil
}

// GetParentChunksByIDs fetches unique parent chunks by their IDs.
func (db *DB) GetParentChunksByIDs(ctx context.Context, ids []int64) ([]models.ParentChunk, error) {
	if len(ids) == 0 {
		return nil, nil
	}

	query := `
		SELECT id, content, season, episode, episode_title, url
		FROM parent_chunks
		WHERE id = ANY($1)`

	rows, err := db.Pool.Query(ctx, query, ids)
	if err != nil {
		return nil, fmt.Errorf("failed to get parent chunks: %w", err)
	}
	defer rows.Close()

	var chunks []models.ParentChunk
	for rows.Next() {
		var c models.ParentChunk
		if err := rows.Scan(&c.ID, &c.Content, &c.Season, &c.Episode, &c.EpisodeTitle, &c.URL); err != nil {
			return nil, fmt.Errorf("failed to scan parent chunk: %w", err)
		}
		chunks = append(chunks, c)
	}

	return chunks, nil
}
