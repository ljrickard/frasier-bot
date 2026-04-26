package database

import (
	"context"
	"fmt"

	"frasier-bot/internal/models"

	"github.com/pgvector/pgvector-go"
)

// CreateChunk inserts a new small dialogue chunk along with its vector embedding.
func (db *DB) CreateChunk(ctx context.Context, chunk *models.Chunk) error {
	var embedding *pgvector.Vector
	if len(chunk.Embedding) > 0 {
		v := pgvector.NewVector(chunk.Embedding)
		embedding = &v
	}

	query := `
		INSERT INTO chunks (show_id, parent_id, content, embedding, season, episode, episode_title, metadata)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
		RETURNING id, created_at_utc, updated_at_utc`

	err := db.Pool.QueryRow(ctx, query,
		chunk.ShowID,
		chunk.ParentID,
		chunk.Content,
		embedding,
		chunk.Season,
		chunk.Episode,
		chunk.EpisodeTitle,
		chunk.Metadata,
	).Scan(&chunk.ID, &chunk.CreatedAt, &chunk.UpdatedAt)

	if err != nil {
		return fmt.Errorf("failed to create chunk: %w", err)
	}

	return nil
}

// GetChunkByID fetches a specific chunk by its ID.
func (db *DB) GetChunkByID(ctx context.Context, id int64) (*models.Chunk, error) {
	query := `
		SELECT id, show_id, parent_id, content, season, episode, episode_title, metadata, created_at_utc, updated_at_utc
		FROM chunks
		WHERE id = $1`

	chunk := &models.Chunk{}
	err := db.Pool.QueryRow(ctx, query, id).Scan(
		&chunk.ID,
		&chunk.ShowID,
		&chunk.ParentID,
		&chunk.Content,
		&chunk.Season,
		&chunk.Episode,
		&chunk.EpisodeTitle,
		&chunk.Metadata,
		&chunk.CreatedAt,
		&chunk.UpdatedAt,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to get chunk by id: %w", err)
	}

	return chunk, nil
}

// ListChunksByShow retrieves all chunks for a specific TV show.
func (db *DB) ListChunksByShow(ctx context.Context, showID int64) ([]*models.Chunk, error) {
	query := `
		SELECT id, show_id, parent_id, content, season, episode, episode_title, metadata, created_at_utc, updated_at_utc
		FROM chunks
		WHERE show_id = $1
		ORDER BY created_at_utc DESC`

	rows, err := db.Pool.Query(ctx, query, showID)
	if err != nil {
		return nil, fmt.Errorf("failed to list chunks: %w", err)
	}
	defer rows.Close()

	var chunks []*models.Chunk
	for rows.Next() {
		chunk := &models.Chunk{}
		err := rows.Scan(
			&chunk.ID,
			&chunk.ShowID,
			&chunk.ParentID,
			&chunk.Content,
			&chunk.Season,
			&chunk.Episode,
			&chunk.EpisodeTitle,
			&chunk.Metadata,
			&chunk.CreatedAt,
			&chunk.UpdatedAt,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan chunk row: %w", err)
		}
		chunks = append(chunks, chunk)
	}

	return chunks, nil
}

// DeleteChunk removes a specific chunk from the database.
func (db *DB) DeleteChunk(ctx context.Context, id int64) error {
	query := `DELETE FROM chunks WHERE id = $1`

	_, err := db.Pool.Exec(ctx, query, id)
	if err != nil {
		return fmt.Errorf("failed to delete chunk: %w", err)
	}

	return nil
}

// SearchChunksDiverse retrieves the top-K semantically similar dialogue chunks but
// limits results to at most perEpisodeLimit chunks per (season, episode) pair,
// ensuring the context window spans a diverse range of episodes.
func (db *DB) SearchChunksWithEpisodeLimit(ctx context.Context, queryEmbedding []float32, limit int, perEpisodeLimit int) ([]models.SearchResult, error) {
	vec := pgvector.NewVector(queryEmbedding)

	// Fetch a larger candidate pool so we have enough to fill `limit` slots
	// after per-episode capping. A multiplier of 5x is generous but bounded.
	candidateLimit := limit * 5

	// We join with parent_chunks to optionally grab the source URL if needed.
	query := `
		SELECT 
			c.episode_title, 
			p.url, 
			c.content, 
			c.parent_id, 
			c.season, 
			c.episode, 
			1 - (c.embedding <=> $1) AS similarity
		FROM chunks c
		LEFT JOIN parent_chunks p ON c.parent_id = p.id
		WHERE c.embedding IS NOT NULL
		ORDER BY c.embedding <=> $1
		LIMIT $2`

	rows, err := db.Pool.Query(ctx, query, vec, candidateLimit)
	if err != nil {
		return nil, fmt.Errorf("failed to search chunks (diverse): %w", err)
	}
	defer rows.Close()

	episodeCounts := make(map[[2]int]int)
	var results []models.SearchResult

	for rows.Next() {
		var r models.SearchResult
		// Scan directly into the SearchResult struct
		if err := rows.Scan(&r.Title, &r.URL, &r.Content, &r.ParentID, &r.Season, &r.Episode, &r.Similarity); err != nil {
			return nil, fmt.Errorf("failed to scan search result (diverse): %w", err)
		}

		key := [2]int{r.Season, r.Episode}
		if episodeCounts[key] < perEpisodeLimit {
			episodeCounts[key]++
			results = append(results, r)
		}

		if len(results) >= limit {
			break
		}
	}

	return results, nil
}

// HasParentChunkForURL checks if a transcript has already been scraped for a given URL.
func (db *DB) HasParentChunkForURL(ctx context.Context, url string) (bool, error) {
	var exists bool
	query := `SELECT EXISTS(SELECT 1 FROM parent_chunks WHERE url = $1)`
	err := db.Pool.QueryRow(ctx, query, url).Scan(&exists)
	if err != nil {
		return false, fmt.Errorf("failed to check parent chunks for url: %w", err)
	}
	return exists, nil
}

// CreateParentChunk inserts a full episode transcript (the parent) and returns its ID.
func (db *DB) CreateParentChunk(ctx context.Context, chunk *models.ParentChunk) error {
	query := `
		INSERT INTO parent_chunks (show_id, content, season, episode, episode_title, url)
		VALUES ($1, $2, $3, $4, $5, $6)
		RETURNING id`

	err := db.Pool.QueryRow(ctx, query,
		chunk.ShowID,
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

// GetParentChunksByIDs fetches unique full transcripts by their IDs.
func (db *DB) GetParentChunksByIDs(ctx context.Context, ids []int64) ([]models.ParentChunk, error) {
	if len(ids) == 0 {
		return nil, nil
	}

	query := `
		SELECT id, show_id, content, season, episode, episode_title, url
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
		if err := rows.Scan(&c.ID, &c.ShowID, &c.Content, &c.Season, &c.Episode, &c.EpisodeTitle, &c.URL); err != nil {
			return nil, fmt.Errorf("failed to scan parent chunk: %w", err)
		}
		chunks = append(chunks, c)
	}

	return chunks, nil
}

// SearchChunks retrieves the top-K semantically similar dialogue chunks
// without any episode diversity limits.
func (db *DB) SearchChunks(ctx context.Context, queryEmbedding []float32, limit int) ([]models.SearchResult, error) {
	vec := pgvector.NewVector(queryEmbedding)

	// We join with parent_chunks to grab the source URL and episode info
	query := `
		SELECT 
			c.episode_title, 
			p.url, 
			c.content, 
			c.parent_id, 
			c.season, 
			c.episode, 
			1 - (c.embedding <=> $1) AS similarity
		FROM chunks c
		LEFT JOIN parent_chunks p ON c.parent_id = p.id
		WHERE c.embedding IS NOT NULL
		ORDER BY c.embedding <=> $1
		LIMIT $2`

	rows, err := db.Pool.Query(ctx, query, vec, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to search chunks: %w", err)
	}
	defer rows.Close()

	var results []models.SearchResult
	for rows.Next() {
		var r models.SearchResult
		if err := rows.Scan(&r.Title, &r.URL, &r.Content, &r.ParentID, &r.Season, &r.Episode, &r.Similarity); err != nil {
			return nil, fmt.Errorf("failed to scan search result: %w", err)
		}
		results = append(results, r)
	}

	return results, nil
}
