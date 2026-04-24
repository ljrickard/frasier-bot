package database

import (
	"context"
	"fmt"

	"frasier-bot/internal/models"
)

func (db *DB) GetOrCreateShow(ctx context.Context, show *models.Show) error {
	query := `
		INSERT INTO shows (title, description)
		VALUES ($1, $2)
		ON CONFLICT (title) DO UPDATE SET title = EXCLUDED.title
		RETURNING id, created_at_utc, updated_at_utc`

	err := db.Pool.QueryRow(ctx, query,
		show.Title,
		show.Description,
	).Scan(&show.ID, &show.CreatedAt, &show.UpdatedAt)
	if err != nil {
		return fmt.Errorf("failed to get or create show: %w", err)
	}

	return nil
}
