package database

import (
	"context"
	"fmt"
	"os"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type DB struct {
	Pool *pgxpool.Pool
}

func New(ctx context.Context) (*DB, error) {
	connStr := os.Getenv("DATABASE_URL")
	if connStr == "" {
		return nil, fmt.Errorf("DATABASE_URL environment variable is not set")
	}

	config, err := pgxpool.ParseConfig(connStr)
	if err != nil {
		return nil, fmt.Errorf("failed to parse connection string: %w", err)
	}

	config.MaxConns = 10

	config.AfterConnect = func(ctx context.Context, conn *pgx.Conn) error {
		_, err := conn.Exec(ctx, "SET timezone = 'UTC'")
		return err
	}

	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create connection pool: %w", err)
	}

	if err := pool.Ping(ctx); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	return &DB{Pool: pool}, nil
}

func (db *DB) Close() {
	db.Pool.Close()
}

func (db *DB) RunMigrations(ctx context.Context) error {
	// 1. Ensure pgvector is installed
	_, err := db.Pool.Exec(ctx, `CREATE EXTENSION IF NOT EXISTS vector`)
	if err != nil {
		return fmt.Errorf("failed to create vector extension: %w", err)
	}

	// 2. Create Shows Table
	_, err = db.Pool.Exec(ctx, `
		CREATE TABLE IF NOT EXISTS shows (
			id              BIGSERIAL PRIMARY KEY,
			title           TEXT NOT NULL UNIQUE,
			description     TEXT,
			created_at_utc  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			updated_at_utc  TIMESTAMPTZ NOT NULL DEFAULT NOW()
		)`)
	if err != nil {
		return fmt.Errorf("failed to create shows table: %w", err)
	}

	// 3. Create Parent Chunks Table (Full episodes/scenes)
	_, err = db.Pool.Exec(ctx, `
		CREATE TABLE IF NOT EXISTS parent_chunks (
			id            BIGSERIAL PRIMARY KEY,
			show_id       BIGINT NOT NULL REFERENCES shows(id) ON DELETE CASCADE,
			content       TEXT NOT NULL,
			season        INTEGER,
			episode       INTEGER,
			episode_title TEXT,
			url           TEXT
		)`)
	if err != nil {
		return fmt.Errorf("failed to create parent_chunks table: %w", err)
	}

	// 4. Create Chunks Table (Embeddable dialogue segments)
	_, err = db.Pool.Exec(ctx, `
		CREATE TABLE IF NOT EXISTS chunks (
			id                 BIGSERIAL PRIMARY KEY,
			show_id            BIGINT NOT NULL REFERENCES shows(id) ON DELETE CASCADE,
			parent_id          BIGINT REFERENCES parent_chunks(id) ON DELETE CASCADE,
			content            TEXT NOT NULL,
			embedding          vector(768),
			season             INTEGER,
			episode            INTEGER,
			episode_title      TEXT,
			metadata           JSONB,
			created_at_utc     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			updated_at_utc     TIMESTAMPTZ NOT NULL DEFAULT NOW()
		)`)
	if err != nil {
		return fmt.Errorf("failed to create chunks table: %w", err)
	}

	// 5. Create Indexes for performance
	_, _ = db.Pool.Exec(ctx, `CREATE INDEX IF NOT EXISTS idx_chunks_show_id ON chunks(show_id)`)

	return nil
}

// ClearDatabase updated to drop all data from the new tables
func (db *DB) ClearDatabase(ctx context.Context) error {
	_, err := db.Pool.Exec(ctx, `DELETE FROM chunks`)
	if err != nil {
		return fmt.Errorf("failed to clear chunks: %w", err)
	}

	_, err = db.Pool.Exec(ctx, `DELETE FROM parent_chunks`)
	if err != nil {
		return fmt.Errorf("failed to clear parent_chunks: %w", err)
	}

	_, err = db.Pool.Exec(ctx, `DELETE FROM shows`)
	if err != nil {
		return fmt.Errorf("failed to clear shows: %w", err)
	}

	return nil
}
