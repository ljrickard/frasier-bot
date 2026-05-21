package database

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/jackc/pgx/v5/pgxpool"
)

// DB wraps the connection pool to provide custom methods later
type DB struct {
	Pool *pgxpool.Pool
}

// Connect establishes a connection pool to the PostgreSQL database using the provided DSN
func Connect(ctx context.Context, dsn string) (*DB, error) {
	slog.Debug("Attempting database connection to Cloud SQL...")

	// 1. Parse the DSN string into a config object
	config, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		return nil, fmt.Errorf("unable to parse database config: %w", err)
	}

	// 2. Create the connection pool
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		return nil, fmt.Errorf("unable to create database pool: %w", err)
	}

	// 3. Ping the database to verify the connection is actually alive
	if err := pool.Ping(ctx); err != nil {
		// If ping fails, close the pool to prevent resource leaks
		pool.Close()
		return nil, fmt.Errorf("database ping failed: %w", err)
	}

	slog.Info("✅ Successfully connected to Cloud SQL (PostgreSQL)")
	return &DB{Pool: pool}, nil
}

// Close cleanly shuts down the database connection pool
// Call this using `defer db.Close()` in your main.go
func (db *DB) Close() {
	if db.Pool != nil {
		slog.Info("🔌 Closing database connection pool")
		db.Pool.Close()
	}
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

	_, err = db.Pool.Exec(ctx, `CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)`)
	if err != nil {
		return fmt.Errorf("failed to create HNSW vector index: %w", err)
	}

	return nil
}

// WipeDatabase drops the entire public schema and recreates it.
// WARNING: This is destructive and will delete all tables, types, and extensions.
func (db *DB) WipeDatabase(ctx context.Context) error {
	// 1. Drop the schema and everything in it
	_, err := db.Pool.Exec(ctx, "DROP SCHEMA public CASCADE;")
	if err != nil {
		return fmt.Errorf("failed to drop public schema: %v", err)
	}

	// 2. Recreate the empty schema
	_, err = db.Pool.Exec(ctx, "CREATE SCHEMA public;")
	if err != nil {
		return fmt.Errorf("failed to recreate public schema: %v", err)
	}

	// 3. Re-grant permissions (default behavior in Postgres)
	_, err = db.Pool.Exec(ctx, "GRANT ALL ON SCHEMA public TO public;")
	if err != nil {
		return fmt.Errorf("failed to grant schema permissions: %v", err)
	}

	return nil
}
