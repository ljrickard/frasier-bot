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
	_, err := db.Pool.Exec(ctx, `CREATE EXTENSION IF NOT EXISTS vector`)
	if err != nil {
		return fmt.Errorf("failed to create vector extension: %w", err)
	}

	_, err = db.Pool.Exec(ctx, `
		CREATE TABLE IF NOT EXISTS companies (
			id             BIGSERIAL PRIMARY KEY,
			name           TEXT NOT NULL,
			ticker         TEXT,
			sector         TEXT,
			description    TEXT,
			created_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			updated_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW()
		)`)
	if err != nil {
		return fmt.Errorf("failed to create companies table: %w", err)
	}

	// Ignore error if constraint already exists
	_, _ = db.Pool.Exec(ctx, `
		ALTER TABLE companies
		ADD CONSTRAINT companies_name_unique UNIQUE (name)`)

	_, err = db.Pool.Exec(ctx, `
		CREATE TABLE IF NOT EXISTS articles (
			id                BIGSERIAL PRIMARY KEY,
			company_id        BIGINT NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
			title             TEXT NOT NULL,
			content           TEXT,
			source            TEXT,
			published_at      TIMESTAMPTZ,
			published_at_local TEXT,
			embedding         vector(768),
			created_at_utc    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			updated_at_utc    TIMESTAMPTZ NOT NULL DEFAULT NOW()
		)`)
	if err != nil {
		return fmt.Errorf("failed to create articles table: %w", err)
	}

	// Add new columns for series metadata (idempotent)
	_, _ = db.Pool.Exec(ctx, `ALTER TABLE articles ADD COLUMN IF NOT EXISTS season INTEGER`)
	_, _ = db.Pool.Exec(ctx, `ALTER TABLE articles ADD COLUMN IF NOT EXISTS episode INTEGER`)
	_, _ = db.Pool.Exec(ctx, `ALTER TABLE articles ADD COLUMN IF NOT EXISTS episode_title TEXT`)
	_, _ = db.Pool.Exec(ctx, `ALTER TABLE articles ADD COLUMN IF NOT EXISTS metadata JSONB`)

	return nil
}
