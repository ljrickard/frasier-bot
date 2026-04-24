SET timezone = 'UTC';
CREATE EXTENSION IF NOT EXISTS vector;

-- Replaces 'companies'
CREATE TABLE IF NOT EXISTS shows (
    id              BIGSERIAL PRIMARY KEY,
    title           TEXT NOT NULL UNIQUE,
    description     TEXT,
    created_at_utc  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at_utc  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Replaces 'parent_chunks' (Holds the full scene or episode text)
CREATE TABLE IF NOT EXISTS parent_chunks (
    id            BIGSERIAL PRIMARY KEY,
    show_id       BIGINT NOT NULL REFERENCES shows(id) ON DELETE CASCADE,
    content       TEXT NOT NULL,
    season        INTEGER,
    episode       INTEGER,
    episode_title TEXT,
    url           TEXT
);

-- Replaces 'articles' (Holds the vector embeddings for search)
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
);

CREATE INDEX IF NOT EXISTS idx_chunks_show_id ON chunks(show_id);