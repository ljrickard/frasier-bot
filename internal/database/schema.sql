SET timezone = 'UTC';

CREATE TABLE IF NOT EXISTS companies (
    id          BIGSERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    ticker      TEXT NOT NULL UNIQUE,
    sector      TEXT,
    description TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS articles (
    id           BIGSERIAL PRIMARY KEY,
    company_id   BIGINT NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    title        TEXT NOT NULL,
    content      TEXT,
    source       TEXT,
    published_at TIMESTAMPTZ,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_articles_company_id ON articles(company_id);
CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles(published_at DESC);
