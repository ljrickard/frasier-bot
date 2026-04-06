package database

import (
	"context"
	"fmt"

	"omnicorp-analyst/internal/models"
)

func (db *DB) CreateCompany(ctx context.Context, company *models.Company) error {
	query := `
		INSERT INTO companies (name, ticker, sector, description)
		VALUES ($1, $2, $3, $4)
		RETURNING id, created_at_utc, updated_at_utc`

	err := db.Pool.QueryRow(ctx, query,
		company.Name,
		company.Ticker,
		company.Sector,
		company.Description,
	).Scan(&company.ID, &company.CreatedAt, &company.UpdatedAt)
	if err != nil {
		return fmt.Errorf("failed to create company: %w", err)
	}

	return nil
}

func (db *DB) GetCompanyByID(ctx context.Context, id int64) (*models.Company, error) {
	query := `
		SELECT id, name, ticker, sector, description, created_at_utc, updated_at_utc
		FROM companies
		WHERE id = $1`

	company := &models.Company{}
	err := db.Pool.QueryRow(ctx, query, id).Scan(
		&company.ID,
		&company.Name,
		&company.Ticker,
		&company.Sector,
		&company.Description,
		&company.CreatedAt,
		&company.UpdatedAt,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to get company by id: %w", err)
	}

	return company, nil
}

func (db *DB) ListCompanies(ctx context.Context) ([]*models.Company, error) {
	query := `
		SELECT id, name, ticker, sector, description, created_at_utc, updated_at_utc
		FROM companies
		ORDER BY name ASC`

	rows, err := db.Pool.Query(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to list companies: %w", err)
	}
	defer rows.Close()

	var companies []*models.Company
	for rows.Next() {
		company := &models.Company{}
		err := rows.Scan(
			&company.ID,
			&company.Name,
			&company.Ticker,
			&company.Sector,
			&company.Description,
			&company.CreatedAt,
			&company.UpdatedAt,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan company row: %w", err)
		}
		companies = append(companies, company)
	}

	return companies, nil
}

func (db *DB) UpdateCompany(ctx context.Context, company *models.Company) error {
	query := `
		UPDATE companies
		SET name = $1, ticker = $2, sector = $3, description = $4, updated_at_utc = NOW()
		WHERE id = $5
		RETURNING updated_at_utc`

	err := db.Pool.QueryRow(ctx, query,
		company.Name,
		company.Ticker,
		company.Sector,
		company.Description,
		company.ID,
	).Scan(&company.UpdatedAt)
	if err != nil {
		return fmt.Errorf("failed to update company: %w", err)
	}

	return nil
}

func (db *DB) DeleteCompany(ctx context.Context, id int64) error {
	query := `DELETE FROM companies WHERE id = $1`

	_, err := db.Pool.Exec(ctx, query, id)
	if err != nil {
		return fmt.Errorf("failed to delete company: %w", err)
	}

	return nil
}
