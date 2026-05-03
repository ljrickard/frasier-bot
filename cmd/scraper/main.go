package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	secretmanager "cloud.google.com/go/secretmanager/apiv1"
	"cloud.google.com/go/secretmanager/apiv1/secretmanagerpb"

	"frasier-bot/internal/database"
	"frasier-bot/internal/embeddings"
	"frasier-bot/internal/gemini"
	"frasier-bot/internal/ingest"
	"frasier-bot/internal/scraper"
)

func main() {
	ctx := context.Background()
	logger := log.New(os.Stdout, "[SCRAPER] ", log.Ldate|log.Ltime|log.Lshortfile)

	// 1. Connect to Database
	dbHost := os.Getenv("DB_HOST")
	dbUser := os.Getenv("DB_USER")
	dbPass := os.Getenv("DB_PASS")
	dbName := os.Getenv("DB_NAME")
	dsn := fmt.Sprintf("host=%s port=5432 user=%s password=%s dbname=%s sslmode=disable",
		dbHost, dbUser, dbPass, dbName)

	logger.Println("Connecting to database...")
	db, err := database.Connect(ctx, dsn)
	if err != nil {
		logger.Fatalf("Failed to connect to database: %v", err)
	}
	defer db.Close()

	// 2. Fetch API Key securely
	projectID := "pisces-12" // Or load from env like os.Getenv("GOOGLE_CLOUD_PROJECT")
	logger.Println("Fetching API key from Secret Manager...")
	apiKey, err := getSecret(ctx, projectID, "gemini-api-key")
	if err != nil {
		logger.Fatalf("Error loading API key: %v", err)
	}

	// 3. Initialize Gemini Client
	logger.Println("Initializing Gemini client...")
	geminiCfg := gemini.Config{
		ProjectID:      projectID,
		Location:       "us-central1",
		APIKey:         apiKey,
		TextModel:      "gemini-2.5-flash",
		EmbeddingModel: "gemini-embedding-001",
		Retry: gemini.RetryConfig{
			MaxRetries: 3,
			BaseDelay:  2 * time.Second,
		},
	}
	aiClient, err := gemini.NewClient(ctx, geminiCfg)
	if err != nil {
		logger.Fatalf("Failed to initialize Gemini client: %v", err)
	}

	// 4. Initialize Embeddings Service
	logger.Println("Initializing Embeddings service...")
	embedService, err := embeddings.New(embeddings.Config{
		Client: aiClient, // Injection happens here!
	})
	if err != nil {
		logger.Fatalf("Failed to init embeddings service: %v", err)
	}

	// 5. Initialize Scraper
	webScraper := scraper.New(logger)

	// 6. Initialize and Execute the Ingestion Pipeline
	logger.Println("Starting Ingestion Runner...")
	runner := &ingest.Runner{
		DB:         db,
		Scraper:    webScraper,
		Embeddings: embedService,
		Logger:     logger,
	}

	if err := runner.Run(ctx); err != nil {
		logger.Fatalf("Pipeline failed: %v", err)
	}

	logger.Println("Pipeline completed successfully.")
}

// getSecret securely fetches the string value of a secret from GCP Secret Manager
func getSecret(ctx context.Context, projectID, secretName string) (string, error) {
	smClient, err := secretmanager.NewClient(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to create Secret Manager client: %v", err)
	}
	defer smClient.Close()

	versionPath := fmt.Sprintf("projects/%s/secrets/%s/versions/latest", projectID, secretName)
	req := &secretmanagerpb.AccessSecretVersionRequest{
		Name: versionPath,
	}

	result, err := smClient.AccessSecretVersion(ctx, req)
	if err != nil {
		return "", fmt.Errorf("failed to access secret version: %v", err)
	}

	return string(result.Payload.Data), nil
}
