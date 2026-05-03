package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"time"

	"frasier-bot/internal/ai"
	"frasier-bot/internal/config"
	"frasier-bot/internal/crossencoder"
	"frasier-bot/internal/database"
	"frasier-bot/internal/gemini"
	"frasier-bot/internal/search"

	secretmanager "cloud.google.com/go/secretmanager/apiv1"
	"cloud.google.com/go/secretmanager/apiv1/secretmanagerpb"
)

var (
	startupTimeout = 30 * time.Second
)

type ChatRequest struct {
	Query     string          `json:"query"`
	SessionID string          `json:"session_id"`
	Config    json.RawMessage `json:"config,omitempty"`
}

func main() {
	level := slog.LevelInfo
	if os.Getenv("LOGGING_LEVEL") == "DEBUG" {
		level = slog.LevelDebug
	}
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: level})))

	slog.Info("🚀 Frasier Chat Service starting...")

	dsn := fmt.Sprintf("host=%s port=5432 user=%s password=%s dbname=%s sslmode=disable",
		os.Getenv("DB_HOST"), os.Getenv("DB_USER"), os.Getenv("DB_PASS"), os.Getenv("DB_NAME"))

	startupCtx, cancel := context.WithTimeout(context.Background(), startupTimeout)
	defer cancel()

	db, err := database.Connect(startupCtx, dsn)
	if err != nil {
		slog.Error("❌ Database connection failed", "error", err)
		os.Exit(1)
	}
	if err := db.Pool.Ping(startupCtx); err != nil {
		slog.Error("❌ Database healthcheck failed", "error", err)
		os.Exit(1)
	}
	defer db.Close()

	baseCfg := config.LoadBaseConfig()

	projectID := "pisces-12" // Or load from env like os.Getenv("GOOGLE_CLOUD_PROJECT")
	slog.Info("Fetching API key from Secret Manager...")
	apiKey, err := getSecret(startupCtx, projectID, "gemini-api-key")
	if err != nil {
		slog.Error("Error loading API key", "error", err)
		os.Exit(1)
	}

	geminiCfg := gemini.Config{
		ProjectID:      os.Getenv("GEMINI_PROJECT"),
		Location:       os.Getenv("GEMINI_LOCATION"),
		TextModel:      os.Getenv("GEMINI_MODEL"),
		APIKey:         apiKey,
		EmbeddingModel: "gemini-embedding-001",
		Retry: gemini.RetryConfig{
			MaxRetries: 3,
			BaseDelay:  2 * time.Second,
		},
	}

	geminiClient, err := gemini.NewClient(context.Background(), geminiCfg)
	if err != nil {
		slog.Error("❌ Failed to initialize Gemini Client", "error", err)
		os.Exit(1)
	}

	encoderURL := os.Getenv("CROSS_ENCODER_URL")
	if encoderURL == "" {
		slog.Warn("⚠️ CROSS_ENCODER_URL not set, local reranking will fail")
	}

	encoderClient := crossencoder.NewClient(encoderURL)
	aiService := ai.NewService(geminiClient, encoderClient)

	mux := http.NewServeMux()

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	mux.HandleFunc("/chat", func(w http.ResponseWriter, r *http.Request) {
		var req ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Bad Request", http.StatusBadRequest)
			return
		}

		// 2. Dereference baseCfg to create a brand new COPY by value
		activeCfg := *baseCfg

		// 3. Unpack the partial JSON directly over our copy!
		if len(req.Config) > 0 {
			if err := json.Unmarshal(req.Config, &activeCfg); err != nil {
				slog.Error("⚠️ Failed to merge config overrides, falling back to base config", "error", err)
			} else {
				slog.Debug("🔧 Merged request-provided RAG overrides", "session_id", req.SessionID)
			}
		}

		// Fail-fast context for the HTTP request to prevent upstream gateway hangs
		reqCtx, reqCancel := context.WithTimeout(r.Context(), 60*time.Second)
		defer reqCancel()

		// 4. Pass a pointer to our newly merged activeCfg
		res, err := search.RunRAGPipeline(reqCtx, db, &activeCfg, aiService, req.Query)
		if err != nil {
			if reqCtx.Err() == context.DeadlineExceeded {
				slog.Error("Pipeline Timed Out", "session_id", req.SessionID)
				http.Error(w, "Request Timed Out", http.StatusGatewayTimeout)
				return
			}
			slog.Error("Pipeline Error", "error", err, "session_id", req.SessionID)
			http.Error(w, "Internal Server Error", http.StatusInternalServerError)
			return
		}

		json.NewEncoder(w).Encode(map[string]any{
			"answer":       res.Answer,
			"reformulated": res.Reformulated,
			"episodes":     res.Contexts,
		})
	})

	slog.Info("🤖 Frasier Bot API Listening on :8080")
	http.ListenAndServe(":8080", mux)
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
