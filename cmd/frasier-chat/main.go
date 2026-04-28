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
)

var (
	startupTimeout = 10 * time.Second
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

	geminiCfg := gemini.Config{
		ProjectID:      os.Getenv("GEMINI_PROJECT"),
		Location:       os.Getenv("GEMINI_LOCATION"),
		Model:          os.Getenv("GEMINI_MODEL"),
		EmbeddingModel: "text-embedding-004",
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
