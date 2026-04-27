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

type ChatRequest struct {
	Query     string            `json:"query"`
	SessionID string            `json:"session_id"`
	Config    *config.RAGConfig `json:"config,omitempty"`
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

	// Startup context for DB connection and Ping
	startupCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
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

	// ---------------------------------------------------------
	// 1. Initialize External AI Clients
	// ---------------------------------------------------------

	// Gemini Configuration with Exponential Backoff
	geminiCfg := gemini.Config{
		ProjectID: os.Getenv("GEMINI_PROJECT"),
		Location:  "us-central1",
		Model:     "gemini-2.5-flash",
		Retry: gemini.RetryConfig{
			MaxRetries: 3,
			BaseDelay:  2 * time.Second,
		},
	}

	// Note: We use a fresh Background context here so the client isn't tied to the startup timeout
	geminiClient, err := gemini.NewClient(context.Background(), geminiCfg)
	if err != nil {
		slog.Error("❌ Failed to initialize Gemini Client", "error", err)
		os.Exit(1)
	}
	// Note: No defer geminiClient.Close() needed for the new unified SDK!

	// Cross-Encoder Configuration
	encoderURL := os.Getenv("CROSS_ENCODER_URL")
	if encoderURL == "" {
		slog.Warn("⚠️ CROSS_ENCODER_URL not set, local reranking will fail")
	}
	encoderClient := crossencoder.NewClient(encoderURL)

	// ---------------------------------------------------------
	// 2. Assemble the AI Service
	// ---------------------------------------------------------

	// Inject the network clients into our domain service
	aiService := ai.NewService(geminiClient, encoderClient)

	// ---------------------------------------------------------
	// 3. HTTP Router Setup
	// ---------------------------------------------------------

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

		activeCfg := baseCfg
		if req.Config != nil {
			activeCfg = req.Config
			slog.Debug("Using request-provided RAG configuration overrides", "session_id", req.SessionID)
		}

		// Fail-fast context for the HTTP request to prevent upstream gateway hangs
		reqCtx, reqCancel := context.WithTimeout(r.Context(), 25*time.Second)
		defer reqCancel()

		slog.Info("🧠 RAG Pipeline Executing", "session_id", req.SessionID)

		// Pass the aiService deep into your pipeline execution
		res, err := search.RunRAGPipeline(reqCtx, db, activeCfg, aiService, req.Query)
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
