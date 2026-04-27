package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"time"

	"frasier-bot/internal/config"
	"frasier-bot/internal/database"
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

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	db, err := database.Connect(ctx, dsn)
	if err != nil {
		slog.Error("❌ Database connection failed", "error", err)
		os.Exit(1)
	}
	if err := db.Pool.Ping(ctx); err != nil {
		slog.Error("❌ Database healthcheck failed", "error", err)
		os.Exit(1)
	}
	defer db.Close()

	baseCfg := config.LoadBaseConfig()

	mux := http.NewServeMux()

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	mux.HandleFunc("/chat", func(w http.ResponseWriter, r *http.Request) {
		var req ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Bad Request", 400)
			return
		}

		activeCfg := baseCfg
		if req.Config != nil {
			activeCfg = req.Config
			slog.Debug("Using request-provided RAG configuration overrides", "session_id", req.SessionID)
		}

		// 1. Create a context that expires after 25 seconds
		// (This is shorter than the Gateway's timeout so we can return a clean error)
		ctx, cancel := context.WithTimeout(r.Context(), 25*time.Second)
		defer cancel()

		slog.Info("🧠 RAG Pipeline Executing", "session_id", req.SessionID)

		// 2. Pass this timed-out context into the pipeline
		res, err := search.RunRAGPipeline(ctx, db, activeCfg, req.Query)
		if err != nil {
			if ctx.Err() == context.DeadlineExceeded {
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
