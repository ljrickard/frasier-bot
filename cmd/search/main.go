package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"frasier-bot/internal/config"
	"frasier-bot/internal/database"
	"frasier-bot/internal/embeddings"
)

func main() {
	ctx := context.Background()

	// Initialize Logger
	logger := log.New(os.Stderr, "", log.Ldate|log.Ltime|log.Lshortfile)

	// Parse Configuration
	cfg := config.ParseFlags()

	// Preflight Checks
	if err := embeddings.Preflight(); err != nil {
		logger.Fatalf("Embedding service preflight check failed: %v", err)
	}

	// Database Connection
	db, err := database.New(ctx)
	if err != nil {
		logger.Fatalf("Failed to connect to database: %v", err)
	}
	defer db.Close()

	// Welcome Banner
	sep := strings.Repeat("=", 80)
	fmt.Println()
	fmt.Println(sep)
	fmt.Println("  🎧 Frasier Chat Session Started")
	fmt.Println("  Ask me anything about Frasier! Type 'exit' or 'quit' to end.")
	fmt.Println(sep)
	fmt.Println()
	cfg.PrintStatus()
	fmt.Println()

	// Kick off the interactive loop
	startChatLoop(ctx, db, cfg, logger)
}
