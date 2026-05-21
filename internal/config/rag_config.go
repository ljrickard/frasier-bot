package config

import (
	"os"
	"strconv"
	"strings"
)

// RAGConfig now has json tags so it can be parsed from the HTTP request body
type RAGConfig struct {
	UseMetadata            bool    `json:"use_metadata"`
	UseQueryClassification bool    `json:"use_query_classification"`
	UseEpisodeLimit        bool    `json:"use_episode_limit"`
	UseQueryExpansion      bool    `json:"use_query_expansion"`
	UseReranker            bool    `json:"use_reranker"`
	RerankerBackend        string  `json:"reranker_backend"`
	UsePersona             bool    `json:"use_persona"`
	UseRAG                 bool    `json:"use_rag"`
	FetchK                 int     `json:"fetch_k"`
	FinalK                 int     `json:"final_k"`
	SpecificScaleFetch     float64 `json:"specific_scale_fetch"`
	SpecificScaleFinal     float64 `json:"specific_scale_final"`
}

// LoadBaseConfig reads the defaults from the Kubernetes environment
func LoadBaseConfig() *RAGConfig {
	return &RAGConfig{
		UseMetadata:            getEnvBool("RAG_USE_METADATA", true),
		UseQueryClassification: getEnvBool("RAG_USE_CLASSIFICATION", true),
		UseEpisodeLimit:        getEnvBool("RAG_USE_EPISODE_LIMIT", true),
		UseQueryExpansion:      getEnvBool("RAG_USE_EXPANSION", true),
		UseReranker:            getEnvBool("RAG_USE_RERANKER", true),
		RerankerBackend:        getEnv("RAG_RERANKER_BACKEND", "local"),
		UsePersona:             getEnvBool("RAG_USE_PERSONA", true),
		UseRAG:                 getEnvBool("RAG_USE_RAG", true),
		FetchK:                 getEnvInt("RAG_FETCH_K", 50),
		FinalK:                 getEnvInt("RAG_FINAL_K", 12),
		SpecificScaleFetch:     getEnvFloat("RAG_SPECIFIC_SCALE_FETCH", 0.60),
		SpecificScaleFinal:     getEnvFloat("RAG_SPECIFIC_SCALE_FINAL", 0.66),
	}
}

// Helper functions for environment variables
func getEnv(key, fallback string) string {
	if value, ok := os.LookupEnv(key); ok {
		return value
	}
	return fallback
}

func getEnvBool(key string, fallback bool) bool {
	s := os.Getenv(key)
	if s == "" {
		return fallback
	}
	return strings.ToLower(s) == "true"
}

// Quick helper for parsing floats from env vars
func getEnvFloat(key string, fallback float64) float64 {
	if val, ok := os.LookupEnv(key); ok {
		if parsed, err := strconv.ParseFloat(val, 64); err == nil {
			return parsed
		}
	}
	return fallback
}

// Helper for parsing integers from environment variables with a fallback
func getEnvInt(key string, fallback int) int {
	if val, ok := os.LookupEnv(key); ok {
		if parsed, err := strconv.Atoi(val); err == nil {
			return parsed
		}
	}
	return fallback
}
