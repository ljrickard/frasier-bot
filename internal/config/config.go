package config

import (
	"os"
	"strings"
)

// RAGConfig now has json tags so it can be parsed from the HTTP request body
type RAGConfig struct {
	UseMetadata            bool   `json:"use_metadata"`
	UseQueryClassification bool   `json:"use_query_classification"`
	UseEpisodeLimit        bool   `json:"use_episode_limit"`
	UseQueryExpansion      bool   `json:"use_query_expansion"`
	UseReranker            bool   `json:"use_reranker"`
	RerankerBackend        string `json:"reranker_backend"`
	UsePersona             bool   `json:"use_persona"`
	UseRAG                 bool   `json:"use_rag"`
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
