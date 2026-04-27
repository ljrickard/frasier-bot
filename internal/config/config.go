package config

import (
	"flag"
	"fmt"
	"log"
	"strings"
)

type RAGConfig struct {
	UseMetadata            bool
	UseQueryClassification bool
	UseEpisodeLimit        bool
	UseQueryExpansion      bool
	UseReranker            bool
	RerankerBackend        string // NEW: "gemini" or "local"
	UsePersona             bool
	UseRAG                 bool
	UseEval                bool
	Debug                  bool
}

func ParseFlags() *RAGConfig {
	cfg := &RAGConfig{}

	flag.BoolVar(&cfg.UseQueryExpansion, "queryexpansion", true, "Expand query keywords to match broader intent")
	flag.BoolVar(&cfg.UseQueryClassification, "switchboard", true, "Dynamically adjust Top-K context size")
	flag.BoolVar(&cfg.UseEpisodeLimit, "diversity", true, "Force search results to span different episodes")
	flag.BoolVar(&cfg.UseMetadata, "metadata", true, "Inject [SxxExx] tags for chronological awareness")
	flag.BoolVar(&cfg.UseReranker, "reranker", true, "Use LLM to re-sort search results for accuracy")
	flag.StringVar(&cfg.RerankerBackend, "reranker-backend", "gemini", "Which reranker to use: 'gemini' or 'local'")
	flag.BoolVar(&cfg.UsePersona, "persona", true, "Apply the Frasier/Crane brother persona to the final output")
	flag.BoolVar(&cfg.UseRAG, "rag", true, "Enable the RAG pipeline (set to false for Vanilla AI)")
	flag.BoolVar(&cfg.UseEval, "eval", true, "Enable the Eval pipeline (Get back a answer_relevancy and faithfulness score)")
	flag.BoolVar(&cfg.Debug, "debug", false, "Enable verbose debug logging in the terminal")

	flag.Parse()

	// THE FIX: Only log.Fatal if the user EXPLICITLY enabled a RAG feature while using -rag=false
	if !cfg.UseRAG {
		conflicts := make([]string, 0)
		flag.Visit(func(f *flag.Flag) {
			switch f.Name {
			case "expansion", "switchboard", "diversity", "metadata", "reranker":
				if val, ok := f.Value.(flag.Getter).Get().(bool); ok && val {
					conflicts = append(conflicts, "-"+f.Name)
				}
			}
		})

		if len(conflicts) > 0 {
			log.Fatalf("🚨 ERROR: Vanilla Mode (-rag=false) cannot be used with explicitly enabled RAG features: %s. Disable these flags or use default RAG.", strings.Join(conflicts, ", "))
		}

		// If no explicit conflict, silently disable defaults for the Vanilla run
		cfg.UseQueryExpansion = false
		cfg.UseQueryClassification = false
		cfg.UseEpisodeLimit = false
		cfg.UseMetadata = false
		cfg.UseReranker = false
	}

	return cfg
}

func (c *RAGConfig) PrintStatus() {
	features := []struct {
		Name    string
		Enabled bool
	}{
		{"Query Query Expansion", c.UseQueryExpansion},
		{"Query Classification", c.UseQueryClassification},
		{"Episode Limit", c.UseEpisodeLimit},
		{"Metadata Enrichment", c.UseMetadata},
		{"Semantic Reranker", c.UseReranker},
		{"Frasier Persona", c.UsePersona},
		{"RAG Pipeline", c.UseRAG},
		{"Evaluate Answer", c.UseEval},
		{"Debug Logging", c.Debug},
	}

	fmt.Println("  ┌─────────────────────────┬──────────┐")
	fmt.Println("  │ Feature                 │ Status   │")
	fmt.Println("  ├─────────────────────────┼──────────┤")
	for _, f := range features {
		status := "\033[32m ENABLED\033[0m"
		if !f.Enabled {
			status = "\033[31mDISABLED\033[0m"
		}
		name := f.Name + strings.Repeat(" ", 23-len(f.Name))
		fmt.Printf("  │ %s │ %s │\n", name, status)
	}
	if c.UseReranker {
		fmt.Printf("  │ %s │ \033[34m%-8s\033[0m │\n", "Reranker Backend       ", c.RerankerBackend)
	}
	fmt.Println("  └─────────────────────────┴──────────┘")
}
