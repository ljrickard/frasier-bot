package config

import (
	"flag"
	"fmt"
	"strings"
)

// RAGConfig holds toggleable feature flags for the RAG pipeline.
type RAGConfig struct {
	UseMetadata    bool
	UseSwitchboard bool
	UseDiversity   bool
	UseExpansion   bool
	UseReranker    bool
	UsePersona     bool
	CompareMode    bool
	Debug          bool
}

// ParseFlags parses command-line flags and returns a RAGConfig.
func ParseFlags() *RAGConfig {
	cfg := &RAGConfig{}

	flag.BoolVar(&cfg.UseExpansion, "expansion", true, "Expand query keywords to match broader user intent")
	flag.BoolVar(&cfg.UseSwitchboard, "switchboard", true, "Dynamically adjust Top-K context size based on query type")
	flag.BoolVar(&cfg.UseDiversity, "diversity", true, "Force search results to span multiple different episodes")
	flag.BoolVar(&cfg.UseMetadata, "metadata", true, "Inject [SxxExx] tags for chronological awareness")
	flag.BoolVar(&cfg.UseReranker, "reranker", true, "Use LLM to grade and re-sort search results for accuracy")
	flag.BoolVar(&cfg.UsePersona, "persona", true, "Apply the Frasier/Crane brother persona to the final output")
	flag.BoolVar(&cfg.CompareMode, "compare", false, "Enable compare mode: Vanilla AI + Evaluation")
	flag.BoolVar(&cfg.Debug, "debug", false, "Enable verbose debug logging in the terminal")

	flag.Parse()
	return cfg
}

// PrintStatus prints a formatted table of feature flags.
func (c *RAGConfig) PrintStatus() {
	features := []struct {
		Name    string
		Enabled bool
	}{
		{"Query Expansion", c.UseExpansion},
		{"Switchboard", c.UseSwitchboard},
		{"Diversity Filter", c.UseDiversity},
		{"Metadata Prefixing", c.UseMetadata},
		{"Semantic Reranker", c.UseReranker},
		{"Frasier Persona", c.UsePersona},
		{"Compare Mode", c.CompareMode},
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
	fmt.Println("  └─────────────────────────┴──────────┘")
}
