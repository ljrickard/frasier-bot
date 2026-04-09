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
}

// ParseFlags parses command-line flags and returns a RAGConfig.
func ParseFlags() *RAGConfig {
	cfg := &RAGConfig{}

	flag.BoolVar(&cfg.UseMetadata, "metadata", true, "Prefix chunks with [SxxExx] episode metadata")
	flag.BoolVar(&cfg.UseSwitchboard, "switchboard", true, "Use LLM query classification (GENERAL/SPECIFIC)")
	flag.BoolVar(&cfg.UseDiversity, "diversity", true, "Apply per-episode diversity capping")
	flag.BoolVar(&cfg.UseExpansion, "expansion", true, "Use LLM query expansion/reformulation")
	flag.BoolVar(&cfg.UseReranker, "reranker", true, "Use LLM-based semantic reranking")
	flag.BoolVar(&cfg.UsePersona, "persona", true, "Use Frasier Crane personality in responses")
	flag.BoolVar(&cfg.CompareMode, "compare", false, "Enable compare mode: Vanilla AI + Evaluation")

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
