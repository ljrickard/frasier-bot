package config

import (
	"flag"
	"fmt"
	"log"
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
	NoRAGMode      bool
	Debug          bool
}

// ParseFlags parses command-line flags and returns a RAGConfig.
func ParseFlags() *RAGConfig {
	cfg := &RAGConfig{}

	flag.BoolVar(&cfg.UseExpansion, "expansion", true, "Expand query keywords...")
	// ... (rest of your flags)
	flag.BoolVar(&cfg.NoRAGMode, "compare", false, "Enable compare mode: Vanilla AI + Evaluation")
	flag.BoolVar(&cfg.Debug, "debug", false, "Enable verbose debug logging")

	flag.Parse()

	// THE FIX: Strict validation for Compare Mode
	if cfg.NoRAGMode {
		if cfg.UseExpansion || cfg.UseSwitchboard || cfg.UseDiversity || cfg.UseMetadata || cfg.UseReranker || cfg.UsePersona {
			log.Fatal("🚨 ERROR: Compare Mode (-compare) must be run completely vanilla. You must disable all other features (e.g., -reranker=false -persona=false).")
		}
	}
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
		{"No RAG Mode", c.NoRAGMode},
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
