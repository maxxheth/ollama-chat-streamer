package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/maxx/ollama-chat-streamer/go_version/chat"
	"github.com/maxx/ollama-chat-streamer/go_version/config"
	"github.com/maxx/ollama-chat-streamer/go_version/db"
)

func main() {
	cfgPath := flag.String("config", "", "Path to YAML config file")
	model := flag.String("model", "", "Ollama model to use")
	ollamaHost := flag.String("host", "", "Ollama host URL")
	contextPath := flag.String("context-path", "", "Path to context file")
	systemPrompt := flag.String("system", "", "System prompt")
	think := flag.String("think", "", "Think mode: auto, true, or false")
	websearch := flag.Bool("websearch", false, "Enable web search tools")
	noWebsearch := flag.Bool("no-websearch", false, "Disable web search tools")
	persist := flag.Bool("persist", false, "Enable DB persistence")
	noPersist := flag.Bool("no-persist", false, "Disable DB persistence")
	dbURL := flag.String("db", "", "Database URL")
	depth := flag.Int("subagent-depth", -1, "Max subagent depth (0 to disable)")
	rounds := flag.Int("subagent-rounds", -1, "Max subagent rounds")
	turnLimit := flag.Int("turn-limit", -1, "Max tool-call turns per chat turn (0 = unlimited)")
	help := flag.Bool("help", false, "Show help")

	flag.Parse()

	if *help {
		printHelp()
		return
	}

	// Auto-detect config if not explicitly provided
	resolvedCfgPath := *cfgPath
	if resolvedCfgPath == "" {
		resolvedCfgPath = os.Getenv("OLLAMA_CONFIG_PATH")
	}
	if resolvedCfgPath == "" {
		if _, err := os.Stat("ollama-chat.yaml"); err == nil {
			resolvedCfgPath = "ollama-chat.yaml"
		}
	}
	if resolvedCfgPath == "" {
		if home, err := os.UserHomeDir(); err == nil {
			path := home + "/.config/ollama-chat/ollama-chat.yaml"
			if _, err := os.Stat(path); err == nil {
				resolvedCfgPath = path
			}
		}
	}

	cfg, err := config.Load(resolvedCfgPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Config error: %v\n", err)
		os.Exit(1)
	}

	if *model != "" {
		cfg.Model = *model
	}
	if *ollamaHost != "" {
		cfg.OllamaHost = *ollamaHost
	}
	if *contextPath != "" {
		cfg.ContextPath = *contextPath
	}
	if *systemPrompt != "" {
		cfg.SystemPrompt = *systemPrompt
	}
	if *think != "" {
		cfg.Think = *think
	}
	if *websearch {
		cfg.ExperimentalWebSearch = true
	}
	if *noWebsearch {
		cfg.ExperimentalWebSearch = false
	}
	if *persist {
		cfg.PersistToDB = true
	}
	if *noPersist {
		cfg.PersistToDB = false
	}
	if *dbURL != "" {
		cfg.DatabaseURL = *dbURL
	}
	if *depth >= 0 {
		cfg.MaxSubagentDepth = *depth
	}
	if *rounds >= 0 {
		cfg.MaxSubagentRounds = *rounds
	}
	if *turnLimit >= 0 {
		cfg.TurnLimit = *turnLimit
	}

	var database *db.Pool
	if cfg.PersistToDB && cfg.DatabaseURL != "" {
		var err error
		database, err = db.Connect(context.Background(), cfg.DatabaseURL)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Database connection failed: %v\n", err)
			fmt.Fprintf(os.Stderr, "Continuing without persistence.\n")
			database = nil
		}
		if database != nil {
			defer database.Close()
		}
	}

	cl := chat.New(cfg, database)
	if err := cl.Run(context.Background()); err != nil {
		if err != context.Canceled {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
	}
}

func printHelp() {
	name := "ollama-chat"
	fmt.Fprintf(os.Stderr, `%[1]s — AI chat with tools and sub-agents

Usage:
  %[1]s [flags]

Flags:
  --help                   Show this help
  -config <path>           Path to YAML config file
  -model <name>            Ollama model (env: OLLAMA_MODEL)
  -host <url>              Ollama host (env: OLLAMA_HOST)
  -context-path <path>     Context file path (env: CONTEXT_PATH)
  -system <text>           System prompt (env: SYSTEM_PROMPT)
  -think <auto|true|false> Think mode  (env: THINK)
  -websearch               Enable web search (env: EXPERIMENTAL_WEBSEARCH)
  -no-websearch            Disable web search
  -persist                 Enable DB persistence (env: PERSIST_TO_DB)
  -no-persist              Disable DB persistence
  -db <url>                Database URL (env: DATABASE_URL)
  -subagent-depth <n>      Max subagent depth (env: MAX_SUBAGENT_DEPTH)
  -subagent-rounds <n>     Max subagent rounds (env: MAX_SUBAGENT_ROUNDS)
  -turn-limit <n>          Max tool-call turns per chat turn (env: TURN_LIMIT, default: 100)

Environment variables (in order of precedence: flag > env > yaml > default):
  OLLAMA_MODEL, OLLAMA_HOST, OLLAMA_MODEL_FALLBACKS, THINK,
  EXPERIMENTAL_WEBSEARCH, PERSIST_TO_DB, DATABASE_URL,
  MAX_SUBAGENT_DEPTH, MAX_SUBAGENT_ROUNDS, TURN_LIMIT, CONTEXT_PATH

Tools available when websearch is enabled:
  web_search, read_file, write_file, append_file, list_directory,
  run_shell, read_json_file, spawn_agent

Sub-agents:
  spawn_agent delegates complex tasks to sub-agents with their own
  isolated message context. Disable with --subagent-depth=0.

Examples:
  %[1]s
  %[1]s -model llama3.2:latest
  %[1]s -model lfm2.5:latest -think false -subagent-depth 0
`, name)

	fmt.Fprintf(os.Stderr, "\nConfig file lookup:\n")
	fmt.Fprintf(os.Stderr, "  1. --config flag\n")
	fmt.Fprintf(os.Stderr, "  2. OLLAMA_CONFIG_PATH env var\n")
	fmt.Fprintf(os.Stderr, "  3. ./ollama-chat.yaml\n")
	fmt.Fprintf(os.Stderr, "  4. ~/.config/ollama-chat/ollama-chat.yaml\n")
	fmt.Fprintf(os.Stderr, "  5. All defaults\n")

	fmt.Fprintf(os.Stderr, "\nExample ollama-chat.yaml:\n")
	fmt.Fprintf(os.Stderr, `  model: llama3.2:latest
  ollama_host: http://localhost:11434
  experimental_websearch: true
  persist_to_db: false
  max_subagent_depth: 1
  think: auto
`)

	fmt.Fprintf(os.Stderr, "\nDatabase (PostgreSQL):\n")
	fmt.Fprintf(os.Stderr, "  Use go_version/run.sh to auto-start Postgres via Docker\n")
	fmt.Fprintf(os.Stderr, "  Default: postgres://postgres:postgres@localhost:5434/chatdb\n")
	fmt.Fprintf(os.Stderr, "  Schema: conversations + messages tables auto-created\n")
}

func init() {
	flag.CommandLine.SetOutput(os.Stderr)
	flag.Usage = func() {
		printHelp()
		os.Exit(0)
	}
}
