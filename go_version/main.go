package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

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
	autoCompactTarget := flag.Int("auto-compact-target", -1, "Target context percentage after compaction (env: AUTO_COMPACT_TARGET, default: 50)")
	autoCompactKeepRecent := flag.Int("auto-compact-keep-recent", -1, "Messages to preserve verbatim during compaction (env: AUTO_COMPACT_KEEP_RECENT, default: 8)")
	toolResultMaxInline := flag.Int("tool-result-max-inline", -1, "Max tool result bytes to keep inline before artifactizing (env: TOOL_RESULT_MAX_INLINE, default: 12000)")
	numCtx := flag.Int("num-ctx", -1, "Override model context window in tokens (env: NUM_CTX, default: model native)")
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
	if *autoCompactTarget >= 0 {
		cfg.AutoCompactTarget = *autoCompactTarget
	}
	if *autoCompactKeepRecent > 0 {
		cfg.AutoCompactKeepRecent = *autoCompactKeepRecent
	}
	if *toolResultMaxInline > 0 {
		cfg.ToolResultMaxInline = *toolResultMaxInline
	}
	if *numCtx > 0 {
		cfg.NumCtx = *numCtx
	}

	var database *db.Pool
	if cfg.PersistToDB && cfg.DatabaseURL != "" {
		// Proactively start Postgres before attempting to connect
		startPostgresViaCompose(cfg.DatabaseURL)

		var err error
		for i := 0; i < 16; i++ {
			database, err = db.Connect(context.Background(), cfg.DatabaseURL)
			if err == nil {
				break
			}
			if i == 0 {
				fmt.Fprintf(os.Stderr, "Database not available (%s)\n", cfg.DatabaseURL)
				fmt.Fprintf(os.Stderr, "Waiting for Postgres to become ready...\n")
			} else if i < 15 {
				fmt.Fprintf(os.Stderr, "  Not ready yet, waiting...\n")
			}
			time.Sleep(2 * time.Second)
		}

		if err != nil {
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

// startPostgresViaCompose looks for docker-compose.yml in the config directory
// and attempts to start the postgres service, passing POSTGRES_PORT to match
// the database URL so the port mapping is always consistent.
func startPostgresViaCompose(databaseURL string) bool {
	home, err := os.UserHomeDir()
	if err != nil {
		return false
	}

	composeFile := filepath.Join(home, ".config", "ollama-chat", "docker-compose.yml")
	if _, err := os.Stat(composeFile); err != nil {
		fmt.Fprintf(os.Stderr, "  Compose file not found at %s\n", composeFile)
		return false
	}

	if _, err := exec.LookPath("docker"); err != nil {
		fmt.Fprintf(os.Stderr, "  Docker is not installed\n")
		return false
	}

	// Extract port from database URL so docker-compose port matches
	postgresPort := extractPort(databaseURL)

	// Determine which docker compose command is available
	composeCmd := ""
	if cmd, err := exec.LookPath("docker-compose"); err == nil {
		composeCmd = cmd
	} else {
		// Check for docker compose plugin
		if err := exec.Command("docker", "compose", "version").Run(); err == nil {
			composeCmd = "docker compose"
		}
	}

	if composeCmd == "" {
		fmt.Fprintf(os.Stderr, "  Neither 'docker compose' nor 'docker-compose' found\n")
		return false
	}

	fmt.Fprintf(os.Stderr, "  Running: %s -f %s up -d postgres (port %s)\n", composeCmd, composeFile, postgresPort)

	// Remove any existing container with the target name so docker-compose
	// can create a fresh one with the current configuration. This handles the
	// case where a container was left behind by a previous run or created
	// outside the compose project.
	exec.Command("docker", "rm", "-f", "ollama-chat-db-go").Run()

	var cmd *exec.Cmd
	if composeCmd == "docker compose" {
		cmd = exec.Command("docker", "compose", "-f", composeFile, "up", "-d", "postgres")
	} else {
		cmd = exec.Command(composeCmd, "-f", composeFile, "up", "-d", "postgres")
	}
	cmd.Env = append(os.Environ(), "POSTGRES_PORT="+postgresPort)

	output, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Fprintf(os.Stderr, "  Failed to start Postgres: %s\n", strings.TrimSpace(string(output)))
		return false
	}

	fmt.Fprintf(os.Stderr, "  %s", strings.TrimSpace(string(output)))
	return true
}

// extractPort parses the port from a postgres:// URL.
// Returns "5432" if the URL has no explicit port or can't be parsed.
func extractPort(databaseURL string) string {
	// Strip postgres:// or postgresql:// prefix
	s := databaseURL
	for _, prefix := range []string{"postgresql://", "postgres://"} {
		if strings.HasPrefix(s, prefix) {
			s = strings.TrimPrefix(s, prefix)
			break
		}
	}

	// Find the port: after the last colon before the first slash
	// Format: user:pass@host:port/dbname
	if atIdx := strings.LastIndex(s, "@"); atIdx >= 0 {
		s = s[atIdx+1:]
	}
	if colonIdx := strings.LastIndex(s, ":"); colonIdx >= 0 {
		slashIdx := strings.Index(s, "/")
		if slashIdx < 0 {
			slashIdx = len(s)
		}
		if colonIdx < slashIdx {
			portStr := s[colonIdx+1 : slashIdx]
			if portStr != "" {
				return portStr
			}
		}
	}

	return "5432"
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
  -auto-compact-target <n> Target context percent after compaction (env: AUTO_COMPACT_TARGET, default: 50)
  -auto-compact-keep-recent <n>
                           Messages preserved verbatim during compaction (env: AUTO_COMPACT_KEEP_RECENT, default: 8)
  -tool-result-max-inline <n>
                           Max tool result bytes before artifact storage (env: TOOL_RESULT_MAX_INLINE, default: 12000)
  -num-ctx <n>             Override model context window in tokens (env: NUM_CTX, default: model native)

Environment variables (in order of precedence: flag > env > yaml > default):
  OLLAMA_MODEL, OLLAMA_HOST, OLLAMA_MODEL_FALLBACKS, THINK,
  EXPERIMENTAL_WEBSEARCH, PERSIST_TO_DB, DATABASE_URL,
  MAX_SUBAGENT_DEPTH, MAX_SUBAGENT_ROUNDS, TURN_LIMIT, CONTEXT_PATH, NUM_CTX,
  AUTO_COMPACT_TARGET, AUTO_COMPACT_KEEP_RECENT, TOOL_RESULT_MAX_INLINE

Tools available when websearch is enabled:
  web_search, read_file, write_file, append_file, list_directory,
  run_shell, read_json_file, spawn_agent, graphify

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
  auto_compact: true
  auto_compact_limit: 75
  auto_compact_target: 50
  auto_compact_keep_recent: 8
  tool_result_max_inline: 12000
`)

	fmt.Fprintf(os.Stderr, "\nDatabase (PostgreSQL):\n")
	fmt.Fprintf(os.Stderr, "  Auto-started via ~/.config/ollama-chat/docker-compose.yml\n")
	fmt.Fprintf(os.Stderr, "  Default: postgres://postgres:postgres@localhost:5432/chatdb\n")
	fmt.Fprintf(os.Stderr, "  Schema: conversations + messages tables auto-created\n")
}

func init() {
	flag.CommandLine.SetOutput(os.Stderr)
	flag.Usage = func() {
		printHelp()
		os.Exit(0)
	}
}
