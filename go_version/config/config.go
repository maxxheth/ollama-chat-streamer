package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/maxx/ollama-chat-streamer/go_version/ollama"
	"gopkg.in/yaml.v3"
)

type Config struct {
	Model                 string `yaml:"model"`
	ModelFallbacks        string `yaml:"model_fallbacks"`
	OllamaHost            string `yaml:"ollama_host"`
	ContextPath           string `yaml:"context_path"`
	SystemPrompt          string `yaml:"system_prompt"`
	ExperimentalWebSearch bool   `yaml:"experimental_websearch"`
	PersistToDB           bool   `yaml:"persist_to_db"`
	DatabaseURL           string `yaml:"database_url"`
	MaxSubagentDepth      int    `yaml:"max_subagent_depth"`
	MaxSubagentRounds     int    `yaml:"max_subagent_rounds"`
	Think                 string `yaml:"think"`
	ReadFileMaxLines      int    `yaml:"read_file_max_lines"`
	ReadFileMaxBytes      int64  `yaml:"read_file_max_bytes"`
	ReadFileMaxSize       int64  `yaml:"read_file_max_size"`
	ModelTimeout          int    `yaml:"model_timeout"`
	TurnLimit             int    `yaml:"turn_limit"`
	InstallPrefix         string `yaml:"install_prefix"`
	Compiled              bool   `yaml:"compiled"`
	AutoCompact           bool   `yaml:"auto_compact"`
	AutoCompactLimit      int    `yaml:"auto_compact_limit"`
	NumCtx                int    `yaml:"num_ctx"`
}

func Default() *Config {
	return &Config{
		Model:                 "llama3.2:latest",
		OllamaHost:            "http://localhost:11434",
		ExperimentalWebSearch: true,
		PersistToDB:           true,
		MaxSubagentDepth:      1,
		MaxSubagentRounds:     10,
		Think:                 "auto",
		ReadFileMaxLines:      200,
		ReadFileMaxBytes:      65536,
		ReadFileMaxSize:       524288,
		ModelTimeout:          120,
		TurnLimit:             100,
		AutoCompact:           true,
		AutoCompactLimit:      80,
	}
}

func Load(path string) (*Config, error) {
	cfg := Default()

	if path != "" {
		data, err := os.ReadFile(path)
		if err == nil {
			if err := yaml.Unmarshal(data, cfg); err != nil {
				return nil, fmt.Errorf("yaml: %w", err)
			}
		}
	}

	cfg.applyEnvOverrides()

	return cfg, nil
}

func (c *Config) applyEnvOverrides() {
	if v := os.Getenv("OLLAMA_MODEL"); v != "" {
		c.Model = v
	}
	if v := os.Getenv("OLLAMA_HOST"); v != "" {
		c.OllamaHost = v
	}
	if v := os.Getenv("OLLAMA_MODEL_FALLBACKS"); v != "" {
		c.ModelFallbacks = v
	}
	if v := os.Getenv("CONTEXT_PATH"); v != "" {
		c.ContextPath = v
	}
	if v := os.Getenv("EXPERIMENTAL_WEBSEARCH"); v != "" {
		c.ExperimentalWebSearch = v == "true"
	}
	if v := os.Getenv("PERSIST_TO_DB"); v != "" {
		c.PersistToDB = v == "true"
	}
	if v := os.Getenv("DATABASE_URL"); v != "" {
		c.DatabaseURL = v
	}
	if v := os.Getenv("THINK"); v != "" {
		c.Think = v
	}
	if v := os.Getenv("MAX_SUBAGENT_DEPTH"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			c.MaxSubagentDepth = n
		}
	}
	if v := os.Getenv("MAX_SUBAGENT_ROUNDS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			c.MaxSubagentRounds = n
		}
	}
	if v := os.Getenv("READ_FILE_MAX_LINES"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			c.ReadFileMaxLines = n
		}
	}
	if v := os.Getenv("READ_FILE_MAX_BYTES"); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil {
			c.ReadFileMaxBytes = n
		}
	}
	if v := os.Getenv("READ_FILE_MAX_SIZE"); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil {
			c.ReadFileMaxSize = n
		}
	}
	if v := os.Getenv("MODEL_TIMEOUT"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			c.ModelTimeout = n
		}
	}
	if v := os.Getenv("TURN_LIMIT"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			c.TurnLimit = n
		}
	}
	if v := os.Getenv("AUTO_COMPACT"); v != "" {
		c.AutoCompact = v == "true"
	}
	if v := os.Getenv("AUTO_COMPACT_LIMIT"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 && n <= 100 {
			c.AutoCompactLimit = n
		}
	}
	if v := os.Getenv("NUM_CTX"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			c.NumCtx = n
		}
	}
}

func (c *Config) GetThinkKwargs(model string) map[string]bool {
	switch strings.ToLower(c.Think) {
	case "true":
		return map[string]bool{"think": true}
	case "false":
		return map[string]bool{"think": false}
	default:
		if strings.HasPrefix(strings.ToLower(model), "lfm2") {
			return map[string]bool{"think": false}
		}
		return nil
	}
}

func (c *Config) BuildOptions(model string) *ollama.Options {
	var opts ollama.Options

	// Think mode
	switch strings.ToLower(c.Think) {
	case "true":
		v := true
		opts.Think = &v
	case "false":
		v := false
		opts.Think = &v
	default:
		if strings.HasPrefix(strings.ToLower(model), "lfm2") {
			v := false
			opts.Think = &v
		}
	}

	// Context window override
	if c.NumCtx > 0 {
		n := c.NumCtx
		opts.NumCtx = &n
	}

	return &opts
}

func (c *Config) IsLFM2() bool {
	return strings.HasPrefix(strings.ToLower(c.Model), "lfm2")
}

func (c *Config) ModelTimeoutDuration() int {
	if c.IsLFM2() && c.ModelTimeout <= 0 {
		return 120
	}
	if c.ModelTimeout > 0 {
		return c.ModelTimeout
	}
	return 60
}
