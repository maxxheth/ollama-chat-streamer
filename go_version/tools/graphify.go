package tools

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

var graphifyTool = Tool{
	Name:        "graphify",
	Description: "Build a knowledge graph of a codebase to understand its structure, dependencies, and design patterns. Analyzes source code, documentation, and diagrams to produce an interactive visualization (graph.html), a queryable JSON graph, and a markdown report. Use this when asked to analyze, document, or understand a project's architecture.",
	Parameters: map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"path": map[string]interface{}{
				"type":        "string",
				"description": "Path to the directory or file to analyze",
			},
			"output": map[string]interface{}{
				"type":        "string",
				"description": "Custom output directory (default: graphify-out/ relative to the analyzed path)",
			},
		},
		"required": []string{"path"},
	},
	Execute: graphifyExecute,
}

func graphifyExecute(ctx context.Context, args map[string]interface{}) (string, error) {
	path := GetString(args, "path")
	if path == "" {
		return "", fmt.Errorf("path is required")
	}

	absPath, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("abs path: %w", err)
	}

	if _, err := os.Stat(absPath); err != nil {
		return "", fmt.Errorf("path does not exist: %s", absPath)
	}

	output := GetString(args, "output")
	if output == "" {
		output = filepath.Join(absPath, "graphify-out")
	} else if !filepath.IsAbs(output) {
		output = filepath.Join(absPath, output)
	}

	// Ensure graphify CLI is available
	if err := ensureGraphify(ctx); err != nil {
		return "", fmt.Errorf("graphify setup: %w", err)
	}

	// Run graphify with a generous timeout for large codebases
	runCtx, cancel := context.WithTimeout(ctx, 120*time.Second)
	defer cancel()

	var stderr bytes.Buffer
	cmd := exec.CommandContext(runCtx, "graphify", absPath, "--output", output)
	cmd.Stderr = &stderr

	outputBytes, err := cmd.Output()
	if err != nil {
		cliOutput := strings.TrimSpace(string(outputBytes))
		cliErr := strings.TrimSpace(stderr.String())
		detail := cliOutput
		if cliErr != "" {
			detail = cliErr
		}
		if runCtx.Err() == context.DeadlineExceeded {
			return "", fmt.Errorf("graphify timed out after 120s (codebase may be too large)")
		}
		return "", fmt.Errorf("graphify failed: %s", detail)
	}

	// Collect results
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Knowledge graph built for: %s\n\n", absPath))

	// Read the report
	reportPath := filepath.Join(output, "GRAPH_REPORT.md")
	if reportData, err := os.ReadFile(reportPath); err == nil {
		sb.WriteString("=== GRAPH_REPORT.md ===\n")
		sb.WriteString(string(reportData))
		sb.WriteString("\n")
	}

	// List output files
	sb.WriteString("\n=== Output Files ===\n")
	entries, _ := os.ReadDir(output)
	for _, entry := range entries {
		fi, _ := entry.Info()
		if fi != nil {
			sb.WriteString(fmt.Sprintf("  %s (%d bytes)\n", entry.Name(), fi.Size()))
		} else {
			sb.WriteString(fmt.Sprintf("  %s\n", entry.Name()))
		}
	}

	sb.WriteString(fmt.Sprintf("\nOpen graph.html in a browser to explore the graph interactively."))
	return sb.String(), nil
}

// ensureGraphify checks if the graphify CLI is available and installs it via uv if not.
func ensureGraphify(ctx context.Context) error {
	if _, err := exec.LookPath("graphify"); err == nil {
		return nil
	}

	// Check for uv
	uvPath, err := exec.LookPath("uv")
	if err != nil {
		return fmt.Errorf("graphify not found and uv is not installed (install uv first: https://docs.astral.sh/uv/)")
	}

	fmt.Fprintf(os.Stderr, "  Installing graphify via uv…\n")
	installCtx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()

	cmd := exec.CommandContext(installCtx, uvPath, "tool", "install", "graphifyy")
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("uv tool install failed: %s", strings.TrimSpace(stderr.String()))
	}

	// Verify it's now available
	if _, err := exec.LookPath("graphify"); err != nil {
		return fmt.Errorf("graphify installed but not found on PATH (try: source <(uv tool list))")
	}

	return nil
}
