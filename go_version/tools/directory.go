package tools

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

var listDirectoryTool = Tool{
	Name:        "list_directory",
	Description: "List the contents of a directory. Returns files and subdirectories with their sizes and modification times. Use for exploring project structure or finding files.",
	Parameters: map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"path": map[string]interface{}{
				"type":        "string",
				"description": "Absolute or relative path to the directory",
			},
		},
		"required": []string{"path"},
	},
	Execute: listDirectoryExecute,
}

var spawnAgentTool = Tool{
	Name:        "spawn_agent",
	Description: "Spawn a sub-agent to handle complex multi-step tasks autonomously. The sub-agent has access to all the same tools (read_file, write_file, run_shell, web_search, etc.) and can perform research, code implementation, refactoring, and analysis. Use for tasks requiring 3+ sequential tool calls, multi-file operations, or iterative problem-solving. The sub-agent receives its own isolated task description and reports back with results.",
	Parameters: map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"task": map[string]interface{}{
				"type":        "string",
				"description": "Detailed description of the task for the sub-agent",
			},
		},
		"required": []string{"task"},
	},
	Execute: spawnAgentExecute,
}

func listDirectoryExecute(ctx context.Context, args map[string]interface{}) (string, error) {
	path := GetString(args, "path")
	if path == "" {
		return "", fmt.Errorf("path is required")
	}

	absPath, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("abs path: %w", err)
	}

	info, err := os.Stat(absPath)
	if err != nil {
		return "", fmt.Errorf("stat: %w", err)
	}

	if !info.IsDir() {
		return "", fmt.Errorf("not a directory: %s", absPath)
	}

	entries, err := os.ReadDir(absPath)
	if err != nil {
		return "", fmt.Errorf("read dir: %w", err)
	}

	sort.Slice(entries, func(i, j int) bool {
		return entries[i].Name() < entries[j].Name()
	})

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Directory: %s\n\n", absPath))

	for _, entry := range entries {
		fi, err := entry.Info()
		name := entry.Name()
		if err != nil {
			sb.WriteString(fmt.Sprintf("  %s\n", name))
			continue
		}

		if entry.IsDir() {
			sb.WriteString(fmt.Sprintf("  %s/\t%s\n", name, fi.ModTime().Format("2006-01-02 15:04")))
		} else {
			size := fi.Size()
			sizeStr := formatSize(size)
			sb.WriteString(fmt.Sprintf("  %s\t%s\t%s\n", name, sizeStr, fi.ModTime().Format("2006-01-02 15:04")))
		}
	}

	sb.WriteString(fmt.Sprintf("\n%d entries", len(entries)))
	return sb.String(), nil
}

func spawnAgentExecute(ctx context.Context, args map[string]interface{}) (string, error) {
	task := GetString(args, "task")
	if task == "" {
		return "", fmt.Errorf("task is required")
	}

	return fmt.Sprintf(`[spawn_agent received task: "%s"]

The sub-agent feature is managed by the chat loop layer. When the main chat loop receives a spawn_agent tool call, it will handle delegation to a sub-agent automatically.

Task to delegate: %s`, truncateText(task, 200), task), nil
}

func truncateText(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

func formatSize(size int64) string {
	switch {
	case size < 1024:
		return fmt.Sprintf("%dB", size)
	case size < 1024*1024:
		return fmt.Sprintf("%.1fKB", float64(size)/1024)
	default:
		return fmt.Sprintf("%.1fMB", float64(size)/(1024*1024))
	}
}
