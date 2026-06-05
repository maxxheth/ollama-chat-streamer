package tools

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const (
	defaultMaxBytes = int64(65536)
	defaultMaxSize  = int64(524288)
	defaultMaxLines = 200
)

var readFileTool = Tool{
	Name:        "read_file",
	Description: "Read the contents of a file. Returns the file content up to a maximum size limit. Use for reading source code, config files, logs, or any text file.",
	Parameters: map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"path": map[string]interface{}{
				"type":        "string",
				"description": "Absolute or relative path to the file",
			},
			"max_lines": map[string]interface{}{
				"type":        "integer",
				"description": "Maximum number of lines to read (default: 200)",
			},
		},
		"required": []string{"path"},
	},
	Execute: readFileExecute,
}

var writeFileTool = Tool{
	Name:        "write_file",
	Description: "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. The content supports escape sequences like \\n for newlines.",
	Parameters: map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"path": map[string]interface{}{
				"type":        "string",
				"description": "Absolute or relative path to the file",
			},
			"content": map[string]interface{}{
				"type":        "string",
				"description": "Content to write to the file",
			},
		},
		"required": []string{"path", "content"},
	},
	Execute: writeFileExecute,
}

var appendFileTool = Tool{
	Name:        "append_file",
	Description: "Append content to an existing file. Creates the file if it doesn't exist. The content supports escape sequences like \\n for newlines.",
	Parameters: map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"path": map[string]interface{}{
				"type":        "string",
				"description": "Absolute or relative path to the file",
			},
			"content": map[string]interface{}{
				"type":        "string",
				"description": "Content to append to the file",
			},
		},
		"required": []string{"path", "content"},
	},
	Execute: appendFileExecute,
}

func readFileExecute(ctx context.Context, args map[string]interface{}) (string, error) {
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

	if info.Size() > defaultMaxSize {
		return "", fmt.Errorf("file too large: %.1f KB (max: %.1f KB)",
			float64(info.Size())/1024, float64(defaultMaxSize)/1024)
	}

	data, err := os.ReadFile(absPath)
	if err != nil {
		return "", fmt.Errorf("read: %w", err)
	}

	if info.Size() > defaultMaxBytes {
		data = data[:defaultMaxBytes]
	}

	maxLines := GetInt(args, "max_lines")
	if maxLines <= 0 {
		maxLines = defaultMaxLines
	}

	lines := strings.Split(string(data), "\n")
	if len(lines) > maxLines {
		lines = lines[:maxLines]
		lines = append(lines, fmt.Sprintf("... (%d more lines)", len(lines)-maxLines))
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("File: %s (%d bytes, %d lines shown)\n\n", absPath, info.Size(), len(lines)))

	if info.Size() > defaultMaxBytes {
		sb.WriteString(fmt.Sprintf("[Note: file truncated to %d bytes]\n\n", defaultMaxBytes))
	}

	sb.WriteString(strings.Join(lines, "\n"))
	return sb.String(), nil
}

func writeFileExecute(ctx context.Context, args map[string]interface{}) (string, error) {
	path := GetString(args, "path")
	if path == "" {
		return "", fmt.Errorf("path is required")
	}

	content := DecodeEscapes(GetString(args, "content"))

	absPath, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("abs path: %w", err)
	}

	dir := filepath.Dir(absPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", fmt.Errorf("mkdir: %w", err)
	}

	if err := os.WriteFile(absPath, []byte(content), 0644); err != nil {
		return "", fmt.Errorf("write: %w", err)
	}

	return fmt.Sprintf("Successfully wrote %d bytes to %s", len(content), absPath), nil
}

func appendFileExecute(ctx context.Context, args map[string]interface{}) (string, error) {
	path := GetString(args, "path")
	if path == "" {
		return "", fmt.Errorf("path is required")
	}

	content := DecodeEscapes(GetString(args, "content"))

	absPath, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("abs path: %w", err)
	}

	dir := filepath.Dir(absPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", fmt.Errorf("mkdir: %w", err)
	}

	f, err := os.OpenFile(absPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return "", fmt.Errorf("open: %w", err)
	}
	defer f.Close()

	n, err := f.WriteString(content)
	if err != nil {
		return "", fmt.Errorf("append: %w", err)
	}

	return fmt.Sprintf("Successfully appended %d bytes to %s", n, absPath), nil
}
