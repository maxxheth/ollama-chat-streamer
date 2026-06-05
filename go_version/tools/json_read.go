package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

var readJSONFileTool = Tool{
	Name:        "read_json_file",
	Description: "Read and parse a JSON file. Returns the parsed content as formatted JSON. Use for reading configuration files, JSON data dumps, or structured data files.",
	Parameters: map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"path": map[string]interface{}{
				"type":        "string",
				"description": "Absolute or relative path to the JSON file",
			},
		},
		"required": []string{"path"},
	},
	Execute: readJSONFileExecute,
}

func readJSONFileExecute(ctx context.Context, args map[string]interface{}) (string, error) {
	path := GetString(args, "path")
	if path == "" {
		return "", fmt.Errorf("path is required")
	}

	absPath, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("abs path: %w", err)
	}

	data, err := os.ReadFile(absPath)
	if err != nil {
		return "", fmt.Errorf("read: %w", err)
	}

	var parsed interface{}
	if err := json.Unmarshal(data, &parsed); err != nil {
		return "", fmt.Errorf("invalid JSON: %w", err)
	}

	formatted, err := json.MarshalIndent(parsed, "", "  ")
	if err != nil {
		return "", fmt.Errorf("format JSON: %w", err)
	}

	return fmt.Sprintf("File: %s (%d bytes)\n\n%s", absPath, len(data), string(formatted)), nil
}
