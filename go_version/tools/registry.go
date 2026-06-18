package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

type ToolResult struct {
	Content string
	Error   error
}

type ToolFunc func(ctx context.Context, args map[string]interface{}) (string, error)

type Tool struct {
	Name        string
	Description string
	Parameters  map[string]interface{}
	Execute     ToolFunc
}

var Registry []Tool

func GetToolSchemas(includeSpawnAgent bool) []map[string]interface{} {
	var schemas []map[string]interface{}
	for _, t := range Registry {
		if !includeSpawnAgent && t.Name == "spawn_agent" {
			continue
		}
		schemas = append(schemas, map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name":        t.Name,
				"description": t.Description,
				"parameters":  t.Parameters,
			},
		})
	}
	return schemas
}

func ExecuteToolCall(ctx context.Context, name string, args json.RawMessage) (string, error) {
	for _, t := range Registry {
		if t.Name == name {
			var parsed map[string]interface{}
			if err := json.Unmarshal(args, &parsed); err != nil {
				return "", fmt.Errorf("tool %s: invalid args: %w", name, err)
			}
			ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
			defer cancel()
			return t.Execute(ctx, parsed)
		}
	}
	return "", fmt.Errorf("unknown tool: %s", name)
}

func init() {
	Registry = []Tool{
		readFileTool,
		writeFileTool,
		appendFileTool,
		listDirectoryTool,
		runShellTool,
		readJSONFileTool,
		webSearchTool,
		spawnAgentTool,
		graphifyTool,
	}
}

func ParseArgs(args json.RawMessage) (map[string]interface{}, error) {
	var parsed map[string]interface{}
	if err := json.Unmarshal(args, &parsed); err != nil {
		return nil, fmt.Errorf("parse args: %w", err)
	}
	return parsed, nil
}

func GetString(args map[string]interface{}, key string) string {
	v, ok := args[key]
	if !ok {
		return ""
	}
	s, ok := v.(string)
	if !ok {
		return fmt.Sprintf("%v", v)
	}
	return s
}

func GetInt(args map[string]interface{}, key string) int {
	v, ok := args[key]
	if !ok {
		return 0
	}
	f, ok := v.(float64)
	if ok {
		return int(f)
	}
	return 0
}

func DecodeEscapes(s string) string {
	s = strings.ReplaceAll(s, "\\n", "\n")
	s = strings.ReplaceAll(s, "\\t", "\t")
	s = strings.ReplaceAll(s, "\\r", "\r")
	s = strings.ReplaceAll(s, "\\\"", "\"")
	s = strings.ReplaceAll(s, "\\\\", "\\")
	return s
}
