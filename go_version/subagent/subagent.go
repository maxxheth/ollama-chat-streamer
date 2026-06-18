package subagent

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/maxx/ollama-chat-streamer/go_version/ollama"
	"github.com/maxx/ollama-chat-streamer/go_version/tools"
)

const SubagentSystemPrompt = `You are a sub-agent with access to tools. Your task is described below.

You can use tools to accomplish your goal. IMPORTANT: You do NOT have spawn_agent — you cannot create sub-agents yourself.

Work step by step:
1. Analyze the task
2. Use tools as needed (one tool call per response)
3. After receiving each tool result, decide what to do next
4. When the task is complete, provide a clear summary of what was done

Do NOT ask the user for help — you have all the tools you need.`

type SubagentConfig struct {
	Model        string
	OllamaHost   string
	Think        string
	MaxRounds    int
	CurrentDepth int
	MaxDepth     int
	ToolSchemas  []map[string]interface{}
}

func RunSubagent(ctx context.Context, task string, history []ollama.Message, cfg SubagentConfig) (string, error) {
	client := ollama.NewClient(cfg.OllamaHost)

	messages := buildSubagentMessages(task, history)

	spawnAgentSchema := findSpawnAgentSchema(cfg.ToolSchemas)
	subagentTools := filterTools(cfg.ToolSchemas)
	hasTools := len(subagentTools) > 0

	if hasTools {
		sysPrompt := ollama.BuildSystemPrompt(true, false)
		if spawnAgentSchema != nil {
			sysPrompt += "\n\nNote: spawn_agent is NOT available to you as a sub-agent."
		}
		messages = ollama.InsertSystemPrompt(messages, sysPrompt)
	}

	thinkKwargs := getThinkKwargs(cfg.Model, cfg.Think)

	for round := 0; round < cfg.MaxRounds; round++ {
		chatTools := convertToolSchemas(subagentTools)

		req := ollama.ChatRequest{
			Model:    cfg.Model,
			Messages: messages,
			Tools:    chatTools,
			Stream:   true,
		}
		if val, ok := thinkKwargs["think"]; ok {
			req.Options = &ollama.Options{Think: &val}
		}

		var collectedMsg ollama.Message
		stream, err := client.ChatStream(ctx, req)
		if err != nil {
			return "", fmt.Errorf("subagent chat: %w", err)
		}

		var assistantContent string
		for chunk := range stream {
			if chunk.Error != "" {
				return "", fmt.Errorf("subagent: %s", chunk.Error)
			}
			assistantContent += chunk.Message.Content
			if len(chunk.Message.ToolCalls) > 0 {
				collectedMsg.ToolCalls = append(collectedMsg.ToolCalls, chunk.Message.ToolCalls...)
			}
		}

		collectedMsg.Role = "assistant"
		collectedMsg.Content = assistantContent

		messages = append(messages, collectedMsg)

		if len(collectedMsg.ToolCalls) == 0 {
			return assistantContent, nil
		}

		for _, tc := range collectedMsg.ToolCalls {
			start := time.Now()
			fmt.Fprintf(os.Stderr, "  subagent: %s(%s)\n", tc.Function.Name, truncateArgs(string(tc.Function.Arguments)))

			result, err := tools.ExecuteToolCall(ctx, tc.Function.Name, tc.Function.Arguments)
			resultStr := result
			if err != nil {
				resultStr = fmt.Sprintf("Error: %v", err)
			}

			elapsed := time.Since(start).Round(time.Millisecond)
			fmt.Fprintf(os.Stderr, "  subagent: %s → done (%v)\n", tc.Function.Name, elapsed)

			messages = append(messages, ollama.Message{
				Role:    "tool",
				Content: resultStr,
			})
		}
	}

	return "", fmt.Errorf("subagent did not complete in %d rounds", cfg.MaxRounds)
}

func buildSubagentMessages(task string, history []ollama.Message) []ollama.Message {
	messages := []ollama.Message{
		{Role: "system", Content: SubagentSystemPrompt},
		{Role: "user", Content: fmt.Sprintf("Task: %s", task)},
	}

	if len(history) > 0 {
		var sb strings.Builder
		sb.WriteString("\n\nRelevant conversation history:\n")
		for _, msg := range history {
			if msg.Content != "" {
				sb.WriteString(fmt.Sprintf("\n[%s]: %s", msg.Role, truncateText(msg.Content, 500)))
			}
		}
		messages = append(messages, ollama.Message{Role: "user", Content: sb.String()})
	}

	return messages
}

func findSpawnAgentSchema(schemas []map[string]interface{}) map[string]interface{} {
	for _, s := range schemas {
		if fn, ok := s["function"].(map[string]interface{}); ok {
			if name, ok := fn["name"].(string); ok && name == "spawn_agent" {
				return s
			}
		}
	}
	return nil
}

func filterTools(schemas []map[string]interface{}) []map[string]interface{} {
	var filtered []map[string]interface{}
	for _, s := range schemas {
		if fn, ok := s["function"].(map[string]interface{}); ok {
			if name, ok := fn["name"].(string); ok && name == "spawn_agent" {
				continue
			}
		}
		filtered = append(filtered, s)
	}
	return filtered
}

func convertToolSchemas(schemas []map[string]interface{}) []ollama.Tool {
	var tools []ollama.Tool
	for _, s := range schemas {
		fn, ok := s["function"].(map[string]interface{})
		if !ok {
			continue
		}
		name, _ := fn["name"].(string)
		desc, _ := fn["description"].(string)

		params, ok := fn["parameters"].(map[string]interface{})
		if !ok {
			params = map[string]interface{}{}
		}

		tools = append(tools, ollama.Tool{
			Type: "function",
			Function: ollama.ToolFunction{
				Name:        name,
				Description: desc,
				Parameters:  params,
			},
		})
	}
	return tools
}

func getThinkKwargs(model, setting string) map[string]bool {
	switch strings.ToLower(setting) {
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

func truncateText(s string, maxLen int) string {
	runes := []rune(s)
	if len(runes) <= maxLen {
		return s
	}
	return string(runes[:maxLen]) + "..."
}

func truncateArgs(args string) string {
	if len(args) > 80 {
		cleaned := strings.ReplaceAll(args[:80], "\n", " ")
		return cleaned + "…"
	}
	return strings.ReplaceAll(args, "\n", " ")
}
