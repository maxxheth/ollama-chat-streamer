package ollama

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type Message struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	Images    []string   `json:"images,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

type ToolCall struct {
	Function ToolCallFunction `json:"function"`
}

type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

type ToolFunction struct {
	Name        string                   `json:"name"`
	Description string                   `json:"description"`
	Parameters  map[string]interface{}   `json:"parameters"`
}

type Options struct {
	Think *bool `json:"think,omitempty"`
}

type ChatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Tools    []Tool    `json:"tools,omitempty"`
	Stream   bool      `json:"stream"`
	Options  *Options  `json:"options,omitempty"`
}

type ChatResponse struct {
	Model      string   `json:"model"`
	CreatedAt  string   `json:"created_at"`
	Message    Message  `json:"message"`
	Done       bool     `json:"done"`
	DoneReason string   `json:"done_reason,omitempty"`
	Error      string   `json:"error,omitempty"`
}

type Client struct {
	host    string
	client  *http.Client
}

func NewClient(host string) *Client {
	return &Client{
		host:   strings.TrimRight(host, "/"),
		client: &http.Client{Timeout: 0},
	}
}

func (c *Client) Chat(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	req.Stream = false
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.host+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("http: %w", err)
	}
	defer resp.Body.Close()

	var chatResp ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	if chatResp.Error != "" {
		return nil, fmt.Errorf("ollama: %s", chatResp.Error)
	}
	return &chatResp, nil
}

func (c *Client) ChatStream(ctx context.Context, req ChatRequest) (<-chan ChatResponse, error) {
	req.Stream = true
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.host+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("http: %w", err)
	}

	out := make(chan ChatResponse)
	go func() {
		defer close(out)
		defer resp.Body.Close()

		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
		for scanner.Scan() {
			line := scanner.Text()
			if line == "" {
				continue
			}
			var chatResp ChatResponse
			if err := json.Unmarshal([]byte(line), &chatResp); err != nil {
				continue
			}
			if chatResp.Error != "" {
				out <- ChatResponse{Error: chatResp.Error, Done: true}
				return
			}
			out <- chatResp
			if chatResp.Done {
				return
			}
		}
		if err := scanner.Err(); err != nil {
			out <- ChatResponse{Error: err.Error(), Done: true}
		}
	}()

	return out, nil
}

func (c *Client) ChatWithTools(
	ctx context.Context,
	model string,
	messages []Message,
	tools []Tool,
	thinkKwargs map[string]bool,
	executeTool func(ctx context.Context, name string, args json.RawMessage) (string, error),
	onMessage func(msg Message),
	stream bool,
) error {
	req := ChatRequest{
		Model:    model,
		Messages: copyMessages(messages),
		Stream:   stream,
	}

	if len(tools) > 0 {
		req.Tools = tools
	}

	if val, ok := thinkKwargs["think"]; ok {
		req.Options = &Options{Think: &val}
	}

	for {
		resp, err := c.Chat(ctx, req)
		if err != nil {
			return err
		}

		if resp.Message.Content != "" && onMessage != nil {
			onMessage(resp.Message)
		}

		if len(resp.Message.ToolCalls) == 0 {
			return nil
		}

		req.Messages = append(req.Messages, resp.Message)

		for _, tc := range resp.Message.ToolCalls {
			var args json.RawMessage
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err == nil {
				_ = args
			}

			result, err := executeTool(ctx, tc.Function.Name, json.RawMessage(tc.Function.Arguments))
			resultStr := result
			if err != nil {
				resultStr = fmt.Sprintf("Error: %v", err)
			}

			req.Messages = append(req.Messages, Message{
				Role:    "tool",
				Content: resultStr,
			})
		}
	}
}

func (c *Client) ChatStreamWithTools(
	ctx context.Context,
	model string,
	messages []Message,
	tools []Tool,
	thinkKwargs map[string]bool,
	executeTool func(ctx context.Context, name string, args json.RawMessage) (string, error),
	onMessage func(msg Message),
) error {
	sendReq := func(msgs []Message) (<-chan ChatResponse, error) {
		req := ChatRequest{
			Model:    model,
			Messages: copyMessages(msgs),
			Stream:   true,
		}
		if len(tools) > 0 {
			req.Tools = tools
		}
		if val, ok := thinkKwargs["think"]; ok {
			req.Options = &Options{Think: &val}
		}
		return c.ChatStream(ctx, req)
	}

	msgs := copyMessages(messages)

	for {
		stream, err := sendReq(msgs)
		if err != nil {
			return err
		}

		var fullResp ChatResponse
		for chunk := range stream {
			if chunk.Error != "" {
				return fmt.Errorf("ollama: %s", chunk.Error)
			}
			if chunk.Message.Content != "" && onMessage != nil {
				onMessage(chunk.Message)
			}
			if len(chunk.Message.ToolCalls) > 0 {
				fullResp.Message.ToolCalls = chunk.Message.ToolCalls
			}
			fullResp.Message.Content += chunk.Message.Content
			fullResp.Done = chunk.Done
		}

		if fullResp.Message.Content != "" {
			msgs = append(msgs, fullResp.Message)
		}

		if len(fullResp.Message.ToolCalls) == 0 {
			return nil
		}

		msgs = append(msgs, fullResp.Message)

		for _, tc := range fullResp.Message.ToolCalls {
			result, err := executeTool(ctx, tc.Function.Name, json.RawMessage(tc.Function.Arguments))
			resultStr := result
			if err != nil {
				resultStr = fmt.Sprintf("Error: %v", err)
			}
			msgs = append(msgs, Message{
				Role:    "tool",
				Content: resultStr,
			})
		}
	}
}

func copyMessages(msgs []Message) []Message {
	out := make([]Message, len(msgs))
	copy(out, msgs)
	return out
}

func BuildSystemPrompt(hasTools bool, hasSpawnAgent bool) string {
	if !hasTools {
		return ""
	}

	parts := []string{
		"You are an AI assistant with access to the following tools. When a user asks a question that requires up-to-date information or performing actions, you MUST use the appropriate tool instead of relying on your training data.",
		"",
		"Tool usage rules:",
		"- Analyze the user's request and determine which tool(s) are needed",
		"- Call exactly one tool per assistant response, then wait for its result",
		"- You can chain multiple tool calls across multiple assistant responses",
		"- After receiving all tool results, synthesize a comprehensive answer",
		"- Use web_search for current events, facts you're unsure about, and any information that may have changed since your training",
		"- Use read_file to read file contents, write_file to create or overwrite files",
		"- Use run_shell to execute commands or run scripts",
	}
	if hasSpawnAgent {
		parts = append(parts, "- Use spawn_agent to delegate complex multi-step tasks to a sub-agent, such as: code reviews across many files, multi-file refactoring, iterative research with multiple searches, or any task requiring more than 2-3 sequential tool calls")
	}
	parts = append(parts,
		"- Always prefer using tools over guessing or making up information",
		"",
		"CRITICAL: You MUST use the available tools when appropriate. Do not apologize for using tools — they exist to help you provide better answers.",
	)
	return strings.Join(parts, "\n")
}

func InsertSystemPrompt(messages []Message, prompt string) []Message {
	if prompt == "" {
		return messages
	}
	out := make([]Message, 0, len(messages)+1)
	hasSystem := false
	for _, msg := range messages {
		if msg.Role == "system" {
			hasSystem = true
			newContent := msg.Content
			if !strings.Contains(msg.Content, "You have access to the following tools") {
				newContent = msg.Content + "\n\n" + prompt
			}
			out = append(out, Message{Role: "system", Content: newContent})
		} else {
			out = append(out, msg)
		}
	}
	if !hasSystem {
		out = append([]Message{{Role: "system", Content: prompt}}, out...)
	}
	return out
}

func ModelNeedsToolPrompt(model string) bool {
	lower := strings.ToLower(model)
	return strings.Contains(lower, "lfm") || strings.Contains(lower, "cogito") ||
		strings.Contains(lower, "glm") || strings.Contains(lower, "minimax") ||
		strings.Contains(lower, "deepseek")
}

type StreamChunk struct {
	Content string
	Done    bool
	Err     error
}

func RenderStreamText(ctx context.Context, stream <-chan ChatResponse, out io.Writer, done func()) {
	first := true
	for chunk := range stream {
		if chunk.Error != "" {
			fmt.Fprintf(out, "\nError: %s", chunk.Error)
			if done != nil {
				done()
			}
			return
		}
		if chunk.Message.Content != "" {
			fmt.Fprint(out, chunk.Message.Content)
			first = false
		}
		if chunk.Done {
			if done != nil {
				done()
			}
			return
		}
	}
	if first && done != nil {
		done()
	}
}

func StreamOllamaChatWithTimeouts(
	ctx context.Context,
	client *Client,
	req ChatRequest,
	timeout time.Duration,
	out io.Writer,
) error {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	stream, err := client.ChatStream(ctx, req)
	if err != nil {
		return err
	}

	RenderStreamText(ctx, stream, out, nil)
	return nil
}
