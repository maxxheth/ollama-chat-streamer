package chat

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/maxx/ollama-chat-streamer/go_version/config"
	"github.com/maxx/ollama-chat-streamer/go_version/db"
	"github.com/maxx/ollama-chat-streamer/go_version/ollama"
	"github.com/maxx/ollama-chat-streamer/go_version/subagent"
	"github.com/maxx/ollama-chat-streamer/go_version/tools"
)

type ChatLoop struct {
	cfg     *config.Config
	client  *ollama.Client
	db      *db.Pool
	messages []ollama.Message
	mu      sync.Mutex
	scanner *bufio.Scanner
}

func New(cfg *config.Config, database *db.Pool) *ChatLoop {
	return &ChatLoop{
		cfg:     cfg,
		client:  ollama.NewClient(cfg.OllamaHost),
		db:      database,
		messages: []ollama.Message{},
		scanner: bufio.NewScanner(os.Stdin),
	}
}

func (cl *ChatLoop) Run(ctx context.Context) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		fmt.Fprintln(os.Stderr, "\nReceived interrupt, shutting down...")
		cancel()
	}()

	if cl.cfg.ContextPath != "" {
		if err := cl.loadContextFile(cl.cfg.ContextPath); err != nil {
			fmt.Fprintf(os.Stderr, "Warning: could not load context file: %v\n", err)
		}
	}

	if cl.cfg.PersistToDB && cl.db != nil {
		if err := cl.migrateMessagesFromDB(ctx); err != nil {
			fmt.Fprintf(os.Stderr, "Warning: could not load previous messages: %v\n", err)
		}
	}

	var conversationID *int
	if cl.cfg.PersistToDB && cl.db != nil {
		id, err := cl.db.SaveConversation(ctx, cl.cfg.Model, nil)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Warning: could not create conversation: %v\n", err)
		}
		conversationID = id
	}

	fmt.Fprintf(os.Stderr, "Model: %s | DB: %v | WebSearch: %v | Subagent depth: %d\n",
		cl.cfg.Model, cl.cfg.PersistToDB, cl.cfg.ExperimentalWebSearch, cl.cfg.MaxSubagentDepth)
	fmt.Fprintf(os.Stderr, "Type /help for commands\n\n")

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		if cl.cfg.PersistToDB && cl.db != nil && conversationID != nil {
			cl.db.UpdateConversation(ctx, *conversationID, map[string]interface{}{
				"model": cl.cfg.Model,
				"websearch": cl.cfg.ExperimentalWebSearch,
				"subagent_depth": cl.cfg.MaxSubagentDepth,
				"messages": len(cl.messages),
			})
		}

		prompt, err := cl.readInput()
		if err != nil {
			return err
		}
		if prompt == "" {
			continue
		}

		if strings.HasPrefix(prompt, "/") {
			if err := cl.handleCommand(ctx, prompt, conversationID); err != nil {
				fmt.Fprintf(os.Stderr, "Command error: %v\n", err)
			}
			continue
		}

		if cl.cfg.PersistToDB && cl.db != nil && conversationID != nil {
			cl.db.SaveMessage(ctx, *conversationID, "user", prompt)
		}

		cl.messages = append(cl.messages, ollama.Message{
			Role:    "user",
			Content: prompt,
		})

		cl.mu.Lock()
		localMessages := make([]ollama.Message, len(cl.messages))
		copy(localMessages, cl.messages)
		cl.mu.Unlock()

		if err := cl.chatTurn(ctx, localMessages); err != nil {
			fmt.Fprintf(os.Stderr, "\nError: %v\n", err)
			if ctx.Err() != nil {
				return ctx.Err()
			}
		}
	}
}

func (cl *ChatLoop) chatTurn(ctx context.Context, messages []ollama.Message) error {
	hasTools := cl.cfg.ExperimentalWebSearch
	hasSpawnAgent := cl.cfg.MaxSubagentDepth > 0

	toolPrompt := ollama.BuildSystemPrompt(hasTools, hasSpawnAgent)
	chatMessages := ollama.InsertSystemPrompt(messages, toolPrompt)

	thinkKwargs := cl.cfg.GetThinkKwargs(cl.cfg.Model)

	toolSchemas := tools.GetToolSchemas(hasSpawnAgent)
	var ollamaTools []ollama.Tool
	for _, s := range toolSchemas {
		fn, ok := s["function"].(map[string]interface{})
		if !ok {
			continue
		}
		name, _ := fn["name"].(string)
		desc, _ := fn["description"].(string)
		params, _ := fn["parameters"].(map[string]interface{})

		ollamaTools = append(ollamaTools, ollama.Tool{
			Type: "function",
			Function: ollama.ToolFunction{
				Name:        name,
				Description: desc,
				Parameters:  params,
			},
		})
	}

	turnLimit := 20
	var done bool
	var err error
	for turn := 0; turn < turnLimit; turn++ {
		func() {
			timeout := time.Duration(cl.cfg.ModelTimeoutDuration()) * time.Second
			turnCtx, turnCancel := context.WithTimeout(ctx, timeout)
			defer turnCancel()

			req := ollama.ChatRequest{
				Model:    cl.cfg.Model,
				Messages: chatMessages,
				Stream:   true,
			}
			if len(ollamaTools) > 0 {
				req.Tools = ollamaTools
			}
			if val, ok := thinkKwargs["think"]; ok {
				req.Options = &ollama.Options{Think: &val}
			}

			stream, err := cl.client.ChatStream(turnCtx, req)
			if err != nil {
				err = fmt.Errorf("chat: %w", err)
				return
			}

			var assistantMsg ollama.Message
			assistantMsg.Role = "assistant"
			fmt.Fprint(os.Stderr, "\n")

			for chunk := range stream {
				if chunk.Error != "" {
					fmt.Fprintf(os.Stderr, "\nError: %s\n", chunk.Error)
					err = fmt.Errorf("ollama: %s", chunk.Error)
					return
				}
				if chunk.Message.Content != "" {
					fmt.Fprint(os.Stdout, chunk.Message.Content)
					assistantMsg.Content += chunk.Message.Content
				}
				if len(chunk.Message.ToolCalls) > 0 {
					assistantMsg.ToolCalls = append(assistantMsg.ToolCalls, chunk.Message.ToolCalls...)
				}
			}
			fmt.Fprint(os.Stdout, "\n")

			chatMessages = append(chatMessages, assistantMsg)
			cl.mu.Lock()
			cl.messages = append(cl.messages, assistantMsg)
			cl.mu.Unlock()

			if len(assistantMsg.ToolCalls) == 0 {
				done = true
				return
			}

			for _, tc := range assistantMsg.ToolCalls {
				toolCtx, toolCancel := context.WithTimeout(ctx, 60*time.Second)

				var result string
				var toolErr error

				fmt.Fprintf(os.Stderr, "  🛠  %s(%s)\n", tc.Function.Name, truncateArgs(tc.Function.Arguments))

				if tc.Function.Name == "spawn_agent" && cl.cfg.MaxSubagentDepth > 0 {
					result, toolErr = cl.runSubAgent(toolCtx, tc.Function.Arguments, messages)
				} else {
					result, toolErr = tools.ExecuteToolCall(toolCtx, tc.Function.Name, json.RawMessage(tc.Function.Arguments))
				}

				toolCancel()

				if toolErr != nil {
					result = fmt.Sprintf("Error: %v", toolErr)
				}
				toolResultPreview := result
				if len(toolResultPreview) > 120 {
					toolResultPreview = toolResultPreview[:120] + "…"
				}
				fmt.Fprintf(os.Stderr, "  ✓ %s → %s\n", tc.Function.Name, strings.ReplaceAll(toolResultPreview, "\n", " "))

				toolMsg := ollama.Message{
					Role:    "tool",
					Content: result,
				}
				chatMessages = append(chatMessages, toolMsg)
				cl.mu.Lock()
				cl.messages = append(cl.messages, toolMsg)
				cl.mu.Unlock()
			}
		}()

		if done {
			return nil
		}
		if err != nil {
			return err
		}
	}

	return fmt.Errorf("tool call loop exceeded %d turns", turnLimit)
}

func (cl *ChatLoop) runSubAgent(ctx context.Context, argsRaw string, history []ollama.Message) (string, error) {
	var args map[string]interface{}
	if err := json.Unmarshal([]byte(argsRaw), &args); err != nil {
		return "", fmt.Errorf("parse spawn_agent args: %w", err)
	}

	task, ok := args["task"].(string)
	if !ok || task == "" {
		return "", fmt.Errorf("spawn_agent requires a task string")
	}

	cfg := subagent.SubagentConfig{
		Model:        cl.cfg.Model,
		OllamaHost:   cl.cfg.OllamaHost,
		Think:        cl.cfg.Think,
		MaxRounds:    cl.cfg.MaxSubagentRounds,
		CurrentDepth: 0,
		MaxDepth:     cl.cfg.MaxSubagentDepth - 1,
		ToolSchemas:  tools.GetToolSchemas(false),
	}

	return subagent.RunSubagent(ctx, task, history, cfg)
}

func (cl *ChatLoop) readInput() (string, error) {
	fmt.Fprint(os.Stderr, "\033[1;32m>>>\033[0m ")
	if !cl.scanner.Scan() {
		return "", cl.scanner.Err()
	}
	return strings.TrimSpace(cl.scanner.Text()), nil
}

func (cl *ChatLoop) loadContextFile(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	content := strings.TrimSpace(string(data))
	if content != "" {
		cl.messages = append(cl.messages, ollama.Message{
			Role:    "user",
			Content: fmt.Sprintf("Context:\n%s", content),
		})
	}
	return nil
}

func (cl *ChatLoop) migrateMessagesFromDB(ctx context.Context) error {
	sessions, err := cl.db.ListSessions(ctx)
	if err != nil {
		return err
	}
	if len(sessions) == 0 {
		return nil
	}

	latest := sessions[0]
	msgs, err := cl.db.ExportSession(ctx, latest.ID)
	if err != nil {
		return err
	}

	for _, m := range msgs {
		cl.messages = append(cl.messages, ollama.Message{
			Role:    m.Role,
			Content: m.Content,
		})
	}

	return nil
}

func (cl *ChatLoop) handleCommand(ctx context.Context, input string, conversationID *int) error {
	parts := strings.Fields(input)
	cmd := strings.TrimPrefix(parts[0], "/")

	switch cmd {
	case "help":
		fmt.Fprint(os.Stderr, `Commands:
  /help       Show this help
  /exit       Exit the program
  /clear      Clear message history
  /save       Save conversation
  /export     Export conversation as JSON
  /sessions   List saved sessions
  /load <id>  Load a saved session
  /model      Show current model
  /history    Show message history count
`)
	case "exit", "quit":
		os.Exit(0)
	case "clear":
		cl.mu.Lock()
		cl.messages = nil
		cl.mu.Unlock()
		fmt.Fprintln(os.Stderr, "Message history cleared.")
	case "save":
		if cl.db != nil && conversationID != nil {
			cl.db.UpdateConversation(ctx, *conversationID, map[string]interface{}{
				"saved_at": time.Now().Format(time.RFC3339),
			})
			fmt.Fprintf(os.Stderr, "Conversation %d saved.\n", *conversationID)
		} else {
			fmt.Fprintln(os.Stderr, "Database not available.")
		}
	case "export":
		cl.mu.Lock()
		msgs := make([]ollama.Message, len(cl.messages))
		copy(msgs, cl.messages)
		cl.mu.Unlock()

		data, _ := json.MarshalIndent(msgs, "", "  ")
		fmt.Println(string(data))
	case "sessions":
		if cl.db == nil {
			fmt.Fprintln(os.Stderr, "Database not available.")
			return nil
		}
		sessions, err := cl.db.ListSessions(ctx)
		if err != nil {
			return err
		}
		if len(sessions) == 0 {
			fmt.Fprintln(os.Stderr, "No saved sessions.")
			return nil
		}
		for _, s := range sessions {
			model, _ := s.Flags["model"].(string)
			if model == "" {
				model = s.Model
			}
			fmt.Fprintf(os.Stderr, "  %d: %s | %s | %s\n", s.ID, model,
				s.CreatedAt.Format("2006-01-02 15:04"),
				s.Flags)
		}
	case "load":
		if len(parts) < 2 {
			fmt.Fprintln(os.Stderr, "Usage: /load <session_id>")
			return nil
		}
		if cl.db == nil {
			fmt.Fprintln(os.Stderr, "Database not available.")
			return nil
		}
		// ... would need to implement load logic
		fmt.Fprintln(os.Stderr, "Load not yet implemented.")
	case "model":
		fmt.Fprintf(os.Stderr, "Current model: %s\n", cl.cfg.Model)
	case "history":
		cl.mu.Lock()
		count := len(cl.messages)
		cl.mu.Unlock()
		fmt.Fprintf(os.Stderr, "Messages in history: %d\n", count)
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: /%s\n", cmd)
	}
	return nil
}

func (cl *ChatLoop) ModelTimeoutDuration() int {
	return cl.cfg.ModelTimeoutDuration()
}

func truncateArgs(args string) string {
	if len(args) > 80 {
		cleaned := strings.ReplaceAll(args[:80], "\n", " ")
		return cleaned + "…"
	}
	return strings.ReplaceAll(args, "\n", " ")
}
