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
	cfg                   *config.Config
	client                *ollama.Client
	db                    *db.Pool
	currentConversationID *int // Tracks the active conversation ID
	messages              []ollama.Message
	mu                    sync.Mutex
	scanner               *bufio.Scanner
	modelInfo             *ollama.ModelInfo // cached model context length
}

func New(cfg *config.Config, database *db.Pool) *ChatLoop {
	return &ChatLoop{
		cfg:      cfg,
		client:   ollama.NewClient(cfg.OllamaHost),
		db:       database,
		messages: []ollama.Message{},
		scanner:  bufio.NewScanner(os.Stdin),
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

	// Initial DB State Setup
	if cl.cfg.PersistToDB && cl.db != nil {
		// Try to find the most recent conversation to resume or create new one later
		sessions, err := cl.db.ListSessions(ctx)
		if err == nil && len(sessions) > 0 {
			latestID := sessions[0].ID
			cl.currentConversationID = &latestID
			fmt.Fprintf(os.Stderr, "Resuming latest session: %d\n", latestID)
		} else {
			fmt.Fprintln(os.Stderr, "No previous sessions found. Will create new upon first message.")
		}
	}

	// Fetch model context length for auto-compact
	if cl.cfg.AutoCompact {
		info, err := cl.client.ShowModel(ctx, cl.cfg.Model)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Warning: could not fetch model info — auto-compact disabled\n")
		} else {
			cl.modelInfo = info
		}
	}

	fmt.Fprintf(os.Stderr, "Model: %s | DB: %v | WebSearch: %v | Subagent depth: %d",
		cl.cfg.Model, cl.cfg.PersistToDB, cl.cfg.ExperimentalWebSearch, cl.cfg.MaxSubagentDepth)
	if cl.modelInfo != nil {
		fmt.Fprintf(os.Stderr, " | Compact: %d%%", cl.cfg.AutoCompactLimit)
	}
	fmt.Fprintln(os.Stderr)
	if cl.currentConversationID != nil {
		fmt.Fprintf(os.Stderr, "Current Session ID: %d\n", *cl.currentConversationID)
	} else {
		fmt.Fprintln(os.Stderr, "Current Session ID: (none - will create on first message)")
	}
	fmt.Fprint(os.Stderr, "Type /help for commands\n\n")

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		prompt, err := cl.readInput()
		if err != nil {
			return err
		}
		if prompt == "" {
			continue
		}

		if strings.HasPrefix(prompt, "/") {
			if err := cl.handleCommand(ctx, prompt); err != nil {
				fmt.Fprintf(os.Stderr, "Command error: %v\n", err)
			}
			continue
		}

		// Ensure conversation exists if it doesn't (first message of fresh session)
		if cl.cfg.PersistToDB && cl.db != nil {
			if cl.currentConversationID == nil {
				id, err := cl.db.SaveConversation(ctx, cl.cfg.Model, nil)
				if err != nil {
					fmt.Fprintf(os.Stderr, "Warning: could not create conversation: %v\n", err)
				} else {
					cl.currentConversationID = id
					fmt.Fprintf(os.Stderr, "Created new session ID: %d\n", *id)
				}
			}

			if cl.currentConversationID != nil {
				cl.db.SaveMessage(ctx, *cl.currentConversationID, "user", prompt)
			}
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

		// Auto-compact if approaching context limit
		cl.compactIfNeeded(ctx)
	}
}

func (cl *ChatLoop) chatTurn(ctx context.Context, messages []ollama.Message) error {
	hasTools := cl.cfg.ExperimentalWebSearch
	hasSpawnAgent := cl.cfg.MaxSubagentDepth > 0

	toolPrompt := ollama.BuildSystemPrompt(hasTools, hasSpawnAgent)
	chatMessages := ollama.InsertSystemPrompt(messages, toolPrompt)

	opts := cl.cfg.BuildOptions(cl.cfg.Model)

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

	turnLimit := cl.cfg.TurnLimit
	if turnLimit <= 0 {
		turnLimit = 100
	}
	for i := 0; i < turnLimit; i++ {
		result := cl.processTurn(ctx, chatMessages, ollamaTools, opts, messages)
		if result.err != nil {
			return result.err
		}
		chatMessages = result.messages
		if result.done {
			return nil
		}
	}

	return fmt.Errorf("tool call loop exceeded %d turns", turnLimit)
}

type turnResult struct {
	done     bool
	err      error
	messages []ollama.Message // updated chat history
}

func (cl *ChatLoop) processTurn(
	ctx context.Context,
	chatMessages []ollama.Message,
	ollamaTools []ollama.Tool,
	opts *ollama.Options,
	originalMessages []ollama.Message,
) turnResult {
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
	req.Options = opts

	stream, streamErr := cl.client.ChatStream(turnCtx, req)
	if streamErr != nil {
		return turnResult{err: fmt.Errorf("chat: %w", streamErr)}
	}

	var assistantMsg ollama.Message
	assistantMsg.Role = "assistant"
	fmt.Fprint(os.Stderr, "\n")

	for chunk := range stream {
		if chunk.Error != "" {
			fmt.Fprintf(os.Stderr, "\nError: %s\n", chunk.Error)
			return turnResult{err: fmt.Errorf("ollama: %s", chunk.Error)}
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

	// Persist Assistant Message if DB is enabled and we have a conversation ID
	if cl.cfg.PersistToDB && cl.db != nil && cl.currentConversationID != nil {
		go func(id int, msg ollama.Message) {
			if err := cl.db.SaveMessage(context.Background(), id, "assistant", msg.Content); err != nil {
				fmt.Fprintf(os.Stderr, "Warning: failed to save assistant message: %v\n", err)
			}
		}(*cl.currentConversationID, assistantMsg)
	}
	cl.mu.Unlock()

	if len(assistantMsg.ToolCalls) == 0 {
		return turnResult{done: true, messages: chatMessages}
	}

	for _, tc := range assistantMsg.ToolCalls {
		toolTimeout := time.Duration(cl.cfg.ModelTimeoutDuration()) * time.Second
		toolCtx, toolCancel := context.WithTimeout(ctx, toolTimeout)

		var result string
		var toolErr error

		fmt.Fprintf(os.Stderr, "  🛠  %s(%s)\n", tc.Function.Name, truncateArgs(string(tc.Function.Arguments)))

		if tc.Function.Name == "spawn_agent" && cl.cfg.MaxSubagentDepth > 0 {
			result, toolErr = cl.runSubAgent(toolCtx, string(tc.Function.Arguments), originalMessages)
		} else {
			result, toolErr = tools.ExecuteToolCall(toolCtx, tc.Function.Name, tc.Function.Arguments)
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

		// Persist Tool Message if DB is enabled
		if cl.cfg.PersistToDB && cl.db != nil && cl.currentConversationID != nil {
			go func(id int, msg ollama.Message) {
				if err := cl.db.SaveMessage(context.Background(), id, "tool", msg.Content); err != nil {
					fmt.Fprintf(os.Stderr, "Warning: failed to save tool message: %v\n", err)
				}
			}(*cl.currentConversationID, toolMsg)
		}
		cl.mu.Unlock()
	}

	return turnResult{messages: chatMessages}
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
		NumCtx:       cl.cfg.NumCtx,
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

// handleCommand processes slash commands.
func (cl *ChatLoop) handleCommand(ctx context.Context, input string) error {
	parts := strings.Fields(input)
	cmd := strings.TrimPrefix(parts[0], "/")

	switch cmd {
	case "help":
		fmt.Fprint(os.Stderr, `Commands:
  /help       Show this help
  /exit       Exit the program
  /clear      Clear message history
  /save       Save conversation (explicit update)
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
		if cl.db != nil && cl.currentConversationID != nil {
			cl.db.UpdateConversation(ctx, *cl.currentConversationID, map[string]interface{}{
				"saved_at": time.Now().Format(time.RFC3339),
			})
			fmt.Fprintf(os.Stderr, "Conversation %d saved.\n", *cl.currentConversationID)
		} else {
			fmt.Fprintln(os.Stderr, "No active conversation to save.")
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
			fmt.Fprintf(os.Stderr, "  %d: %s | %s | Created: %s\n", s.ID, model,
				s.CreatedAt.Format("2006-01-02 15:04"),
				s.CreatedAt.Format("2006-01-02"))
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

		var sessionID int
		_, err := fmt.Sscanf(parts[1], "%d", &sessionID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Invalid session ID: %s\n", parts[1])
			return nil
		}

		msgs, err := cl.db.ExportSession(ctx, sessionID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading session %d: %v\n", sessionID, err)
			return nil
		}

		// Clear current messages and load new ones
		cl.mu.Lock()
		cl.messages = make([]ollama.Message, len(msgs))
		for i, m := range msgs {
			cl.messages[i] = ollama.Message{
				Role:    m.Role,
				Content: m.Content,
			}
		}
		cl.mu.Unlock()

		// Set the active conversation ID so future messages go to this session
		cl.currentConversationID = &sessionID

		fmt.Fprintf(os.Stderr, "Loaded session %d (%d messages). Messages restored.\n", sessionID, len(msgs))

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

// compactIfNeeded checks if the conversation is approaching the model's context
// limit and compacts it if necessary. Uses the model itself to summarize older
// messages, keeping recent context intact.
func (cl *ChatLoop) compactIfNeeded(ctx context.Context) {
	if cl.modelInfo == nil || !cl.cfg.AutoCompact {
		return
	}

	cl.mu.Lock()
	msgs := make([]ollama.Message, len(cl.messages))
	copy(msgs, cl.messages)
	cl.mu.Unlock()

	if !ollama.ShouldCompact(msgs, cl.modelInfo.ContextLength, cl.cfg.AutoCompactLimit) {
		return
	}

	estimated := ollama.EstimateTokens(msgs)
	fmt.Fprintf(os.Stderr, "\n  ⚡ Context: ~%d / %d tokens (%.0f%%) — compacting…\n",
		estimated, cl.modelInfo.ContextLength,
		float64(estimated)/float64(cl.modelInfo.ContextLength)*100)

	compacted, err := ollama.CompactMessages(ctx, cl.client, cl.cfg.Model, msgs, 6)
	if err != nil {
		fmt.Fprintf(os.Stderr, "  ⚠ Compact failed: %v\n", err)
		return
	}

	cl.mu.Lock()
	cl.messages = compacted
	cl.mu.Unlock()

	newEstimated := ollama.EstimateTokens(compacted)
	fmt.Fprintf(os.Stderr, "  ✓ Compacted: %d → %d messages (~%d → ~%d tokens)\n\n",
		len(msgs), len(compacted), estimated, newEstimated)
}

func truncateArgs(args string) string {
	if len(args) > 80 {
		cleaned := strings.ReplaceAll(args[:80], "\n", " ")
		return cleaned + "…"
	}
	return strings.ReplaceAll(args, "\n", " ")
}
