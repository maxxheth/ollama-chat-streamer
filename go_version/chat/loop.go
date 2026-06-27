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
	lastMessageID         int  // Last persisted message ID for checkpoint boundaries
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

	// Initial DB State Setup. Resume the latest session from the newest durable
	// working-memory/checkpoint snapshot plus only the recent messages after it.
	if cl.cfg.PersistToDB && cl.db != nil {
		sessions, err := cl.db.ListSessions(ctx)
		if err == nil && len(sessions) > 0 {
			latestID := sessions[0].ID
			cl.currentConversationID = &latestID
			if err := cl.loadSessionCompact(ctx, latestID); err != nil {
				fmt.Fprintf(os.Stderr, "Warning: could not compact-resume session %d: %v\n", latestID, err)
			} else {
				fmt.Fprintf(os.Stderr, "Resuming latest session: %d (%d active context messages)\n", latestID, len(cl.messages))
			}
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
				if msgID, err := cl.db.SaveMessageID(ctx, *cl.currentConversationID, "user", prompt); err != nil {
					fmt.Fprintf(os.Stderr, "Warning: could not save user message: %v\n", err)
				} else if msgID != nil {
					cl.lastMessageID = *msgID
				}
			}
		}

		cl.messages = append(cl.messages, ollama.Message{
			Role:    "user",
			Content: prompt,
		})

		// Preflight compaction before the next model call. Post-turn compaction is
		// too late for very large prompts/tool histories.
		cl.compactIfNeeded(ctx)

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
	basePrompt := strings.TrimSpace(strings.Join([]string{cl.cfg.SystemPrompt, toolPrompt}, "\n\n"))
	chatMessages := ollama.InsertSystemPrompt(messages, basePrompt)

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
		chatMessages = cl.compactLocalIfNeeded(ctx, chatMessages)
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
	cl.mu.Unlock()
	cl.persistMessage(ctx, "assistant", assistantMsg.Content)

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
		modelResult := cl.maybeStoreArtifact(ctx, tc.Function.Name, string(tc.Function.Arguments), result)
		toolResultPreview := modelResult
		if len(toolResultPreview) > 120 {
			toolResultPreview = toolResultPreview[:120] + "…"
		}
		fmt.Fprintf(os.Stderr, "  ✓ %s → %s\n", tc.Function.Name, strings.ReplaceAll(toolResultPreview, "\n", " "))

		toolMsg := ollama.Message{
			Role:    "tool",
			Content: modelResult,
		}
		chatMessages = append(chatMessages, toolMsg)
		cl.mu.Lock()
		cl.messages = append(cl.messages, toolMsg)
		cl.mu.Unlock()
		cl.persistMessage(ctx, "tool", toolMsg.Content)

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
  /save          Save conversation metadata
  /checkpoint    Save a durable working-memory checkpoint
  /status        Show context/session status
  /export        Export active context as JSON
  /sessions      List saved sessions
  /load <id>     Load a saved session from latest snapshot + recent messages
  /artifacts     List recent large tool-output artifacts
  /artifact <id> Print a stored artifact
  /model         Show current model
  /history       Show message history count
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

		if err := cl.loadSessionCompact(ctx, sessionID); err != nil {
			fmt.Fprintf(os.Stderr, "Error loading session %d: %v\n", sessionID, err)
			return nil
		}
		cl.currentConversationID = &sessionID
		fmt.Fprintf(os.Stderr, "Loaded session %d (%d active context messages, last message id %d).\n", sessionID, len(cl.messages), cl.lastMessageID)

	case "checkpoint":
		if err := cl.saveCheckpoint(ctx); err != nil {
			fmt.Fprintf(os.Stderr, "Checkpoint error: %v\n", err)
			return nil
		}
	case "status":
		fmt.Fprint(os.Stderr, cl.statusString())
	case "artifacts":
		if cl.db == nil {
			fmt.Fprintln(os.Stderr, "Database not available.")
			return nil
		}
		artifacts, err := cl.db.ListArtifacts(ctx, cl.currentConversationID, 20)
		if err != nil {
			return err
		}
		if len(artifacts) == 0 {
			fmt.Fprintln(os.Stderr, "No artifacts.")
			return nil
		}
		for _, a := range artifacts {
			fmt.Fprintf(os.Stderr, "  #%d %s %s (%d bytes) %s\n", a.ID, a.Kind, a.Name, a.SizeBytes, a.CreatedAt.Format("2006-01-02 15:04"))
			if strings.TrimSpace(a.Summary) != "" {
				fmt.Fprintf(os.Stderr, "      %s\n", firstLine(a.Summary))
			}
		}
	case "artifact":
		if len(parts) < 2 {
			fmt.Fprintln(os.Stderr, "Usage: /artifact <id>")
			return nil
		}
		if cl.db == nil {
			fmt.Fprintln(os.Stderr, "Database not available.")
			return nil
		}
		var artifactID int
		if _, err := fmt.Sscanf(parts[1], "%d", &artifactID); err != nil {
			fmt.Fprintf(os.Stderr, "Invalid artifact ID: %s\n", parts[1])
			return nil
		}
		artifact, err := cl.db.GetArtifact(ctx, artifactID)
		if err != nil {
			return err
		}
		if artifact == nil {
			fmt.Fprintf(os.Stderr, "Artifact %d not found.\n", artifactID)
			return nil
		}
		fmt.Printf("Artifact #%d %s %s (%d bytes, sha256 %s)\n\n%s\n", artifact.ID, artifact.Kind, artifact.Name, artifact.SizeBytes, artifact.ContentHash, artifact.Content)

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

func (cl *ChatLoop) effectiveContextLength() int {
	if cl.cfg.NumCtx > 0 {
		return cl.cfg.NumCtx
	}
	if cl.modelInfo != nil && cl.modelInfo.ContextLength > 0 {
		return cl.modelInfo.ContextLength
	}
	return 0
}

func (cl *ChatLoop) compactTargetTokens() int {
	contextLength := cl.effectiveContextLength()
	if contextLength <= 0 {
		return 0
	}
	target := cl.cfg.AutoCompactTarget
	if target <= 0 || target >= cl.cfg.AutoCompactLimit {
		target = 50
	}
	return contextLength * target / 100
}

func (cl *ChatLoop) compactKeepRecent() int {
	if cl.cfg.AutoCompactKeepRecent > 0 {
		return cl.cfg.AutoCompactKeepRecent
	}
	return 8
}

// compactLocalIfNeeded protects tool-call loops where local tool results can
// grow beyond the context budget before the top-level cl.messages is compacted.
func (cl *ChatLoop) compactLocalIfNeeded(ctx context.Context, msgs []ollama.Message) []ollama.Message {
	if !cl.cfg.AutoCompact {
		return msgs
	}
	contextLength := cl.effectiveContextLength()
	if contextLength <= 0 || !ollama.ShouldCompact(msgs, contextLength, cl.cfg.AutoCompactLimit) {
		return msgs
	}

	compacted, err := ollama.CompactMessagesWithOptions(ctx, cl.client, cl.cfg.Model, msgs, ollama.CompactOptions{
		KeepRecent:       cl.compactKeepRecent(),
		TargetTokens:     cl.compactTargetTokens(),
		MaxSummaryTokens: contextLength / 10,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "  ⚠ Local compact failed: %v\n", err)
		return msgs
	}
	fmt.Fprintf(os.Stderr, "  ✓ Local context compacted for tool loop: %d → %d messages\n", len(msgs), len(compacted))
	return compacted
}

// compactIfNeeded checks if the conversation is approaching the model's context
// limit and compacts it into a structured working-memory snapshot.
func (cl *ChatLoop) compactIfNeeded(ctx context.Context) {
	if !cl.cfg.AutoCompact {
		return
	}
	contextLength := cl.effectiveContextLength()
	if contextLength <= 0 {
		return
	}

	cl.mu.Lock()
	msgs := make([]ollama.Message, len(cl.messages))
	copy(msgs, cl.messages)
	cl.mu.Unlock()

	if !ollama.ShouldCompact(msgs, contextLength, cl.cfg.AutoCompactLimit) {
		return
	}

	estimated := ollama.EstimateTokens(msgs)
	fmt.Fprintf(os.Stderr, "\n  ⚡ Context: ~%d / %d tokens (%.0f%%) — compacting to working memory…\n",
		estimated, contextLength,
		float64(estimated)/float64(contextLength)*100)

	compacted, err := ollama.CompactMessagesWithOptions(ctx, cl.client, cl.cfg.Model, msgs, ollama.CompactOptions{
		KeepRecent:       cl.compactKeepRecent(),
		TargetTokens:     cl.compactTargetTokens(),
		MaxSummaryTokens: contextLength / 10,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "  ⚠ Compact failed: %v\n", err)
		return
	}

	cl.mu.Lock()
	cl.messages = compacted
	cl.mu.Unlock()

	newEstimated := ollama.EstimateTokens(compacted)
	fmt.Fprintf(os.Stderr, "  ✓ Compacted: %d → %d messages (~%d → ~%d tokens)\n",
		len(msgs), len(compacted), estimated, newEstimated)
	cl.persistWorkingMemorySnapshot(ctx, "working_memory", compacted, newEstimated)
	fmt.Fprintln(os.Stderr)
}

func (cl *ChatLoop) persistMessage(ctx context.Context, role, content string) {
	if !cl.cfg.PersistToDB || cl.db == nil || cl.currentConversationID == nil || content == "" {
		return
	}
	msgID, err := cl.db.SaveMessageID(ctx, *cl.currentConversationID, role, content)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: failed to save %s message: %v\n", role, err)
		return
	}
	if msgID != nil {
		cl.lastMessageID = *msgID
	}
}

func (cl *ChatLoop) persistWorkingMemorySnapshot(ctx context.Context, kind string, messages []ollama.Message, tokenEstimate int) {
	if cl.db == nil || cl.currentConversationID == nil {
		return
	}
	memory := ollama.ExtractWorkingMemory(messages)
	if strings.TrimSpace(memory) == "" {
		return
	}
	messageIDThrough := cl.lastMessageID
	if _, err := cl.db.SaveSnapshot(ctx, *cl.currentConversationID, &messageIDThrough, kind, memory, tokenEstimate); err != nil {
		fmt.Fprintf(os.Stderr, "Warning: failed to save %s snapshot: %v\n", kind, err)
	}
}

func (cl *ChatLoop) saveCheckpoint(ctx context.Context) error {
	if cl.db == nil || cl.currentConversationID == nil {
		return fmt.Errorf("no active persisted conversation")
	}

	cl.mu.Lock()
	msgs := make([]ollama.Message, len(cl.messages))
	copy(msgs, cl.messages)
	cl.mu.Unlock()

	// Force a structured working memory update even if we are not at the compact
	// threshold, so checkpoints are useful resume points.
	compacted, err := ollama.CompactMessagesWithOptions(ctx, cl.client, cl.cfg.Model, msgs, ollama.CompactOptions{
		KeepRecent:       cl.compactKeepRecent(),
		TargetTokens:     cl.compactTargetTokens(),
		MaxSummaryTokens: maxInt(cl.effectiveContextLength()/10, 1200),
	})
	if err != nil {
		return err
	}

	memory := ollama.ExtractWorkingMemory(compacted)
	if strings.TrimSpace(memory) == "" {
		memory = manualCheckpointMemory(compacted)
		compacted = append([]ollama.Message{{Role: "system", Content: ollama.WorkingMemoryPrefix + "\n" + memory}}, compacted...)
	}

	cl.mu.Lock()
	cl.messages = compacted
	cl.mu.Unlock()

	messageIDThrough := cl.lastMessageID
	if _, err := cl.db.SaveSnapshot(ctx, *cl.currentConversationID, &messageIDThrough, "checkpoint", memory, ollama.EstimateTokens(compacted)); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "Checkpoint saved for conversation %d at message %d.\n", *cl.currentConversationID, cl.lastMessageID)
	return nil
}

func manualCheckpointMemory(messages []ollama.Message) string {
	start := len(messages) - 20
	if start < 0 {
		start = 0
	}
	var sb strings.Builder
	sb.WriteString("# Current Goal\n- Continue from the recent session context.\n")
	sb.WriteString("# User Preferences / Constraints\n- None captured.\n")
	sb.WriteString("# Repo Facts\n- None captured.\n")
	sb.WriteString("# Important Files\n- None captured.\n")
	sb.WriteString("# Decisions Made\n- None captured.\n")
	sb.WriteString("# Changes Made\n- None captured.\n")
	sb.WriteString("# Commands Run\n")
	wroteCommand := false
	for _, msg := range messages[start:] {
		if msg.Role == "tool" {
			wroteCommand = true
			sb.WriteString("- ")
			sb.WriteString(firstLine(msg.Content))
			sb.WriteString("\n")
		}
	}
	if !wroteCommand {
		sb.WriteString("- None captured.\n")
	}
	sb.WriteString("# Failing Tests / Errors\n- None captured.\n")
	sb.WriteString("# Artifacts\n- See /artifacts if large tool outputs were stored.\n")
	sb.WriteString("# Open Questions\n- None captured.\n")
	sb.WriteString("# Next Steps\n- Review recent messages and continue.\n")
	return sb.String()
}

func (cl *ChatLoop) loadSessionCompact(ctx context.Context, sessionID int) error {
	var messages []ollama.Message
	cl.lastMessageID = 0
	afterID := 0
	if snap, err := cl.db.LatestSnapshot(ctx, sessionID, "checkpoint"); err != nil {
		return err
	} else if snap != nil {
		messages = append(messages, ollama.Message{Role: "system", Content: ollama.WorkingMemoryPrefix + "\n" + snap.Content})
		if snap.MessageIDThrough != nil {
			afterID = *snap.MessageIDThrough
		}
	} else if snap, err := cl.db.LatestSnapshot(ctx, sessionID, "working_memory"); err != nil {
		return err
	} else if snap != nil {
		messages = append(messages, ollama.Message{Role: "system", Content: ollama.WorkingMemoryPrefix + "\n" + snap.Content})
		if snap.MessageIDThrough != nil {
			afterID = *snap.MessageIDThrough
		}
	}

	dbMessages, err := cl.db.ExportSessionAfterMessageID(ctx, sessionID, afterID)
	if err != nil {
		return err
	}
	for _, m := range dbMessages {
		messages = append(messages, ollama.Message{Role: m.Role, Content: m.Content})
		if m.ID > cl.lastMessageID {
			cl.lastMessageID = m.ID
		}
	}
	if len(dbMessages) == 0 && afterID > cl.lastMessageID {
		cl.lastMessageID = afterID
	}

	cl.mu.Lock()
	cl.messages = messages
	cl.mu.Unlock()
	return nil
}

func (cl *ChatLoop) maybeStoreArtifact(ctx context.Context, toolName, args, content string) string {
	limit := cl.cfg.ToolResultMaxInline
	if limit <= 0 {
		limit = 12000
	}
	if len(content) <= limit {
		return content
	}

	summary := summarizeToolOutput(content, 3000)
	if cl.db == nil {
		return fmt.Sprintf("Large %s tool result (%d bytes) was summarized because persistence is unavailable.\nArguments: %s\n\nSummary:\n%s", toolName, len(content), truncateArgs(args), summary)
	}

	name := fmt.Sprintf("%s %s", toolName, time.Now().Format("2006-01-02 15:04:05"))
	artifact, err := cl.db.SaveArtifact(ctx, cl.currentConversationID, "tool_output", name, summary, content)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: failed to save large tool output artifact: %v\n", err)
		return fmt.Sprintf("Large %s tool result (%d bytes) could not be persisted; summarized inline.\nArguments: %s\n\nSummary:\n%s", toolName, len(content), truncateArgs(args), summary)
	}
	return fmt.Sprintf("Large %s tool result stored as artifact #%d (%d bytes, sha256 %s).\nArguments: %s\n\nSummary:\n%s\n\nUse /artifact %d to inspect the full output.", toolName, artifact.ID, artifact.SizeBytes, artifact.ContentHash, truncateArgs(args), summary, artifact.ID)
}

func summarizeToolOutput(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	keywords := []string{"error", "failed", "failure", "panic", "undefined", "cannot", "denied", "warning", "FAIL", "PASS", "modified", "created", "deleted"}
	lines := strings.Split(s, "\n")
	var picked []string
	for _, line := range lines {
		lower := strings.ToLower(line)
		for _, kw := range keywords {
			if strings.Contains(lower, strings.ToLower(kw)) {
				picked = append(picked, line)
				break
			}
		}
		if len(strings.Join(picked, "\n")) > maxLen {
			break
		}
	}
	if len(picked) > 0 {
		out := strings.Join(picked, "\n")
		if len(out) > maxLen {
			return out[:maxLen] + "…"
		}
		return out
	}
	headLen := maxLen / 2
	tailLen := maxLen - headLen
	return s[:headLen] + "\n... [middle omitted] ...\n" + s[len(s)-tailLen:]
}

func (cl *ChatLoop) statusString() string {
	cl.mu.Lock()
	count := len(cl.messages)
	estimated := ollama.EstimateTokens(cl.messages)
	memory := ollama.ExtractWorkingMemory(cl.messages)
	cl.mu.Unlock()
	contextLength := cl.effectiveContextLength()
	var sb strings.Builder
	sb.WriteString("Session status:\n")
	if cl.currentConversationID != nil {
		sb.WriteString(fmt.Sprintf("  Conversation: %d\n", *cl.currentConversationID))
	} else {
		sb.WriteString("  Conversation: none\n")
	}
	sb.WriteString(fmt.Sprintf("  Messages in active context: %d\n", count))
	sb.WriteString(fmt.Sprintf("  Estimated tokens: %d", estimated))
	if contextLength > 0 {
		sb.WriteString(fmt.Sprintf(" / %d (%.0f%%)", contextLength, float64(estimated)/float64(contextLength)*100))
	}
	sb.WriteString("\n")
	sb.WriteString(fmt.Sprintf("  Last persisted message ID: %d\n", cl.lastMessageID))
	if strings.TrimSpace(memory) != "" {
		sb.WriteString("  Working memory: present\n")
	} else {
		sb.WriteString("  Working memory: none yet\n")
	}
	return sb.String()
}

func firstLine(s string) string {
	s = strings.TrimSpace(s)
	if idx := strings.Index(s, "\n"); idx >= 0 {
		s = s[:idx]
	}
	if len(s) > 100 {
		return s[:100] + "…"
	}
	return s
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func truncateArgs(args string) string {
	if len(args) > 80 {
		cleaned := strings.ReplaceAll(args[:80], "\n", " ")
		return cleaned + "…"
	}
	return strings.ReplaceAll(args, "\n", " ")
}
