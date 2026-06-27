package ollama

import (
	"context"
	"fmt"
	"strings"
)

const WorkingMemoryPrefix = "Session Working Memory:"

// EstimateTokens returns a conservative rough token count for a slice of messages.
// It intentionally includes per-message/tool overhead so long-running sessions
// compact before they are dangerously close to the real model limit.
func EstimateTokens(messages []Message) int {
	chars := 0
	for _, msg := range messages {
		chars += len(msg.Role) + len(msg.Content) + 64
		for _, tc := range msg.ToolCalls {
			chars += len(tc.Function.Name) + len(tc.Function.Arguments) + 64
		}
		if len(msg.Images) > 0 {
			chars += len(msg.Images) * 256
		}
	}
	if chars == 0 {
		return 0
	}
	return chars/4 + 1
}

type CompactOptions struct {
	KeepRecent       int
	TargetTokens     int
	MaxSummaryTokens int
}

// CompactMessages summarizes old messages into a structured coding-session
// working memory. It preserves non-memory system messages and recent messages.
func CompactMessages(
	ctx context.Context,
	client *Client,
	model string,
	messages []Message,
	keepRecent int,
) ([]Message, error) {
	return CompactMessagesWithOptions(ctx, client, model, messages, CompactOptions{KeepRecent: keepRecent})
}

func CompactMessagesWithOptions(
	ctx context.Context,
	client *Client,
	model string,
	messages []Message,
	opts CompactOptions,
) ([]Message, error) {
	keepRecent := opts.KeepRecent
	if keepRecent <= 0 {
		keepRecent = 8
	}
	if len(messages) <= keepRecent+2 {
		return messages, nil
	}

	baseSystem, existingMemory, nonSystem := splitForCompaction(messages)
	if len(nonSystem) <= keepRecent+1 {
		return messages, nil
	}

	oldEnd := len(nonSystem) - keepRecent
	if oldEnd <= 0 {
		return messages, nil
	}
	oldMessages := nonSystem[:oldEnd]
	recentMessages := nonSystem[oldEnd:]

	summary, err := buildWorkingMemory(ctx, client, model, existingMemory, oldMessages, opts.MaxSummaryTokens)
	if err != nil {
		return nil, err
	}

	compacted := make([]Message, 0, len(baseSystem)+1+len(recentMessages))
	compacted = append(compacted, baseSystem...)
	compacted = append(compacted, Message{
		Role:    "system",
		Content: WorkingMemoryPrefix + "\n" + strings.TrimSpace(summary),
	})
	compacted = append(compacted, recentMessages...)

	// If a target was provided and the result is still too large, trim recent
	// context progressively. The durable working memory is the source of truth.
	if opts.TargetTokens > 0 {
		for EstimateTokens(compacted) > opts.TargetTokens && len(recentMessages) > 2 {
			recentMessages = recentMessages[1:]
			compacted = compacted[:0]
			compacted = append(compacted, baseSystem...)
			compacted = append(compacted, Message{Role: "system", Content: WorkingMemoryPrefix + "\n" + strings.TrimSpace(summary)})
			compacted = append(compacted, recentMessages...)
		}
	}

	return compacted, nil
}

func splitForCompaction(messages []Message) (baseSystem []Message, existingMemory string, nonSystem []Message) {
	for _, msg := range messages {
		if msg.Role != "system" {
			nonSystem = append(nonSystem, msg)
			continue
		}
		if IsWorkingMemory(msg) || strings.Contains(msg.Content, "Previous conversation summary:") {
			if existingMemory != "" {
				existingMemory += "\n\n"
			}
			existingMemory += msg.Content
			continue
		}
		baseSystem = append(baseSystem, msg)
	}
	return baseSystem, existingMemory, nonSystem
}

func IsWorkingMemory(msg Message) bool {
	return msg.Role == "system" && strings.HasPrefix(strings.TrimSpace(msg.Content), WorkingMemoryPrefix)
}

func ExtractWorkingMemory(messages []Message) string {
	for _, msg := range messages {
		if IsWorkingMemory(msg) {
			return strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(msg.Content), WorkingMemoryPrefix))
		}
	}
	return ""
}

func buildWorkingMemory(ctx context.Context, client *Client, model string, existingMemory string, oldMessages []Message, maxSummaryTokens int) (string, error) {
	var sb strings.Builder
	sb.WriteString("Update the coding-session working memory using the existing memory and the conversation/tool history below.\n")
	sb.WriteString("Return ONLY markdown with these exact headings:\n")
	sb.WriteString("# Current Goal\n# User Preferences / Constraints\n# Repo Facts\n# Important Files\n# Decisions Made\n# Changes Made\n# Commands Run\n# Failing Tests / Errors\n# Artifacts\n# Open Questions\n# Next Steps\n\n")
	sb.WriteString("Rules:\n")
	sb.WriteString("- Preserve concrete filenames, symbols, commands, error messages, decisions, and TODOs.\n")
	sb.WriteString("- Do not include chit-chat.\n")
	sb.WriteString("- If a section has nothing useful, write '- None'.\n")
	sb.WriteString("- Merge and deduplicate facts from existing memory.\n")
	if maxSummaryTokens > 0 {
		sb.WriteString(fmt.Sprintf("- Keep the result under roughly %d tokens.\n", maxSummaryTokens))
	}

	if strings.TrimSpace(existingMemory) != "" {
		sb.WriteString("\nExisting memory:\n")
		sb.WriteString(truncateForSummary(existingMemory, 12000))
		sb.WriteString("\n")
	}

	sb.WriteString("\nHistory to distill:\n")
	for _, msg := range oldMessages {
		sb.WriteString("\n--- ")
		sb.WriteString(msg.Role)
		sb.WriteString(" ---\n")
		if len(msg.ToolCalls) > 0 {
			for _, tc := range msg.ToolCalls {
				sb.WriteString(fmt.Sprintf("tool_call: %s(%s)\n", tc.Function.Name, truncateForSummary(string(tc.Function.Arguments), 1200)))
			}
		}
		if strings.TrimSpace(msg.Content) != "" {
			sb.WriteString(salientForSummary(msg.Content, msg.Role))
			sb.WriteString("\n")
		}
	}

	summaryReq := ChatRequest{
		Model: model,
		Messages: []Message{
			{Role: "system", Content: "You are a meticulous coding-session memory curator. You produce compact, factual, structured markdown memory for future continuation."},
			{Role: "user", Content: sb.String()},
		},
		Stream: false,
	}

	resp, err := client.Chat(ctx, summaryReq)
	if err != nil {
		return "", fmt.Errorf("compact working memory: %w", err)
	}

	summary := strings.TrimSpace(resp.Message.Content)
	if summary == "" {
		summary = fallbackWorkingMemory(existingMemory, oldMessages)
	}
	return summary, nil
}

func fallbackWorkingMemory(existingMemory string, oldMessages []Message) string {
	var sb strings.Builder
	if strings.TrimSpace(existingMemory) != "" {
		sb.WriteString(strings.TrimSpace(existingMemory))
		sb.WriteString("\n\n")
	}
	sb.WriteString("# Current Goal\n- Unknown\n")
	sb.WriteString("# User Preferences / Constraints\n- None\n")
	sb.WriteString("# Repo Facts\n- None\n")
	sb.WriteString("# Important Files\n- None\n")
	sb.WriteString("# Decisions Made\n- None\n")
	sb.WriteString("# Changes Made\n- None\n")
	sb.WriteString("# Commands Run\n")
	for _, msg := range oldMessages {
		if msg.Role == "tool" || strings.Contains(msg.Content, "Command") {
			sb.WriteString("- ")
			sb.WriteString(truncateForSummary(strings.ReplaceAll(msg.Content, "\n", " "), 300))
			sb.WriteString("\n")
		}
	}
	sb.WriteString("# Failing Tests / Errors\n- Unknown\n")
	sb.WriteString("# Artifacts\n- None\n")
	sb.WriteString("# Open Questions\n- None\n")
	sb.WriteString("# Next Steps\n- Continue from recent messages.\n")
	return sb.String()
}

func salientForSummary(s string, role string) string {
	limit := 1800
	if role == "tool" {
		return extractImportantLines(s, 2200)
	}
	return truncateForSummary(s, limit)
}

func extractImportantLines(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	keywords := []string{"error", "failed", "failure", "panic", "undefined", "cannot", "no such", "denied", "diff", "modified", "created", "deleted", "artifact", "warning", "TODO", "FAIL", "PASS"}
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
	}
	if len(picked) == 0 {
		head := truncateForSummary(s, maxLen/2)
		tail := s[len(s)-maxLen/2:]
		return head + "\n...\n" + tail
	}
	out := strings.Join(picked, "\n")
	if len(out) > maxLen {
		out = truncateForSummary(out, maxLen)
	}
	return out
}

// ShouldCompact checks whether the estimated token count exceeds the threshold
// percentage of the model's context window.
func ShouldCompact(messages []Message, contextLength int, thresholdPercent int) bool {
	if contextLength <= 0 || thresholdPercent <= 0 {
		return false
	}
	estimated := EstimateTokens(messages)
	limit := contextLength * thresholdPercent / 100
	return estimated >= limit
}

func truncateForSummary(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	if maxLen <= 3 {
		return s[:maxLen]
	}
	return s[:maxLen] + "..."
}
