package ollama

import (
	"context"
	"fmt"
	"strings"
)

// EstimateTokens returns a rough token count for a slice of messages.
// Uses ~4 characters per token heuristic for English text.
func EstimateTokens(messages []Message) int {
	total := 0
	for _, msg := range messages {
		total += len(msg.Content)
		for _, tc := range msg.ToolCalls {
			total += len(tc.Function.Name) + len(tc.Function.Arguments)
		}
	}
	return total / 4
}

// CompactMessages summarizes old messages and replaces them with a condensed
// system message. Keeps the system prompt and the most recent messages intact.
//
// Strategy:
//   - Preserve the first system message (if any)
//   - Keep the last keepRecent messages (default: 6, i.e. 3 exchanges)
//   - Summarize everything in between into a single system message
//
// Returns the compacted message list and whether compaction was performed.
func CompactMessages(
	ctx context.Context,
	client *Client,
	model string,
	messages []Message,
	keepRecent int,
) ([]Message, error) {
	if len(messages) <= keepRecent+2 {
		// Not enough messages to compact
		return messages, nil
	}

	// Find the system message (always keep it)
	sysIdx := -1
	for i, msg := range messages {
		if msg.Role == "system" {
			sysIdx = i
			break
		}
	}

	// Determine the boundary: everything after sysIdx+1 up to keepRecent
	// from the end is "old" and will be summarized.
	oldStart := 0
	if sysIdx >= 0 {
		oldStart = sysIdx + 1
	}
	oldEnd := len(messages) - keepRecent
	if oldEnd <= oldStart {
		return messages, nil // nothing to compact
	}

	oldMessages := messages[oldStart:oldEnd]
	recentMessages := messages[oldEnd:]

	// Build a summary prompt
	var sb strings.Builder
	sb.WriteString("Summarize the following conversation history concisely. ")
	sb.WriteString("Capture key facts, decisions, user preferences, and important context. ")
	sb.WriteString("Keep it brief — this summary will replace the full history to save context space.\n\n")
	sb.WriteString("Conversation to summarize:\n")
	for _, msg := range oldMessages {
		if msg.Content != "" {
			sb.WriteString(fmt.Sprintf("[%s]: %s\n", msg.Role, truncateForSummary(msg.Content, 300)))
		}
	}

	// Call the model to generate the summary
	summaryReq := ChatRequest{
		Model: model,
		Messages: []Message{
			{Role: "system", Content: "You are a conversation summarizer. Produce a concise, factual summary."},
			{Role: "user", Content: sb.String()},
		},
		Stream: false,
	}

	resp, err := client.Chat(ctx, summaryReq)
	if err != nil {
		return nil, fmt.Errorf("compact summary: %w", err)
	}

	summary := strings.TrimSpace(resp.Message.Content)
	if summary == "" {
		summary = "(conversation history compacted)"
	}

	// Rebuild the message list
	var compacted []Message

	// Keep the original system message
	if sysIdx >= 0 {
		compacted = append(compacted, messages[sysIdx])
	}

	// Insert the summary as a system message
	compacted = append(compacted, Message{
		Role:    "system",
		Content: fmt.Sprintf("Previous conversation summary:\n%s", summary),
	})

	// Append recent messages
	compacted = append(compacted, recentMessages...)

	return compacted, nil
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
	return s[:maxLen] + "..."
}
