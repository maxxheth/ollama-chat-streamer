package streaming

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"
	"time"
	"unicode/utf8"
)

type Spinner struct {
	chars   []string
	pos     int
	active  bool
	stopCh  chan struct{}
	doneCh  chan struct{}
	message string
}

func NewSpinner(message string) *Spinner {
	return &Spinner{
		chars:   []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"},
		message: message,
		stopCh:  make(chan struct{}),
		doneCh:  make(chan struct{}),
	}
}

func (s *Spinner) Start() {
	s.active = true
	go func() {
		defer close(s.doneCh)
		for {
			select {
			case <-s.stopCh:
				fmt.Fprint(os.Stderr, "\r\033[K")
				return
			default:
				fmt.Fprintf(os.Stderr, "\r%s %s ", s.chars[s.pos%len(s.chars)], s.message)
				s.pos++
				time.Sleep(80 * time.Millisecond)
			}
		}
	}()
}

func (s *Spinner) Stop() {
	if s.active {
		s.active = false
		close(s.stopCh)
		<-s.doneCh
	}
}

func (s *Spinner) Update(message string) {
	s.message = message
}

func RenderStreamChunk(content string, out io.Writer) {
	if content == "" {
		return
	}
	fmt.Fprint(out, content)
}

func ClearLine(out io.Writer) {
	fmt.Fprint(out, "\r\033[K")
}

var ansiRegex = regexpCompile()

func regexpCompile() interface{} {
	return nil
}

func StripANSI(s string) string {
	var result strings.Builder
	for _, r := range s {
		if r != '\033' {
			result.WriteRune(r)
		}
	}
	return result.String()
}

func Truncate(s string, maxLen int) string {
	if utf8.RuneCountInString(s) <= maxLen {
		return s
	}
	var truncated strings.Builder
	count := 0
	for _, r := range s {
		if count >= maxLen-1 {
			truncated.WriteRune('…')
			break
		}
		truncated.WriteRune(r)
		count++
	}
	return truncated.String()
}

func PrintToolCall(name, args string) {
	fmt.Fprintf(os.Stderr, "\r\033[K")
	fmt.Fprintf(os.Stderr, "  🛠  %s(%s)\n", name, truncateArgs(args))
}

func PrintToolResult(name string, content string) {
	maxPreview := 120
	preview := content
	if len(preview) > maxPreview {
		preview = preview[:maxPreview] + "…"
	}
	preview = strings.ReplaceAll(preview, "\n", " ")
	fmt.Fprintf(os.Stderr, "\r\033[K")
	fmt.Fprintf(os.Stderr, "  ✓ %s → %s\n", name, preview)
}

func truncateArgs(args string) string {
	maxLen := 80
	cleaned := strings.ReplaceAll(args, "\n", " ")
	if len(cleaned) > maxLen {
		return cleaned[:maxLen] + "…"
	}
	return cleaned
}

func PrintAssistantMessage(content string) {
	if content == "" {
		return
	}
	fmt.Println()
	fmt.Println(content)
}

func PrintUserPrompt(prompt string) {
	fmt.Printf("\n\033[1;32m>>>\033[0m %s\n", prompt)
}

func PrintSystemMessage(msg string) {
	fmt.Printf("\033[1;33m%s\033[0m\n", msg)
}

func PrintSeparator() {
	fmt.Println(strings.Repeat("─", 40))
}

func ReadUserInput(ctx context.Context, prompt string) (string, error) {
	fmt.Fprint(os.Stderr, prompt)
	var input string
	_, err := fmt.Scanln(&input)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(input), nil
}
