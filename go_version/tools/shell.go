package tools

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"strings"
	"time"
)

var runShellTool = Tool{
	Name:        "run_shell",
	Description: "Execute a shell command and return its output. Use for running scripts, git commands, build tools, or any command-line operation. The command runs with a 30-second timeout.",
	Parameters: map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"command": map[string]interface{}{
				"type":        "string",
				"description": "The shell command to execute",
			},
			"timeout": map[string]interface{}{
				"type":        "integer",
				"description": "Timeout in seconds (default: 30)",
			},
		},
		"required": []string{"command"},
	},
	Execute: runShellExecute,
}

func runShellExecute(ctx context.Context, args map[string]interface{}) (string, error) {
	command := GetString(args, "command")
	if command == "" {
		return "", fmt.Errorf("command is required")
	}

	timeoutSec := GetInt(args, "timeout")
	if timeoutSec <= 0 {
		timeoutSec = 30
	}

	ctx, cancel := context.WithTimeout(ctx, time.Duration(timeoutSec)*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "bash", "-c", command)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()

	var sb strings.Builder
	if stdout.Len() > 0 {
		sb.WriteString(stdout.String())
	}
	if stderr.Len() > 0 {
		if sb.Len() > 0 {
			sb.WriteString("\n")
		}
		sb.WriteString("stderr:\n")
		sb.WriteString(stderr.String())
	}

	output := sb.String()

	if err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return output, fmt.Errorf("command timed out after %ds", timeoutSec)
		}
		if output == "" {
			return "", fmt.Errorf("command failed: %w", err)
		}
		return output, fmt.Errorf("command failed (exit code %d): %w", cmd.ProcessState.ExitCode(), err)
	}

	if output == "" {
		return "Command completed successfully (no output)", nil
	}
	return strings.TrimRight(output, "\n"), nil
}
