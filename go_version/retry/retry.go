package retry

import (
	"context"
	"fmt"
	"math"
	"os"
	"time"
)

type RetryConfig struct {
	MaxRetries int
	BaseDelay  time.Duration
	MaxDelay   time.Duration
}

func DefaultConfig() *RetryConfig {
	return &RetryConfig{
		MaxRetries: 3,
		BaseDelay:  2 * time.Second,
		MaxDelay:   30 * time.Second,
	}
}

type StreamItem[T any] struct {
	Value T
	Err   error
}

func Retry[T any](ctx context.Context, cfg *RetryConfig, label string, fn func(context.Context) (T, error)) (T, error) {
	var zero T
	var lastErr error

	for attempt := 0; attempt <= cfg.MaxRetries; attempt++ {
		if attempt > 0 {
			delay := time.Duration(math.Min(
				float64(cfg.BaseDelay)*math.Pow(2, float64(attempt-1)),
				float64(cfg.MaxDelay),
			))
			fmt.Fprintf(nil, "  retry %d/%d (%v)…\n", attempt, cfg.MaxRetries, lastErr)
			select {
			case <-ctx.Done():
				return zero, ctx.Err()
			case <-time.After(delay):
			}
		}

		result, err := fn(ctx)
		if err == nil {
			return result, nil
		}
		lastErr = err
	}

	return zero, fmt.Errorf("%s failed after %d retries: %w", label, cfg.MaxRetries, lastErr)
}

func RetryStream[T any](
	ctx context.Context,
	cfg *RetryConfig,
	label string,
	fn func(context.Context) (<-chan StreamItem[T], error),
) (<-chan StreamItem[T], error) {
	out := make(chan StreamItem[T])

	go func() {
		defer close(out)

		var lastErr error
		for attempt := 0; attempt <= cfg.MaxRetries; attempt++ {
			if attempt > 0 {
				delay := time.Duration(math.Min(
					float64(cfg.BaseDelay)*math.Pow(2, float64(attempt-1)),
					float64(cfg.MaxDelay),
				))
			fmt.Fprintf(os.Stderr, "  retry %d/%d (%v)…\n", attempt, cfg.MaxRetries, lastErr)
				select {
				case <-ctx.Done():
					return
				case <-time.After(delay):
				}
			}

			stream, err := fn(ctx)
			if err != nil {
				lastErr = err
				continue
			}

			sent := 0
			for item := range stream {
				if item.Err != nil {
					lastErr = item.Err
					break
				}
				out <- item
				sent++
			}

			if lastErr == nil {
				return
			}
		}

		if lastErr != nil {
			out <- StreamItem[T]{Err: fmt.Errorf("%s failed after %d retries: %w", label, cfg.MaxRetries, lastErr)}
		}
	}()

	return out, nil
}

func RunWithTimeout[T any](ctx context.Context, timeout time.Duration, fn func(context.Context) (T, error)) (T, error) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	return fn(ctx)
}
