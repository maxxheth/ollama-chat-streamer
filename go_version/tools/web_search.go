package tools

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/PuerkitoBio/goquery"
)

var webSearchTool = Tool{
	Name:        "web_search",
	Description: "Search the web using DuckDuckGo. Returns a list of results with titles, snippets, and URLs.",
	Parameters: map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"query": map[string]interface{}{
				"type":        "string",
				"description": "The search query",
			},
			"max_results": map[string]interface{}{
				"type":        "integer",
				"description": "Maximum number of results (default: 5)",
			},
		},
		"required": []string{"query"},
	},
	Execute: webSearchExecute,
}

func webSearchExecute(ctx context.Context, args map[string]interface{}) (string, error) {
	query := GetString(args, "query")
	if query == "" {
		return "", fmt.Errorf("query is required")
	}
	maxResults := GetInt(args, "max_results")
	if maxResults <= 0 {
		maxResults = 5
	}

	results, err := ddgSearch(ctx, query, maxResults)
	if err != nil {
		return "", fmt.Errorf("search failed: %w", err)
	}

	if len(results) == 0 {
		return "No results found.", nil
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Search results for \"%s\":\n\n", query))
	for i, r := range results {
		sb.WriteString(fmt.Sprintf("%d. %s\n", i+1, r.Title))
		sb.WriteString(fmt.Sprintf("   URL: %s\n", r.URL))
		if r.Snippet != "" {
			sb.WriteString(fmt.Sprintf("   %s\n", r.Snippet))
		}
		sb.WriteString("\n")
	}

	return sb.String(), nil
}

type searchResult struct {
	Title   string
	URL     string
	Snippet string
}

func ddgSearch(ctx context.Context, query string, maxResults int) ([]searchResult, error) {
	searchURL := fmt.Sprintf("https://html.duckduckgo.com/html/?q=%s", url.QueryEscape(query))

	client := &http.Client{Timeout: 10 * time.Second}
	req, err := http.NewRequestWithContext(ctx, "GET", searchURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	doc, err := goquery.NewDocumentFromReader(resp.Body)
	if err != nil {
		return nil, err
	}

	var results []searchResult
	doc.Find(".result").Each(func(i int, s *goquery.Selection) {
		if len(results) >= maxResults {
			return
		}
		titleSel := s.Find(".result__title a")
		title := strings.TrimSpace(titleSel.Text())
		href, _ := titleSel.Attr("href")

		snippet := strings.TrimSpace(s.Find(".result__snippet").Text())

		if title != "" {
			results = append(results, searchResult{
				Title:   title,
				URL:     href,
				Snippet: snippet,
			})
		}
	})

	return results, nil
}
