package db

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

type Pool struct {
	*pgxpool.Pool
}

type Conversation struct {
	ID        int              `json:"id"`
	Model     string           `json:"model"`
	Flags     map[string]interface{} `json:"flags"`
	CreatedAt time.Time        `json:"created_at"`
}

type Message struct {
	ID             int       `json:"id"`
	ConversationID int       `json:"conversation_id"`
	Role           string    `json:"role"`
	Content        string    `json:"content"`
	CreatedAt      time.Time `json:"created_at"`
}

func Connect(ctx context.Context, databaseURL string) (*Pool, error) {
	cfg, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		return nil, fmt.Errorf("pgx config: %w", err)
	}

	cfg.MaxConns = 10
	cfg.MinConns = 2

	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("pgx pool: %w", err)
	}

	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("pgx ping: %w", err)
	}

	p := &Pool{pool}
	if err := p.createSchema(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("schema: %w", err)
	}

	return p, nil
}

func (p *Pool) createSchema(ctx context.Context) error {
	_, err := p.Exec(ctx, `
		CREATE TABLE IF NOT EXISTS conversations (
			id SERIAL PRIMARY KEY,
			model TEXT NOT NULL,
			flags JSONB DEFAULT '{}',
			created_at TIMESTAMPTZ DEFAULT NOW()
		);
	`)
	if err != nil {
		return err
	}

	_, err = p.Exec(ctx, `
		CREATE TABLE IF NOT EXISTS messages (
			id SERIAL PRIMARY KEY,
			conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
			role TEXT NOT NULL,
			content TEXT NOT NULL,
			created_at TIMESTAMPTZ DEFAULT NOW()
		);
	`)
	return err
}

func (p *Pool) SaveConversation(ctx context.Context, model string, conversationID *int) (*int, error) {
	if conversationID != nil {
		return conversationID, nil
	}

	var id int
	err := p.QueryRow(ctx,
		`INSERT INTO conversations (model) VALUES ($1) RETURNING id`,
		model,
	).Scan(&id)
	if err != nil {
		return nil, fmt.Errorf("insert conversation: %w", err)
	}
	return &id, nil
}

func (p *Pool) UpdateConversation(ctx context.Context, conversationID int, flags map[string]interface{}) error {
	flagsJSON, err := json.Marshal(flags)
	if err != nil {
		return fmt.Errorf("marshal flags: %w", err)
	}

	_, err = p.Exec(ctx,
		`UPDATE conversations SET flags = $1 WHERE id = $2`,
		flagsJSON, conversationID,
	)
	return err
}

func (p *Pool) SaveMessage(ctx context.Context, conversationID int, role, content string) error {
	_, err := p.Exec(ctx,
		`INSERT INTO messages (conversation_id, role, content) VALUES ($1, $2, $3)`,
		conversationID, role, content,
	)
	return err
}

func (p *Pool) ListSessions(ctx context.Context) ([]Conversation, error) {
	rows, err := p.Query(ctx,
		`SELECT id, model, flags, created_at FROM conversations ORDER BY created_at DESC`,
	)
	if err != nil {
		return nil, fmt.Errorf("list sessions: %w", err)
	}
	defer rows.Close()

	var sessions []Conversation
	for rows.Next() {
		var s Conversation
		var flagsJSON []byte
		if err := rows.Scan(&s.ID, &s.Model, &flagsJSON, &s.CreatedAt); err != nil {
			return nil, err
		}
		if len(flagsJSON) > 0 {
			json.Unmarshal(flagsJSON, &s.Flags)
		}
		if s.Flags == nil {
			s.Flags = make(map[string]interface{})
		}
		sessions = append(sessions, s)
	}
	return sessions, nil
}

func (p *Pool) ExportSession(ctx context.Context, conversationID int) ([]Message, error) {
	rows, err := p.Query(ctx,
		`SELECT id, conversation_id, role, content, created_at
		 FROM messages WHERE conversation_id = $1
		 ORDER BY id`,
		conversationID,
	)
	if err != nil {
		return nil, fmt.Errorf("export session: %w", err)
	}
	defer rows.Close()

	var messages []Message
	for rows.Next() {
		var m Message
		if err := rows.Scan(&m.ID, &m.ConversationID, &m.Role, &m.Content, &m.CreatedAt); err != nil {
			return nil, err
		}
		messages = append(messages, m)
	}
	return messages, nil
}

func (p *Pool) Close() {
	if p.Pool != nil {
		p.Pool.Close()
	}
}
