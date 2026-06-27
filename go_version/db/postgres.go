package db

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type Pool struct {
	*pgxpool.Pool
}

type Conversation struct {
	ID        int                    `json:"id"`
	Model     string                 `json:"model"`
	Flags     map[string]interface{} `json:"flags"`
	CreatedAt time.Time              `json:"created_at"`
	UpdatedAt time.Time              `json:"updated_at"`
}

type Message struct {
	ID             int       `json:"id"`
	ConversationID int       `json:"conversation_id"`
	Role           string    `json:"role"`
	Content        string    `json:"content"`
	CreatedAt      time.Time `json:"created_at"`
}

type ConversationSnapshot struct {
	ID               int       `json:"id"`
	ConversationID   int       `json:"conversation_id"`
	MessageIDThrough *int      `json:"message_id_through,omitempty"`
	Kind             string    `json:"kind"`
	Content          string    `json:"content"`
	TokenEstimate    int       `json:"token_estimate"`
	CreatedAt        time.Time `json:"created_at"`
}

type Artifact struct {
	ID             int       `json:"id"`
	ConversationID *int      `json:"conversation_id,omitempty"`
	Kind           string    `json:"kind"`
	Name           string    `json:"name"`
	Summary        string    `json:"summary"`
	Content        string    `json:"content,omitempty"`
	ContentHash    string    `json:"content_hash"`
	SizeBytes      int       `json:"size_bytes"`
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

	_, err = p.Exec(ctx, `ALTER TABLE conversations ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();`)
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
	if err != nil {
		return err
	}

	_, err = p.Exec(ctx, `
		CREATE TABLE IF NOT EXISTS conversation_snapshots (
			id SERIAL PRIMARY KEY,
			conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
			message_id_through INTEGER,
			kind TEXT NOT NULL,
			content TEXT NOT NULL,
			token_estimate INTEGER DEFAULT 0,
			created_at TIMESTAMPTZ DEFAULT NOW()
		);
		CREATE INDEX IF NOT EXISTS idx_snapshots_conversation_kind_created
			ON conversation_snapshots (conversation_id, kind, created_at DESC);
	`)
	if err != nil {
		return err
	}

	_, err = p.Exec(ctx, `
		CREATE TABLE IF NOT EXISTS artifacts (
			id SERIAL PRIMARY KEY,
			conversation_id INTEGER REFERENCES conversations(id) ON DELETE SET NULL,
			kind TEXT NOT NULL,
			name TEXT NOT NULL,
			summary TEXT NOT NULL,
			content TEXT NOT NULL,
			content_hash TEXT NOT NULL,
			size_bytes INTEGER NOT NULL,
			created_at TIMESTAMPTZ DEFAULT NOW()
		);
		CREATE INDEX IF NOT EXISTS idx_artifacts_conversation_created
			ON artifacts (conversation_id, created_at DESC);
	`)
	return err
}

func (p *Pool) SaveConversation(ctx context.Context, model string, conversationID *int) (*int, error) {
	if conversationID != nil {
		return conversationID, nil
	}

	var id int
	err := p.QueryRow(ctx,
		`INSERT INTO conversations (model, updated_at) VALUES ($1, NOW()) RETURNING id`,
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
		`UPDATE conversations SET flags = $1, updated_at = NOW() WHERE id = $2`,
		flagsJSON, conversationID,
	)
	return err
}

func (p *Pool) SaveMessage(ctx context.Context, conversationID int, role, content string) error {
	_, err := p.SaveMessageID(ctx, conversationID, role, content)
	return err
}

func (p *Pool) SaveMessageID(ctx context.Context, conversationID int, role, content string) (*int, error) {
	tx, err := p.Begin(ctx)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback(ctx)

	var id int
	err = tx.QueryRow(ctx,
		`INSERT INTO messages (conversation_id, role, content) VALUES ($1, $2, $3) RETURNING id`,
		conversationID, role, content,
	).Scan(&id)
	if err != nil {
		return nil, err
	}

	if _, err := tx.Exec(ctx, `UPDATE conversations SET updated_at = NOW() WHERE id = $1`, conversationID); err != nil {
		return nil, err
	}

	if err := tx.Commit(ctx); err != nil {
		return nil, err
	}
	return &id, nil
}

func (p *Pool) ListSessions(ctx context.Context) ([]Conversation, error) {
	rows, err := p.Query(ctx,
		`SELECT id, model, flags, created_at, COALESCE(updated_at, created_at)
		 FROM conversations ORDER BY COALESCE(updated_at, created_at) DESC`,
	)
	if err != nil {
		return nil, fmt.Errorf("list sessions: %w", err)
	}
	defer rows.Close()

	var sessions []Conversation
	for rows.Next() {
		var s Conversation
		var flagsJSON []byte
		if err := rows.Scan(&s.ID, &s.Model, &flagsJSON, &s.CreatedAt, &s.UpdatedAt); err != nil {
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
	return sessions, rows.Err()
}

func (p *Pool) ExportSession(ctx context.Context, conversationID int) ([]Message, error) {
	return p.ExportSessionAfterMessageID(ctx, conversationID, 0)
}

func (p *Pool) ExportSessionAfterMessageID(ctx context.Context, conversationID int, afterID int) ([]Message, error) {
	rows, err := p.Query(ctx,
		`SELECT id, conversation_id, role, content, created_at
		 FROM messages WHERE conversation_id = $1 AND id > $2
		 ORDER BY id`,
		conversationID, afterID,
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
	return messages, rows.Err()
}

func (p *Pool) SaveSnapshot(ctx context.Context, conversationID int, messageIDThrough *int, kind, content string, tokenEstimate int) (*int, error) {
	var id int
	err := p.QueryRow(ctx,
		`INSERT INTO conversation_snapshots (conversation_id, message_id_through, kind, content, token_estimate)
		 VALUES ($1, $2, $3, $4, $5) RETURNING id`,
		conversationID, nullableInt(messageIDThrough), kind, content, tokenEstimate,
	).Scan(&id)
	if err != nil {
		return nil, fmt.Errorf("insert snapshot: %w", err)
	}
	_, _ = p.Exec(ctx, `UPDATE conversations SET updated_at = NOW() WHERE id = $1`, conversationID)
	return &id, nil
}

func (p *Pool) LatestSnapshot(ctx context.Context, conversationID int, kind string) (*ConversationSnapshot, error) {
	query := `SELECT id, conversation_id, message_id_through, kind, content, token_estimate, created_at
		FROM conversation_snapshots WHERE conversation_id = $1`
	args := []interface{}{conversationID}
	if kind != "" {
		query += ` AND kind = $2`
		args = append(args, kind)
	}
	query += ` ORDER BY created_at DESC, id DESC LIMIT 1`

	var s ConversationSnapshot
	var messageID sql.NullInt64
	err := p.QueryRow(ctx, query, args...).Scan(&s.ID, &s.ConversationID, &messageID, &s.Kind, &s.Content, &s.TokenEstimate, &s.CreatedAt)
	if err == pgx.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("latest snapshot: %w", err)
	}
	if messageID.Valid {
		v := int(messageID.Int64)
		s.MessageIDThrough = &v
	}
	return &s, nil
}

func (p *Pool) SaveArtifact(ctx context.Context, conversationID *int, kind, name, summary, content string) (*Artifact, error) {
	h := sha256.Sum256([]byte(content))
	artifact := &Artifact{
		ConversationID: conversationID,
		Kind:           kind,
		Name:           name,
		Summary:        summary,
		Content:        content,
		ContentHash:    hex.EncodeToString(h[:]),
		SizeBytes:      len([]byte(content)),
	}

	err := p.QueryRow(ctx,
		`INSERT INTO artifacts (conversation_id, kind, name, summary, content, content_hash, size_bytes)
		 VALUES ($1, $2, $3, $4, $5, $6, $7)
		 RETURNING id, created_at`,
		nullableInt(conversationID), kind, name, summary, content, artifact.ContentHash, artifact.SizeBytes,
	).Scan(&artifact.ID, &artifact.CreatedAt)
	if err != nil {
		return nil, fmt.Errorf("insert artifact: %w", err)
	}
	if conversationID != nil {
		_, _ = p.Exec(ctx, `UPDATE conversations SET updated_at = NOW() WHERE id = $1`, *conversationID)
	}
	return artifact, nil
}

func (p *Pool) ListArtifacts(ctx context.Context, conversationID *int, limit int) ([]Artifact, error) {
	if limit <= 0 {
		limit = 20
	}
	query := `SELECT id, conversation_id, kind, name, summary, content_hash, size_bytes, created_at FROM artifacts`
	args := []interface{}{}
	if conversationID != nil {
		query += ` WHERE conversation_id = $1`
		args = append(args, *conversationID)
		query += ` ORDER BY created_at DESC, id DESC LIMIT $2`
		args = append(args, limit)
	} else {
		query += ` ORDER BY created_at DESC, id DESC LIMIT $1`
		args = append(args, limit)
	}

	rows, err := p.Query(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("list artifacts: %w", err)
	}
	defer rows.Close()

	var artifacts []Artifact
	for rows.Next() {
		var a Artifact
		var conversationID sql.NullInt64
		if err := rows.Scan(&a.ID, &conversationID, &a.Kind, &a.Name, &a.Summary, &a.ContentHash, &a.SizeBytes, &a.CreatedAt); err != nil {
			return nil, err
		}
		if conversationID.Valid {
			v := int(conversationID.Int64)
			a.ConversationID = &v
		}
		artifacts = append(artifacts, a)
	}
	return artifacts, rows.Err()
}

func (p *Pool) GetArtifact(ctx context.Context, id int) (*Artifact, error) {
	var a Artifact
	var conversationID sql.NullInt64
	err := p.QueryRow(ctx,
		`SELECT id, conversation_id, kind, name, summary, content, content_hash, size_bytes, created_at
		 FROM artifacts WHERE id = $1`,
		id,
	).Scan(&a.ID, &conversationID, &a.Kind, &a.Name, &a.Summary, &a.Content, &a.ContentHash, &a.SizeBytes, &a.CreatedAt)
	if err == pgx.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("get artifact: %w", err)
	}
	if conversationID.Valid {
		v := int(conversationID.Int64)
		a.ConversationID = &v
	}
	return &a, nil
}

func nullableInt(v *int) interface{} {
	if v == nil {
		return nil
	}
	return *v
}

func (p *Pool) Close() {
	if p.Pool != nil {
		p.Pool.Close()
	}
}
