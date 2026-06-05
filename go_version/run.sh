#!/usr/bin/env bash
set -euo pipefail

APP_NAME="ollama-chat"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Colors ──────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}==>${NC} $*"; }
ok()    { echo -e "${GREEN}  ✓${NC} $*"; }
warn()  { echo -e "${YELLOW}  ⚠${NC} $*"; }
err()   { echo -e "${RED}  ✗${NC} $*"; }

# ── Load .env ───────────────────────────────────────────────────────
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  source "$PROJECT_ROOT/.env"
  set +a
  info "Loaded .env from $PROJECT_ROOT/.env"
else
  warn "No .env found at $PROJECT_ROOT/.env — using defaults"
fi

# ── Defaults ────────────────────────────────────────────────────────
: "${POSTGRES_USER:=postgres}"
: "${POSTGRES_PASSWORD:=postgres}"
: "${POSTGRES_DB:=chatdb}"
: "${POSTGRES_PORT:=5434}"
: "${GO_BUILD_FLAGS:=}"

: "${OLLAMA_MODEL:=llama3.2:latest}"
: "${OLLAMA_HOST:=http://localhost:11434}"
: "${EXPERIMENTAL_WEBSEARCH:=true}"
: "${PERSIST_TO_DB:=true}"
: "${THINK:=auto}"
: "${MAX_SUBAGENT_DEPTH:=1}"
: "${MAX_SUBAGENT_ROUNDS:=10}"
: "${READ_FILE_MAX_BYTES:=65536}"
: "${READ_FILE_MAX_SIZE:=524288}"

# ── Find free port ──────────────────────────────────────────────────
find_free_port() {
    local port=$1 max=${2:-50}
    for ((i=0; i<max; i++)); do
        if command -v ss &>/dev/null; then
            if ss -tln 2>/dev/null | grep -qE "[:.]${port}\b"; then
                port=$((port + 1))
                continue
            fi
        elif (: < "/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
            port=$((port + 1))
            continue
        fi
        echo "$port"
        return 0
    done
    echo "$1"
}
POSTGRES_PORT=$(find_free_port "$POSTGRES_PORT")

# Export for docker-compose to see
export POSTGRES_USER POSTGRES_PASSWORD POSTGRES_DB POSTGRES_PORT

DATABASE_URL="postgres://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:${POSTGRES_PORT}/${POSTGRES_DB}"

# ── Prerequisites ───────────────────────────────────────────────────
if ! command -v go &>/dev/null; then
  err "Go is not installed. Install Go 1.21+ first."
  exit 1
fi
GO_VERSION="$(go version | grep -oP 'go\K[0-9]+\.[0-9]+')"
if awk "BEGIN {exit !($GO_VERSION < 1.21)}"; then
  err "Go 1.21+ required (found $GO_VERSION)"
  exit 1
fi

if ! command -v docker &>/dev/null; then
  err "Docker is not installed."
  exit 1
fi

# ── Start Postgres ──────────────────────────────────────────────────
info "Starting Postgres ($POSTGRES_USER@localhost:$POSTGRES_PORT/$POSTGRES_DB)…"
docker compose -f "$SCRIPT_DIR/docker-compose.yml" up -d postgres 2>/dev/null || \
  docker-compose -f "$SCRIPT_DIR/docker-compose.yml" up -d postgres

# ── Wait for Postgres ───────────────────────────────────────────────
MAX_TRIES=15
for ((i=1; i<=MAX_TRIES; i++)); do
  if docker exec ollama-chat-db-go pg_isready -U "$POSTGRES_USER" &>/dev/null; then
    ok "Postgres is ready"
    break
  fi
  if [[ $i -eq $MAX_TRIES ]]; then
    err "Postgres did not become ready in time"
    docker compose -f "$SCRIPT_DIR/docker-compose.yml" down 2>/dev/null || true
    exit 1
  fi
  sleep 2
done

# ── Cleanup on exit ─────────────────────────────────────────────────
cleanup() {
  info "Shutting down…"
  docker compose -f "$SCRIPT_DIR/docker-compose.yml" down 2>/dev/null || true
}
trap cleanup EXIT SIGINT SIGTERM

# ── Build ───────────────────────────────────────────────────────────
info "Building ${APP_NAME}…"
pushd "$SCRIPT_DIR" >/dev/null
go build $GO_BUILD_FLAGS -o "$APP_NAME" .
popd >/dev/null
ok "Build complete"

# ── Run ─────────────────────────────────────────────────────────────
info "Starting ${APP_NAME} (model: ${OLLAMA_MODEL}, db: ${PERSIST_TO_DB})…"
echo

export OLLAMA_MODEL
export OLLAMA_HOST
export EXPERIMENTAL_WEBSEARCH
export PERSIST_TO_DB
export DATABASE_URL
export THINK
export MAX_SUBAGENT_DEPTH
export MAX_SUBAGENT_ROUNDS
export READ_FILE_MAX_BYTES
export READ_FILE_MAX_SIZE

exec "$SCRIPT_DIR/$APP_NAME"
