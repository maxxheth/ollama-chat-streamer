#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy.sh — Build and install ollama-chat for production use
#
# Usage:
#   ./deploy.sh                    # Build + install to default prefix
#   ./deploy.sh --prefix /opt/bin  # Custom install directory
#   ./deploy.sh --help             # Show help
#
# Config precedence (same as the Go binary):
#   CLI flag > env var > YAML config > default
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

APP_NAME="ollama-chat"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${CYAN}==>${NC} $*"; }
ok()    { echo -e "${GREEN}  ✓${NC} $*"; }
warn()  { echo -e "${YELLOW}  ⚠${NC} $*"; }
err()   { echo -e "${RED}  ✗${NC} $*"; }

# ── Help ───────────────────────────────────────────────────────────────────
usage() {
  cat <<EOF
${BOLD}deploy.sh${NC} — Build and install ollama-chat

${BOLD}Usage:${NC}
  $(basename "$0") [options]

${BOLD}Options:${NC}
  --prefix <dir>      Install binary to <dir> (default: \$HOME/.local/bin)
  --config <path>     Path to YAML config file (default: auto-detect)
  --no-config         Skip generating default config file
  --compose <path>    Docker Compose file for DB (default: go_version/docker-compose.yml)
  --no-compose        Skip docker-compose setup entirely
  --no-persist        Skip starting the database service
  --help              Show this help

${BOLD}Config precedence:${NC}
  CLI flag > env var > YAML config > built-in default

${BOLD}Environment variables honored:${NC}
  OLLAMA_MODEL, OLLAMA_HOST, EXPERIMENTAL_WEBSEARCH, PERSIST_TO_DB,
  THINK, MAX_SUBAGENT_DEPTH, MAX_SUBAGENT_ROUNDS, TURN_LIMIT,
  DATABASE_URL, CONTEXT_PATH, GO_BUILD_FLAGS, INSTALL_PREFIX
EOF
  exit 0
}

# ── Parse CLI flags ───────────────────────────────────────────────────────
CLI_PREFIX=""
CLI_CONFIG=""
CLI_COMPOSE=""
SKIP_CONFIG=false
SKIP_COMPOSE=false
START_DB=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)     CLI_PREFIX="$2"; shift 2 ;;
    --config)     CLI_CONFIG="$2"; shift 2 ;;
    --no-config)  SKIP_CONFIG=true; shift ;;
    --compose)    CLI_COMPOSE="$2"; shift 2 ;;
    --no-compose) SKIP_COMPOSE=true; shift ;;
    --no-persist) START_DB=false; shift ;;
    --help|-h)    usage ;;
    *)            err "Unknown option: $1"; usage ;;
  esac
done

# ── Load .env from project root ───────────────────────────────────────────
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  source "$PROJECT_ROOT/.env"
  set +a
  info "Loaded .env from $PROJECT_ROOT/.env"
else
  warn "No .env found at $PROJECT_ROOT/.env — using defaults"
fi

# ── Detect YAML config ─────────────────────────────────────────────────────
# Lookup order matches main.go: --config flag > OLLAMA_CONFIG_PATH > ./ > ~/
YAML_CONFIG=""
if [[ -n "$CLI_CONFIG" ]]; then
  YAML_CONFIG="$CLI_CONFIG"
elif [[ -n "${OLLAMA_CONFIG_PATH:-}" ]]; then
  YAML_CONFIG="$OLLAMA_CONFIG_PATH"
elif [[ -f "$PROJECT_ROOT/ollama-chat.yaml" ]]; then
  YAML_CONFIG="$PROJECT_ROOT/ollama-chat.yaml"
elif [[ -f "$HOME/.config/ollama-chat/ollama-chat.yaml" ]]; then
  YAML_CONFIG="$HOME/.config/ollama-chat/ollama-chat.yaml"
fi

if [[ -n "$YAML_CONFIG" ]]; then
  info "Using config: $YAML_CONFIG"
fi

# ── Helper: read a value from YAML ────────────────────────────────────────
# Uses yq if available, otherwise falls back to grep+sed.
read_yaml() {
  local key="$1" default="${2:-}"
  local val=""

  if [[ -z "$YAML_CONFIG" || ! -f "$YAML_CONFIG" ]]; then
    echo "$default"
    return
  fi

  if command -v yq &>/dev/null; then
    val="$(yq ".$key" "$YAML_CONFIG" 2>/dev/null || true)"
    if [[ "$val" = "null" || -z "$val" ]]; then
      echo "$default"
      return
    fi
    # Strip surrounding quotes if yq returns them
    val="${val#\"}"
    val="${val%\"}"
    echo "$val"
  else
    # Simple grep-based fallback for flat YAML (no nesting)
    val="$(grep -E "^${key}:" "$YAML_CONFIG" 2>/dev/null | head -1 | sed -E 's/^[^:]+:[[:space:]]*//' | sed -E 's/^"//;s/"$//')"
    echo "${val:-$default}"
  fi
}

# ── Determine install prefix ───────────────────────────────────────────────
# Precedence: --prefix CLI > INSTALL_PREFIX env > YAML > default
if [[ -n "$CLI_PREFIX" ]]; then
  INSTALL_PREFIX="$CLI_PREFIX"
elif [[ -z "${INSTALL_PREFIX:-}" ]]; then
  INSTALL_PREFIX="$(read_yaml "install_prefix" "")"
fi
if [[ -z "${INSTALL_PREFIX:-}" ]]; then
  INSTALL_PREFIX="$HOME/.local/bin"
fi

# ── Prerequisites ─────────────────────────────────────────────────────────
if ! command -v go &>/dev/null; then
  err "Go is not installed. Install Go 1.21+ first."
  exit 1
fi
GO_VERSION="$(go version | grep -oP 'go\K[0-9]+\.[0-9]+')"
if awk "BEGIN {exit !($GO_VERSION < 1.21)}"; then
  err "Go 1.21+ required (found $GO_VERSION)"
  exit 1
fi

# ── Build ─────────────────────────────────────────────────────────────────
info "Building ${APP_NAME} (release mode)…"

BUILD_FLAGS=()
if [[ -n "${GO_BUILD_FLAGS:-}" ]]; then
  # shellcheck disable=SC2206
  BUILD_FLAGS=($GO_BUILD_FLAGS)
else
  BUILD_FLAGS=(-ldflags="-s -w" -trimpath)
fi

pushd "$SCRIPT_DIR" >/dev/null
go build "${BUILD_FLAGS[@]}" -o "$APP_NAME" .
popd >/dev/null

BINARY="$SCRIPT_DIR/$APP_NAME"
if [[ ! -f "$BINARY" ]]; then
  err "Build failed — binary not found at $BINARY"
  exit 1
fi

BUILD_SIZE="$(stat -c%s "$BINARY" 2>/dev/null || stat -f%z "$BINARY" 2>/dev/null)"
ok "Build complete ($(numfmt --to=iec "$BUILD_SIZE" 2>/dev/null || echo "${BUILD_SIZE}B"))"

# ── Install ───────────────────────────────────────────────────────────────
info "Installing to $INSTALL_PREFIX…"

mkdir -p "$INSTALL_PREFIX"
install -m 755 "$BINARY" "$INSTALL_PREFIX/$APP_NAME"
rm -f "$BINARY"

INSTALLED="$INSTALL_PREFIX/$APP_NAME"
if [[ ! -f "$INSTALLED" ]]; then
  err "Install failed — binary not found at $INSTALLED"
  exit 1
fi

ok "Installed: $INSTALLED"

# ── Check PATH ────────────────────────────────────────────────────────────
if ! command -v "$APP_NAME" &>/dev/null; then
  warn "$INSTALL_PREFIX is not in your PATH."
  warn "Add it with:  export PATH=\"\$PATH:$INSTALL_PREFIX\""
fi

# ── Seed default config ────────────────────────────────────────────────────
if [[ "$SKIP_CONFIG" != "true" ]]; then
  CONFIG_DIR="$HOME/.config/ollama-chat"
  CONFIG_FILE="$CONFIG_DIR/ollama-chat.yaml"

  if [[ -f "$CONFIG_FILE" ]]; then
    ok "Config already exists at $CONFIG_FILE (not overwritten)"
  else
    info "Generating default config at $CONFIG_FILE…"

    # Read values with precedence: env > YAML > default
    CFG_MODEL="${OLLAMA_MODEL:-$(read_yaml "model" "qwen3.6:35b")}"
    CFG_HOST="${OLLAMA_HOST:-$(read_yaml "ollama_host" "http://localhost:11434")}"
    CFG_WEBSEARCH="${EXPERIMENTAL_WEBSEARCH:-$(read_yaml "experimental_websearch" "true")}"
    CFG_PERSIST="${PERSIST_TO_DB:-$(read_yaml "persist_to_db" "true")}"
    CFG_THINK="${THINK:-$(read_yaml "think" "auto")}"
    CFG_DEPTH="${MAX_SUBAGENT_DEPTH:-$(read_yaml "max_subagent_depth" "1")}"
    CFG_ROUNDS="${MAX_SUBAGENT_ROUNDS:-$(read_yaml "max_subagent_rounds" "10")}"
    CFG_TURNS="${TURN_LIMIT:-$(read_yaml "turn_limit" "100")}"
    CFG_AUTO_COMPACT="${AUTO_COMPACT:-$(read_yaml "auto_compact" "true")}"
    CFG_AUTO_COMPACT_LIMIT="${AUTO_COMPACT_LIMIT:-$(read_yaml "auto_compact_limit" "75")}"
    CFG_AUTO_COMPACT_TARGET="${AUTO_COMPACT_TARGET:-$(read_yaml "auto_compact_target" "50")}"
    CFG_AUTO_COMPACT_KEEP_RECENT="${AUTO_COMPACT_KEEP_RECENT:-$(read_yaml "auto_compact_keep_recent" "8")}"
    CFG_TOOL_RESULT_MAX_INLINE="${TOOL_RESULT_MAX_INLINE:-$(read_yaml "tool_result_max_inline" "12000")}"
    CFG_NUM_CTX="${NUM_CTX:-$(read_yaml "num_ctx" "")}"
    CFG_DB_URL="${DATABASE_URL:-$(read_yaml "database_url" "")}"
    CFG_CONTEXT="${CONTEXT_PATH:-$(read_yaml "context_path" "")}"

    mkdir -p "$CONFIG_DIR"
    cat > "$CONFIG_FILE" <<YAMLEOF
# ollama-chat configuration
# Created by deploy.sh on $(date +%Y-%m-%d)
# See 'ollama-chat --help' for all available options.

model: ${CFG_MODEL}
ollama_host: ${CFG_HOST}
experimental_websearch: ${CFG_WEBSEARCH}
persist_to_db: ${CFG_PERSIST}
think: ${CFG_THINK}
max_subagent_depth: ${CFG_DEPTH}
max_subagent_rounds: ${CFG_ROUNDS}
turn_limit: ${CFG_TURNS}
auto_compact: ${CFG_AUTO_COMPACT}
auto_compact_limit: ${CFG_AUTO_COMPACT_LIMIT}
auto_compact_target: ${CFG_AUTO_COMPACT_TARGET}
auto_compact_keep_recent: ${CFG_AUTO_COMPACT_KEEP_RECENT}
tool_result_max_inline: ${CFG_TOOL_RESULT_MAX_INLINE}
install_prefix: ${INSTALL_PREFIX}
compiled: true
YAMLEOF

    if [[ -n "$CFG_NUM_CTX" ]]; then
      echo "num_ctx: ${CFG_NUM_CTX}" >> "$CONFIG_FILE"
    fi
    if [[ -n "$CFG_DB_URL" ]]; then
      echo "database_url: ${CFG_DB_URL}" >> "$CONFIG_FILE"
    fi
    if [[ -n "$CFG_CONTEXT" ]]; then
      echo "context_path: ${CFG_CONTEXT}" >> "$CONFIG_FILE"
    fi

    ok "Config written to $CONFIG_FILE"
  fi
fi

# ── Docker Compose (database persistence) ────────────────────────────────
COMPOSE_FILE=""
if [[ "$SKIP_COMPOSE" != "true" ]]; then
  if [[ -n "$CLI_COMPOSE" ]]; then
    # Custom path — use as-is, no copying
    COMPOSE_FILE="$CLI_COMPOSE"
    if [[ ! -f "$COMPOSE_FILE" ]]; then
      warn "Compose file not found: $COMPOSE_FILE"
      warn "Database persistence will not be available."
      COMPOSE_FILE=""
    fi
  else
    # Default: copy built-in compose file to config directory
    SOURCE_COMPOSE="$SCRIPT_DIR/docker-compose.yml"
    if [[ ! -f "$SOURCE_COMPOSE" ]]; then
      warn "Built-in compose file not found: $SOURCE_COMPOSE"
      warn "Database persistence will not be available."
    else
      COMPOSE_DIR="$HOME/.config/ollama-chat"
      mkdir -p "$COMPOSE_DIR"
      cp "$SOURCE_COMPOSE" "$COMPOSE_DIR/docker-compose.yml"
      COMPOSE_FILE="$COMPOSE_DIR/docker-compose.yml"
      ok "Compose file: $COMPOSE_FILE"
    fi
  fi

  if [[ -n "$COMPOSE_FILE" && "$START_DB" == "true" ]]; then
    if ! command -v docker &>/dev/null; then
      err "Docker is not installed. Cannot start database."
    else
      info "Starting database service…"
      if docker compose version &>/dev/null; then
        DOCKER_COMPOSE="docker compose"
      elif command -v docker-compose &>/dev/null; then
        DOCKER_COMPOSE="docker-compose"
      else
        err "Neither 'docker compose' nor 'docker-compose' found."
        DOCKER_COMPOSE=""
      fi
      if [[ -n "$DOCKER_COMPOSE" ]]; then
        if $DOCKER_COMPOSE -f "$COMPOSE_FILE" up -d postgres; then
          ok "Database service started"
        else
          warn "Database service failed to start (Docker daemon not running? Port in use?)"
          warn "Start it later with: $DOCKER_COMPOSE -f $COMPOSE_FILE up -d postgres"
        fi
      fi
    fi
  fi
fi

# ── Summary ────────────────────────────────────────────────────────────────
echo
echo -e "${GREEN}${BOLD}Deployment complete!${NC}"
echo -e "  ${BOLD}Binary:${NC}  $INSTALLED"
if [[ -n "${CONFIG_FILE:-}" ]]; then
  echo -e "  ${BOLD}Config:${NC}  $CONFIG_FILE"
fi
if [[ -n "${COMPOSE_FILE:-}" ]]; then
  echo -e "  ${BOLD}Compose:${NC} $COMPOSE_FILE"
fi
echo
echo -e "  Run ${CYAN}ollama-chat${NC} to start chatting."
echo -e "  Run ${CYAN}ollama-chat --help${NC} for all options."
echo
if [[ -n "${COMPOSE_FILE:-}" && "$START_DB" != "true" ]]; then
  echo -e "  ${YELLOW}Note:${NC} Start Postgres for DB persistence:"
  echo -e "       ${CYAN}docker compose -f $COMPOSE_FILE up -d postgres${NC}"
fi
echo -e "  ${YELLOW}Note:${NC} Make sure Ollama is running at ${CFG_HOST:-http://localhost:11434}"
echo
