#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ──────────────────────────────────────────────────────────────────
APP_NAME="ollama-chat"
ENTRY_POINT="stream_chat.py"
INSTALL_PREFIX=
PREFIX_FROM_CLI=0
NVIDIA_MODE=
VERSION="$(date +%Y%m%d)"

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { printf "${CYAN}[INFO]${NC}  %s\n" "$*"; }
ok()    { printf "${GREEN}[OK]${NC}    %s\n" "$*"; }
warn()  { printf "${YELLOW}[WARN]${NC}  %s\n" "$*"; }
err()   { printf "${RED}[ERROR]${NC} %s\n" "$*" >&2; exit 1; }
step()  { printf "${BOLD}==>${NC}    %s\n" "$*"; }

# ── Usage ─────────────────────────────────────────────────────────────────────
usage() {
cat <<EOF
$(basename "$0") — Build and distribute ollama-chat-streamer

Usage: $(basename "$0") [OPTIONS]

Build options:
  --portable         Build a portable directory (no system Python needed)
  --nvidia           Enable CUDA/nvidia plugin support
  --clean            Remove build artifacts and dist/ directory

Install options:
  --install          Install after build (binary + Docker Compose for Postgres)
                      Prompts for install path (default: /usr/local/bin)
  --uninstall        Remove the installed binary, Docker services, and data
  --prefix=PATH      Installation prefix (default: /usr/local)
  --help             Show this help message

Examples:
  $(basename "$0")                           # Build onefile binary, leave in dist/
  $(basename "$0") --install                 # Build + install (prompts for path)
  $(basename "$0") --portable --install       # Portable build + install
  $(basename "$0") --uninstall               # Remove everything
  $(basename "$0") --clean                   # Clean build artifacts
EOF
exit 0
}

# ── Argument parsing ─────────────────────────────────────────────────────────
DO_INSTALL=0
DO_UNINSTALL=0
DO_PORTABLE=0
DO_CLEAN=0

for arg in "$@"; do
  case "$arg" in
    --install)    DO_INSTALL=1 ;;
    --uninstall)  DO_UNINSTALL=1 ;;
    --portable)   DO_PORTABLE=1 ;;
    --nvidia)     NVIDIA_MODE=1 ;;
    --clean)      DO_CLEAN=1 ;;
    --prefix=*)   INSTALL_PREFIX="${arg#--prefix=}"; PREFIX_FROM_CLI=1 ;;
    --help|-h)    usage ;;
    *)            err "Unknown option: $arg" ;;
  esac
done

# ── Interactive install path prompt ───────────────────────────────────────────
ask_install_path() {
  local default_path="/usr/local"
  printf "\n${BOLD}Where should ollama-chat be installed?${NC}\n"
  printf "  The binary will be symlinked to \${PREFIX}/bin/ollama-chat\n"
  printf "  Config and Docker Compose go to \${PREFIX}/lib/ollama-chat/\n\n"
  printf "  [%s]: " "$default_path"
  read -r user_path
  INSTALL_PREFIX="${user_path:-$default_path}"
}

# ── Ask install path upfront (before the build) ──────────────────────────────
# ── Ask install path upfront for --install (before the build) ─────────────────
if [ "$DO_INSTALL" -eq 1 ] && [ "$PREFIX_FROM_CLI" -eq 0 ]; then
  ask_install_path
fi

# ── Uninstall ──────────────────────────────────────────────────────────────────
if [ "$DO_UNINSTALL" -eq 1 ]; then
  if [ "$PREFIX_FROM_CLI" -eq 0 ]; then
    ask_install_path
  fi
  step "Uninstalling $APP_NAME from ${INSTALL_PREFIX}..."

  BIN="${INSTALL_PREFIX}/bin/${APP_NAME}"
  LIB_DIR="${INSTALL_PREFIX}/lib/${APP_NAME}"

  # Remove binary
  if [ -L "$BIN" ] || [ -f "$BIN" ]; then
    sudo rm -f "$BIN"
    ok "Removed $BIN"
  else
    warn "$BIN not found"
  fi

  # Remove lib directory
  if [ -d "$LIB_DIR" ]; then
    sudo rm -rf "$LIB_DIR"
    ok "Removed $LIB_DIR"
  else
    warn "$LIB_DIR not found"
  fi

  # Stop and remove Docker Postgres
  if command -v docker >/dev/null 2>&1; then
    if [ -f "${LIB_DIR}/docker-compose.db.yml" ]; then
      docker compose -f "${LIB_DIR}/docker-compose.db.yml" down --volumes 2>/dev/null && ok "Stopped and removed Docker Postgres service" || true
    else
      # Try the project-local compose file as fallback
      docker compose -f "${SCRIPT_DIR}/docker-compose.yml" down --volumes 2>/dev/null && ok "Stopped and removed Docker Postgres service" || true
    fi
  fi

  printf "\n${GREEN}Uninstall complete.${NC}\n"
  exit 0
fi

# ── Clean ─────────────────────────────────────────────────────────────────────
if [ "$DO_CLEAN" -eq 1 ]; then
  step "Cleaning build artifacts..."
  rm -rf "${SCRIPT_DIR}/dist"
  rm -rf "${SCRIPT_DIR}/${APP_NAME}.build"
  rm -rf "${SCRIPT_DIR}/${APP_NAME}.dist"
  rm -rf "${SCRIPT_DIR}/__pycache__"
  rm -rf "${SCRIPT_DIR}/core/__pycache__"
  ok "Clean complete"
  exit 0
fi

# ── Preflight checks ──────────────────────────────────────────────────────────
step "Running preflight checks..."

command -v uv >/dev/null 2>&1 || err "uv is not installed. Get it from https://docs.astral.sh/uv/"
command -v docker >/dev/null 2>&1 || warn "Docker not found — Postgres service won't be available until Docker is installed"

# Check for patchelf (required by Nuitka on Linux for standalone/onefile)
if [ "$(uname -s)" = "Linux" ]; then
  if ! command -v patchelf >/dev/null 2>&1; then
    step "Installing patchelf (required by Nuitka on Linux)..."
    if command -v apt-get >/dev/null 2>&1; then
      sudo apt-get install -y patchelf
    elif command -v dnf >/dev/null 2>&1; then
      sudo dnf install -y patchelf
    elif command -v yum >/dev/null 2>&1; then
      sudo yum install -y patchelf
    elif command -v pacman >/dev/null 2>&1; then
      sudo pacman -S --noconfirm patchelf
    elif command -v brew >/dev/null 2>&1; then
      brew install patchelf
    else
      err "patchelf is required on Linux. Install it with your package manager, e.g.: sudo apt install patchelf"
    fi
  fi
  ok "patchelf available"
fi

info "Installing Nuitka and build dependencies..."
uv pip install nuitka ordered-set zstandard

PYTHON="$(uv run python -c 'import sys; print(sys.executable)')"
info "Using Python: $PYTHON"

# ── Build the Nuitka command ──────────────────────────────────────────────────
NUITKA_ARGS=(
  --follow-imports
  --follow-import-to=core
  --follow-import-to=ddgs
  --follow-import-to=duckduckgo_search
  --follow-import-to=dotenv
  --follow-import-to=ollama
  --follow-import-to=httpx
  --follow-import-to=psycopg2
  --follow-import-to=asyncpg
  --follow-import-to=questionary
  --follow-import-to=yaml
  --enable-plugin=anti-bloat
  --assume-yes-for-downloads
  --output-dir="${SCRIPT_DIR}/dist"
  --output-filename="${APP_NAME}"
)

if [ "$DO_PORTABLE" -eq 1 ]; then
  step "Portable mode — building standalone directory"
  NUITKA_ARGS+=(--standalone)
else
  step "Building single-file executable"
  NUITKA_ARGS+=(--onefile)
fi

if [ -n "${NVIDIA_MODE}" ]; then
  info "NVIDIA mode — including CUDA plugins"
  NUITKA_ARGS+=(--enable-plugin=nvidia-cuda)
fi

# ── Compile ────────────────────────────────────────────────────────────────────
step "Compiling ${ENTRY_POINT} with Nuitka..."

uv run python -m nuitka "${NUITKA_ARGS[@]}" "${ENTRY_POINT}"

# ── Locate the output ─────────────────────────────────────────────────────────
if [ "$DO_PORTABLE" -eq 1 ]; then
  BUILD_OUT="${SCRIPT_DIR}/dist/${APP_NAME}.dist"
  [ -d "$BUILD_OUT" ] || err "Expected portable directory not found at $BUILD_OUT"
else
  BUILD_OUT="${SCRIPT_DIR}/dist/${APP_NAME}"
  [ -f "$BUILD_OUT" ] || BUILD_OUT="${SCRIPT_DIR}/dist/${APP_NAME}.bin"
  [ -f "$BUILD_OUT" ] || BUILD_OUT=$(find "${SCRIPT_DIR}/dist" -name "${APP_NAME}*" -type f ! -name '*.tar.gz' | head -1)
  [ -f "$BUILD_OUT" ] || err "Compiled binary not found in dist/"
fi

ok "Build succeeded: $BUILD_OUT"

# ── Create distribution package ────────────────────────────────────────────────
step "Assembling distribution package..."

DIST_DIR="${SCRIPT_DIR}/dist/${APP_NAME}-package"
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

# Copy compiled binary
if [ "$DO_PORTABLE" -eq 1 ]; then
  cp -r "$BUILD_OUT" "${DIST_DIR}/"
else
  cp "$BUILD_OUT" "${DIST_DIR}/"
fi

# ── Bundle .env ───────────────────────────────────────────────────────────────
if [ -f "${SCRIPT_DIR}/.env" ]; then
  cp "${SCRIPT_DIR}/.env" "${DIST_DIR}/.env"
  info "Bundled .env"
else
  cat > "${DIST_DIR}/.env" << 'ENVEOF'
# Ollama Chat Streamer Configuration
# Edit this file to match your environment.

EXPERIMENTAL=true
EXPERIMENTAL_WEBSEARCH=true
OLLAMA_MODEL=lfm2.5:latest
OLLAMA_HOST=http://localhost:11434
PERSIST_TO_DB=true
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/chatdb
ENVEOF
  info "Generated default .env"
fi

# ── Bundle Docker Compose for Postgres ─────────────────────────────────────────
cat > "${DIST_DIR}/docker-compose.db.yml" << 'DCOMPOSE'
# Docker Compose for the ollama-chat-streamer database layer.
# Start with:  docker compose -f docker-compose.db.yml up -d
# Stop with:   docker compose -f docker-compose.db.yml down
# Reset data:  docker compose -f docker-compose.db.yml down -v

services:
  postgres:
    image: postgres:15-alpine
    container_name: ollama-chat-db
    restart: unless-stopped
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: chatdb
    ports:
      - "5432:5432"
    volumes:
      - ollama_chat_db_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  ollama_chat_db_data:
DCOMPOSE

info "Bundled docker-compose.db.yml (Postgres service)"

cat > "${DIST_DIR}/docker-compose.yml" << 'DCOMPOSE2'
services:
  postgres:
    image: postgres:15-alpine
    container_name: ollama-chat-db
    restart: unless-stopped
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: chatdb
    ports:
      - "5432:5432"
    volumes:
      - ollama_chat_db_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  ollama_chat_db_data:
DCOMPOSE2

info "Bundled docker-compose.yml (Postgres service)"

# ── Bundle pyproject.toml ─────────────────────────────────────────────────────
[ -f "${SCRIPT_DIR}/pyproject.toml" ] && cp "${SCRIPT_DIR}/pyproject.toml" "${DIST_DIR}/"

# ── Write wrapper script for portable mode ─────────────────────────────────────
if [ "$DO_PORTABLE" -eq 1 ]; then
  cat > "${DIST_DIR}/${APP_NAME}" << 'WRAPPER'
#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/${APP_NAME}.dist/${APP_NAME}" "$@"
WRAPPER
  chmod +x "${DIST_DIR}/${APP_NAME}"
fi

# ── Write install.sh ──────────────────────────────────────────────────────────
cat > "${DIST_DIR}/install.sh" << 'INSTALLER'
#!/usr/bin/env bash
set -euo pipefail

APP_NAME="ollama-chat"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Interactive install path ────────────────────────────────────────────────────
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

step()  { printf "${BOLD}==>${NC}    %s\n" "$*"; }
ok()    { printf "${GREEN}[OK]${NC}    %s\n" "$*"; }

printf "\n${BOLD}Where should ollama-chat be installed?${NC}\n"
printf "  The binary will be symlinked to \${PREFIX}/bin/ollama-chat\n"
printf "  Config and Docker Compose go to \${PREFIX}/lib/ollama-chat/\n\n"
printf "  [/usr/local]: "
read -r user_prefix
PREFIX="${user_prefix:-/usr/local}"

BIN="${PREFIX}/bin"
LIB="${PREFIX}/lib/${APP_NAME}"

step "Installing ${APP_NAME} to ${PREFIX}..."

# Create dirs (may need sudo for /usr/local)
if [ -w "${PREFIX}" ] 2>/dev/null; then
  mkdir -p "$BIN" "$LIB"
else
  sudo mkdir -p "$BIN" "$LIB"
fi

# Copy files
if [ -w "$LIB" ]; then
  cp -r "${SCRIPT_DIR}"/* "$LIB/" 2>/dev/null || true
else
  sudo cp -r "${SCRIPT_DIR}"/* "$LIB/" 2>/dev/null || true
fi

# Symlink binary
if [ -w "$BIN" ]; then
  ln -sf "${LIB}/${APP_NAME}" "${BIN}/${APP_NAME}"
  chmod +x "${BIN}/${APP_NAME}"
else
  sudo ln -sf "${LIB}/${APP_NAME}" "${BIN}/${APP_NAME}"
  sudo chmod +x "${BIN}/${APP_NAME}"
fi
ok "Installed binary: ${BIN}/${APP_NAME}"

ok "Config: ${LIB}/.env"

# ── Start Postgres via Docker ─────────────────────────────────────────────────
step "Setting up database layer..."

if command -v docker >/dev/null 2>&1; then
  if docker compose -f "${LIB}/docker-compose.db.yml" up -d 2>/dev/null; then
    ok "Postgres database started"
    for i in $(seq 1 30); do
      if docker exec ollama-chat-db pg_isready -U postgres >/dev/null 2>&1; then
        ok "Postgres is ready"
        break
      fi
      sleep 1
    done
  else
    printf "${CYAN}[INFO]${NC}  Could not start Docker Postgres automatically.\n"
    printf "${CYAN}[INFO]${NC}  Start it manually:\n"
    printf "         docker compose -f ${LIB}/docker-compose.db.yml up -d\n"
  fi
else
  printf "${CYAN}[INFO]${NC}  Docker not found. Install Docker to use the managed Postgres service.\n"
  printf "${CYAN}[INFO]${NC}  Alternatively, set DATABASE_URL in .env to point to your own Postgres.\n"
fi

# ── Print summary ─────────────────────────────────────────────────────────────
printf "\n${BOLD}Installation complete!${NC}\n\n"
printf "  Binary:       ${BIN}/${APP_NAME}\n"
printf "  Config:       ${LIB}/.env\n"
printf "  Database:     docker compose -f ${LIB}/docker-compose.db.yml up -d\n\n"
printf "  Make sure ${BIN} is in your \$PATH.\n\n"
printf "  Run with:\n"
printf "    ${APP_NAME} --experimental-websearch --persist-to-db\n\n"
printf "  To uninstall:\n"
printf "    ${LIB}/uninstall.sh\n"
INSTALLER
chmod +x "${DIST_DIR}/install.sh"

# ── Write uninstall.sh ─────────────────────────────────────────────────────────
cat > "${DIST_DIR}/uninstall.sh" << 'UNINSTALLER'
#!/usr/bin/env bash
set -euo pipefail

APP_NAME="ollama-chat"
PREFIX="${1:-/usr/local}"
BIN="${PREFIX}/bin/${APP_NAME}"
LIB="${PREFIX}/lib/${APP_NAME}"

step()  { printf "\033[1m==>\033[0m    %s\n" "$*"; }
ok()    { printf "\033[0;32m[OK]\033[0m    %s\n" "$*"; }

step "Uninstalling ${APP_NAME}..."

# Stop Docker Postgres
if [ -f "${LIB}/docker-compose.db.yml" ]; then
  if command -v docker >/dev/null 2>&1; then
    docker compose -f "${LIB}/docker-compose.db.yml" down 2>/dev/null && ok "Stopped Postgres service" || true
  fi
fi

# Remove binary and lib — may need sudo for /usr/local
if [ -w "${PREFIX}" ] 2>/dev/null; then
  rm -f "$BIN" 2>/dev/null && ok "Removed $BIN" || true
  rm -rf "$LIB" 2>/dev/null && ok "Removed $LIB" || true
else
  sudo rm -f "$BIN" 2>/dev/null && ok "Removed $BIN" || true
  sudo rm -rf "$LIB" 2>/dev/null && ok "Removed $LIB" || true
fi

printf "\nUninstall complete.\n"
UNINSTALLER
chmod +x "${DIST_DIR}/uninstall.sh"

# ── Create tarball ─────────────────────────────────────────────────────────────
TARBALL="${SCRIPT_DIR}/dist/${APP_NAME}-${VERSION}-$(uname -m).tar.gz"
tar -czf "$TARBALL" -C "${SCRIPT_DIR}/dist" "${APP_NAME}-package"
ok "Tarball created: $TARBALL"

# ── Install ───────────────────────────────────────────────────────────────────
if [ "$DO_INSTALL" -eq 1 ]; then
  step "Installing to ${INSTALL_PREFIX}..."
  BIN_DIR="${INSTALL_PREFIX}/bin"
  LIB_DIR="${INSTALL_PREFIX}/lib/${APP_NAME}"

  # Create dirs — use sudo if needed (e.g. /usr/local)
  if [ -w "${INSTALL_PREFIX}" ] 2>/dev/null; then
    mkdir -p "$BIN_DIR" "$LIB_DIR"
  else
    sudo mkdir -p "$BIN_DIR" "$LIB_DIR"
  fi

  # Copy files
  if [ -w "$LIB_DIR" ]; then
    cp -r "${DIST_DIR}"/* "$LIB_DIR/" 2>/dev/null || true
    ln -sf "${LIB_DIR}/${APP_NAME}" "${BIN_DIR}/${APP_NAME}"
    chmod +x "${BIN_DIR}/${APP_NAME}"
  else
    sudo cp -r "${DIST_DIR}"/* "$LIB_DIR/" 2>/dev/null || true
    sudo ln -sf "${LIB_DIR}/${APP_NAME}" "${BIN_DIR}/${APP_NAME}"
    sudo chmod +x "${BIN_DIR}/${APP_NAME}"
  fi

  ok "Installed: ${BIN_DIR}/${APP_NAME}"

  # Start Postgres
  if command -v docker >/dev/null 2>&1; then
    step "Starting Postgres database..."
    if docker compose -f "${LIB_DIR}/docker-compose.db.yml" up -d 2>/dev/null; then
      ok "Postgres database started"
      for i in $(seq 1 30); do
        if docker exec ollama-chat-db pg_isready -U postgres >/dev/null 2>&1; then
          ok "Postgres is ready"
          break
        fi
        sleep 1
      done
    else
      warn "Could not start Postgres automatically — start it with:"
      warn "  docker compose -f ${LIB_DIR}/docker-compose.db.yml up -d"
    fi
  else
    warn "Docker not found — install Docker to use the managed Postgres service"
    warn "Or set DATABASE_URL in ${LIB_DIR}/.env to point to your own Postgres"
  fi

  printf "\n${BOLD}${GREEN}Installation complete!${NC}\n\n"
  printf "  Binary:       ${BIN_DIR}/${APP_NAME}\n"
  printf "  Config:       ${LIB_DIR}/.env\n"
  printf "  Database:     docker compose -f ${LIB_DIR}/docker-compose.db.yml up -d\n\n"
  printf "  Make sure ${BIN_DIR} is in your \$PATH.\n\n"
  printf "  Run with:\n"
  printf "    ${APP_NAME} --experimental-websearch --persist-to-db\n"
fi

printf "\nDone!\n"