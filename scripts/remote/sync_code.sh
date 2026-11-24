#!/usr/bin/env bash
set -euo pipefail

# Safely sync tracked code, configs, docs, tests, scripts, and reports from local to rl-server.
# Never overwrites remote data/, runs/, .venv/, or .git/.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-rl-server}"
REMOTE_DIR="${REMOTE_DIR:-/root/ToolCall-SFT}"

usage() {
  cat <<'EOF'
Usage: sync_code.sh [--dry-run | -n] [--help | -h]

Safely sync controlled project files/directories to rl-server without touching
remote data/, runs/, .venv/, or .git/. Any unknown option fails immediately.
EOF
}

DRY_RUN=""
for arg in "$@"; do
  case "$arg" in
    --dry-run|-n)
      DRY_RUN="--dry-run"
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument '$arg'" >&2
      usage >&2
      exit 2
      ;;
  esac
done

RSYNC_OPTS=(
  -avz
  --delete
  --exclude="__pycache__/"
  --exclude="*.py[cod]"
  --exclude="*.egg-info/"
  --exclude=".pytest_cache/"
  --exclude=".ruff_cache/"
)

if [[ -n "$DRY_RUN" ]]; then
  RSYNC_OPTS+=("$DRY_RUN")
fi

echo "==> Syncing code & configs to ${REMOTE_HOST}:${REMOTE_DIR} ${DRY_RUN}"

# 1. Sync root metadata files (without --delete at root)
ROOT_RSYNC_OPTS=(-avz)
if [[ -n "$DRY_RUN" ]]; then
  ROOT_RSYNC_OPTS+=("$DRY_RUN")
fi

rsync "${ROOT_RSYNC_OPTS[@]}" \
  "${ROOT_DIR}/AGENTS.md" \
  "${ROOT_DIR}/README.md" \
  "${ROOT_DIR}/pyproject.toml" \
  "${ROOT_DIR}/.gitignore" \
  "${REMOTE_HOST}:${REMOTE_DIR}/"

# 2. Sync controlled project directories (safe --delete within each subdir)
for dir in configs docs reports scripts src tests; do
  if [[ -d "${ROOT_DIR}/${dir}" ]]; then
    rsync "${RSYNC_OPTS[@]}" "${ROOT_DIR}/${dir}/" "${REMOTE_HOST}:${REMOTE_DIR}/${dir}/"
  fi
done

echo "==> Done. Active tmux snapshots under runs/snapshots/ remain isolated from this sync."
echo "    Remember: if pyproject.toml changed, update editable install on ${REMOTE_HOST}."
