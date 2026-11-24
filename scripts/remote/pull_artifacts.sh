#!/usr/bin/env bash
set -euo pipefail

# Pull lightweight experiment artifacts (metrics, summaries, logs, charts) from rl-server to local.
# Excludes model weights, checkpoints, virtualenvs, and upstream caches.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-rl-server}"
REMOTE_DIR="${REMOTE_DIR:-/root/ToolCall-SFT}"

usage() {
  cat <<'EOF'
Usage: pull_artifacts.sh [all|eval|train|services|install-logs|tmux-logs] [--dry-run | -n] [--help | -h]
EOF
}

DRY_RUN=""
SCOPE=""
for arg in "$@"; do
  case "$arg" in
    --dry-run|-n)
      DRY_RUN="--dry-run"
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    eval|train|services|install-logs|tmux-logs|all)
      if [[ -n "$SCOPE" ]]; then
        echo "Error: multiple scopes specified ('$SCOPE' and '$arg')" >&2
        usage >&2
        exit 2
      fi
      SCOPE="$arg"
      ;;
    *)
      echo "Error: unknown argument '$arg'" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCOPE="${SCOPE:-all}"

RSYNC_BASE=(
  -avz
  --prune-empty-dirs
  --exclude="__pycache__/"
  --exclude="*.safetensors"
  --exclude="*.bin"
  --exclude="*.pt"
  --exclude="*.pth"
  --exclude="*.gguf"
  --exclude="*.ckpt"
  --exclude="*.onnx"
  --exclude="checkpoint-*/"
  --exclude="merged-*/"
  --exclude="final_adapter/"
  --exclude="tokenizer.json"
  --exclude="vocab.json"
  --exclude="merges.txt"
  --exclude=".file_locks/"
)

if [[ -n "$DRY_RUN" ]]; then
  RSYNC_BASE+=("$DRY_RUN")
fi

pull_dir() {
  local subdir="$1"
  mkdir -p "${ROOT_DIR}/${subdir}"
  echo "==> Pulling ${REMOTE_HOST}:${REMOTE_DIR}/${subdir}/ -> ${ROOT_DIR}/${subdir}/ ${DRY_RUN}"
  rsync "${RSYNC_BASE[@]}" "${REMOTE_HOST}:${REMOTE_DIR}/${subdir}/" "${ROOT_DIR}/${subdir}/"
}

case "$SCOPE" in
  eval)
    pull_dir "runs/eval"
    ;;
  train)
    pull_dir "runs/train"
    ;;
  services)
    pull_dir "runs/services"
    ;;
  install-logs)
    pull_dir "runs/install-logs"
    ;;
  tmux-logs)
    pull_dir "runs/logs/tmux"
    ;;
  all)
    pull_dir "runs/eval"
    pull_dir "runs/train"
    pull_dir "runs/services"
    pull_dir "runs/install-logs"
    pull_dir "runs/logs/tmux"
    ;;
esac

echo "==> Done."
