#!/usr/bin/env bash
set -euo pipefail

# Launch a long-running experiment on rl-server inside an isolated code/config snapshot
# under runs/snapshots/<session>-<timestamp>/ so subsequent code syncs never contaminate
# an in-flight run.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-rl-server}"
REMOTE_DIR="${REMOTE_DIR:-/root/ToolCall-SFT}"

usage() {
  cat <<'EOF'
Usage: run_tmux.sh [--sync] [--dry-run | -n] [--help | -h] <session-name> [--] <command...>

Creates an immutable snapshot of src/, configs/, scripts/, and pyproject.toml under
<REMOTE_DIR>/runs/snapshots/<session-name>-<UTC_STAMP>/ with symlinks to shared data/,
runs/, and .venv/, then starts a detached tmux session running <command...> inside
that snapshot with PYTHONPATH pointed at <SNAPSHOT>/src.

Options:
  --sync         Run scripts/remote/sync_code.sh before snapshotting and launching.
  --dry-run, -n  Print the remote actions without creating a snapshot or tmux session.
  --help, -h     Show this help message.
EOF
}

DO_SYNC=0
DRY_RUN=0
POSITIONAL=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sync)
      DO_SYNC=1
      shift
      ;;
    --dry-run|-n)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      POSITIONAL+=("$@")
      break
      ;;
    -*)
      echo "Error: unknown option '$1'" >&2
      usage >&2
      exit 2
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      if [[ $# -gt 0 && "$1" == "--" ]]; then
        shift
      fi
      POSITIONAL+=("$@")
      break
      ;;
  esac
done

if [[ ${#POSITIONAL[@]} -lt 2 ]]; then
  echo "Error: <session-name> and <command...> are required." >&2
  usage >&2
  exit 2
fi

SESSION="${POSITIONAL[0]}"
if [[ ! "$SESSION" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Error: invalid tmux session name '$SESSION' (use [A-Za-z0-9._-]+)." >&2
  exit 2
fi

CMD_ARGS=("${POSITIONAL[@]:1}")
CMD_STR=""
for arg in "${CMD_ARGS[@]}"; do
  printf -v q_arg '%q' "$arg"
  CMD_STR+="${CMD_STR:+ }${q_arg}"
done

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SNAP_REL="runs/snapshots/${SESSION}-${STAMP}"
LOG_REL="runs/logs/tmux/${SESSION}-${STAMP}.log"
LOCAL_HEAD="$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"

if [[ "$DO_SYNC" -eq 1 ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    bash "${SCRIPT_DIR}/sync_code.sh" --dry-run
  else
    bash "${SCRIPT_DIR}/sync_code.sh"
  fi
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[DRY-RUN] Host:      ${REMOTE_HOST}:${REMOTE_DIR}"
  echo "[DRY-RUN] Session:   ${SESSION}"
  echo "[DRY-RUN] Snapshot:  ${REMOTE_DIR}/${SNAP_REL}"
  echo "[DRY-RUN] Log file:  ${REMOTE_DIR}/${LOG_REL}"
  echo "[DRY-RUN] Local git: ${LOCAL_HEAD}"
  echo "[DRY-RUN] Command:   ${CMD_STR}"
  exit 0
fi

REMOTE_ARGS=()
for v in "$REMOTE_HOST" "$REMOTE_DIR" "$SESSION" "$SNAP_REL" "$LOG_REL" "$LOCAL_HEAD" "$STAMP" "$CMD_STR"; do
  printf -v q_v '%q' "$v"
  REMOTE_ARGS+=("$q_v")
done

ssh -o BatchMode=yes "$REMOTE_HOST" "bash -s -- ${REMOTE_ARGS[*]}" <<'REMOTE_EOF'
set -euo pipefail

REMOTE_HOST="$1"
REMOTE_DIR="$2"
SESSION="$3"
SNAP_REL="$4"
LOG_REL="$5"
LOCAL_HEAD="$6"
STAMP="$7"
CMD_STR="$8"

cd "$REMOTE_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Error: tmux session '$SESSION' already exists on $(hostname)." >&2
  exit 1
fi

SNAP_DIR="${REMOTE_DIR}/${SNAP_REL}"
LOG_FILE="${REMOTE_DIR}/${LOG_REL}"

mkdir -p "$SNAP_DIR" "$(dirname "$LOG_FILE")"

for item in src configs scripts pyproject.toml; do
  if [[ -e "${REMOTE_DIR}/${item}" ]]; then
    cp -a "${REMOTE_DIR}/${item}" "${SNAP_DIR}/${item}"
  fi
done

for shared in data runs .venv; do
  if [[ -e "${REMOTE_DIR}/${shared}" ]]; then
    ln -s "${REMOTE_DIR}/${shared}" "${SNAP_DIR}/${shared}"
  fi
done

cat > "${SNAP_DIR}/SNAPSHOT_META.txt" <<EOF
session=${SESSION}
created_utc=${STAMP}
local_git_head=${LOCAL_HEAD}
remote_dir=${REMOTE_DIR}
snapshot_dir=${SNAP_DIR}
log_file=${LOG_FILE}
command=${CMD_STR}
EOF

printf -v INNER_CMD 'set -o pipefail; cd %q && export PYTHONPATH=%q && export TOOLCALL_SNAPSHOT_DIR=%q && { %s; } 2>&1 | tee -a %q' \
  "$SNAP_DIR" \
  "${SNAP_DIR}/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$SNAP_DIR" \
  "$CMD_STR" \
  "$LOG_FILE"

tmux new-session -d -s "$SESSION" "bash -lc $(printf '%q' "$INNER_CMD")"

echo "==> Started tmux session '${SESSION}'"
echo "    Snapshot: ${SNAP_DIR}"
echo "    Log file: ${LOG_FILE}"
echo "    Inspect:  ssh ${REMOTE_HOST} \"tmux capture-pane -pt ${SESSION} | tail -n 30\""
REMOTE_EOF
