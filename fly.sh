#!/usr/bin/env bash
# Fly one flight and upload its log to GitHub (branch flight-logs), see LAB_DEPLOY.md.
#
#   ./fly.sh                     # = python scripts/deploy.py --config level2.toml
#   ./fly.sh --render True       # extra arguments are passed to deploy.py
#
# After the flight (also after Ctrl+C) the new flight_logs/*.npz are committed to the local branch
# flight-logs (without touching the checked-out code branch) and pushed to adr_project in the background.
# Env: FLY_REMOTE (default: adr_project on GitHub, no matter which repo was cloned),
#      FLY_CONFIG (default level2.toml), FLY_NO_PUSH=1 (commit only),
#      FLY_DEPLOY_CMD (replace the deploy command, e.g. a sim for testing).
set -u
cd "$(dirname "$0")"
REMOTE=${FLY_REMOTE:-https://github.com/Xiaolong-Li1904/adr_project.git}   # logs always go to adr_project
BRANCH=flight-logs
CONFIG=${FLY_CONFIG:-level2.toml}
GITDIR=$(git rev-parse --git-dir)
PUSHLOG="$GITDIR/flight-logs-push.log"

# Keep flight_logs/ out of the code branch (local exclude, no change to .gitignore).
grep -qx "flight_logs/" "$GITDIR/info/exclude" 2>/dev/null || echo "flight_logs/" >> "$GITDIR/info/exclude"

# Result of the previous background upload.
if [ -f "$PUSHLOG" ] && grep -qiE "error|fatal|rejected" "$PUSHLOG"; then
    echo "[fly] WARNING: last upload failed (will retry now): $(tail -1 "$PUSHLOG")"
fi

# ---- fly -------------------------------------------------------------------------------------------
if [ -n "${FLY_DEPLOY_CMD:-}" ]; then
    CMD=($FLY_DEPLOY_CMD)
elif [ "${PIXI_ENVIRONMENT_NAME:-}" = "deploy" ]; then
    CMD=(python scripts/deploy.py --config "$CONFIG" "$@")
else
    CMD=(pixi run -e deploy python scripts/deploy.py --config "$CONFIG" "$@")
fi
echo "[fly] policy=${ADR_RL_POLICY:-<default general_tilt1.0>}  kappa=$(cat "${ADR_KAPPA_FILE:-/nonexistent}" 2>/dev/null || echo "${ADR_KAPPA:-0}")"
trap 'echo "[fly] Ctrl+C -> saving log"' INT   # a handler (not ignore), so deploy.py still gets Ctrl+C
"${CMD[@]}"
trap - INT

# ---- commit the new logs to branch flight-logs (separate index, code branch untouched) --------------
shopt -s nullglob
logs=(flight_logs/*.npz)
[ ${#logs[@]} -eq 0 ] && { echo "[fly] no flight log"; exit 0; }

export GIT_INDEX_FILE="$GITDIR/flight-logs.index"
if ! git rev-parse -q --verify "refs/heads/$BRANCH" >/dev/null; then
    git fetch -q "$REMOTE" "$BRANCH:$BRANCH" 2>/dev/null || true   # continue the remote branch if it exists
fi
parent=$(git rev-parse -q --verify "refs/heads/$BRANCH" || true)
if [ -n "$parent" ]; then git read-tree "$parent"; else git read-tree --empty; fi
git add -f flight_logs/*.npz
tree=$(git write-tree)
if [ -n "$parent" ] && [ "$tree" = "$(git rev-parse "$parent^{tree}")" ]; then
    echo "[fly] no new log"
else
    new=$(git diff --cached --name-only ${parent:+"$parent"} 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')
    [ -z "$parent" ] && new=$(git ls-files | xargs -n1 basename | tr '\n' ' ')
    commit=$(git commit-tree "$tree" ${parent:+-p "$parent"} -m "flight $(date +%Y-%m-%d_%H:%M:%S): $new")
    git update-ref "refs/heads/$BRANCH" "$commit"
    echo "[fly] committed to $BRANCH: $new"
fi
unset GIT_INDEX_FILE

# ---- quick result of the newest log -------------------------------------------------------------------
newest=$(ls -t flight_logs/*.npz | head -1)
if [ -n "${FLY_DEPLOY_CMD:-}" ] || [ "${PIXI_ENVIRONMENT_NAME:-}" = "deploy" ]; then PY=(python); else PY=(pixi run -e deploy python); fi
"${PY[@]}" -W ignore scripts/analyze_flight_log.py "$newest" 2>/dev/null | head -1 || true

# ---- upload in the background --------------------------------------------------------------------------
if [ "${FLY_NO_PUSH:-0}" != "1" ]; then
    ( git push -q "$REMOTE" "$BRANCH:$BRANCH" > "$PUSHLOG" 2>&1 && echo "ok $(date +%T)" >> "$PUSHLOG" ) &
    echo "[fly] uploading to adr_project/$BRANCH in the background (status: $PUSHLOG)"
fi
