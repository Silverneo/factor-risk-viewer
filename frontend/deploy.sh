#!/usr/bin/env bash
# Tiny wrapper so `./deploy.sh <verb>` works from a Git Bash session that
# doesn't have node on PATH (e.g. when fnm hasn't been activated for the
# current shell). Uses the durable fnm-managed node install path.
#
# Usage:
#   ./deploy.sh login     # interactive Vercel login (do once per machine)
#   ./deploy.sh deploy    # production deploy
#   ./deploy.sh preview   # preview deploy
#   ./deploy.sh logs      # tail the latest deploy's logs

set -euo pipefail

# Hunt for a usable node — prefer whatever's already on PATH, otherwise
# fall back to the fnm install path.
if ! command -v node >/dev/null 2>&1; then
  FNM_NODE="/c/Users/CMZHA/AppData/Roaming/fnm/node-versions/v24.15.0/installation"
  if [ -x "$FNM_NODE/node.exe" ] || [ -x "$FNM_NODE/node" ]; then
    export PATH="$FNM_NODE:$PATH"
  else
    echo "node not found on PATH and fnm install missing at $FNM_NODE" >&2
    echo "edit deploy.sh to point at your node install, or activate fnm in this shell." >&2
    exit 1
  fi
fi

cd "$(dirname "$0")"

VERB="${1:-deploy}"
shift || true

case "$VERB" in
  login)   exec ./node_modules/.bin/vercel login "$@" ;;
  deploy)  exec ./node_modules/.bin/vercel --prod --yes "$@" ;;
  preview) exec ./node_modules/.bin/vercel --yes "$@" ;;
  logs)    exec ./node_modules/.bin/vercel logs "$@" ;;
  *)       exec ./node_modules/.bin/vercel "$VERB" "$@" ;;
esac
