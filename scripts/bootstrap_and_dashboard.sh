#!/usr/bin/env bash
set -euo pipefail

# One-command onboarding after clone:
# - create venv
# - install package and dashboard-focused dependencies
# - expose `zane` command from ~/.local/bin
# - launch dashboard

MODE="lite"

usage() {
  cat <<'EOF'
Usage: bash scripts/bootstrap_and_dashboard.sh [--full] [--] [dashboard args...]

Options:
  --full   Install full project dependencies (default: lite dashboard deps only)
  -h, --help
           Show this help text

Examples:
  bash scripts/bootstrap_and_dashboard.sh
  bash scripts/bootstrap_and_dashboard.sh --full
  bash scripts/bootstrap_and_dashboard.sh -- --static --query "cold cough"
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)
      MODE="full"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

cd "$(dirname "$0")/.."

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

.venv/bin/pip install --upgrade pip

if [[ "$MODE" == "full" ]]; then
  .venv/bin/pip install -e .
else
  .venv/bin/pip install -e . --no-deps
  .venv/bin/pip install -r requirements-dashboard.txt
fi

mkdir -p "$HOME/.local/bin"
ln -sf "$PWD/.venv/bin/zane" "$HOME/.local/bin/zane"

if [[ ":${PATH}:" != *":$HOME/.local/bin:"* ]]; then
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
fi

export PATH="$HOME/.local/bin:$PATH"

echo "Bootstrap complete (mode=$MODE). Launching dashboard..."
zane dashboard --interactive-query "$@"
