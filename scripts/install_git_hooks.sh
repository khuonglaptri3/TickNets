#!/usr/bin/env bash
# Install git hooks into local .git/hooks/
set -e

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
HOOKS_DIR="$REPO_ROOT/.git/hooks"

if [ ! -d "$HOOKS_DIR" ]; then
    echo "Error: .git/hooks directory not found at $HOOKS_DIR"
    exit 1
fi

echo "Installing post-merge hook into $HOOKS_DIR/post-merge..."
cp "$REPO_ROOT/scripts/hooks/post-merge" "$HOOKS_DIR/post-merge"
chmod +x "$HOOKS_DIR/post-merge"

echo "[✓] Git hooks successfully installed! 'git pull' will now automatically verify CIFAR datasets."
