#!/usr/bin/env bash
# Fetch the Coppermind markdown mirror at a pinned commit.
# See plans/cosmere-rag-plan.md and project memory for why we vendor a
# third-party scrape instead of hitting the live MediaWiki API.
set -euo pipefail

PINNED_SHA="2a1945c24f48c313a32b20483a8160e86aa1c047"
DEST="data/coppermind-mirror"

repo_root="$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
cd "$repo_root"

if [[ -d "$DEST/.git" ]]; then
    echo "Mirror already cloned at $DEST"
else
    mkdir -p data
    git clone --filter=blob:none "https://github.com/Malthemester/CoppermindScraper" "$DEST"
fi

git -C "$DEST" fetch --depth=1 origin "$PINNED_SHA"
git -C "$DEST" checkout --detach "$PINNED_SHA"
echo "$PINNED_SHA" > "$DEST/COMMIT.txt"

count=$(find "$DEST/Cosmere" -maxdepth 1 -name '*.md' | wc -l | tr -d ' ')
echo "Mirror checked out at $PINNED_SHA (${count} top-level .md files)"
