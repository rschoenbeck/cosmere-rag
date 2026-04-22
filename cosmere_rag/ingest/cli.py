"""CLI: build a chunked JSONL corpus for one Cosmere scope.

Currently the only supported scope is `mistborn-era-1`. Output is a single
JSONL file plus a sidecar `era1_titles.txt` of selected article titles for
human inspection.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from cosmere_rag.ingest import coppermind
from cosmere_rag.ingest.mistborn_era1_seeds import SEED_TITLES

LICENSE_TAG = "CC BY-SA 4.0"
LICENSE_ATTRIBUTION = "Coppermind contributors"


def _read_corpus_snapshot(corpus_dir: Path) -> str:
    """Return the pinned mirror SHA written by scripts/fetch_corpus.sh."""
    commit_file = corpus_dir.parent / "COMMIT.txt"
    if commit_file.is_file():
        return commit_file.read_text(encoding="utf-8").strip()
    return "unknown"


def _series_mentioned(featured_in: str | None) -> list[str]:
    """Map a `Featured In` cell to a small set of series codes for metadata."""
    if not featured_in:
        return []
    tokens = coppermind.featured_in_tokens(featured_in)
    out: list[str] = []
    seen: set[str] = set()
    series_map = [
        ("MB-Era1", {
            "mistborn", "mistborn era 1", "mistborn (series)",
            "mistborn series", "mistborn trilogy",
            "mistborn: the final empire", "mistborn: secret history",
        }),
        ("MB-Era2", {"mistborn era 2"}),
        ("Stormlight", {"the stormlight archive", "stormlight archive"}),
        ("Elantris", {"elantris"}),
        ("Warbreaker", {"warbreaker"}),
        ("Emperor's Soul", {"the emperor's soul"}),
        ("White Sand", {"white sand"}),
        ("Sixth of the Dusk", {"sixth of the dusk"}),
        ("Shadows for Silence", {
            "shadows for silence",
            "shadows for silence in the forests of hell",
        }),
    ]
    for token in tokens:
        for code, names in series_map:
            if token in names and code not in seen:
                out.append(code)
                seen.add(code)
    return out


def run_ingest(
    corpus_dir: Path,
    out_path: Path,
    titles_path: Path,
    synthetic_dir: Path | None = None,
) -> int:
    selection = coppermind.select_era1_titles(
        corpus_dir, SEED_TITLES, synthetic_dir=synthetic_dir,
    )
    titles_path.parent.mkdir(parents=True, exist_ok=True)
    titles_path.write_text("\n".join(selection.titles) + "\n", encoding="utf-8")
    print(
        f"Selected {len(selection.titles)} Era 1 titles -> {titles_path}",
        file=sys.stderr,
    )
    synthetic_count = sum(
        1 for p in selection.provenance.values()
        if p == coppermind.PROVENANCE_SYNTHETIC
    )
    if synthetic_count:
        print(
            f"  (including {synthetic_count} synthetic-claude articles from {synthetic_dir})",
            file=sys.stderr,
        )
    if selection.missing_seeds:
        print(
            f"WARNING: {len(selection.missing_seeds)} seed titles missing from corpus snapshot: "
            + ", ".join(selection.missing_seeds),
            file=sys.stderr,
        )

    snapshot = _read_corpus_snapshot(corpus_dir)
    ingested_at = datetime.now(timezone.utc).isoformat()

    chunks_written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for title in selection.titles:
            md_path = selection.paths[title]
            provenance = selection.provenance[title]
            text = md_path.read_text(encoding="utf-8")
            article = coppermind.parse_article(title, text)
            series = _series_mentioned(article.infobox.get("featured in"))
            for chunk in coppermind.chunk_article(article):
                record = {
                    "article_title": chunk.article_title,
                    "heading_path": chunk.heading_path,
                    "series_mentioned": series,
                    "spoiler_scope": "MB-Era1",
                    "source_url": article.source_url,
                    "license": LICENSE_TAG,
                    "license_attribution": LICENSE_ATTRIBUTION,
                    "content_type": "wiki",
                    "content_provenance": provenance,
                    "corpus_snapshot": snapshot,
                    "ingested_at": ingested_at,
                    "token_count": chunk.token_count,
                    "text": chunk.text,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                chunks_written += 1
    print(f"Wrote {chunks_written} chunks -> {out_path}", file=sys.stderr)
    return chunks_written


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cosmere-ingest-coppermind")
    p.add_argument("--corpus-dir", type=Path, required=True,
                   help="Path to the Cosmere/ folder inside the cloned mirror.")
    p.add_argument("--synthetic-dir", type=Path, default=None,
                   help="Optional path to a directory of synthetic gap-fill articles "
                        "(same markdown shape as the mirror). Mirror files always "
                        "win on title collisions; synthetic fills gaps only.")
    p.add_argument("--scope", choices=["mistborn-era-1"], default="mistborn-era-1")
    p.add_argument("--out", type=Path, required=True,
                   help="Path to the JSONL chunks output file.")
    p.add_argument("--titles-out", type=Path, default=None,
                   help="Path to write the selected titles list (default: alongside --out).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    titles_path = args.titles_out or args.out.with_name("era1_titles.txt")
    run_ingest(args.corpus_dir, args.out, titles_path, synthetic_dir=args.synthetic_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
