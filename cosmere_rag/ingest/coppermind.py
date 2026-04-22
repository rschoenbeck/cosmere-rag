"""Parse, filter, and chunk articles from the Malthemester/CoppermindScraper
markdown mirror for the Mistborn Era 1 ingestion slice.

The mirror is a one-.md-per-article Obsidian vault. Each file's leading
markdown table is the rendered Coppermind infobox; section headings are
preserved as `##` / `###`; wikilinks are Obsidian-style `[[Target|Label]]`;
the canonical Coppermind URL is appended as the last line.

This module deliberately doesn't try to recover information the HTML->MD
conversion threw away (wikitext spoiler templates, categories, revision IDs).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import tiktoken

# Featured In tokens (case-insensitive, italic-stripped) that mark Era 1.
# `Mistborn` alone covers the era when no other suffix existed; `(series)` and
# `series`/`trilogy` are the umbrella forms. Era 2 explicitly excluded.
_ERA1_FEATURED_IN_TOKENS: frozenset[str] = frozenset({
    "mistborn",
    "mistborn era 1",
    "mistborn (series)",
    "mistborn series",
    "mistborn trilogy",
    "mistborn: the final empire",
    "mistborn: secret history",
})

_FOOTNOTE_RE = re.compile(r"\[\d+\]")
_WIKILINK_RE = re.compile(r"\[\[([^\]|\\]+)(?:\\?\|[^\]]*)?\]\]")
_INFOBOX_ROW_RE = re.compile(r"^\|(?P<key>[^|]*)\|(?P<value>.*)\|\s*$")
_HEADING_RE = re.compile(r"^(?P<hashes>#{2,3})\s+(?P<text>.+?)\s*$")
_IMAGE_CREDIT_RE = re.compile(r"^\s*by\s+[A-Z].*$")
_BARE_FOOTNOTE_LINE_RE = re.compile(r"^\s*\[\d+\]\s*$")
_NOISE_HEADINGS: frozenset[str] = frozenset({
    "contents", "notes", "references", "trivia gallery", "gallery",
})


# --------- Infobox parsing ---------

def parse_infobox(text: str) -> dict[str, str]:
    """Return {field: value} from the leading markdown table.

    Field keys are unwrapped from `**...**` bold and lowercased-trimmed.
    Values keep their inner markdown (italics, wikilinks); strip those at the
    callsite if needed. Empty-value rows and the `|-|-|` separator row are
    skipped. Stops at the first blank line or non-table line after the table
    has started.
    """
    fields: dict[str, str] = {}
    in_table = False
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.startswith("|"):
            if in_table:
                break
            continue
        in_table = True
        if set(line) <= {"|", "-", " "}:
            continue
        m = _INFOBOX_ROW_RE.match(line)
        if not m:
            continue
        key = m.group("key").strip().strip("*").strip()
        value = m.group("value").strip()
        if not key or not value:
            continue
        fields[key.lower()] = value
    return fields


def featured_in_tokens(value: str) -> list[str]:
    """Tokenize a `Featured In` cell value into lowercase series tokens."""
    cleaned = value.strip().strip("*").strip()
    cleaned = _FOOTNOTE_RE.sub("", cleaned)
    return [t.strip().lower() for t in cleaned.split(",") if t.strip()]


def is_era1_featured_in(value: str) -> bool:
    """True if any tokenized `Featured In` value matches an Era 1 series."""
    return any(t in _ERA1_FEATURED_IN_TOKENS for t in featured_in_tokens(value))


PROVENANCE_MIRROR = "coppermind-mirror"
PROVENANCE_SYNTHETIC = "synthetic-claude"


def _scan_era1_dir(
    directory: Path, seeds: set[str]
) -> tuple[set[str], dict[str, Path]]:
    """Scan one directory. Return (matched_titles, title_to_path)."""
    matched: set[str] = set()
    paths: dict[str, Path] = {}
    for md_path in sorted(directory.glob("*.md")):
        title = md_path.stem
        try:
            text = md_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        paths[title] = md_path
        if title in seeds:
            matched.add(title)
            continue
        infobox = parse_infobox(text)
        fi = infobox.get("featured in")
        if fi and is_era1_featured_in(fi):
            matched.add(title)
    return matched, paths


@dataclass
class Era1Selection:
    """Outcome of Era 1 title selection across one or more corpus roots."""
    titles: list[str]
    missing_seeds: list[str]
    paths: dict[str, Path]
    provenance: dict[str, str]


def select_era1_titles(
    corpus_dir: Path,
    seeds: Iterable[str],
    *,
    synthetic_dir: Path | None = None,
) -> Era1Selection:
    """Walk the corpus directory (and optional synthetic dir) and return
    an Era1Selection describing which titles are in scope, which seed
    titles are missing from *all* sources, and the chosen file path and
    provenance per title.

    A title is included if it's a seed (regardless of infobox) OR if its
    infobox `Featured In` field tokenizes to at least one Era 1 series.

    Mirror articles beat synthetic articles on title collisions; synthetic
    only fills gaps.
    """
    seeds_set = set(seeds)
    mirror_matched, mirror_paths = _scan_era1_dir(corpus_dir, seeds_set)

    synthetic_matched: set[str] = set()
    synthetic_paths: dict[str, Path] = {}
    if synthetic_dir is not None and synthetic_dir.is_dir():
        synthetic_matched, synthetic_paths = _scan_era1_dir(synthetic_dir, seeds_set)

    all_matched = mirror_matched | synthetic_matched
    paths: dict[str, Path] = {}
    provenance: dict[str, str] = {}
    for title in all_matched:
        if title in mirror_paths:
            paths[title] = mirror_paths[title]
            provenance[title] = PROVENANCE_MIRROR
        else:
            paths[title] = synthetic_paths[title]
            provenance[title] = PROVENANCE_SYNTHETIC

    present_seeds = {t for t in seeds_set if t in mirror_paths or t in synthetic_paths}
    missing_seeds = sorted(seeds_set - present_seeds)
    return Era1Selection(
        titles=sorted(all_matched),
        missing_seeds=missing_seeds,
        paths=paths,
        provenance=provenance,
    )


# --------- Body parsing & chunking ---------

@dataclass
class Section:
    heading_path: list[str]
    body: str


@dataclass
class ParsedArticle:
    title: str
    source_url: str | None
    infobox: dict[str, str]
    sections: list[Section]
    wikilinks: list[str] = field(default_factory=list)


def extract_source_url(text: str) -> str | None:
    """The scraper appends the canonical `https://coppermind.net/wiki/<...>`
    URL on the last non-blank line of every article."""
    for line in reversed(text.splitlines()):
        s = line.strip()
        if s.startswith("https://coppermind.net/"):
            return s
    return None


def extract_wikilinks(text: str) -> list[str]:
    """Return the unique list of wikilink targets (left-hand side of [[T|L]])
    in document order."""
    seen: dict[str, None] = {}
    for m in _WIKILINK_RE.finditer(text):
        target = m.group(1).strip()
        if target and target not in seen:
            seen[target] = None
    return list(seen.keys())


def _strip_infobox_block(text: str) -> str:
    lines = text.splitlines()
    started = False
    cut = 0
    for i, raw in enumerate(lines):
        if raw.startswith("|"):
            started = True
            cut = i + 1
        elif started:
            break
    return "\n".join(lines[cut:]) if started else text


def _strip_trailing_url(text: str) -> str:
    lines = text.splitlines()
    while lines and (not lines[-1].strip() or lines[-1].strip().startswith("https://coppermind.net/")):
        lines.pop()
    return "\n".join(lines)


def _clean_body_line(line: str) -> str | None:
    """Drop UI noise lines. Returns None for lines that should be removed."""
    s = line.strip()
    if not s:
        return ""
    if _BARE_FOOTNOTE_LINE_RE.match(s):
        return None
    if _IMAGE_CREDIT_RE.match(s) and len(s) < 80:
        return None
    if "🐱" in s:
        return None
    return line


def split_sections(body: str) -> list[Section]:
    """Split a body (infobox already stripped) into sections by ## / ###.

    Heading_path is a list: [h2_text] for a `##` section, [h2_text, h3_text]
    for a `###` section nested under the most recent h2. A leading lead
    paragraph (text before the first heading) becomes a section with
    heading_path == [].
    """
    sections: list[Section] = []
    current_h2: str | None = None
    current_path: list[str] = []
    current_lines: list[str] = []

    def flush() -> None:
        if current_lines or sections == []:
            text = "\n".join(current_lines).strip()
            if text:
                sections.append(Section(heading_path=list(current_path), body=text))

    for raw in body.splitlines():
        m = _HEADING_RE.match(raw)
        if m:
            flush()
            current_lines = []
            level = len(m.group("hashes"))
            heading = m.group("text").strip()
            if heading.lower() in _NOISE_HEADINGS:
                # Skip noise sections by setting a sentinel path that we'll
                # filter on flush (we just stop appending until next heading).
                current_path = ["__skip__", heading]
                continue
            if level == 2:
                current_h2 = heading
                current_path = [heading]
            else:  # level == 3
                current_path = [current_h2, heading] if current_h2 else [heading]
            continue
        if current_path and current_path[0] == "__skip__":
            continue
        cleaned = _clean_body_line(raw)
        if cleaned is None:
            continue
        current_lines.append(cleaned)
    flush()
    return [s for s in sections if s.body]


def parse_article(title: str, text: str) -> ParsedArticle:
    infobox = parse_infobox(text)
    source_url = extract_source_url(text)
    wikilinks = extract_wikilinks(text)
    body = _strip_trailing_url(_strip_infobox_block(text))
    sections = split_sections(body)
    return ParsedArticle(
        title=title,
        source_url=source_url,
        infobox=infobox,
        sections=sections,
        wikilinks=wikilinks,
    )


# --------- Token-aware chunking ---------

@dataclass
class Chunk:
    article_title: str
    heading_path: list[str]
    text: str
    token_count: int


_ENCODER = tiktoken.get_encoding("cl100k_base")
_DEFAULT_MAX_TOKENS = 800
_DEFAULT_MERGE_FLOOR = 80


def _count_tokens(s: str) -> int:
    return len(_ENCODER.encode(s, disallowed_special=()))


def _split_long_section(prefix: str, body: str, max_tokens: int) -> list[str]:
    """Split a too-long section body on paragraph boundaries, packing
    paragraphs into chunks under max_tokens (counting the heading prefix).
    Falls back to splitting a single oversized paragraph on sentence-ish
    boundaries (`. `) if needed."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = _count_tokens(prefix)

    def flush() -> None:
        if current:
            chunks.append(prefix + "\n\n".join(current))
            current.clear()

    for para in paragraphs:
        ptok = _count_tokens(para)
        if ptok > max_tokens:
            # Mega-paragraph: split on sentence-ish boundaries.
            flush()
            current_tokens = _count_tokens(prefix)
            sentences = re.split(r"(?<=[.!?])\s+", para)
            buf: list[str] = []
            buf_tok = _count_tokens(prefix)
            for sent in sentences:
                stok = _count_tokens(sent)
                if buf and buf_tok + stok > max_tokens:
                    chunks.append(prefix + " ".join(buf))
                    buf = [sent]
                    buf_tok = _count_tokens(prefix) + stok
                else:
                    buf.append(sent)
                    buf_tok += stok
            if buf:
                chunks.append(prefix + " ".join(buf))
            continue
        if current and current_tokens + ptok > max_tokens:
            flush()
            current_tokens = _count_tokens(prefix)
        current.append(para)
        current_tokens += ptok
    flush()
    return chunks


def chunk_article(
    article: ParsedArticle,
    *,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    merge_floor: int = _DEFAULT_MERGE_FLOOR,
) -> list[Chunk]:
    """Produce chunks: one per `### subsection`, or per `## section` if
    no subsections; merge undersized siblings into their parent's prose
    where adjacent; split oversize sections on paragraph boundaries.

    Every chunk's `text` is prefixed with `Title › heading › subheading\\n\\n`
    to give retrieval lexical anchors for proper-noun-heavy queries.
    """
    if not article.sections:
        return []

    # First merge: any section under the floor merges into the previous
    # section if siblings under the same parent.
    merged: list[Section] = []
    for sec in article.sections:
        tok = _count_tokens(sec.body)
        if (
            merged
            and tok < merge_floor
            and merged[-1].heading_path[: max(len(sec.heading_path) - 1, 0)]
                == sec.heading_path[: max(len(sec.heading_path) - 1, 0)]
        ):
            merged[-1] = Section(
                heading_path=merged[-1].heading_path,
                body=merged[-1].body + "\n\n" + sec.body,
            )
        else:
            merged.append(sec)

    chunks: list[Chunk] = []
    for sec in merged:
        path_str = " › ".join(sec.heading_path) if sec.heading_path else ""
        prefix_label = f"{article.title} › {path_str}" if path_str else article.title
        prefix = f"{prefix_label}\n\n"
        body_tokens = _count_tokens(sec.body)
        full_tokens = _count_tokens(prefix) + body_tokens
        if full_tokens <= max_tokens:
            text = prefix + sec.body
            chunks.append(Chunk(
                article_title=article.title,
                heading_path=sec.heading_path,
                text=text,
                token_count=_count_tokens(text),
            ))
        else:
            for piece in _split_long_section(prefix, sec.body, max_tokens):
                chunks.append(Chunk(
                    article_title=article.title,
                    heading_path=sec.heading_path,
                    text=piece,
                    token_count=_count_tokens(piece),
                ))
    return chunks
