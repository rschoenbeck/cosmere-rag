import json
from pathlib import Path

import pytest

from cosmere_rag.ingest import coppermind
from cosmere_rag.ingest.cli import _series_mentioned, run_ingest

FIXTURES = Path(__file__).parent / "fixtures"
SYNTHETIC_FIXTURES = FIXTURES / "synthetic"


def _read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _read_synth(name: str) -> str:
    return (SYNTHETIC_FIXTURES / name).read_text(encoding="utf-8")


# --------- Infobox parsing ---------

def test_infobox_kelsier():
    box = coppermind.parse_infobox(_read("Kelsier.md"))
    assert box["world"].startswith("[[Scadrial")
    assert box["featured in"] == "*Mistborn*"
    assert "spouse" in box


def test_infobox_atium_lowercase_field():
    """Atium uses `Featured in` (lowercase i) — case-insensitive match required."""
    box = coppermind.parse_infobox(_read("Atium.md"))
    assert box["featured in"] == "*Mistborn (series)*"


# --------- Featured In tokenization & Era 1 match ---------

@pytest.mark.parametrize("value,expected", [
    ("*Mistborn*", True),
    ("*Mistborn Era 1*", True),
    ("*Mistborn (series)*", True),
    ("*Mistborn Era 2*", False),
    ("*The Stormlight Archive, Mistborn Era 1*", True),
    ("*The Stormlight Archive*", False),
    ("*Mistborn,[1] Warbreaker*", True),
])
def test_is_era1_featured_in(value, expected):
    assert coppermind.is_era1_featured_in(value) is expected


# --------- Series-code mapping ---------

def test_series_mentioned_cross_cosmere():
    codes = _series_mentioned("*The Stormlight Archive, Mistborn Era 1*")
    assert codes == ["Stormlight", "MB-Era1"]


def test_series_mentioned_era2_excluded_from_era1_code():
    codes = _series_mentioned("*Mistborn Era 2*")
    assert "MB-Era1" not in codes
    assert "MB-Era2" in codes


# --------- Article parsing ---------

def test_parse_article_kelsier_structure():
    article = coppermind.parse_article("Kelsier", _read("Kelsier.md"))
    assert article.title == "Kelsier"
    assert article.source_url == "https://coppermind.net/wiki/Survivor"
    assert article.infobox.get("featured in") == "*Mistborn*"
    headings = {tuple(s.heading_path) for s in article.sections}
    assert ("Appearance and Personality",) in headings
    assert ("Attributes and Abilities", "Allomancy") in headings


def test_parse_article_drops_noise():
    article = coppermind.parse_article("Kelsier", _read("Kelsier.md"))
    full_body = "\n".join(s.body for s in article.sections)
    # The "Contents" auto-TOC section should be skipped entirely.
    assert "1 Appearance and Personality" not in full_body
    # Bare footnote lines like `[8]` on their own should be gone.
    for line in full_body.splitlines():
        assert line.strip() != "[8]"


def test_extract_wikilinks_includes_aliases():
    links = coppermind.extract_wikilinks(_read("Kelsier.md"))
    assert "Mare" in links
    assert "Allomancy" in links


# --------- Stub article handles a tiny body ---------

def test_parse_stub_article_jedal():
    article = coppermind.parse_article("Jedal", _read("Jedal.md"))
    assert article.infobox["featured in"] == "*Mistborn Era 1*"
    chunks = coppermind.chunk_article(article)
    assert chunks
    # Stub fits in one chunk under the cap.
    assert len(chunks) <= 2
    assert all(c.token_count <= 800 for c in chunks)
    assert all(c.text.startswith("Jedal") for c in chunks)


# --------- Chunking guarantees ---------

def test_chunks_respect_token_cap():
    article = coppermind.parse_article("Kelsier", _read("Kelsier.md"))
    chunks = coppermind.chunk_article(article, max_tokens=400)
    assert chunks
    for c in chunks:
        assert c.token_count <= 450, f"chunk over cap: {c.token_count} tokens"


def test_chunk_text_starts_with_title_and_path():
    article = coppermind.parse_article("Kelsier", _read("Kelsier.md"))
    chunks = coppermind.chunk_article(article)
    for c in chunks:
        if c.heading_path:
            expected_prefix = f"Kelsier › {' › '.join(c.heading_path)}"
            assert c.text.startswith(expected_prefix)
        else:
            assert c.text.startswith("Kelsier")


def test_chunk_merge_floor_combines_tiny_sections():
    article = coppermind.parse_article("Kelsier", _read("Kelsier.md"))
    chunks_no_merge = coppermind.chunk_article(article, merge_floor=0)
    chunks_merged = coppermind.chunk_article(article, merge_floor=200)
    assert len(chunks_merged) <= len(chunks_no_merge)


# --------- End-to-end filter selection ---------

def test_select_era1_titles_from_fixtures(tmp_path):
    """All fixtures except Atium are Era 1 (Atium is `Mistborn (series)`,
    which we do count). Hammond is Era 1 only. Allomancy and Atium include
    cross-Cosmere mentions but qualify."""
    # Copy fixtures into a tmp dir so we control the corpus exactly.
    corpus = tmp_path / "Cosmere"
    corpus.mkdir()
    for name in ("Kelsier.md", "Allomancy.md", "Atium.md", "Hammond.md", "Jedal.md"):
        (corpus / name).write_text(_read(name), encoding="utf-8")
    selection = coppermind.select_era1_titles(corpus, seeds=["Kelsier"])
    assert set(selection.titles) == {"Kelsier", "Allomancy", "Atium", "Hammond", "Jedal"}
    assert selection.missing_seeds == []
    assert all(
        p == coppermind.PROVENANCE_MIRROR for p in selection.provenance.values()
    )


# --------- Synthetic corpus merge ---------

def _make_corpus(tmp_path: Path) -> tuple[Path, Path]:
    """Spin up a mirror+synthetic pair under tmp_path and return (mirror, synth)."""
    corpus = tmp_path / "Cosmere"
    corpus.mkdir()
    for name in ("Kelsier.md", "Hammond.md"):
        (corpus / name).write_text(_read(name), encoding="utf-8")
    synth = tmp_path / "synthetic"
    synth.mkdir()
    for name in ("Ham.md", "Breeze.md"):
        (synth / name).write_text(_read_synth(name), encoding="utf-8")
    return corpus, synth


def test_synthetic_stub_fills_gap(tmp_path):
    """Ham.md is synthetic-only; Breeze.md is synthetic-only; both show up
    as Era 1 titles with synthetic provenance."""
    corpus, synth = _make_corpus(tmp_path)
    selection = coppermind.select_era1_titles(corpus, seeds=[], synthetic_dir=synth)
    assert "Ham" in selection.titles
    assert "Breeze" in selection.titles
    assert selection.provenance["Ham"] == coppermind.PROVENANCE_SYNTHETIC
    assert selection.provenance["Breeze"] == coppermind.PROVENANCE_SYNTHETIC
    assert selection.provenance["Hammond"] == coppermind.PROVENANCE_MIRROR


def test_mirror_beats_synthetic_on_collision(tmp_path):
    """If a title exists in both dirs, the mirror file wins."""
    corpus, synth = _make_corpus(tmp_path)
    # Add a Hammond-named synthetic file that would shadow the real one.
    (synth / "Hammond.md").write_text(_read_synth("Ham.md"), encoding="utf-8")
    selection = coppermind.select_era1_titles(corpus, seeds=[], synthetic_dir=synth)
    assert selection.provenance["Hammond"] == coppermind.PROVENANCE_MIRROR
    assert selection.paths["Hammond"] == corpus / "Hammond.md"


def test_synthetic_redirect_stub_chunks(tmp_path):
    """A one-sentence redirect stub should parse and produce at least one chunk."""
    article = coppermind.parse_article("Ham", _read_synth("Ham.md"))
    assert article.infobox["featured in"] == "*Mistborn Era 1*"
    chunks = coppermind.chunk_article(article)
    assert chunks
    assert all(c.text.startswith("Ham") for c in chunks)
    # Redirect stub should fit easily in a single chunk.
    assert len(chunks) == 1


def test_run_ingest_tags_content_provenance(tmp_path):
    """End-to-end: chunks from the synthetic dir carry
    content_provenance=synthetic-claude; mirror chunks carry coppermind-mirror."""
    corpus, synth = _make_corpus(tmp_path)
    out = tmp_path / "out" / "chunks.jsonl"
    titles_out = tmp_path / "out" / "titles.txt"
    run_ingest(corpus, out, titles_out, synthetic_dir=synth)
    records = [json.loads(line) for line in out.read_text().splitlines() if line]
    by_title: dict[str, set[str]] = {}
    for rec in records:
        by_title.setdefault(rec["article_title"], set()).add(rec["content_provenance"])
    assert by_title["Hammond"] == {coppermind.PROVENANCE_MIRROR}
    assert by_title["Ham"] == {coppermind.PROVENANCE_SYNTHETIC}
    assert by_title["Breeze"] == {coppermind.PROVENANCE_SYNTHETIC}
