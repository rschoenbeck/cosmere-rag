"""Behavioural checks for the noise baseline embedder."""
from __future__ import annotations

import math

import pytest

from cosmere_rag.eval.baselines import NoiseEmbedder


def test_same_text_produces_same_vector():
    e = NoiseEmbedder(dim=64)
    assert e.embed_query("Vin") == e.embed_query("Vin")
    assert e.embed_documents(["Vin"])[0] == e.embed_query("Vin")


def test_different_text_produces_different_vector():
    e = NoiseEmbedder(dim=64)
    assert e.embed_query("Vin") != e.embed_query("Kelsier")


def test_vectors_are_l2_normalized():
    e = NoiseEmbedder(dim=128)
    vec = e.embed_query("Allomancy")
    norm = math.sqrt(sum(x * x for x in vec))
    assert norm == pytest.approx(1.0, abs=1e-5)


def test_dim_matches_request():
    e = NoiseEmbedder(dim=256)
    assert len(e.embed_query("x")) == 256
    assert all(len(v) == 256 for v in e.embed_documents(["a", "b", "c"]))


def test_invalid_dim_rejected():
    with pytest.raises(ValueError):
        NoiseEmbedder(dim=0)
    with pytest.raises(ValueError):
        NoiseEmbedder(dim=-3)
