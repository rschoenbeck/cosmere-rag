"""Combined eval report: metrics + per-query records, JSON + markdown."""
from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from cosmere_rag.eval.runner import RetrievalResult


class EvalReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_name: str
    embedding_model: str
    collection: str
    backend: str
    k: int
    num_queries: int
    ir_metrics: dict[str, float] = Field(default_factory=dict)
    deepeval_metrics: dict[str, float] = Field(default_factory=dict)
    per_query: list[dict] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, ensure_ascii=False)

    def write_markdown(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._render_markdown(), encoding="utf-8")

    def _render_markdown(self) -> str:
        lines: list[str] = []
        lines.append(f"# Eval report: {self.run_name}")
        lines.append("")
        lines.append(f"- Embedding model: `{self.embedding_model}`")
        lines.append(f"- Backend: `{self.backend}`  Collection: `{self.collection}`")
        lines.append(f"- k = {self.k}, queries = {self.num_queries}")
        lines.append(f"- Created: {self.created_at.isoformat()}")
        lines.append("")
        if self.ir_metrics:
            lines.append("## IR metrics")
            lines.append("")
            lines.append("| Metric | Score |")
            lines.append("|--------|-------|")
            for name, value in self.ir_metrics.items():
                lines.append(f"| {name} | {value:.4f} |")
            lines.append("")
        if self.deepeval_metrics:
            lines.append("## DeepEval metrics")
            lines.append("")
            lines.append("| Metric | Score |")
            lines.append("|--------|-------|")
            for name, value in self.deepeval_metrics.items():
                lines.append(f"| {name} | {value:.4f} |")
            lines.append("")
        return "\n".join(lines)


def per_query_record(
    result: RetrievalResult,
    ir_scores: Mapping[str, float] | None = None,
    deepeval_scores: Mapping[str, float] | None = None,
) -> dict:
    return {
        "query_id": result.query_id,
        "query": result.query,
        "retrieved_chunk_ids": list(result.retrieved_chunk_ids),
        "scores": list(result.scores),
        "ir": dict(ir_scores or {}),
        "deepeval": dict(deepeval_scores or {}),
    }
