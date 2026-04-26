"""Upload a JSONL golden set into a LangSmith dataset.

JSONL on disk stays the canonical, PR-reviewable source. The LangSmith
dataset is the runtime copy that `evaluate()` reads from. Re-running
this module upserts: new `query_id`s are created, existing ones get
their inputs/outputs/metadata refreshed.

Identity is keyed on `metadata.query_id` (not the LangSmith example
UUID, which we don't store anywhere).
"""
from __future__ import annotations

from pathlib import Path

from langsmith import Client
from langsmith.schemas import Dataset, ExampleCreate, ExampleUpdate

from cosmere_rag.eval.dataset import EvalQuery, load_golden_set


def upload_golden_set(
    path: Path,
    dataset_name: str | None = None,
    *,
    client: Client | None = None,
    description: str | None = None,
) -> Dataset:
    """Upsert a JSONL golden set into a LangSmith dataset.

    `dataset_name` defaults to the file stem (e.g. `mistborn_era1.jsonl`
    → `mistborn_era1`). Returns the resulting `Dataset`.
    """
    queries = load_golden_set(path)
    name = dataset_name or path.stem
    client = client or Client()

    dataset = _get_or_create_dataset(client, name, description)

    existing_by_qid: dict[str, str] = {}
    for ex in client.list_examples(dataset_id=dataset.id):
        qid = (ex.metadata or {}).get("query_id")
        if qid:
            existing_by_qid[qid] = str(ex.id)

    to_create: list[ExampleCreate] = []
    to_update: list[ExampleUpdate] = []
    for q in queries:
        inputs, outputs, metadata = _example_payload(q)
        existing_id = existing_by_qid.get(q.query_id)
        if existing_id is None:
            to_create.append(
                ExampleCreate(inputs=inputs, outputs=outputs, metadata=metadata)
            )
        else:
            to_update.append(
                ExampleUpdate(
                    id=existing_id,
                    inputs=inputs,
                    outputs=outputs,
                    metadata=metadata,
                )
            )

    if to_create:
        client.create_examples(dataset_id=dataset.id, examples=to_create)
    if to_update:
        client.update_examples(dataset_id=dataset.id, updates=to_update)

    return dataset


def _get_or_create_dataset(
    client: Client, name: str, description: str | None
) -> Dataset:
    if client.has_dataset(dataset_name=name):
        return client.read_dataset(dataset_name=name)
    return client.create_dataset(dataset_name=name, description=description)


def _example_payload(q: EvalQuery) -> tuple[dict, dict, dict]:
    inputs = {
        "query": q.query,
        "spoiler_scope": q.spoiler_scope,
        "series_filter": q.series_filter,
    }
    outputs = {
        "expected_answer": q.expected_answer,
        "relevant_chunk_ids": q.relevant_chunk_ids,
    }
    metadata = {"query_id": q.query_id, "notes": q.notes}
    return inputs, outputs, metadata
