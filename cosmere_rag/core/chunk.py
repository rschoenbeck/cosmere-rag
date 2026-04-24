from pydantic import BaseModel, ConfigDict
from datetime import datetime


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str            # stable hash of (article_title, heading_path, text)
    text: str
    article_title: str
    heading_path: list[str]
    spoiler_scope: str
    series_mentioned: list[str]
    source_url: str
    content_provenance: str
    corpus_snapshot: str
    ingested_at: datetime
    token_count: int