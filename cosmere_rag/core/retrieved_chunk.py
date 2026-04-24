from pydantic import BaseModel, ConfigDict

from cosmere_rag.core.chunk import Chunk


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float             # cosine similarity, higher = better