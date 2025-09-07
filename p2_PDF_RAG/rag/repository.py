# rag/repository.py
from __future__ import annotations

from typing import Any
from uuid import uuid4

from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import ScoredPoint


class VectorRepository:
    def __init__(self, host: str = "localhost", port: int = 6333) -> None:
        # Initialize the repository with a connection to the Qdrant database
        self.db_client = AsyncQdrantClient(host=host, port=port)

    async def aclose(self) -> None:
        # Close the connection to the Qdrant database
        await self.db_client.close()

    async def create_collection(self, collection_name: str, size: int) -> bool:
        # Create a new collection in the Qdrant database if it doesn't already exist
        resp = await self.db_client.get_collections()
        exists = any(c.name == collection_name for c in resp.collections)

        if exists:
            logger.debug(f"Collection {collection_name} already exists; skipping create.")
            return True

        logger.debug(f"Creating collection {collection_name}")
        return await self.db_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
        )

    async def delete_collection(self, name: str) -> bool:
        # Delete a collection from the Qdrant database
        logger.debug(f"Deleting collection {name}")
        return await self.db_client.delete_collection(name)

    async def create(
        self,
        collection_name: str,
        embedding_vector: list[float],
        original_text: str,
        source: str,
        point_id: str | None = None,
        payload_extra: dict[str, Any] | None = None,
    ) -> None:
        # Insert or update a point in the specified collection with the given embedding vector and metadata
        pid = point_id or str(uuid4())
        payload = {
            "source": source,
            "original_text": original_text,
        }
        if payload_extra:
            payload.update(payload_extra)

        logger.debug(f"Upserting point {pid} into {collection_name}")
        await self.db_client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=pid,
                    vector=embedding_vector,
                    payload=payload,
                )
            ],
        )

    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        retrieval_limit: int = 5,
        score_threshold: float | None = None,
    ) -> list[ScoredPoint]:
        # Perform a vector similarity search in the specified collection
        logger.debug(f"Searching in {collection_name} (limit={retrieval_limit})")
        resp = await self.db_client.query_points(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=retrieval_limit,
            score_threshold=score_threshold,
        )
        return resp.points
