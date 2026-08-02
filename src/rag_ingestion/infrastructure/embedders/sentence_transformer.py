from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast


from rag_ingestion.config import Settings
from rag_ingestion.domain.entities.chunk import Chunk, EmbeddedChunk
from rag_ingestion.domain.protocols.embedder import Embedder



class SentenceTransformerEmbedder(Embedder):
    """
    Embed text chunks using Sentence Transformers.

    The model is loaded lazily on first use so that the application can be
    imported even when the embedding dependencies are not yet installed.
    """

    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.embedding_model
        self._batch_size = settings.embedding_batch_size
        self._device = settings.embedding_device
        self._normalize = settings.embedding_normalize
        self._model: Any | None = None

    def embed(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        if not chunks:
            return []

        try:
            model = self._get_model()
            embeddings = self._embed_chunks(model, chunks)
        except RuntimeError:
            raise
        except Exception as exc:  # pragma: no cover - defensive guard for unexpected failures
            raise RuntimeError(
                f"SentenceTransformerEmbedder failed to embed {len(chunks)} chunk(s) "
                f"with model {self._model_name!r}."
            ) from exc

        if len(embeddings) != len(chunks):
            raise RuntimeError(
                f"SentenceTransformerEmbedder produced {len(embeddings)} embeddings for "
                f"{len(chunks)} chunk(s) with model {self._model_name!r}."
            )

        return [
            EmbeddedChunk(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = self._build_model()
        return self._model

    def _build_model(self) -> Any:
        try:
            from sentence_transformers import SentenceTransformer  
        except Exception as exc:  # pragma: no cover - exercised from runtime import failures
            raise RuntimeError(
                "SentenceTransformerEmbedder requires sentence-transformers to be installed."
            ) from exc

        try:
            model = SentenceTransformer(self._model_name, device=self._device)
            model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"SentenceTransformerEmbedder failed to load model {self._model_name!r} "
                f"on device {self._device!r}."
            ) from exc

        return model

    def _embed_chunks(self, model: Any, chunks: list[Chunk]) -> list[tuple[float, ...]]:
        embeddings: list[tuple[float, ...]] = []

        for batch_number, start_index in enumerate(range(0, len(chunks), self._batch_size), start=1):
            batch_chunks = chunks[start_index : start_index + self._batch_size]
            batch_embeddings = self._encode_batch(model, batch_chunks, batch_number)
            embeddings.extend(batch_embeddings)

        return embeddings

    def _encode_batch(
        self,
        model: Any,
        batch_chunks: list[Chunk],
        batch_number: int,
    ) -> list[tuple[float, ...]]:
        try:
            encoded_embeddings = model.encode(
                [chunk.text for chunk in batch_chunks],
                batch_size=self._batch_size,
                convert_to_numpy=False,
                normalize_embeddings=self._normalize,
                device=self._device,
            )
        except Exception as exc:
            raise RuntimeError(
                f"SentenceTransformerEmbedder failed while encoding batch {batch_number} "
                f"for model {self._model_name!r}."
            ) from exc

        batch_embeddings = self._coerce_batch_embeddings(encoded_embeddings, len(batch_chunks))
        if len(batch_embeddings) != len(batch_chunks):
            raise RuntimeError(
                f"SentenceTransformerEmbedder returned {len(batch_embeddings)} embeddings for "
                f"batch {batch_number} containing {len(batch_chunks)} chunk(s)."
            )

        return batch_embeddings

    @staticmethod
    def _coerce_batch_embeddings(embeddings: Iterable[object], expected_count: int) -> list[tuple[float, ...]]:
        values = SentenceTransformerEmbedder._materialize_embeddings(embeddings)

        if values and SentenceTransformerEmbedder._looks_like_vector(values[0]):
            return [SentenceTransformerEmbedder._coerce_vector(vector) for vector in values]

        if expected_count == 1:
            return [SentenceTransformerEmbedder._coerce_vector(values)]

        return [SentenceTransformerEmbedder._coerce_vector(values)]

    @staticmethod
    def _materialize_embeddings(embeddings: Iterable[object]) -> list[object]:
        try:
            return cast(list[object], list(embeddings))
        except TypeError as exc:
            raise RuntimeError("SentenceTransformerEmbedder received non-iterable embeddings.") from exc

    @staticmethod
    def _coerce_vector(vector: object) -> tuple[float, ...]:
        try:
            values = list(cast(Iterable[object], vector))
        except TypeError as exc:
            raise RuntimeError("SentenceTransformerEmbedder received a non-iterable vector.") from exc

        return tuple(float(cast(Any, value)) for value in values)

    @staticmethod
    def _looks_like_vector(value: object) -> bool:
        return isinstance(value, Iterable) and not isinstance(value, (str, bytes))
