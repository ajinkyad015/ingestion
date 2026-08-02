from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rag_ingestion.config import Settings
from rag_ingestion.domain.entities.chunk import Chunk, ChunkMetadata, EmbeddedChunk
from rag_ingestion.domain.protocols.embedder import Embedder
from rag_ingestion.infrastructure.embedders.sentence_transformer import SentenceTransformerEmbedder


def _make_settings(
    *,
    model: str = "demo-model",
    batch_size: int = 2,
    device: str = "cpu",
    normalize: bool = True,
) -> Settings:
    return Settings(
        EMBEDDING_MODEL=model,
        EMBEDDING_BATCH_SIZE=batch_size,
        EMBEDDING_DEVICE=device,
        EMBEDDING_NORMALIZE=normalize,
    )


def _make_chunk(
    text: str,
    *,
    document_hash: str = "abc123",
    chunk_index: int = 0,
    page_number: int = 0,
) -> Chunk:
    metadata = ChunkMetadata(
        document_hash=document_hash,
        source_path=Path("/docs/report.pdf"),
        chunk_index=chunk_index,
        total_chunks=3,
        heading="Overview",
        page_number=page_number,
        char_start=0,
        char_end=len(text),
        source_file="report.pdf",
        section_hierarchy=("Overview",),
        title="Quarterly Report",
    )
    return Chunk(chunk_id=f"{document_hash}_{chunk_index:06d}", text=text, metadata=metadata)


def _make_model(encode_side_effect: object | None = None) -> MagicMock:
    model = MagicMock()
    model.eval.return_value = None
    if encode_side_effect is not None:
        model.encode.side_effect = encode_side_effect
    return model


class TestSentenceTransformerEmbedderProtocol:
    def test_protocol_conformance(self) -> None:
        embedder = SentenceTransformerEmbedder(_make_settings())

        assert isinstance(embedder, Embedder)


class TestSentenceTransformerEmbedderInitialization:
    def test_settings_are_stored(self) -> None:
        settings = _make_settings(model="demo-model", batch_size=4, device="cuda", normalize=False)
        embedder = SentenceTransformerEmbedder(settings)

        assert embedder._model_name == "demo-model"
        assert embedder._batch_size == 4
        assert embedder._device == "cuda"
        assert embedder._normalize is False

    def test_model_is_loaded_lazily(self) -> None:
        embedder = SentenceTransformerEmbedder(_make_settings())

        with patch("sentence_transformers.SentenceTransformer") as mock_sentence_transformer:
            assert embedder.embed([]) == []

        mock_sentence_transformer.assert_not_called()


class TestSentenceTransformerEmbedderEmbed:
    def test_empty_input_returns_empty_list_without_loading_model(self) -> None:
        embedder = SentenceTransformerEmbedder(_make_settings())

        with patch("sentence_transformers.SentenceTransformer") as mock_sentence_transformer:
            result = embedder.embed([])

        mock_sentence_transformer.assert_not_called()
        assert result == []

    def test_model_initialization_comes_from_settings(self) -> None:
        settings = _make_settings(model="custom-model", 
                                  batch_size=3, 
                                  device="cuda:0", 
                                  normalize=True
                                  )
        embedder = SentenceTransformerEmbedder(settings)
        chunks = [_make_chunk("hello")]
        model = _make_model([[0.1, 0.2, 0.3]])

        with patch("sentence_transformers.SentenceTransformer", return_value=model) as mock_cls:
            embedded = embedder.embed(chunks)

        mock_cls.assert_called_once_with("custom-model", device="cuda:0")
        model.eval.assert_called_once_with()
        model.encode.assert_called_once_with(
            ["hello"],
            batch_size=3,
            convert_to_numpy=False,
            normalize_embeddings=True,
            device="cuda:0",
        )
        assert embedded[0].chunk is chunks[0]
        assert embedded[0].embedding == (0.1, 0.2, 0.3)

    def test_batches_requests_and_preserves_order(self) -> None:
        embedder = SentenceTransformerEmbedder(_make_settings(batch_size=2, normalize=False))
        chunks = [
            _make_chunk("alpha", chunk_index=0),
            _make_chunk("beta", chunk_index=1),
            _make_chunk("gamma", chunk_index=2),
        ]
        model = _make_model(
            [
                [[0.1, 0.2], [0.3, 0.4]],
                [[0.5, 0.6]],
            ]
        )

        with patch("sentence_transformers.SentenceTransformer", return_value=model):
            embedded = embedder.embed(chunks)

        assert model.encode.call_count == 2
        assert model.encode.call_args_list[0].args[0] == ["alpha", "beta"]
        assert model.encode.call_args_list[1].args[0] == ["gamma"]
        assert [item.chunk.chunk_id for item in embedded] == [chunk.chunk_id for chunk in chunks]
        assert [item.embedding for item in embedded] == [
            (0.1, 0.2),
            (0.3, 0.4),
            (0.5, 0.6),
        ]

    @pytest.mark.parametrize("normalize", [True, False])
    def test_normalization_flag_is_forwarded(self, normalize: bool) -> None:
        embedder = SentenceTransformerEmbedder(_make_settings(normalize=normalize))
        model = _make_model([[1.0, 2.0]])
        chunks = [_make_chunk("normalized")]

        with patch("sentence_transformers.SentenceTransformer", return_value=model):
            embedder.embed(chunks)

        model.encode.assert_called_once_with(
            ["normalized"],
            batch_size=2,
            convert_to_numpy=False,
            normalize_embeddings=normalize,
            device="cpu",
        )

    def test_metadata_is_preserved_unchanged(self) -> None:
        embedder = SentenceTransformerEmbedder(_make_settings())
        chunks = [_make_chunk("one", chunk_index=0), _make_chunk("two", chunk_index=1)]
        model = _make_model([[[0.1, 0.2], [0.3, 0.4]]])

        with patch("sentence_transformers.SentenceTransformer", return_value=model):
            embedded = embedder.embed(chunks)

        assert all(isinstance(item, EmbeddedChunk) for item in embedded)
        for original, result in zip(chunks, embedded, strict=True):
            assert result.chunk is original
            assert result.chunk.text == original.text
            assert result.chunk.metadata is original.metadata
            assert result.chunk.metadata == original.metadata

    def test_model_is_cached_per_instance(self) -> None:
        embedder = SentenceTransformerEmbedder(_make_settings())
        model = _make_model()
        model.encode.return_value = [[0.1, 0.2]]
        chunks = [_make_chunk("first")]

        with patch("sentence_transformers.SentenceTransformer", return_value=model) as mock_cls:
            first = embedder.embed(chunks)
            second = embedder.embed(chunks)

        mock_cls.assert_called_once_with("demo-model", device="cpu")
        assert first == second
        assert model.encode.call_count == 2

    def test_load_failure_raises_runtime_error_with_context(self) -> None:
        embedder = SentenceTransformerEmbedder(_make_settings(model="broken-model", device="cpu"))

        with patch("sentence_transformers.SentenceTransformer", 
                   side_effect=ValueError("load failed")
                   ):
            with pytest.raises(RuntimeError,
                                match="failed to load model 'broken-model'.*'cpu'"
                                ) as excinfo:
                embedder.embed([_make_chunk("x")])

        assert isinstance(excinfo.value.__cause__, ValueError)

    def test_inference_failure_raises_runtime_error_with_context(self) -> None:
        embedder = SentenceTransformerEmbedder(_make_settings())
        model = _make_model()
        model.encode.side_effect = RuntimeError("encode failed")

        with patch("sentence_transformers.SentenceTransformer", return_value=model):
            with pytest.raises(RuntimeError, match="failed while encoding batch 1") as excinfo:
                embedder.embed([_make_chunk("x")])

        assert isinstance(excinfo.value.__cause__, RuntimeError)
