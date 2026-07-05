from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from langchain_openai import OpenAIEmbeddings


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CHUNKS_PATH = DATA_DIR / "meditations_chunks.jsonl"
EMBEDDINGS_PATH = DATA_DIR / "meditations_embeddings.npy"
EMBEDDINGS_META_PATH = DATA_DIR / "meditations_embeddings_meta.json"
NOTES_PATH = ROOT / "docs" / "stoic_interpretive_notes.md"


TOKEN_RE = re.compile(r"[a-z][a-z'-]{1,}", re.IGNORECASE)


@dataclass(frozen=True)
class SearchResult:
    chunk: dict[str, Any]
    vector_score: float
    bm25_score: float
    hybrid_score: float


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower().strip("'") for match in TOKEN_RE.finditer(text)]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    if math.isclose(min_score, max_score):
        return np.ones_like(scores) if max_score > 0 else np.zeros_like(scores)
    return (scores - min_score) / (max_score - min_score)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


class BM25Index:
    def __init__(self, documents: list[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(document) for document in documents]
        self.doc_lengths = np.asarray([len(tokens) for tokens in self.doc_tokens], dtype=np.float32)
        self.avg_doc_length = float(np.mean(self.doc_lengths)) if len(self.doc_lengths) else 0.0
        self.term_freqs: list[dict[str, int]] = []
        document_freqs: dict[str, int] = {}

        for tokens in self.doc_tokens:
            freqs: dict[str, int] = {}
            for token in tokens:
                freqs[token] = freqs.get(token, 0) + 1
            self.term_freqs.append(freqs)
            for token in freqs:
                document_freqs[token] = document_freqs.get(token, 0) + 1

        doc_count = len(self.doc_tokens)
        self.idf = {
            token: math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
            for token, df in document_freqs.items()
        }

    def score(self, query: str) -> np.ndarray:
        query_tokens = tokenize(query)
        scores = np.zeros(len(self.doc_tokens), dtype=np.float32)
        if not query_tokens or self.avg_doc_length == 0:
            return scores

        for index, freqs in enumerate(self.term_freqs):
            doc_length = self.doc_lengths[index]
            score = 0.0
            for token in query_tokens:
                if token not in freqs:
                    continue
                tf = freqs[token]
                idf = self.idf.get(token, 0.0)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                score += idf * (tf * (self.k1 + 1)) / denominator
            scores[index] = score
        return scores


class HybridRetriever:
    def __init__(
        self,
        *,
        chunks_path: Path = CHUNKS_PATH,
        embeddings_path: Path = EMBEDDINGS_PATH,
        meta_path: Path = EMBEDDINGS_META_PATH,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ) -> None:
        self.chunks = load_jsonl(chunks_path)
        self.embeddings = np.load(embeddings_path)
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

        if len(self.chunks) != self.embeddings.shape[0]:
            raise ValueError("Chunk count does not match embedding row count.")

        self.bm25 = BM25Index([chunk["text"] for chunk in self.chunks])
        self.embedding_model = self.meta.get("model", "text-embedding-3-large")
        requested_dimensions = self.meta.get("requested_dimensions")
        dimensions = requested_dimensions if requested_dimensions is not None else None
        self.query_embedder = OpenAIEmbeddings(model=self.embedding_model, dimensions=dimensions)

    def retrieve(
        self,
        *,
        semantic_query: str,
        keyword_query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        query_vector = np.asarray(self.query_embedder.embed_query(semantic_query), dtype=np.float32)
        query_vector = normalize_vector(query_vector)

        vector_scores = self.embeddings @ query_vector
        bm25_scores = self.bm25.score(keyword_query)

        vector_norm = normalize_scores(vector_scores)
        bm25_norm = normalize_scores(bm25_scores)
        hybrid_scores = self.vector_weight * vector_norm + self.bm25_weight * bm25_norm

        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        return [
            SearchResult(
                chunk=self.chunks[int(index)],
                vector_score=float(vector_scores[int(index)]),
                bm25_score=float(bm25_scores[int(index)]),
                hybrid_score=float(hybrid_scores[int(index)]),
            )
            for index in top_indices
        ]


def format_results_for_prompt(results: list[SearchResult]) -> str:
    parts: list[str] = []
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        parts.append(
            "\n".join(
                [
                    f"[{index}] {chunk['source']}",
                    f"id: {chunk['id']}",
                    f"hybrid_score: {result.hybrid_score:.4f}",
                    chunk["text"],
                ]
            )
        )
    return "\n\n".join(parts)


def load_concept_cards(path: Path = NOTES_PATH) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    cards: list[dict[str, str]] = []
    matches = list(re.finditer(r"^###\s+(.+)$", text, flags=re.MULTILINE))
    for index, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            cards.append({"title": title, "body": body})
    return cards


def select_concept_notes(query_payload: dict[str, Any], *, max_cards: int = 3) -> str:
    cards = load_concept_cards()
    if not cards:
        return ""

    query_text = " ".join(
        [
            str(query_payload.get("intent", "")),
            " ".join(query_payload.get("themes", []) or []),
            str(query_payload.get("semantic_query", "")),
            str(query_payload.get("keyword_query", "")),
        ]
    )
    bm25 = BM25Index([f"{card['title']}\n{card['body']}" for card in cards])
    scores = bm25.score(query_text)
    top_indices = np.argsort(scores)[::-1][:max_cards]

    selected = []
    for index in top_indices:
        if scores[int(index)] <= 0 and selected:
            continue
        card = cards[int(index)]
        selected.append(f"### {card['title']}\n{card['body']}")
    return "\n\n".join(selected)

