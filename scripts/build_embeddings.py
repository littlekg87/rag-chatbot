from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CHUNKS_PATH = DATA_DIR / "meditations_chunks.jsonl"
EMBEDDINGS_PATH = DATA_DIR / "meditations_embeddings.npy"
META_PATH = DATA_DIR / "meditations_embeddings_meta.json"

DEFAULT_MODEL = "text-embedding-3-large"
DEFAULT_BATCH_SIZE = 64


def load_chunks(path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunks.append(json.loads(line))
    if not chunks:
        raise RuntimeError(f"No chunks found in {path}.")
    return chunks


def chunk_fingerprint(chunks: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk["id"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk["text"].encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return matrix / norms


def batched(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def embed_texts(
    texts: list[str],
    *,
    model: str,
    dimensions: int | None,
    batch_size: int,
) -> np.ndarray:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it before running this script, "
            "for example: $env:OPENAI_API_KEY='sk-...'"
        )

    client = OpenAI()
    vectors: list[list[float]] = []
    batches = batched(texts, batch_size)

    for index, batch in enumerate(batches, start=1):
        kwargs: dict[str, Any] = {
            "model": model,
            "input": batch,
            "encoding_format": "float",
        }
        if dimensions is not None:
            kwargs["dimensions"] = dimensions

        response = client.embeddings.create(**kwargs)
        vectors.extend(item.embedding for item in response.data)
        print(f"embedded batch {index}/{len(batches)} ({len(vectors)}/{len(texts)})")

    return np.asarray(vectors, dtype=np.float32)


def write_meta(
    *,
    path: Path,
    chunks: list[dict[str, Any]],
    model: str,
    dimensions: int,
    requested_dimensions: int | None,
    normalized: bool,
    fingerprint: str,
) -> None:
    meta = {
        "model": model,
        "dimensions": dimensions,
        "requested_dimensions": requested_dimensions,
        "normalized": normalized,
        "chunk_count": len(chunks),
        "chunk_fingerprint": fingerprint,
        "chunks_path": str(CHUNKS_PATH.relative_to(ROOT)).replace("\\", "/"),
        "embeddings_path": str(EMBEDDINGS_PATH.relative_to(ROOT)).replace("\\", "/"),
        "created_at_unix": int(time.time()),
        "ids": [chunk["id"] for chunk in chunks],
    }
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OpenAI embeddings for Meditations chunks.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI embedding model.")
    parser.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help="Optional shortened embedding dimension. Omit to use the model default.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if metadata appears to match the current chunk file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = load_chunks(CHUNKS_PATH)
    fingerprint = chunk_fingerprint(chunks)

    if not args.force and META_PATH.exists() and EMBEDDINGS_PATH.exists():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        if (
            meta.get("model") == args.model
            and meta.get("chunk_fingerprint") == fingerprint
            and meta.get("chunk_count") == len(chunks)
            and meta.get("requested_dimensions") == args.dimensions
        ):
            print("Embeddings are already up to date. Use --force to rebuild.")
            return

    texts = [chunk["text"].replace("\n", " ") for chunk in chunks]
    embeddings = embed_texts(
        texts,
        model=args.model,
        dimensions=args.dimensions,
        batch_size=args.batch_size,
    )
    embeddings = normalize_rows(embeddings)

    DATA_DIR.mkdir(exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    write_meta(
        path=META_PATH,
        chunks=chunks,
        model=args.model,
        dimensions=int(embeddings.shape[1]),
        requested_dimensions=args.dimensions,
        normalized=True,
        fingerprint=fingerprint,
    )

    print(f"wrote: {EMBEDDINGS_PATH.relative_to(ROOT)} {embeddings.shape}")
    print(f"wrote: {META_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
