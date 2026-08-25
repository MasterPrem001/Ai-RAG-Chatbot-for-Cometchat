"""
Knowledge Base Indexer for Aster & Row RAG Support Agent.

Parses markdown files with YAML front matter, splits them into chunks
by heading, preserves metadata, and builds both vector (ChromaDB) and
lexical (BM25) indexes.
"""

# Prevent transformers from trying to import TensorFlow (avoids Keras conflict)
import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import re
import hashlib
from pathlib import Path
from typing import Optional

import yaml
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from src import config


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Chunk:
    """A single retrievable unit of knowledge-base content."""

    def __init__(
        self,
        chunk_id: str,
        text: str,
        source_file: str,
        heading: str,
        metadata: dict,
    ):
        self.chunk_id = chunk_id
        self.text = text
        self.source_file = source_file
        self.heading = heading
        self.metadata = metadata  # parsed YAML front matter

    def __repr__(self):
        return (
            f"Chunk(id={self.chunk_id!r}, "
            f"source={self.source_file!r}, "
            f"heading={self.heading!r}, "
            f"status={self.metadata.get('status')!r})"
        )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_front_matter(content: str) -> tuple[dict, str]:
    """
    Extract YAML front matter from markdown content.

    Returns (metadata_dict, body_text). If no front matter is found,
    returns an empty dict and the full content.
    """
    pattern = r"^---\s*\n(.*?)\n---\s*\n"
    match = re.match(pattern, content, re.DOTALL)
    if not match:
        return {}, content

    yaml_str = match.group(1)
    body = content[match.end():]
    try:
        metadata = yaml.safe_load(yaml_str) or {}
    except yaml.YAMLError:
        metadata = {}

    return metadata, body


def split_by_heading(body: str, level: str = "## ") -> list[tuple[str, str]]:
    """
    Split markdown body into sections at the given heading level.

    Returns a list of (heading_text, section_content) tuples.
    Content before the first heading is assigned heading "Introduction".
    """
    sections = []
    current_heading = "Introduction"
    current_lines = []

    for line in body.split("\n"):
        if line.startswith(level):
            # Save the previous section
            text = "\n".join(current_lines).strip()
            if text:
                sections.append((current_heading, text))
            # Start a new section
            current_heading = line.lstrip("#").strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    # Save the last section
    text = "\n".join(current_lines).strip()
    if text:
        sections.append((current_heading, text))

    return sections


def make_chunk_id(source_file: str, heading: str) -> str:
    """Create a deterministic, readable chunk ID."""
    slug = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")
    # Short hash for uniqueness if slug collides
    h = hashlib.md5(f"{source_file}:{heading}".encode()).hexdigest()[:6]
    return f"{Path(source_file).stem}_{slug}_{h}"


# ---------------------------------------------------------------------------
# Main indexing logic
# ---------------------------------------------------------------------------

def parse_knowledge_base(kb_dir: Path | None = None) -> list[Chunk]:
    """
    Parse all markdown files in the knowledge base directory.

    Returns a list of Chunk objects with metadata preserved.
    """
    if kb_dir is None:
        kb_dir = config.KNOWLEDGE_BASE_DIR

    chunks = []

    for md_file in sorted(kb_dir.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        metadata, body = parse_front_matter(content)

        # Add source file info to metadata
        metadata["source_file"] = md_file.name

        # Split into sections by ## headings
        sections = split_by_heading(body)

        for heading, section_text in sections:
            chunk_id = make_chunk_id(md_file.name, heading)

            # Build the chunk text: include the document title for context
            doc_title = metadata.get("title", md_file.stem)
            chunk_text = f"[Document: {doc_title}]\n\n{section_text}"

            chunk = Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                source_file=md_file.name,
                heading=heading,
                metadata=metadata.copy(),
            )
            chunks.append(chunk)

    return chunks


class KnowledgeBaseIndex:
    """
    Combined vector + lexical index over the knowledge base.

    Builds:
    1. ChromaDB collection with sentence-transformer embeddings
    2. BM25 index for lexical matching
    """

    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks: list[Chunk] = []
        self.chunk_map: dict[str, Chunk] = {}  # chunk_id -> Chunk
        self.bm25: BM25Okapi | None = None
        self.chroma_collection = None
        self.embedding_model: SentenceTransformer | None = None

        if chunks is not None:
            self.build(chunks)

    def build(self, chunks: list[Chunk]):
        """Build both vector and BM25 indexes from chunks."""
        self.chunks = chunks
        self.chunk_map = {c.chunk_id: c for c in chunks}

        self._build_bm25()
        self._build_chroma()

    def _build_bm25(self):
        """Build BM25 index over chunk texts."""
        tokenized = [self._tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

    def _build_chroma(self):
        """Build ChromaDB vector index with sentence-transformer embeddings."""
        # Load embedding model
        print(f"Loading embedding model: {config.EMBEDDING_MODEL}...")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)

        # Create in-memory Chroma client (no persistence needed for 14 docs)
        client = chromadb.Client()

        # Delete collection if it exists (re-indexing)
        try:
            client.delete_collection("knowledge_base")
        except Exception:
            pass

        self.chroma_collection = client.create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"},
        )

        # Generate embeddings and add to collection
        texts = [c.text for c in self.chunks]
        embeddings = self.embedding_model.encode(texts).tolist()

        # ChromaDB metadata must be flat strings/numbers/bools
        flat_metadatas = []
        for c in self.chunks:
            flat_meta = {
                "source_file": c.source_file,
                "heading": c.heading,
                "document_id": c.metadata.get("document_id", ""),
                "title": c.metadata.get("title", ""),
                "status": c.metadata.get("status", "unknown"),
                "effective_date": str(c.metadata.get("effective_date", "")),
                "audience": c.metadata.get("audience", ""),
                "policy_authority": c.metadata.get("policy_authority", ""),
                "supersedes": c.metadata.get("supersedes", ""),
                "superseded_by": c.metadata.get("superseded_by", ""),
            }
            flat_metadatas.append(flat_meta)

        self.chroma_collection.add(
            ids=[c.chunk_id for c in self.chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=flat_metadatas,
        )

        print(f"Indexed {len(self.chunks)} chunks into ChromaDB.")

    def search_vector(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        """Search using vector similarity. Returns (chunk, score) pairs."""
        if self.chroma_collection is None or self.embedding_model is None:
            return []

        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.chroma_collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, len(self.chunks)),
        )

        scored = []
        for chunk_id, distance in zip(results["ids"][0], results["distances"][0]):
            if chunk_id in self.chunk_map:
                # ChromaDB returns cosine distance; convert to similarity
                score = 1.0 - distance
                scored.append((self.chunk_map[chunk_id], score))

        return scored

    def search_bm25(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        """Search using BM25 lexical matching. Returns (chunk, score) pairs."""
        if self.bm25 is None:
            return []

        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # Get top-k indices
        indexed_scores = [(i, s) for i, s in enumerate(scores) if s > 0]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in indexed_scores[:top_k]:
            results.append((self.chunks[idx], score))

        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + lowercasing tokenizer for BM25."""
        # Remove punctuation, lowercase, split
        text = re.sub(r"[^\w\s]", " ", text.lower())
        return text.split()


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def build_index(kb_dir: Path | None = None) -> KnowledgeBaseIndex:
    """Parse the knowledge base and build the combined index."""
    chunks = parse_knowledge_base(kb_dir)
    print(f"Parsed {len(chunks)} chunks from knowledge base.")
    index = KnowledgeBaseIndex(chunks)
    return index


# ---------------------------------------------------------------------------
# CLI entry point for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building knowledge base index...")
    index = build_index()

    print(f"\n{'='*60}")
    print(f"Total chunks: {len(index.chunks)}")
    print(f"{'='*60}\n")

    # Print all chunks with metadata
    for chunk in index.chunks:
        status = chunk.metadata.get("status", "?")
        audience = chunk.metadata.get("audience", "?")
        authority = chunk.metadata.get("policy_authority", "?")
        print(
            f"  [{status:>10}] [{audience:>10}] [{authority:>10}]  "
            f"{chunk.source_file} > {chunk.heading}"
        )

    # Quick test search
    print(f"\n{'='*60}")
    print("Test search: 'return window'")
    print(f"{'='*60}\n")

    bm25_results = index.search_bm25("return window", top_k=5)
    print("BM25 results:")
    for chunk, score in bm25_results:
        print(f"  {score:.3f}  {chunk.source_file} > {chunk.heading}")

    vector_results = index.search_vector("return window", top_k=5)
    print("\nVector results:")
    for chunk, score in vector_results:
        print(f"  {score:.3f}  {chunk.source_file} > {chunk.heading}")
