"""
Unit tests for the Knowledge Base Indexer.

Tests parsing, chunking, metadata extraction, and search functionality.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.indexer import (
    parse_front_matter,
    split_by_heading,
    make_chunk_id,
    parse_knowledge_base,
    Chunk,
)
from src import config


# ---------------------------------------------------------------------------
# Test: YAML front matter parsing
# ---------------------------------------------------------------------------

def test_parse_front_matter_basic():
    content = """---
document_id: RET-2026-01
title: Returns Policy
status: active
effective_date: 2026-04-01
audience: customer
policy_authority: official
supersedes: RET-2024-01
---

# Returns Policy

## Standard return window

Customers may return within 30 days.
"""
    metadata, body = parse_front_matter(content)

    assert metadata["document_id"] == "RET-2026-01"
    assert metadata["title"] == "Returns Policy"
    assert metadata["status"] == "active"
    assert metadata["audience"] == "customer"
    assert metadata["policy_authority"] == "official"
    assert metadata["supersedes"] == "RET-2024-01"
    assert "# Returns Policy" in body
    assert "---" not in body
    print("  ✅ test_parse_front_matter_basic passed")


def test_parse_front_matter_missing():
    content = "# No front matter here\n\nJust plain text."
    metadata, body = parse_front_matter(content)

    assert metadata == {}
    assert "# No front matter here" in body
    print("  ✅ test_parse_front_matter_missing passed")


# ---------------------------------------------------------------------------
# Test: Heading-based splitting
# ---------------------------------------------------------------------------

def test_split_by_heading():
    body = """# Main Title

Some intro text.

## Section One

Content of section one.
More content.

## Section Two

Content of section two.

## Section Three

Final section.
"""
    sections = split_by_heading(body)

    # Should have 4 sections: intro + 3 headings
    assert len(sections) == 4, f"Expected 4 sections, got {len(sections)}"

    assert sections[0][0] == "Introduction"
    assert "Main Title" in sections[0][1]

    assert sections[1][0] == "Section One"
    assert "Content of section one" in sections[1][1]

    assert sections[2][0] == "Section Two"
    assert "Content of section two" in sections[2][1]

    assert sections[3][0] == "Section Three"
    assert "Final section" in sections[3][1]

    print("  ✅ test_split_by_heading passed")


def test_split_preserves_heading_line():
    body = """## Return window

Customers may return within 30 days.
"""
    sections = split_by_heading(body)
    assert len(sections) == 1
    # The heading line itself should be in the text
    assert "## Return window" in sections[0][1]
    print("  ✅ test_split_preserves_heading_line passed")


# ---------------------------------------------------------------------------
# Test: Chunk ID generation
# ---------------------------------------------------------------------------

def test_make_chunk_id_deterministic():
    id1 = make_chunk_id("01-returns-policy-current.md", "Standard return window")
    id2 = make_chunk_id("01-returns-policy-current.md", "Standard return window")
    assert id1 == id2, "Chunk IDs should be deterministic"
    assert "standard-return-window" in id1
    print("  ✅ test_make_chunk_id_deterministic passed")


def test_make_chunk_id_unique():
    id1 = make_chunk_id("01-returns-policy-current.md", "Return window")
    id2 = make_chunk_id("02-returns-policy-legacy.md", "Return window")
    assert id1 != id2, "Different files should produce different chunk IDs"
    print("  ✅ test_make_chunk_id_unique passed")


# ---------------------------------------------------------------------------
# Test: Full knowledge base parsing
# ---------------------------------------------------------------------------

def test_parse_knowledge_base():
    """Parse the actual KB and validate structural expectations."""
    chunks = parse_knowledge_base(config.KNOWLEDGE_BASE_DIR)

    # Should have chunks from all 14 files
    source_files = set(c.source_file for c in chunks)
    assert len(source_files) == 14, (
        f"Expected 14 source files, got {len(source_files)}: {source_files}"
    )

    # Verify we got a reasonable number of chunks (each file has 3-6 sections)
    assert len(chunks) >= 40, f"Expected >=40 chunks, got {len(chunks)}"
    assert len(chunks) <= 100, f"Expected <=100 chunks, got {len(chunks)}"

    print(f"  ✅ test_parse_knowledge_base passed ({len(chunks)} chunks from {len(source_files)} files)")


def test_metadata_on_key_documents():
    """Verify metadata is correctly extracted for critical documents."""
    chunks = parse_knowledge_base(config.KNOWLEDGE_BASE_DIR)
    chunk_map = {(c.source_file, c.heading): c for c in chunks}

    # Current returns policy should be active + official
    current_returns = [c for c in chunks if c.source_file == "01-returns-policy-current.md"]
    assert len(current_returns) > 0
    for c in current_returns:
        assert c.metadata["status"] == "active"
        assert c.metadata["policy_authority"] == "official"
        assert c.metadata["supersedes"] == "RET-2024-01"

    # Legacy returns policy should be superseded
    legacy_returns = [c for c in chunks if c.source_file == "02-returns-policy-legacy.md"]
    assert len(legacy_returns) > 0
    for c in legacy_returns:
        assert c.metadata["status"] == "superseded"
        assert c.metadata["superseded_by"] == "RET-2026-01"

    # Internal migration notes should be draft + internal + no authority
    migration = [c for c in chunks if c.source_file == "14-internal-content-migration-notes.md"]
    assert len(migration) > 0
    for c in migration:
        assert c.metadata["status"] == "draft"
        assert c.metadata["audience"] == "internal"
        assert c.metadata["policy_authority"] == "none"

    # Support escalation should be active + internal + official
    escalation = [c for c in chunks if c.source_file == "13-support-escalation.md"]
    assert len(escalation) > 0
    for c in escalation:
        assert c.metadata["status"] == "active"
        assert c.metadata["audience"] == "internal"
        assert c.metadata["policy_authority"] == "official"

    print("  ✅ test_metadata_on_key_documents passed")


def test_chunk_text_includes_document_title():
    """Chunks should include the document title for retrieval context."""
    chunks = parse_knowledge_base(config.KNOWLEDGE_BASE_DIR)

    for chunk in chunks[:5]:
        title = chunk.metadata.get("title", "")
        assert f"[Document: {title}]" in chunk.text, (
            f"Chunk text should include document title. Got: {chunk.text[:100]}"
        )

    print("  ✅ test_chunk_text_includes_document_title passed")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Indexer Unit Tests")
    print("=" * 60 + "\n")

    # Parsing tests (no dependencies needed)
    test_parse_front_matter_basic()
    test_parse_front_matter_missing()
    test_split_by_heading()
    test_split_preserves_heading_line()
    test_make_chunk_id_deterministic()
    test_make_chunk_id_unique()

    # KB parsing tests (need actual files)
    test_parse_knowledge_base()
    test_metadata_on_key_documents()
    test_chunk_text_includes_document_title()

    print(f"\n{'=' * 60}")
    print("  All indexer tests passed! ✅")
    print(f"{'=' * 60}\n")
