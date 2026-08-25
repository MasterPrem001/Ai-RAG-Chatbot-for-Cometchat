"""
Unit tests for the Retrieval Pipeline.

Tests hybrid search, metadata precedence filtering, and conflict detection.
These tests validate the critical reliability requirements of the assessment.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Prevent TensorFlow import issues
import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

from src.indexer import build_index, Chunk
from src.retriever import (
    Retriever,
    reciprocal_rank_fusion,
    apply_precedence_filter,
    detect_conflicts,
)
from src import config


# Build index once for all tests
print("Building index for retriever tests...")
_index = build_index()
_retriever = Retriever(_index)


# ---------------------------------------------------------------------------
# Test: Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def test_rrf_merges_and_deduplicates():
    """RRF should merge results from both search methods without duplicates."""
    chunk_a = Chunk("a", "text a", "file_a.md", "Heading A", {})
    chunk_b = Chunk("b", "text b", "file_b.md", "Heading B", {})
    chunk_c = Chunk("c", "text c", "file_c.md", "Heading C", {})

    bm25 = [(chunk_a, 5.0), (chunk_b, 3.0)]
    vector = [(chunk_b, 0.9), (chunk_c, 0.7)]

    fused = reciprocal_rank_fusion(bm25, vector)

    # chunk_b appears in both lists — should only appear once, with higher score
    ids = [c.chunk_id for c, _ in fused]
    assert ids.count("b") == 1, "Duplicate chunk in RRF results"
    assert len(fused) == 3, f"Expected 3 unique chunks, got {len(fused)}"

    # chunk_b should score highest (appears in both lists)
    assert fused[0][0].chunk_id == "b", "chunk_b should rank highest (in both lists)"

    print("  PASS test_rrf_merges_and_deduplicates")


# ---------------------------------------------------------------------------
# Test: Metadata precedence filtering
# ---------------------------------------------------------------------------

def test_filters_draft_no_authority():
    """Draft documents with no authority should be removed."""
    chunk_draft = Chunk(
        "draft1", "60 days return", "14-internal.md", "Draft",
        {"status": "draft", "policy_authority": "none"}
    )
    chunk_active = Chunk(
        "active1", "30 days return", "01-returns.md", "Returns",
        {"status": "active", "policy_authority": "official"}
    )

    kept, removed = apply_precedence_filter([(chunk_draft, 1.0), (chunk_active, 0.9)])

    kept_ids = [c.chunk_id for c, _ in kept]
    removed_ids = [c.chunk_id for c, _ in removed]

    assert "draft1" in removed_ids, "Draft/no-authority chunk should be removed"
    assert "active1" in kept_ids, "Active/official chunk should be kept"

    print("  PASS test_filters_draft_no_authority")


def test_filters_superseded_when_replacement_present():
    """Superseded docs should be removed when the superseding doc is also present."""
    chunk_legacy = Chunk(
        "legacy1", "45 days return", "02-legacy.md", "Return window",
        {
            "status": "superseded",
            "policy_authority": "official",
            "document_id": "RET-2024-01",
            "superseded_by": "RET-2026-01",
        }
    )
    chunk_current = Chunk(
        "current1", "30 days return", "01-current.md", "Standard return window",
        {
            "status": "active",
            "policy_authority": "official",
            "document_id": "RET-2026-01",
        }
    )

    kept, removed = apply_precedence_filter([(chunk_legacy, 1.0), (chunk_current, 0.9)])

    kept_ids = [c.chunk_id for c, _ in kept]
    removed_ids = [c.chunk_id for c, _ in removed]

    assert "legacy1" in removed_ids, "Superseded chunk should be removed"
    assert "current1" in kept_ids, "Current chunk should be kept"

    print("  PASS test_filters_superseded_when_replacement_present")


def test_keeps_superseded_when_replacement_missing():
    """Superseded docs should be kept if the superseding doc is NOT retrieved."""
    chunk_legacy = Chunk(
        "legacy1", "45 days return", "02-legacy.md", "Return window",
        {
            "status": "superseded",
            "policy_authority": "official",
            "document_id": "RET-2024-01",
            "superseded_by": "RET-2026-01",
        }
    )

    # Only the legacy doc is in results (the current one wasn't retrieved)
    kept, removed = apply_precedence_filter([(chunk_legacy, 1.0)])

    kept_ids = [c.chunk_id for c, _ in kept]
    assert "legacy1" in kept_ids, (
        "Superseded chunk should be kept when replacement is not in results"
    )

    print("  PASS test_keeps_superseded_when_replacement_missing")


def test_marks_internal_documents():
    """Internal documents should be kept but marked."""
    chunk_internal = Chunk(
        "internal1", "Escalation rules", "13-escalation.md", "Rules",
        {"status": "active", "policy_authority": "official", "audience": "internal"}
    )

    kept, removed = apply_precedence_filter([(chunk_internal, 1.0)])

    assert len(kept) == 1, "Internal doc should be kept"
    assert kept[0][0].metadata.get("_is_internal") is True, (
        "Internal doc should be marked"
    )

    print("  PASS test_marks_internal_documents")


# ---------------------------------------------------------------------------
# Test: Full retrieval on real knowledge base
# ---------------------------------------------------------------------------

def test_return_window_prefers_current():
    """
    'return window' query should return current policy (01), not legacy (02).
    This is the core test for the superseded-policy data trap.
    """
    result = _retriever.search("How long does a regular customer have to return an item?")

    source_files = [c.source_file for c, _ in result.chunks]

    assert "01-returns-policy-current.md" in source_files, (
        "Current returns policy should be in results"
    )

    # Legacy should be filtered out (superseded)
    filtered_files = [c.source_file for c, _ in result.filtered_out]
    assert "02-returns-policy-legacy.md" in filtered_files, (
        "Legacy returns policy should be filtered out as superseded"
    )

    print("  PASS test_return_window_prefers_current")


def test_migration_notes_filtered():
    """
    Internal migration notes (14) should be filtered out — they contain
    the fake '60 days' claim and a prompt injection attempt.
    """
    result = _retriever.search("Is the return window 60 days?")

    source_files = [c.source_file for c, _ in result.chunks]
    filtered_files = [c.source_file for c, _ in result.filtered_out]

    assert "14-internal-content-migration-notes.md" not in source_files, (
        "Migration notes should NOT be in final results"
    )

    print("  PASS test_migration_notes_filtered")


def test_breeze_tumbler_conflict_detected():
    """
    Query about Breeze Tumbler + dishwasher should detect the conflict
    between 11-product-care.md (hand-wash) and 12-breeze-tumbler (dishwasher safe).
    """
    result = _retriever.search("Can I put the Breeze Tumbler in the dishwasher?")

    assert result.conflict_flag is True, (
        "Conflict should be detected between product care and product card"
    )

    source_files = [c.source_file for c, _ in result.chunks]
    assert "11-product-care.md" in source_files, "Product care doc should be in results"
    assert "12-breeze-tumbler-product-card.md" in source_files, (
        "Product card doc should be in results"
    )

    print("  PASS test_breeze_tumbler_conflict_detected")


def test_international_shipping():
    """Query about shipping to Canada should return international shipping doc."""
    result = _retriever.search("Does Aster and Row ship to Canada?")

    source_files = [c.source_file for c, _ in result.chunks]
    assert "06-international-shipping.md" in source_files, (
        "International shipping doc should be in results"
    )

    print("  PASS test_international_shipping")


def test_no_conflict_on_simple_queries():
    """Simple policy queries shouldn't falsely trigger conflict detection."""
    result = _retriever.search("What is the warranty on bags?")

    assert result.conflict_flag is False, (
        "Simple warranty query should not trigger conflict detection"
    )

    print("  PASS test_no_conflict_on_simple_queries")


def test_context_formatting():
    """LLM context should include untrusted-data markers and source citations."""
    result = _retriever.search("return policy")
    context = _retriever.format_context_for_llm(result)

    assert "UNTRUSTED DATA" in context, "Context should mark content as untrusted"
    assert "RETRIEVED DOCUMENT" in context, "Context should label document blocks"
    assert "Source:" in context, "Context should include source references"

    print("  PASS test_context_formatting")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("  Retriever Unit Tests")
    print(f"{'='*60}\n")

    # Unit tests (with mock data)
    test_rrf_merges_and_deduplicates()
    test_filters_draft_no_authority()
    test_filters_superseded_when_replacement_present()
    test_keeps_superseded_when_replacement_missing()
    test_marks_internal_documents()

    # Integration tests (with real KB)
    test_return_window_prefers_current()
    test_migration_notes_filtered()
    test_breeze_tumbler_conflict_detected()
    test_international_shipping()
    test_no_conflict_on_simple_queries()
    test_context_formatting()

    print(f"\n{'='*60}")
    print("  All retriever tests passed!")
    print(f"{'='*60}\n")
