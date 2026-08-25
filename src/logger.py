"""
Structured observability logger for Aster & Row RAG Support Agent.

Writes one JSON line per turn to a log file. In debug mode, also
prints a human-readable trace to stderr.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src import config


class StructuredLogger:
    """
    JSON-lines logger for agent observability.

    Each turn produces one log entry with:
    - User message and rewritten query
    - Intent classification
    - Retrieved chunks with scores and sources
    - Tool calls and sanitized results
    - Conflict/handoff flags
    - Final response
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.turn_count = 0

        # Log file
        self.log_file = config.LOG_DIR / f"session_{session_id}.jsonl"

    def log_turn(
        self,
        user_message: str,
        rewritten_query: str | None = None,
        intent: str | None = None,
        retrieved_chunks: list[dict] | None = None,
        tool_call: dict | None = None,
        tool_result_summary: str | None = None,
        conflict_flag: bool = False,
        conflict_details: str = "",
        handoff_flag: bool = False,
        filtered_out: list[dict] | None = None,
        response: str = "",
        error: str | None = None,
    ):
        """Log a complete turn."""
        self.turn_count += 1

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "turn": self.turn_count,
            "user_message": user_message,
            "rewritten_query": rewritten_query,
            "intent": intent,
            "retrieved_chunks": retrieved_chunks or [],
            "filtered_out": filtered_out or [],
            "tool_call": tool_call,
            "tool_result_summary": tool_result_summary,
            "conflict_flag": conflict_flag,
            "conflict_details": conflict_details,
            "handoff_flag": handoff_flag,
            "response": response,
            "error": error,
        }

        # Scrub potential PII from user message before logging
        scrubbed_message = self._scrub_pii(entry.get("user_message", ""))
        entry["user_message"] = scrubbed_message

        # Write to log file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @staticmethod
    def _scrub_pii(text: str) -> str:
        """Remove common PII patterns from text before logging."""
        import re
        # Scrub email addresses
        text = re.sub(r"[\w.-]+@[\w.-]+\.\w+", "[REDACTED_EMAIL]", text)
        # Scrub phone numbers (US-style)
        text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[REDACTED_PHONE]", text)
        # Scrub SSN-like patterns
        text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]", text)
        # Scrub credit card-like patterns (13-19 digits)
        text = re.sub(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{1,7}\b", "[REDACTED_CC]", text)
        return text

        # Debug mode: print human-readable trace
        if config.DEBUG:
            self._print_debug(entry)

    def _print_debug(self, entry: dict):
        """Print a human-readable trace to stderr."""
        print(f"\n{'─'*60}", file=sys.stderr)
        print(f"  TRACE: Turn {entry['turn']}", file=sys.stderr)
        print(f"{'─'*60}", file=sys.stderr)
        print(f"  Message:  {entry['user_message']}", file=sys.stderr)

        if entry["rewritten_query"] and entry["rewritten_query"] != entry["user_message"]:
            print(f"  Rewritten: {entry['rewritten_query']}", file=sys.stderr)

        print(f"  Intent:   {entry['intent']}", file=sys.stderr)

        if entry["retrieved_chunks"]:
            print(f"  Retrieved ({len(entry['retrieved_chunks'])}):", file=sys.stderr)
            for chunk in entry["retrieved_chunks"]:
                print(f"    {chunk.get('score', '?'):.4f}  {chunk.get('source', '?')}", file=sys.stderr)

        if entry["filtered_out"]:
            print(f"  Filtered ({len(entry['filtered_out'])}):", file=sys.stderr)
            for chunk in entry["filtered_out"]:
                print(f"    x {chunk.get('source', '?')} - {chunk.get('reason', '?')}", file=sys.stderr)

        if entry["tool_call"]:
            print(f"  Tool:     {entry['tool_call']}", file=sys.stderr)
        if entry["tool_result_summary"]:
            print(f"  Tool Result: {entry['tool_result_summary']}", file=sys.stderr)

        if entry["conflict_flag"]:
            print(f"  CONFLICT: {entry['conflict_details']}", file=sys.stderr)
        if entry["handoff_flag"]:
            print(f"  HANDOFF:  Recommending human assistance", file=sys.stderr)
        if entry["error"]:
            print(f"  ERROR:    {entry['error']}", file=sys.stderr)

        print(f"{'─'*60}", file=sys.stderr)
