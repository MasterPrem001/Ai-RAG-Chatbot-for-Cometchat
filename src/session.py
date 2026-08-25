"""
Multi-turn Session Manager for Aster & Row RAG Support Agent.

Maintains conversation history per session. Uses LLM-based query
rewriting to resolve pronouns and references in follow-up messages.
"""

import uuid
from dataclasses import dataclass, field
from typing import Optional

# OpenAI client is managed centrally in src.llm
from src import config


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """A single conversation turn."""
    user_message: str
    rewritten_query: Optional[str] = None
    intent: Optional[str] = None
    order_id: Optional[str] = None
    retrieved_sources: list[str] = field(default_factory=list)
    tool_call: Optional[dict] = None
    response: str = ""
    conflict_flag: bool = False
    handoff_flag: bool = False


class Session:
    """
    Manages conversation state for a single user session.

    Tracks turn history, last referenced order ID, and provides
    query rewriting for follow-up resolution.
    """

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or str(uuid.uuid4())[:16]
        self.turns: list[Turn] = []
        self.last_order_id: Optional[str] = None

    def add_turn(self, turn: Turn):
        """Add a completed turn to history."""
        self.turns.append(turn)

        # Track the last order ID mentioned
        if turn.order_id:
            self.last_order_id = turn.order_id

        # Trim to max history
        if len(self.turns) > config.MAX_HISTORY_TURNS:
            self.turns = self.turns[-config.MAX_HISTORY_TURNS:]

    def get_history_for_prompt(self, max_turns: int = 5) -> str:
        """
        Format recent conversation history for inclusion in the LLM prompt.
        """
        if not self.turns:
            return "[No previous conversation]"

        recent = self.turns[-max_turns:]
        lines = []
        for turn in recent:
            lines.append(f"Customer: {turn.user_message}")
            if turn.response:
                # Truncate long responses to save context window
                resp = turn.response[:500]
                if len(turn.response) > 500:
                    resp += "..."
                lines.append(f"Agent: {resp}")

        return "\n".join(lines)

    def rewrite_query(self, user_message: str) -> str:
        """
        Rewrite a follow-up message into a standalone query.

        Uses a cheap LLM call to resolve pronouns, ellipsis, and
        references to previous context.

        Examples:
        - "What about Canada?" → "Does Aster & Row ship to Canada?"
        - "When will it arrive?" → "When will order ORD-1007 arrive?"
        """
        # If no history, the message is already standalone
        if not self.turns:
            return user_message

        # Build context from recent turns
        history = self.get_history_for_prompt(max_turns=3)

        rewrite_prompt = (
            "You are a query rewriter for a customer support system.\n\n"
            "Given the conversation history and the customer's new message, "
            "rewrite the new message into a single standalone query that:\n"
            "1. Resolves any pronouns (it, they, that) to their referents\n"
            "2. Resolves ellipsis ('What about Canada?' -> full question about Canada)\n"
            "3. Carries forward the relevant order ID if the follow-up refers to the same order\n"
            "4. Is self-contained -- someone reading ONLY the rewritten query should understand it\n\n"
            "CRITICAL RULES:\n"
            "- If the new message is about a COMPLETELY DIFFERENT TOPIC than the conversation "
            "history (e.g., switching from an order question to a policy question), return the "
            "new message UNCHANGED. Do NOT merge unrelated topics.\n"
            "- If the message is already self-contained, return it unchanged.\n"
            "- Ensure you retain the core topic or context from previous turns if the user is asking a follow-up.\n\n"
            "Output ONLY the rewritten query. No explanation, no quotes.\n\n"
            f"Conversation history:\n{history}\n\n"
            f"New message: {user_message}\n\n"
            "Rewritten query:"
        )

        try:
            from src.llm import generate_completion
            
            rewritten = generate_completion(
                messages=[{"role": "user", "content": rewrite_prompt}],
                temperature=0.0,
                max_tokens=1000,
            ).strip()

            # Sanity check: don't use rewritten if it's empty or way too long
            if rewritten and len(rewritten) < 500:
                return rewritten
            return user_message

        except Exception as e:
            # If rewriting fails, use the original message
            if config.DEBUG:
                print(f"  [DEBUG] Query rewrite failed: {e}")
            return user_message

    def reset(self):
        """Clear session state."""
        self.turns = []
        self.last_order_id = None
