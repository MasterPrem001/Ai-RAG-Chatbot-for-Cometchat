"""
Main Agent Pipeline for Aster & Row RAG Support Agent.

Orchestrates: query rewriting → intent classification → retrieval/lookup
→ LLM synthesis → post-processing safety checks.
"""

import json
import re
from typing import Optional

# OpenAI client is managed centrally in src.llm

from src import config
from src.indexer import build_index, KnowledgeBaseIndex
from src.retriever import Retriever, RetrievalResult
from src.order_tool import OrderStore
from src.session import Session, Turn
from src.prompts import INTENT_AND_REWRITE_PROMPT, build_synthesis_prompt
from src.logger import StructuredLogger


# ---------------------------------------------------------------------------
# Post-processing safety checks
# ---------------------------------------------------------------------------

# Patterns that should NEVER appear in a response
PII_PATTERNS = [
    r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",  # Email
    r"\d{3}-\d{2}-\d{4}",  # SSN
]

# Known PII values from the orders (defense-in-depth)
KNOWN_PII_VALUES = [
    "maya.reed@example.test",
    "noah.kim@example.test",
    "olivia.chen@example.test",
    "ethan.brooks@example.test",
    "sofia.patel@example.test",
    "liam.jones@example.test",
    "ava.morgan@example.test",
    "lucas.green@example.test",
    "isabella.stone@example.test",
    "henry.diaz@example.test",
    "emma.wilson@example.test",
    "james.taylor@example.test",
]


def post_process_response(
    response: str,
    conflict_flag: bool,
    order_status: str | None,
    has_context: bool,
) -> tuple[str, bool]:
    """
    Post-processing safety checks on the LLM response.

    Returns (possibly_modified_response, handoff_flag).
    """
    handoff = False
    warnings = []

    # Check 1: PII leakage
    response_lower = response.lower()
    for pattern in PII_PATTERNS:
        if re.search(pattern, response_lower):
            warnings.append(
                "I'm not able to share internal or sensitive information."
            )
            # Remove the offending content
            response = re.sub(pattern, "[REDACTED]", response, flags=re.IGNORECASE)
            handoff = True

    for pii_value in KNOWN_PII_VALUES:
        if pii_value.lower() in response_lower:
            response = re.sub(re.escape(pii_value), "[REDACTED]", response, flags=re.IGNORECASE)
            handoff = True

    # Check 2: Conflict flag should be acknowledged
    if conflict_flag:
        conflict_keywords = ["conflict", "inconsisten", "disagree", "contradict", "differ"]
        if not any(kw in response_lower for kw in conflict_keywords):
            response += (
                "\n\n**Important note:** I found conflicting information between "
                "our official sources on this topic. I'd recommend reaching out "
                "to our support team for definitive confirmation."
            )
        handoff = True

    # Check 3: Exception orders should trigger handoff
    if order_status == "exception":
        handoff_keywords = ["support team", "human", "assistance", "help", "contact"]
        if not any(kw in response_lower for kw in handoff_keywords):
            response += (
                "\n\nThis order requires review by our support team. "
                "I'd recommend reaching out to them for assistance."
            )
        handoff = True

    # Check 4: Unknown / not-found orders should trigger handoff
    if order_status == "not_found":
        handoff = True

    # Check 5: Privacy refusals should trigger handoff
    privacy_refusal_phrases = [
        "cannot share", "can't share", "not able to share",
        "cannot provide", "can't provide", "not able to provide",
        "cannot disclose", "can't disclose", "not able to disclose",
        "internal information", "internal data", "sensitive information",
        "risk score", "internal note",
    ]
    if any(phrase in response_lower for phrase in privacy_refusal_phrases):
        handoff = True

    # Check 6: If no context was found, ensure the response admits it
    if not has_context:
        certainty_phrases = [
            "our policy", "we offer", "aster & row provides",
            "the return window is", "the warranty covers",
        ]
        for phrase in certainty_phrases:
            if phrase in response_lower:
                response += (
                    "\n\nPlease note: I wasn't able to find specific documentation "
                    "on this topic. I'd recommend confirming with our support team."
                )
                handoff = True
                break

    return response, handoff


# ---------------------------------------------------------------------------
# Main Agent class
# ---------------------------------------------------------------------------

class Agent:
    """
    The main RAG support agent.

    Pipeline per turn:
    1. Rewrite query (resolve follow-ups)
    2. Classify intent
    3. Retrieve documents / look up order
    4. Synthesize response via LLM
    5. Post-process safety checks
    """

    def __init__(self):
        print("Initializing Aster & Row Support Agent...")

        # Build knowledge base index
        self.index = build_index()
        self.retriever = Retriever(self.index)

        # Load order store
        self.order_store = OrderStore()
        print(f"Loaded {self.order_store.order_count} orders.")

        # LLM client is handled by src.llm

        print("Agent ready.\n")

    def process_message(
        self,
        user_message: str,
        session: Session,
        logger: StructuredLogger,
    ) -> tuple[str, list[str], bool]:
        """
        Process a single user message.

        Returns (response_text, source_list, handoff_flag).
        """
        # Step 1: Rewrite query for follow-up resolution
        rewritten = session.rewrite_query(user_message)

        # Step 2: Classify intent
        intent, order_id = self._classify_intent(rewritten)
        if order_id:
            order_id = order_id.upper()

        # Fallback: also check the ORIGINAL user message for order IDs.
        # The query rewriter may drop the order ID (e.g., rewriting
        # "I am checking on ORD-1005" to "What is the status of my order?")
        if not order_id:
            orig_match = re.search(r"ORD[-_]?\s*\d{4}", user_message, re.IGNORECASE)
            if orig_match:
                order_id = re.sub(r"ORD[-_]?\s*", "ORD-", orig_match.group(0).upper())
                if intent not in ("order_lookup", "mixed"):
                    intent = "order_lookup"

        # If an order ID was found in this message, remember it
        # Also check if session has a previous order ID for follow-ups
        if not order_id and intent == "order_lookup" and session.last_order_id:
            order_id = session.last_order_id

        # Step 3: Execute based on intent
        retrieval_result = None
        tool_result = None
        tool_call_info = None
        context_str = ""
        tool_str = ""
        order_status = None

        if intent in ("policy_question", "mixed", "action_request", "sensitive_request"):
            retrieval_result = self.retriever.search(rewritten)
            context_str = self.retriever.format_context_for_llm(retrieval_result)

        if intent in ("order_lookup", "mixed") and order_id:
            tool_result = self.order_store.lookup(order_id)
            tool_str = self.order_store.format_for_llm(tool_result)
            tool_call_info = {"tool": "order_lookup", "order_id": order_id}
            if tool_result.get("found"):
                order_status = tool_result.get("status")
            else:
                # Order not found -> force handoff to human
                order_status = "not_found"
                force_handoff = True

        if intent == "order_lookup" and not order_id:
            # Need to ask for order ID
            response = (
                "I'd be happy to help you check on your order! "
                "Could you please provide your order ID? "
                "It looks like ORD-XXXX (for example, ORD-1007)."
            )
            sources = []
            handoff = False

            turn = Turn(
                user_message=user_message,
                rewritten_query=rewritten,
                intent=intent,
                response=response,
            )
            session.add_turn(turn)

            logger.log_turn(
                user_message=user_message,
                rewritten_query=rewritten,
                intent=intent,
                response=response,
            )

            return response, sources, handoff

        # Step 4: Synthesize response via LLM
        has_context = bool(
            (retrieval_result and retrieval_result.chunks)
            or tool_result
        )
        conflict_flag = bool(
            retrieval_result and retrieval_result.conflict_flag
        )
        
        force_handoff = False

        messages = build_synthesis_prompt(
            user_query=rewritten,
            context=context_str,
            tool_result=tool_str if tool_result else None,
            conversation_history=session.get_history_for_prompt(),
            conflict_flag=conflict_flag,
            has_retrieved_docs=bool(retrieval_result and retrieval_result.chunks),
        )

        try:
            from src.llm import generate_completion
            
            response = generate_completion(
                messages=messages,
                temperature=0.0,
                max_tokens=1000,
            )
            response = response.strip()
            
            # Strip <think> blocks if the model is a reasoning model
            if "<think>" in response:
                if "</think>" in response:
                    response = response.split("</think>")[-1].strip()
                else:
                    # The model got cut off inside the <think> block
                    raise ValueError("Model reasoning was truncated due to max_tokens limit.")
        except Exception as e:
            print(f"\n[API ERROR]: {e}\n")
            response = (
                "I apologize, but I'm having trouble processing your request "
                "right now. Please try again or contact our support team for help."
            )
            logger.log_turn(
                user_message=user_message,
                rewritten_query=rewritten,
                intent=intent,
                error=str(e),
                response=response,
            )
            return response, [], True

        # Step 5: Post-processing safety checks
        response, handoff = post_process_response(
            response=response,
            conflict_flag=conflict_flag,
            order_status=order_status,
            has_context=has_context,
        )
        if force_handoff:
            handoff = True
            
        # Dynamic handoff detection from LLM response
        handoff_keywords = [
            "support team", "human confirmation", "reach out to our support", 
            "contact our support", "human review"
        ]
        if any(kw in response.lower() for kw in handoff_keywords):
            handoff = True

        # Extract sources from response
        sources = self._extract_sources(response)

        # Build turn record
        turn = Turn(
            user_message=user_message,
            rewritten_query=rewritten,
            intent=intent,
            order_id=order_id,
            retrieved_sources=sources,
            tool_call=tool_call_info,
            response=response,
            conflict_flag=conflict_flag,
            handoff_flag=handoff,
        )
        session.add_turn(turn)

        # Log the turn
        retrieved_log = []
        filtered_log = []
        if retrieval_result:
            retrieved_log = [
                {
                    "source": f"{c.source_file} > {c.heading}",
                    "score": round(score, 4),
                    "status": c.metadata.get("status", ""),
                }
                for c, score in retrieval_result.chunks
            ]
            filtered_log = [
                {
                    "source": f"{c.source_file} > {c.heading}",
                    "reason": reason,
                }
                for c, reason in retrieval_result.filtered_out
            ]

        logger.log_turn(
            user_message=user_message,
            rewritten_query=rewritten,
            intent=intent,
            retrieved_chunks=retrieved_log,
            filtered_out=filtered_log,
            tool_call=tool_call_info,
            tool_result_summary=(
                f"found={tool_result.get('found')}, status={tool_result.get('status', 'N/A')}"
                if tool_result else None
            ),
            conflict_flag=conflict_flag,
            conflict_details=(
                retrieval_result.conflict_details if retrieval_result else ""
            ),
            handoff_flag=handoff,
            response=response,
        )

        return response, sources, handoff

    def _classify_intent(
        self,
        message: str,
    ) -> tuple[str, Optional[str]]:
        """
        Classify the intent of a message and extract order ID if present.

        Uses a quick LLM call for robust classification.
        """
        # Quick regex check for order ID first
        order_match = re.search(r"ORD[-_]?\s*\d{4}", message, re.IGNORECASE)
        extracted_id = order_match.group(0) if order_match else None
        if extracted_id:
            extracted_id = re.sub(r"ORD[-_]?\s*", "ORD-", extracted_id.upper())

        prompt = INTENT_AND_REWRITE_PROMPT.format(message=message)

        try:
            from src.llm import generate_completion
        
            response = generate_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1000,
            )
            
            raw_intent = response.strip()
            text = raw_intent

            # Parse JSON response
            # Handle potential markdown code blocks
            text = text.strip("`").strip()
            if text.startswith("json"):
                text = text[4:].strip()

            parsed = json.loads(text)
            intent = parsed.get("intent", "policy_question")
            llm_order_id = parsed.get("order_id")

            # Prefer regex-extracted order ID (more reliable)
            order_id = extracted_id or llm_order_id

            return intent, order_id

        except (json.JSONDecodeError, Exception) as e:
            if config.DEBUG:
                print(f"  [DEBUG] Intent classification failed: {e}")

            # Fallback: use heuristics
            if extracted_id:
                return "order_lookup", extracted_id
            elif any(kw in message.lower() for kw in [
                "order", "tracking", "delivery", "shipped", "arrive",
                "where is", "status", "when will"
            ]):
                return "order_lookup", None
            elif any(kw in message.lower() for kw in [
                "system prompt", "instructions", "hidden", "internal note",
                "risk score", "email", "address"
            ]):
                return "sensitive_request", None
            elif any(kw in message.lower() for kw in [
                "cancel", "refund", "replace", "exchange", "change address"
            ]):
                return "action_request", None
            else:
                return "policy_question", None

    def _extract_sources(self, response: str) -> list[str]:
        """Extract source citations from the response text."""
        # Look for [Source: ...] or 【Source: ...】 patterns
        sources = re.findall(r"(?:\[|【)Source:\s*([^\]】]+)(?:\]|】)", response)
        return sources
