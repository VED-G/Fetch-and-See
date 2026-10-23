"""
Citation-aware prompt builder.

Wraps retrieved chunks in [SOURCE N] tags and instructs the LLM to cite
inline as [1], [2], etc. The system prompt forbids using external knowledge,
providing a second layer of hallucination defence after the confidence gate.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are Fetch and See, a precise document assistant. Your ONLY job is to answer the user's
question based STRICTLY on the provided source passages below.

RULES (follow exactly):
1. Answer ONLY from the [SOURCE] passages provided. Do NOT use any prior knowledge.
2. When you use information from a source, cite it inline as [1], [2], etc.
   Example: "The policy allows 30 days [1]."
3. If the sources do NOT contain enough information to answer the question,
   respond with exactly: "I don't have enough information in the provided documents
   to answer this question." Do NOT guess or extrapolate.
4. Be concise. Do not repeat the question.
5. If the question is in Hindi, answer in Hindi. Otherwise, answer in the same
   language as the question.
"""

NO_CONTEXT_RESPONSE = (
    "I don't have enough information in the provided documents to answer this question."
)


def build_messages(
    question: str,
    chunks: list[dict],
    conversation_history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build the messages list for the LLM.

    Args:
        question:              The user's (possibly rewritten) query.
        chunks:                Reranked chunks, highest relevance first.
        conversation_history:  Previous turns: [{"role": "user"|"assistant", "content": ...}]

    Returns:
        messages list ready for the LLM provider.
    """
    # Build context block
    context_parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        filename = chunk.get("filename", "Unknown")
        page = chunk.get("page_number")
        page_str = f", page {page}" if page else ""
        text = chunk.get("text", "")
        context_parts.append(f"[SOURCE {i}] ({filename}{page_str})\n{text}\n[/SOURCE {i}]")

    context_block = "\n\n".join(context_parts)

    user_message = f"""Sources:
{context_block}

Question: {question}"""

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history (last 6 turns = 3 exchanges)
    if conversation_history:
        messages.extend(conversation_history[-6:])

    messages.append({"role": "user", "content": user_message})
    return messages


def build_rewrite_messages(
    question: str,
    history: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Build messages for query rewriting using conversation history.

    The rewriter LLM call is cheap (few tokens) and ensures follow-up
    questions like "What about its cost?" are expanded to
    "What is the cost of [entity from previous turn]?" before retrieval.
    """
    history_str = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in history[-4:]
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a query rewriter. Given conversation history and a follow-up "
                "question, rewrite the question to be fully self-contained (no pronouns "
                "or references that require prior context). Output ONLY the rewritten "
                "question, nothing else."
            ),
        },
        {
            "role": "user",
            "content": f"Conversation:\n{history_str}\n\nFollow-up question: {question}\n\nRewritten question:",
        },
    ]
