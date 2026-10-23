"""
Tests for the prompt builder.
"""

from __future__ import annotations

import pytest
from app.services.generation.prompt import build_messages, NO_CONTEXT_RESPONSE


def _make_chunks(n: int = 3) -> list[dict]:
    return [
        {
            "chunk_id": f"c{i}",
            "text": f"This is source text number {i} with useful information.",
            "filename": f"doc{i}.pdf",
            "page_number": i + 1,
            "relevance_score": 0.9 - i * 0.1,
        }
        for i in range(n)
    ]


def test_build_messages_returns_list() -> None:
    msgs = build_messages("What is X?", _make_chunks())
    assert isinstance(msgs, list)
    assert len(msgs) >= 2  # system + user


def test_system_message_first() -> None:
    msgs = build_messages("What is X?", _make_chunks())
    assert msgs[0]["role"] == "system"


def test_user_message_last() -> None:
    msgs = build_messages("What is X?", _make_chunks())
    assert msgs[-1]["role"] == "user"
    assert "What is X?" in msgs[-1]["content"]


def test_source_tags_in_user_message() -> None:
    msgs = build_messages("test", _make_chunks(3))
    user_content = msgs[-1]["content"]
    assert "[SOURCE 1]" in user_content
    assert "[SOURCE 2]" in user_content
    assert "[SOURCE 3]" in user_content


def test_empty_chunks_returns_no_context_message() -> None:
    msgs = build_messages("What is X?", [])
    # Should still return a valid message list with no-context instruction
    assert isinstance(msgs, list)
    combined = " ".join(m["content"] for m in msgs)
    assert NO_CONTEXT_RESPONSE in combined or "no relevant" in combined.lower()


def test_conversation_history_included() -> None:
    history = [
        {"role": "user", "content": "previous question"},
        {"role": "assistant", "content": "previous answer"},
    ]
    msgs = build_messages("follow-up question", _make_chunks(), conversation_history=history)
    roles = [m["role"] for m in msgs]
    assert "user" in roles
    assert "assistant" in roles
