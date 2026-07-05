from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


DEFAULT_REWRITE_MODEL = "gpt-5.5"
DEFAULT_ANSWER_MODEL = "gpt-5.5"

ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = ROOT / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", load_prompt("rewrite_system_prompt.md")),
        (
            "human",
            """RECENT CONVERSATION:
{conversation_context}

LATEST USER MESSAGE:
{user_query}
""",
        ),
    ]
)


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", load_prompt("answer_system_prompt.md")),
        (
            "human",
            """USER QUESTION:
{user_query}

RECENT CONVERSATION:
{conversation_context}

REWRITTEN SEARCH PAYLOAD:
{query_payload}

RETRIEVED MEDITATIONS PASSAGES:
{retrieved_passages}

STOIC INTERPRETIVE NOTES:
{concept_notes}

TASK:
Answer as a Marcus-like conversation partner, not as a lecturer.
Continue naturally from RECENT CONVERSATION when it is relevant.
Keep the answer grounded in the retrieved Meditations.
The answer must be in Korean.
Use the required section structure: 사건, 판단, 행동, 로마 황제의 한 마디, 관련 구절.
Under "로마 황제의 한 마디", write exactly one short warm conversational sentence that summarizes the answer.
Mention 1-2 relevant Meditations passages at the end under "관련 구절".
Each cited passage must include a one-line Korean translated quote.
Use only sources shown in RETRIEVED MEDITATIONS PASSAGES.
""",
        ),
    ]
)


def make_llm(model: str, *, reasoning_effort: str = "low", temperature: float | None = None) -> ChatOpenAI:
    kwargs: dict[str, Any] = {
        "model": model,
        "reasoning_effort": reasoning_effort,
        "use_responses_api": True,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    return ChatOpenAI(**kwargs)


def rewrite_query(user_query: str, *, conversation_context: str = "") -> dict[str, Any]:
    model = os.getenv("MARCUS_REWRITE_MODEL", DEFAULT_REWRITE_MODEL)
    chain = REWRITE_PROMPT | make_llm(model, reasoning_effort="low") | JsonOutputParser()
    try:
        payload = chain.invoke(
            {
                "user_query": user_query,
                "conversation_context": conversation_context or "(none)",
            }
        )
    except Exception:
        payload = {
            "intent": "User asks for Stoic guidance related to their concern.",
            "themes": ["judgement", "virtue", "self-command"],
            "semantic_query": user_query,
            "keyword_query": user_query,
        }

    required = ["intent", "themes", "semantic_query", "keyword_query"]
    for key in required:
        if key not in payload:
            payload[key] = [] if key == "themes" else user_query
    return payload


def generate_answer(
    *,
    user_query: str,
    conversation_context: str,
    query_payload: dict[str, Any],
    retrieved_passages: str,
    concept_notes: str,
) -> str:
    model = os.getenv("MARCUS_ANSWER_MODEL", DEFAULT_ANSWER_MODEL)
    chain = ANSWER_PROMPT | make_llm(model, reasoning_effort="medium") | StrOutputParser()
    return chain.invoke(
        {
            "user_query": user_query,
            "conversation_context": conversation_context or "(none)",
            "query_payload": json.dumps(query_payload, ensure_ascii=False, indent=2),
            "retrieved_passages": retrieved_passages,
            "concept_notes": concept_notes,
        }
    )
