from __future__ import annotations

import html
import os
import re

import streamlit as st

from src.meditations_rag.chains import generate_answer, rewrite_query
from src.meditations_rag.retriever import (
    HybridRetriever,
    format_results_for_prompt,
    select_concept_notes,
)


st.set_page_config(page_title="Marcus Aurelius", page_icon="M", layout="wide")


@st.cache_resource(show_spinner=False)
def get_retriever() -> HybridRetriever:
    return HybridRetriever(vector_weight=0.7, bm25_weight=0.3)


def sync_openai_key_from_secrets() -> None:
    if os.getenv("OPENAI_API_KEY"):
        return
    try:
        key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        key = None
    if key:
        os.environ["OPENAI_API_KEY"] = str(key)


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&display=swap');

        :root {
          --ink-void: #0D1420;
          --ink-panel: #141C29;
          --ink-raise: #1D2736;
          --lumen-cyan: #4DF0D0;
          --lumen-lime: #C8F04D;
          --text-bright: #E8EDF2;
          --text-dim: #7A8494;
        }

        html, body, [data-testid="stAppViewContainer"], .stApp {
          background: var(--ink-void);
          color: var(--text-bright);
          font-family: 'Inter', system-ui, sans-serif;
        }

        .stApp::before {
          content: "";
          position: fixed;
          inset: 0;
          pointer-events: none;
          background-image:
            radial-gradient(circle at 12% 18%, rgba(232, 237, 242, 0.18) 0 1px, transparent 1.3px),
            radial-gradient(circle at 28% 72%, rgba(77, 240, 208, 0.24) 0 1px, transparent 1.4px),
            radial-gradient(circle at 46% 31%, rgba(232, 237, 242, 0.18) 0 1px, transparent 1.2px),
            radial-gradient(circle at 67% 62%, rgba(232, 237, 242, 0.22) 0 1px, transparent 1.3px),
            radial-gradient(circle at 82% 24%, rgba(77, 240, 208, 0.2) 0 1px, transparent 1.3px),
            radial-gradient(circle at 91% 78%, rgba(232, 237, 242, 0.18) 0 1px, transparent 1.2px),
            linear-gradient(180deg, rgba(13, 20, 32, 0.94), rgba(8, 13, 21, 0.98));
          background-size: 360px 360px, 420px 420px, 520px 520px, 460px 460px, 580px 580px, 500px 500px, 100% 100%;
          background-position: 0 0, 40px 80px, 120px 20px, 20px 160px, 180px 60px, 80px 220px, 0 0;
          opacity: 0.9;
        }

        [data-testid="stHeader"] {
          background: transparent;
        }

        [data-testid="stToolbar"] {
          display: none;
        }

        .block-container {
          max-width: 980px;
          padding: 2.6rem 2rem 7.5rem;
        }

        section[data-testid="stSidebar"] {
          background: #0A1019;
          border-right: 1px solid rgba(77, 240, 208, 0.08);
        }

        section[data-testid="stSidebar"] * {
          color: var(--text-bright);
          font-family: 'Inter', system-ui, sans-serif;
        }

        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] [data-baseweb="input"],
        section[data-testid="stSidebar"] [data-baseweb="base-input"] {
          background: var(--ink-panel) !important;
          border-color: rgba(77, 240, 208, 0.16) !important;
          color: var(--text-bright) !important;
          border-radius: 8px !important;
          box-shadow: none !important;
          outline: none !important;
        }

        section[data-testid="stSidebar"] input:focus,
        section[data-testid="stSidebar"] [data-baseweb="input"]:focus-within,
        section[data-testid="stSidebar"] [data-baseweb="base-input"]:focus-within {
          border-color: rgba(77, 240, 208, 0.5) !important;
          box-shadow: 0 0 0 1px rgba(77, 240, 208, 0.1) !important;
        }

        section[data-testid="stSidebar"] [data-baseweb="input"] button,
        section[data-testid="stSidebar"] [data-baseweb="base-input"] button,
        section[data-testid="stSidebar"] [data-testid="stTextInput"] button {
          background: transparent !important;
          border: 0 !important;
          box-shadow: none !important;
        }

        section[data-testid="stSidebar"] [data-baseweb="input"] svg,
        section[data-testid="stSidebar"] [data-testid="stTextInput"] svg {
          color: var(--text-dim) !important;
          fill: var(--text-dim) !important;
        }

        .study-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin: 0 auto 3.4rem;
          border-bottom: 1px solid rgba(232, 237, 242, 0.06);
          padding-bottom: 1rem;
        }

        .study-name {
          font-family: 'Newsreader', Georgia, serif;
          font-size: clamp(2rem, 4vw, 3.3rem);
          line-height: 1;
          color: var(--text-bright);
          letter-spacing: 0;
        }

        .study-state {
          display: flex;
          align-items: center;
          gap: 0.55rem;
          color: var(--text-dim);
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.78rem;
          text-transform: uppercase;
        }

        .study-state::before {
          content: "";
          width: 0.5rem;
          height: 0.5rem;
          border-radius: 999px;
          background: var(--lumen-cyan);
          box-shadow: 0 0 18px rgba(77, 240, 208, 0.7);
        }

        .empty-study {
          margin: 8vh auto 0;
          max-width: 620px;
          color: var(--text-dim);
          font-family: 'Newsreader', Georgia, serif;
          font-size: clamp(1.4rem, 2.4vw, 2rem);
          line-height: 1.45;
        }

        .chat-flow {
          display: flex;
          flex-direction: column;
          gap: 1.35rem;
        }

        .chat-row {
          display: flex;
          width: 100%;
        }

        .chat-row.assistant {
          justify-content: flex-start;
        }

        .chat-row.user {
          justify-content: flex-end;
        }

        .bubble {
          position: relative;
          max-width: min(720px, 86%);
          border-radius: 8px;
          padding: 1rem 1.08rem;
          color: var(--text-bright);
          overflow-wrap: anywhere;
        }

        .assistant-bubble {
          background: rgba(17, 23, 33, 0.92);
          border: 1px solid rgba(232, 237, 242, 0.06);
          border-left: 1px solid rgba(77, 240, 208, 0.55);
          box-shadow: 0 18px 60px rgba(0, 0, 0, 0.18);
        }

        .assistant-bubble::before {
          content: "";
          position: absolute;
          left: -1px;
          top: 0;
          width: 1px;
          height: 100%;
          background: linear-gradient(180deg, transparent, var(--lumen-cyan), transparent);
          transform-origin: top;
          animation: lumen-draw 780ms ease-out both;
          box-shadow: 0 0 18px rgba(77, 240, 208, 0.8);
        }

        @keyframes lumen-draw {
          from { transform: scaleY(0); opacity: 0; }
          to { transform: scaleY(1); opacity: 1; }
        }

        .user-bubble {
          background: var(--ink-raise);
          border: 1px solid rgba(232, 237, 242, 0.06);
        }

        .speaker {
          margin-bottom: 0.48rem;
          color: var(--text-dim);
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.72rem;
          letter-spacing: 0;
        }

        .assistant-bubble .message-content {
          font-family: 'Newsreader', Georgia, serif;
          font-size: 1.13rem;
          line-height: 1.62;
        }

        .user-bubble .message-content {
          font-family: 'Inter', system-ui, sans-serif;
          font-size: 0.98rem;
          line-height: 1.58;
        }

        .message-content p {
          margin: 0 0 0.85rem;
        }

        .message-content p:last-child {
          margin-bottom: 0;
        }

        .message-content .sources {
          margin-top: 1rem;
          padding-top: 0.78rem;
          border-top: 1px solid rgba(77, 240, 208, 0.12);
          color: var(--text-dim);
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.78rem;
          line-height: 1.55;
        }

        .message-content strong {
          color: var(--lumen-cyan);
          font-weight: 500;
        }

        div[data-testid="stChatInput"] {
          background: linear-gradient(180deg, rgba(13, 20, 32, 0), var(--ink-void) 34%);
          padding-bottom: 1.2rem;
        }

        div[data-testid="stBottom"] {
          background: linear-gradient(180deg, rgba(13, 20, 32, 0), var(--ink-void) 30%) !important;
        }

        div[data-testid="stBottom"] > div,
        div[data-testid="stBottomBlockContainer"],
        div[data-testid="stChatInput"] > div,
        div[data-testid="stChatInput"] > div > div {
          background: transparent !important;
        }

        div[data-testid="stChatInput"] > div,
        div[data-testid="stChatInput"] > div > div,
        div[data-testid="stChatInput"] > div > div > div {
          border: 0 !important;
          border-color: transparent !important;
          outline: none !important;
          box-shadow: none !important;
        }

        div[data-testid="stChatInput"] textarea {
          background: var(--ink-panel);
          border: 1px solid rgba(77, 240, 208, 0.22) !important;
          color: var(--text-bright);
          border-radius: 8px;
          font-family: 'Inter', system-ui, sans-serif;
          outline: none !important;
          box-shadow: none !important;
        }

        div[data-testid="stChatInput"] textarea:focus {
          border-color: rgba(77, 240, 208, 0.62) !important;
          outline: none !important;
          box-shadow: 0 0 0 1px rgba(77, 240, 208, 0.12), 0 0 24px rgba(77, 240, 208, 0.06) !important;
        }

        div[data-testid="stChatInput"] textarea,
        div[data-testid="stChatInput"] textarea:focus,
        div[data-testid="stChatInput"] [data-baseweb="textarea"],
        div[data-testid="stChatInput"] [data-baseweb="textarea"]:focus-within {
          border-color: rgba(77, 240, 208, 0.22) !important;
          caret-color: var(--lumen-cyan);
        }

        div[data-testid="stChatInput"] [data-baseweb="textarea"] {
          background: var(--ink-panel) !important;
          border-radius: 8px !important;
          box-shadow: none !important;
          outline: none !important;
        }

        div[data-testid="stChatInput"]:focus-within > div,
        div[data-testid="stChatInput"]:focus-within > div > div,
        div[data-testid="stChatInput"]:focus-within > div > div > div {
          border: 0 !important;
          border-color: transparent !important;
          outline: none !important;
          box-shadow: none !important;
        }

        button[kind="primary"], .stButton button {
          background: var(--lumen-lime);
          color: #10140C;
          border: 0;
          border-radius: 8px;
          font-weight: 600;
        }

        .stExpander {
          background: rgba(17, 23, 33, 0.66);
          border-radius: 8px;
          border: 1px solid rgba(232, 237, 242, 0.06);
        }

        code, pre {
          font-family: 'JetBrains Mono', monospace !important;
        }

        .stAlert {
          background: rgba(17, 23, 33, 0.9);
          color: var(--text-bright);
          border: 1px solid rgba(200, 240, 77, 0.18);
        }

        @media (max-width: 760px) {
          .block-container {
            padding: 1.35rem 1rem 7rem;
          }

          .study-header {
            align-items: flex-start;
            gap: 0.55rem;
            margin-bottom: 2.4rem;
            padding-bottom: 0.85rem;
          }

          .study-name {
            font-size: 2.35rem;
            line-height: 0.96;
          }

          .study-state {
            font-size: 0.66rem;
            margin-top: 0.18rem;
          }

          .empty-study {
            margin-top: 5vh;
            max-width: 100%;
            font-size: 1.42rem;
            line-height: 1.5;
          }

          .chat-flow {
            gap: 1rem;
          }

          .bubble {
            max-width: 94%;
            padding: 0.9rem 0.95rem;
          }

          .chat-row.user .bubble {
            max-width: 88%;
          }

          .assistant-bubble .message-content {
            font-size: 1.04rem;
            line-height: 1.58;
          }

          .user-bubble .message-content {
            font-size: 0.94rem;
          }

          .message-content .sources {
            font-size: 0.72rem;
          }

          div[data-testid="stBottomBlockContainer"] {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
          }

          div[data-testid="stChatInput"] {
            padding-bottom: 0.9rem;
          }

          div[data-testid="stChatInput"] textarea {
            min-height: 2.55rem !important;
            font-size: 0.95rem !important;
          }
        }

        @media (max-width: 430px) {
          .block-container {
            padding-left: 0.85rem;
            padding-right: 0.85rem;
          }

          .study-header {
            flex-direction: column;
          }

          .study-name {
            font-size: 2.08rem;
          }

          .bubble,
          .chat-row.user .bubble {
            max-width: 100%;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="study-header">
          <div class="study-name">Marcus Aurelius</div>
          <div class="study-state">night study</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_message_content(content: str) -> str:
    safe = html.escape(content.strip())
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    blocks = re.split(r"\n\s*\n", safe)
    rendered: list[str] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        block = block.replace("\n", "<br>")
        css_class = "sources" if block.startswith("관련 구절") else ""
        rendered.append(f'<p class="{css_class}">{block}</p>')
    return "".join(rendered)


def render_message(role: str, content: str) -> None:
    is_assistant = role == "assistant"
    row_class = "assistant" if is_assistant else "user"
    bubble_class = "assistant-bubble" if is_assistant else "user-bubble"
    speaker = "Marcus" if is_assistant else "You"
    st.markdown(
        f"""
        <div class="chat-row {row_class}">
          <div class="bubble {bubble_class}">
            <div class="speaker">{speaker}</div>
            <div class="message-content">{format_message_content(content)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(result, index: int) -> None:
    chunk = result.chunk
    with st.expander(f"{index}. {chunk['source']} · score {result.hybrid_score:.3f}"):
        st.caption(
            f"id={chunk['id']} | vector={result.vector_score:.3f} | bm25={result.bm25_score:.3f}"
        )
        st.write(chunk["text"])


def format_conversation_context(messages: list[dict[str, object]], *, limit: int = 6) -> str:
    recent = messages[-limit:]
    lines: list[str] = []
    for message in recent:
        role = "User" if message.get("role") == "user" else "Marcus"
        content = str(message.get("content", "")).strip()
        content = re.sub(r"\s+", " ", content)
        if len(content) > 900:
            content = content[:900].rstrip() + "..."
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


sync_openai_key_from_secrets()
apply_theme()
render_header()

with st.sidebar:
    st.subheader("설정")
    show_debug = st.toggle("검색 로그", value=False)
    api_key_input = st.text_input(
        "OpenAI API key",
        type="password",
        placeholder="sk-...",
        help="환경 변수 OPENAI_API_KEY가 없을 때만 입력하세요. 이 로컬 세션에만 사용됩니다.",
    )
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input
    st.divider()
    st.caption("rewrite · gpt-5.5")
    st.caption("answer · gpt-5.5")
    st.caption("embedding · text-embedding-3-large")
    st.caption("retrieval · hybrid top-5")

if not os.getenv("OPENAI_API_KEY"):
    st.warning("OPENAI_API_KEY가 설정되어 있어야 대화할 수 있습니다.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-study">
          마음에 가장 오래 머문 생각 하나를 가져오라.
          우리는 그것을 사건과 판단과 행동으로 조용히 나누어 볼 것이다.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="chat-flow">', unsafe_allow_html=True)
for message in st.session_state.messages:
    render_message(message["role"], message["content"])
    if message["role"] == "assistant" and show_debug and message.get("results"):
        for idx, result in enumerate(message["results"], start=1):
            render_result(result, idx)
        if message.get("query_payload"):
            with st.expander("Rewrite payload"):
                st.json(message["query_payload"])
st.markdown("</div>", unsafe_allow_html=True)

user_query = st.chat_input("지금 마음에 걸리는 것을 적어보세요")

if user_query:
    conversation_context = format_conversation_context(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": user_query})
    render_message("user", user_query)

    with st.status("관련 구절을 찾고 있습니다...", expanded=False) as status:
        query_payload = rewrite_query(user_query, conversation_context=conversation_context)
        retriever = get_retriever()
        results = retriever.retrieve(
            semantic_query=query_payload["semantic_query"],
            keyword_query=query_payload["keyword_query"],
            top_k=5,
        )
        retrieved_passages = format_results_for_prompt(results)
        concept_notes = select_concept_notes(query_payload, max_cards=3)
        status.update(label="답변을 고르고 있습니다...", state="running")
        answer = generate_answer(
            user_query=user_query,
            conversation_context=conversation_context,
            query_payload=query_payload,
            retrieved_passages=retrieved_passages,
            concept_notes=concept_notes,
        )
        status.update(label="완료", state="complete")

    render_message("assistant", answer)
    if show_debug:
        for idx, result in enumerate(results, start=1):
            render_result(result, idx)
        with st.expander("Rewrite payload"):
            st.json(query_payload)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "results": results,
            "query_payload": query_payload,
        }
    )
