import streamlit as st
import uuid
import os
from datetime import datetime, timezone, timedelta

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from supabase import create_client

st.set_page_config(page_title="RAG 챗봇", page_icon="🤖", layout="centered")
st.title("🤖 논문 RAG 챗봇")

KST = timezone(timedelta(hours=9))
PDF_PATH = os.path.join(os.path.dirname(__file__), "thesis.pdf")

# ── Supabase ─────────────────────────────────────────────────
@st.cache_resource
def get_supabase():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

def save_to_db(session_id, question, answer):
    try:
        get_supabase().table("conversations").insert({
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "created_at": datetime.now(timezone.utc).isoformat()
        }).execute()
    except Exception as e:
        st.warning(f"DB 저장 실패: {e}")

def load_history():
    try:
        return get_supabase().table("conversations")\
            .select("*").order("created_at", desc=True).limit(100).execute().data
    except Exception as e:
        st.error(f"DB 조회 실패: {e}")
        return []

# ── 벡터스토어 (앱 시작 시 1회만 빌드) ──────────────────────
@st.cache_resource
def build_chain():
    loader = PyPDFLoader(PDF_PATH)
    docs   = loader.load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).split_documents(docs)

    embeddings  = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm    = ChatOpenAI(model="gpt-4o-mini", api_key=st.secrets["OPENAI_API_KEY"], temperature=0)
    prompt = ChatPromptTemplate.from_template("""
다음 문서 내용을 바탕으로 질문에 답하세요. 문서에 없는 내용은 모른다고 하세요.

문서 내용:
{context}

질문: {question}
""")
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# ── 세션 초기화 ──────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── 탭 ──────────────────────────────────────────────────────
tab_chat, tab_history = st.tabs(["💬 챗봇", "📋 대화 기록"])

with tab_chat:
    with st.spinner("논문 로딩 중..."):
        chain = build_chain()

    st.success("✅ 준비 완료! 논문에 대해 질문하세요.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if question := st.chat_input("질문을 입력하세요"):
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                answer = chain.invoke(question)
            st.write(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        save_to_db(st.session_state.session_id, question, answer)

with tab_history:
    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader("전체 대화 기록")
    with col2:
        if st.button("🔄 새로고침"):
            st.rerun()

    records = load_history()
    if not records:
        st.info("아직 대화 기록이 없습니다.")
    else:
        st.caption(f"총 {len(records)}개의 대화")
        for row in records:
            try:
                dt = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")).astimezone(KST)
                kst_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                kst_time = row["created_at"]

            with st.expander(f"🕐 {kst_time}  |  Q: {row['question'][:50]}{'...' if len(row['question']) > 50 else ''}"):
                st.markdown(f"**질문**\n\n{row['question']}")
                st.divider()
                st.markdown(f"**답변**\n\n{row['answer']}")
                st.caption(f"세션 ID: {row['session_id'][:8]}...")
