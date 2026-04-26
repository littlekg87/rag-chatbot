import streamlit as st
import uuid
from datetime import datetime, timezone, timedelta
import tempfile
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from supabase import create_client

st.set_page_config(page_title="RAG 챗봇", page_icon="🤖", layout="centered")
st.title("🤖 PDF RAG 챗봇")

KST = timezone(timedelta(hours=9))

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
        result = get_supabase().table("conversations")\
            .select("*").order("created_at", desc=True).limit(100).execute()
        return result.data
    except Exception as e:
        st.error(f"DB 조회 실패: {e}")
        return []

# ── 세션 초기화 ──────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chain" not in st.session_state:
    st.session_state.chain = None

# ── 탭 구성 ─────────────────────────────────────────────────
tab_chat, tab_history = st.tabs(["💬 챗봇", "📋 대화 기록"])

# ── 챗봇 탭 ─────────────────────────────────────────────────
with tab_chat:
    uploaded_file = st.file_uploader("📄 PDF 파일을 업로드하세요", type="pdf")

    if uploaded_file and st.session_state.chain is None:
        with st.spinner("PDF 처리 중..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            os.unlink(tmp_path)

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(docs)

            embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
            vectorstore = FAISS.from_documents(chunks, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            llm = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=st.secrets["OPENAI_API_KEY"],
                temperature=0
            )
            prompt = ChatPromptTemplate.from_template("""
다음 문서 내용을 바탕으로 질문에 답하세요. 문서에 없는 내용은 모른다고 하세요.

문서 내용:
{context}

질문: {question}
""")
            st.session_state.chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
        st.success("✅ PDF 준비 완료! 질문하세요.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if st.session_state.chain is None:
        st.info("PDF를 업로드하면 질문할 수 있습니다.")
    else:
        if question := st.chat_input("질문을 입력하세요"):
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("답변 생성 중..."):
                    answer = st.session_state.chain.invoke(question)
                st.write(answer)

            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            save_to_db(st.session_state.session_id, question, answer)

# ── 대화 기록 탭 ─────────────────────────────────────────────
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
            # KST 변환
            raw_time = row["created_at"]
            try:
                dt = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
                kst_time = dt.astimezone(KST).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                kst_time = raw_time

            with st.expander(f"🕐 {kst_time}  |  Q: {row['question'][:50]}{'...' if len(row['question']) > 50 else ''}"):
                st.markdown(f"**질문**\n\n{row['question']}")
                st.divider()
                st.markdown(f"**답변**\n\n{row['answer']}")
                st.caption(f"세션 ID: {row['session_id'][:8]}...")
