import streamlit as st
import uuid
from datetime import datetime

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from supabase import create_client

import tempfile
import os

# ── 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(page_title="RAG 챗봇", page_icon="🤖", layout="centered")
st.title("🤖 PDF RAG 챗봇")

# ── Supabase 연결 ────────────────────────────────────────────
@st.cache_resource
def get_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

def save_to_db(session_id, question, answer):
    try:
        supabase = get_supabase()
        supabase.table("conversations").insert({
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "created_at": datetime.utcnow().isoformat()
        }).execute()
    except Exception as e:
        st.warning(f"DB 저장 실패: {e}")

# ── 세션 초기화 ──────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chain" not in st.session_state:
    st.session_state.chain = None

# ── PDF 업로드 및 처리 ───────────────────────────────────────
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

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=st.secrets["OPENAI_API_KEY"],
            temperature=0
        )
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        st.session_state.chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=False
        )
    st.success("✅ PDF 준비 완료! 질문하세요.")

# ── 채팅 UI ──────────────────────────────────────────────────
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
                result = st.session_state.chain.invoke({"question": question})
                answer = result["answer"]
            st.write(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        save_to_db(st.session_state.session_id, question, answer)
