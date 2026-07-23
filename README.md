# Marcus Aurelius RAG Chatbot

[한국어](./README.md) | [English](./README.en.md)

## 🏛️ 지금, 마르쿠스 아우렐리우스와 이야기해 보세요

마음에 걸리는 일을 적으면 《명상록》에서 관련 구절을 찾아 사건·판단·행동으로 차분하게 나누어 답해 주는 RAG 챗봇입니다.

### [👉 명상록 챗봇과 대화하러 가기](https://rag-chatbot-km9hgveen2rfuexsp777cf.streamlit.app/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rag-chatbot-km9hgveen2rfuexsp777cf.streamlit.app/)

한국어와 English를 모두 지원합니다. 심심할 때 들어와서 요즘 마음에 오래 머무는 생각 하나를 건네 보세요.

---

## 이 브랜치는 무엇인가요?

마르쿠스 아우렐리우스의 《명상록》을 바탕으로 답하는 Streamlit RAG 챗봇의 코드와 데이터를 관리합니다.

사용자의 질문을 검색에 적합한 영어 질의로 재작성한 뒤, 사전 생성된 임베딩과 BM25를 결합한 하이브리드 검색으로 관련 구절을 찾습니다. 검색 결과와 스토아 해석 노트를 근거로 대화형 답변을 생성하며, 화면과 답변은 한국어와 영어 사이에서 전환할 수 있습니다.

OpenAI API 키는 저장소에 포함하지 않으며 Streamlit Secrets를 통해 서버에서 관리합니다.

## 폴더 구조

```text
.
├─ app.py
├─ requirements.txt
├─ data/
├─ docs/
├─ prompts/
├─ scripts/
├─ src/
│  └─ meditations_rag/
└─ text/
```

## 파일과 폴더의 역할

| 경로 | 역할 |
|---|---|
| `app.py` | Streamlit 화면, 한·영 전환, 채팅 세션과 전체 RAG 호출 흐름 |
| `requirements.txt` | 앱 실행과 데이터 처리에 필요한 Python 패키지 목록 |
| `.gitignore` | API 키, 로그, 캐시와 로컬 가상환경 등 커밋 제외 대상 |
| `data/meditations.xml` | PDF에서 추출·정리한 《명상록》 구조화 원문 |
| `data/meditations_chunks.jsonl` | 검색에 사용할 구절 단위 청크 |
| `data/meditations_embeddings.npy` | 청크별 사전 생성 벡터 임베딩 |
| `data/meditations_embeddings_meta.json` | 임베딩 모델, 차원, 청크 ID 등 검증용 메타데이터 |
| `docs/marcus_response_style_guide.md` | 마르쿠스풍 답변의 말투와 응답 원칙 |
| `docs/stoic_interpretive_notes.md` | 검색 결과 해석을 보조하는 스토아 개념 노트 |
| `prompts/rewrite_system_prompt.md` | 사용자 질문을 명상록 검색 질의로 바꾸는 프롬프트 |
| `prompts/answer_system_prompt.md` | 한국어 답변 형식과 근거 규칙을 정의하는 프롬프트 |
| `prompts/answer_system_prompt_en.md` | 영어 답변 형식과 근거 규칙을 정의하는 프롬프트 |
| `scripts/build_meditations_corpus.py` | PDF/XML 원문을 검색용 청크로 만드는 스크립트 |
| `scripts/build_embeddings.py` | 청크 임베딩과 메타데이터를 생성하는 스크립트 |
| `src/meditations_rag/__init__.py` | Python 패키지 초기화 파일 |
| `src/meditations_rag/retriever.py` | 벡터 검색과 BM25를 결합한 하이브리드 검색기 |
| `src/meditations_rag/chains.py` | 질문 재작성 및 한·영 답변 생성 체인 |
| `text/Marcus-Aurelius-Meditations.pdf` | 코퍼스 구축에 사용한 《명상록》 원문 PDF |
