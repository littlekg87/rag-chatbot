# RAG Chatbot Lab

## 🏛️ 먼저, 마르쿠스 아우렐리우스와 잠깐 이야기해 보세요

마음에 걸리는 일이 있거나, 머릿속 생각을 조용히 정리하고 싶을 때 사용할 수 있는 **마르쿠스 아우렐리우스의 《명상록》 기반 RAG 챗봇**입니다.

### [👉 명상록 챗봇과 대화하러 가기](https://rag-chatbot-km9hgveen2rfuexsp777cf.streamlit.app/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rag-chatbot-km9hgveen2rfuexsp777cf.streamlit.app/)

한국어와 English를 모두 지원합니다. 심심할 때 한 번 들어와서 지금 마음에 걸리는 생각 하나를 건네 보세요.

---

## 이 저장소는 무엇인가요?

여러 종류의 RAG(Retrieval-Augmented Generation) 챗봇을 만들고 실험한 기록을 모아두는 저장소입니다.

각 프로젝트는 목적과 데이터가 서로 섞이지 않도록 별도의 브랜치에서 관리합니다. 기본 브랜치인 `master`는 전체 프로젝트를 소개하고 각 작업으로 안내하는 허브 역할만 합니다.

## 브랜치 안내

| 브랜치 | 설명 | 상태 |
|---|---|---|
| [`master`](https://github.com/littlekg87/rag-chatbot/tree/master) | 저장소 소개와 프로젝트 링크를 제공하는 기본 허브 | 유지 |
| [`marcus-aurelius-chatbot`](https://github.com/littlekg87/rag-chatbot/tree/marcus-aurelius-chatbot) | 《명상록》 원문, 사전 생성 임베딩, 하이브리드 검색을 사용하는 한·영 지원 Streamlit 챗봇 | 운영 중 |
| [`thesis-rag-chatbot`](https://github.com/littlekg87/rag-chatbot/tree/thesis-rag-chatbot) | 논문 PDF와 Supabase를 사용했던 초기 RAG 챗봇 실험 | 보관 |

## 작업 방식

- 앱 코드와 데이터는 각 프로젝트 브랜치에서 관리합니다.
- 할 일, 버그, 개선 아이디어는 [GitHub Issues](https://github.com/littlekg87/rag-chatbot/issues)에서 기록합니다.
- API 키와 서비스 인증정보는 GitHub에 커밋하지 않고 배포 서비스의 Secrets에서 관리합니다.
