# RAG Chatbot Lab

[한국어](./README.md) | [English](./README.en.md)

## 🏛️ First, take a moment to talk with Marcus Aurelius

This RAG chatbot is grounded in Marcus Aurelius' *Meditations*. Use it when something is weighing on your mind or when you want a quiet place to sort through your thoughts.

### [👉 Talk with the Meditations chatbot](https://rag-chatbot-km9hgveen2rfuexsp777cf.streamlit.app/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rag-chatbot-km9hgveen2rfuexsp777cf.streamlit.app/)

The app supports both 한국어 and English. If you have a quiet moment, drop in and share one thought that has been staying with you lately.

---

## What is this repository?

This repository collects experiments and working projects for different RAG (Retrieval-Augmented Generation) chatbots.

Each project lives on its own branch so that its purpose, code, and data remain separate. The default `master` branch serves only as a landing page that introduces the repository and points to each project.

## Branches

| Branch | Description | Status |
|---|---|---|
| [`master`](https://github.com/littlekg87/rag-chatbot/tree/master) | Default landing page with repository information and project links | Maintained |
| [`marcus-aurelius-chatbot`](https://github.com/littlekg87/rag-chatbot/tree/marcus-aurelius-chatbot) | Bilingual Streamlit chatbot using *Meditations*, precomputed embeddings, and hybrid retrieval | Live |
| [`thesis-rag-chatbot`](https://github.com/littlekg87/rag-chatbot/tree/thesis-rag-chatbot) | Early RAG chatbot experiment using a thesis PDF and Supabase | Archived |

## How work is organized

- Application code and data are maintained on their respective project branches.
- Tasks, bugs, and improvement ideas are tracked in [GitHub Issues](https://github.com/littlekg87/rag-chatbot/issues).
- API keys and service credentials are never committed to GitHub; they are managed through the deployment platform's Secrets.
