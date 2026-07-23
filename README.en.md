# Marcus Aurelius RAG Chatbot

[한국어](./README.md) | [English](./README.en.md)

## 🏛️ Take a moment to talk with Marcus Aurelius

Share whatever has been weighing on your mind. This RAG chatbot finds relevant passages from *Meditations* and gently separates the situation into event, judgment, and action.

### [👉 Talk with the Meditations chatbot](https://rag-chatbot-km9hgveen2rfuexsp777cf.streamlit.app/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rag-chatbot-km9hgveen2rfuexsp777cf.streamlit.app/)

The app supports both 한국어 and English. If you have a quiet moment, drop in and share one thought that has been staying with you lately.

---

## What is this branch?

This branch contains the code and data for a Streamlit RAG chatbot grounded in Marcus Aurelius' *Meditations*.

It rewrites a user's question into an English retrieval query, then uses hybrid search—precomputed vector embeddings combined with BM25—to find relevant passages. The retrieved text and supporting Stoic interpretation notes ground a conversational response. Both the interface and generated answers can be switched between Korean and English.

The OpenAI API key is not stored in this repository. It is managed server-side through Streamlit Secrets.

## Project structure

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

## Files and folders

| Path | Purpose |
|---|---|
| `app.py` | Streamlit interface, language switching, chat sessions, and the complete RAG flow |
| `requirements.txt` | Python packages required to run the app and process its data |
| `.gitignore` | Excludes API keys, logs, caches, and local virtual environments |
| `data/meditations.xml` | Structured *Meditations* text extracted and cleaned from the PDF |
| `data/meditations_chunks.jsonl` | Passage-level chunks used for retrieval |
| `data/meditations_embeddings.npy` | Precomputed vector embeddings for all chunks |
| `data/meditations_embeddings_meta.json` | Validation metadata including model, dimensions, and chunk IDs |
| `docs/marcus_response_style_guide.md` | Voice and response principles for the Marcus-inspired conversation style |
| `docs/stoic_interpretive_notes.md` | Stoic concept notes used to support interpretation |
| `prompts/rewrite_system_prompt.md` | Rewrites user messages into retrieval queries suitable for *Meditations* |
| `prompts/answer_system_prompt.md` | Korean response format and grounding rules |
| `prompts/answer_system_prompt_en.md` | English response format and grounding rules |
| `scripts/build_meditations_corpus.py` | Converts PDF/XML source material into retrieval chunks |
| `scripts/build_embeddings.py` | Generates chunk embeddings and their metadata |
| `src/meditations_rag/__init__.py` | Python package initializer |
| `src/meditations_rag/retriever.py` | Hybrid retriever combining vector similarity and BM25 |
| `src/meditations_rag/chains.py` | Query rewriting and bilingual answer-generation chains |
| `text/Marcus-Aurelius-Meditations.pdf` | Source edition of *Meditations* used to build the corpus |
