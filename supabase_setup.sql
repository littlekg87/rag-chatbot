-- Supabase SQL Editor에 붙여넣고 실행하세요
CREATE TABLE conversations (
  id         SERIAL PRIMARY KEY,
  session_id TEXT NOT NULL,
  question   TEXT NOT NULL,
  answer     TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
