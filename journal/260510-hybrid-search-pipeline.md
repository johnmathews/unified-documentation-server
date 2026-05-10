# 2026-05-10 — Hybrid search pipeline (BM25 + dense + cross-encoder rerank)

## Problem

Pure-vector search ranked irrelevant docs at position 1 for queries on rare
named entities. Reproduced on prod (`infra:8085`):

```
GET /api/search?q=strava&source=journal-insights-webapp&limit=10
→ Top: journal-insights-webapp:CLAUDE.md, score 0.7465
       snippet: "Vue 3 frontend for the Journal Analysis Tool…"
       literal occurrences of "strava" in body: 0
```

`all-mpnet-base-v2` has no isolating training signal for the token "strava";
the embedding sits near the broader fitness/journal cluster and any
general-journal-app chunk wins on cosine. The pre-existing keyword fallback
(`_keyword_search_title_path`) only matched `title` and `file_path` and could
not rescue this — the strava-mentioning docs had unrelated titles.

## Decision

Two-stage hybrid pipeline, no flag, no fallback to the old behaviour.
Captured in detail in `docs/adr/0004-hybrid-search-pipeline.md`.

- **L1**: SQLite FTS5 BM25 (title-weighted 2x via `bm25(chunks_fts, 2.0, 1.0)`)
  + ChromaDB cosine, fused with RRF (k=60). Top 50 chunks → L2.
- **L2**: `cross-encoder/ms-marco-MiniLM-L6-v2`, ONNX int8, ~23 MB on disk,
  ~85 MB resident. Reorders L1 candidates with full cross-attention.
- **Deleted**: `_keyword_search_title_path`, the synthetic-0.5 score boost,
  and the four tests that exercised them. FTS5 with the title column does
  the job correctly.
- **Score field semantics flipped** from "lower = better cosine distance" to
  "higher = better rerank logit". Webapp not affected (verified: no
  score-based sort in `documentation-webapp/src/`).
- **Container memory**: `mem_limit` 768→1024 MB, `mem_reservation` 256→320 MB.
  cgroup-OOM-before-global-OOM design intent preserved (1024 MB still well
  below the 1.95 GB VM ceiling).

## Why these specific choices

- **FTS5 over `rank_bm25`**: zero new deps, persistent, native `bm25()`
  ranking, supports per-column weights so title-bias is one number.
- **`unicode61 remove_diacritics 2` tokenizer, no stemming**: the trigger for
  the redesign was a proper-noun query. Stemming would mutate proper nouns
  and break exact matches.
- **MiniLM-L6-v2 over L12 / bge-reranker / LLM**: 1 s latency budget;
  L6 reranks 50 pairs in <150 ms on CPU and adds <100 MB resident. L12 is
  ~2× the latency for marginal quality gain on this corpus. LLM rerank
  costs Anthropic API calls per search and adds ~500 ms.
- **RRF over weighted-sum**: no `α` to tune, no score normalisation needed,
  robust across embedding-model swaps.
- **Both `search()` and `search_documents()` adopt the new pipeline**: the
  chat agent benefits from better retrieval too. Single code path, single
  test surface.
- **Dedup-to-parent after L2**: if you dedup before, the reranker only sees
  one chunk per parent and loses signal about which section of a long doc
  is actually relevant.

## Implementation notes

- `chunks_fts` is a non-contentless FTS5 table (default mode), not external-
  content. Rationale: `documents.doc_id` is `TEXT PRIMARY KEY`, not a stable
  INTEGER rowid, and `INSERT OR REPLACE` shifts the implicit rowid on every
  upsert. External-content FTS5 would silently desync. Default-mode FTS5
  stores its own copy of `title` and `content`; for current corpus size the
  storage cost is negligible (~12-16 MB at ~10k chunks).
- FTS sync at write time uses application-level inserts in `upsert_document`
  / `upsert_documents_batch` / `delete_document` / `delete_source_documents`
  / `rename_source`, mirroring the existing ChromaDB-write pattern. Triggers
  on `documents` were rejected as harder to debug.
- Backfill on first init: `_init_sqlite()` populates `chunks_fts` from
  existing `documents` rows when `chunks_fts` is empty. Idempotent.
- Reranker module mirrors `embedding.py` patterns: lazy load, pinned HF
  revision, cached_property session/tokenizer, image-cache seeding into
  `/data/models/ms-marco-MiniLM-L6-v2`.
- Reranker failure absorbed in a `try/except` in `rerank()`: returns
  L1 results unchanged on any error so search never 500s because rerank
  is sick.

## Verification

Headline acceptance test:
`tests/test_knowledge_base.py::test_search_documents_strava_reproduction`
inserts a CLAUDE.md-like decoy chunk and a journal entry that literally
mentions "strava"; asserts the top result is the strava doc.

Full suite: 456 passing (was 450 before; +6 reranker tests, +7 hybrid
tests, –4 deleted keyword-search-title-path tests).

## Follow-ups

- Watch first-search latency in prod logs after deploy. Target: <2 s cold
  (model load from baked image), <500 ms warm.
- If FTS5 tokenizer behaviour surprises users on compound words or
  CJK, consider `porter` (English stemming) or `trigram` tokenizer.
- `_RRF_CANDIDATE_K = 100` and `_L1_OUTPUT_SIZE = 50` are conventional
  starting values; revisit only if relevance is poor.
- The chunks_fts schema doesn't include `file_path`. If users want to
  search by file path text (e.g. find docs in "networking/" path), we'd
  add it as another indexed column. Not in scope for this change.
