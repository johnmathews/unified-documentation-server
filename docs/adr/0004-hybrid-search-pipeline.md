# ADR 0004 — Hybrid search pipeline (BM25 + dense + cross-encoder rerank)

**Status:** Accepted (2026-05-10)

## Context

Until this ADR, `KnowledgeBase.search_documents()` ran pure ChromaDB cosine
similarity over chunk embeddings, optionally complemented by a content-blind
SQLite `LIKE`-on-title-and-path keyword fallback. The keyword fallback assigned
a synthetic score of 0.5 so its results "sort after strong semantic matches but
before weak ones".

Reproduced on prod (`infra:8085`) on 2026-05-10:

```
GET /api/search?q=strava&source=journal-insights-webapp&limit=10
→ Top result: journal-insights-webapp:CLAUDE.md
  score 0.7465, snippet: "Vue 3 frontend for the Journal Analysis Tool…"
  Literal occurrences of "strava" in document body: 0
```

The query ranks docs that never mention "strava" above journal entries that
mention it repeatedly. This is a textbook failure mode of pure dense retrieval
on rare named entities: `all-mpnet-base-v2` has no isolating training signal
for the token "strava" — its embedding sits in the broader fitness/journal
neighbourhood, and any chunk dominated by general journal-app content scores
close to it. A purely lexical signal (BM25, TF-IDF, even substring count)
would flip this trivially.

The keyword fallback didn't help because it only matches `title` and
`file_path`. Documents that mention "strava" only in their content (not the
title) are invisible to it. And the synthetic 0.5 score collides with the
cosine-distance scale — title hits land mid-ranking by accident, not by merit.

## Decision

Replace the pipeline with two stages:

**L1 (retrieval).** Two parallel candidate generators, fused with Reciprocal
Rank Fusion:

- **BM25** via SQLite FTS5. A new `chunks_fts` virtual table mirrors chunk
  content + title. `bm25(chunks_fts, 2.0, 1.0)` weights the title 2× the
  body so title-only matches still surface. Tokenizer:
  `unicode61 remove_diacritics 2` — no stemming, so proper-noun queries
  match the surface form.
- **Dense** via the existing ChromaDB cosine search.
- Each leg returns its top 100 chunks. Results are fused with RRF
  (k=60, the standard default). Top 50 chunks pass to L2.

**L2 (reranking).** A cross-encoder (`cross-encoder/ms-marco-MiniLM-L6-v2`,
ONNX int8, ~23 MB on disk, ~85 MB resident) scores `(query, chunk)` pairs
with full cross-attention. Output replaces the L1 RRF score with a
relevance logit; results are dedup'd to parent docs and the top-N are
returned.

**Removed.** `_keyword_search_title_path` and the synthetic 0.5 score are
deleted entirely. FTS5 with the title column subsumes their intent.
**Score field semantics flip** from "lower = better cosine distance" to
"higher = better rerank logit". Webapp unaffected (no score-based sort).

**Container memory limit** raised from 768 MB to 1024 MB; reservation from
256 MB to 320 MB to accommodate the additional model.

No feature flag. Both `search()` (chat agent) and `search_documents()`
(REST) adopt the new pipeline together.

## Consequences

**Update (2026-05-10, post-deploy):** the initial release shipped with
`_RERANK_BATCH_SIZE = 50`, which spiked transient activation memory past
the 1024 MB cgroup ceiling on every search and caused immediate OOM kills
on `infra:8085`. Dropped to 8 (matches the embedding model's batch size,
~5–10 ms throughput cost at 50 L1 candidates). See journal entry
`260510-hybrid-search-pipeline.md` for the full post-mortem. The
`_RERANK_BATCH_SIZE` knob is now documented in
`docs/architecture.md`.

Good:

- Top result for `q=strava` becomes a doc that actually mentions "strava".
  The acceptance test (`test_search_documents_strava_reproduction`) is the
  smoking-gun proof.
- The score-scale hack is gone. Score field has consistent semantics.
- No new Python dependencies — FTS5 is bundled in CPython's SQLite, and
  `tokenizers` / `huggingface-hub` / `onnxruntime` were already deps for
  the embedding model.
- Both retrieval legs run independently; failure of either degrades
  gracefully (RRF with one empty list still ranks the other). Reranker
  failure falls back to L1 order via a `try/except` wrapper.
- The chat agent benefits from the same retrieval improvement without
  extra integration work.

Bad / accepted:

- +85 MB resident once the reranker loads. Container memory limit raised
  to 1024 MB; cgroup-OOM-before-global-OOM design intent preserved.
- +23 MB on the Docker image (pre-baked reranker). One-time storage cost.
- Search latency increases by ~80–150 ms on warm calls (50-pair scoring
  on CPU). Well within the 1 s target the operator stipulated.
- FTS5 backfill on first deploy: a one-shot
  `INSERT INTO chunks_fts SELECT … FROM documents` runs once on init when
  `chunks_fts` is empty. Idempotent and fast for current corpus size.
- Score field semantics changed. Anything downstream that assumed
  ascending-score ordering would break. The webapp does not.

## Alternatives considered

- **Pure BM25 (drop dense entirely).** Would fix the strava bug but lose
  the paraphrase / conceptual recall that dense provides. Both legs are
  cheap; running both is strictly better.
- **Weighted-sum fusion** instead of RRF. Requires tuning `α` and re-tuning
  whenever the embedding model changes. RRF needs no normalisation.
- **`rank_bm25` Python library** instead of FTS5. Rebuilds the index on
  every restart, adds a Python dep, no `bm25()` ranking function. FTS5
  is the obvious choice.
- **LLM-as-reranker** (Anthropic API call per search). Highest possible
  quality ceiling but ~300–800 ms per call and external API dependency on
  every typed-search-box request. Reasonable for the chat-agent path
  (already LLM-bound) but overkill for the REST path.
- **Larger reranker** (`MiniLM-L12-v2`, `bge-reranker-base`). Marginal
  quality gain at 2× the latency or 2-10× the size. Revisit only if
  L6 quality is observed to be insufficient.
- **Feature flag for rollback.** Rejected. Both code paths would need
  test coverage and operational surface; rollback via `git revert` is
  simpler.
