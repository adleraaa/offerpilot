# OfferPilot — Design Spec

Date: 2026-07-24
Status: Approved by user (chat review)

## Purpose

A local, human-in-the-loop job-search agent. It collects real job postings
from public sources, scores them against a personal knowledge base
(resume, projects, preferences) using retrieval + LLM reasoning, drafts
tailored outreach materials for high-scoring roles, and queues everything
for explicit human approval. Nothing is ever sent or submitted
automatically.

Secondary purpose: a portfolio project demonstrating agent engineering —
LangGraph orchestration, tool use, RAG, structured outputs, and eval
discipline — for AI engineering / forward-deployed engineering
applications.

## Hard boundaries

1. **No autonomous outreach.** The agent's output terminates at local
   drafts in a review queue. A human approves, edits, and sends manually.
2. **Polite collection only.** Public APIs first. Playwright scraping of
   careers pages only for companies without an ATS API: no login, no
   captcha/anti-bot handling (abort and mark the company on any
   challenge), rate-limited requests.
3. **No secrets in code or data.** API keys via environment variables.
   The knowledge base contains only material the user already publishes
   (resume, project READMEs) plus a preferences file.
4. **Spending fuse.** A configurable daily LLM-spend cap halts the graph
   runner when exceeded.

## Stack

- Python 3.11+, LangGraph for the agent graph, DeepSeek (OpenAI-compatible
  API) as the LLM, Pydantic for structured outputs.
- sentence-transformers (local embeddings) + Chroma for the knowledge base.
- SQLite as the single store (jobs, runs, review queue, labels).
- FastAPI + one static HTML page for the review panel.
- Playwright (Python) for the careers-page tool.
- pytest for tests.

Repo: `D:\offerpilot`, public on GitHub (github.com/adleraaa/offerpilot).

## Module layout

```
src/offerpilot/
  collectors/    greenhouse.py, lever.py, ashby.py, hn.py  (pure Python, no LLM)
  tools/         browse_careers.py (Playwright), fetch_detail.py
  graph/         nodes.py, graph.py, prompts.py, schemas.py
  kb/            ingest.py, retrieve.py
  store/         db.py (SQLite schema + queries)
  review/        app.py (FastAPI), static/index.html
  evals/         run_eval.py
config.yaml      company list, score threshold, rate limits, spend cap
data/            offerpilot.db, chroma/  (gitignored)
evals/dataset.jsonl   labeled jobs (committed)
evals/results/        timestamped eval runs (committed)
```

## Data flow

1. **Collect** (cron or manual): collectors pull postings for companies in
   `config.yaml` → normalize to a common Job record → dedupe by
   (source, external_id) and URL → insert into SQLite with
   `status=new`.
2. **Run graph** on each `status=new` job:
   - **research node**: LLM sees the job record; may call tools
     `fetch_job_detail` (full ATS description) or `browse_careers_page`
     (Playwright, for `ats: none` companies) when information is thin.
     Output: enriched job context.
   - **match node**: retrieve top-k chunks from the knowledge base;
     LLM returns `MatchResult` (Pydantic): `score 0-100`,
     `reasons: list[str]`, `cited_experience: list[str]`.
   - **gate**: `score >= threshold` (config) continues; otherwise job is
     marked `scored_low` and stops (saves tokens).
   - **draft node**: retrieve the most relevant experience chunks; LLM
     writes a tailored cold email / application blurb (`DraftResult`).
   - Job + score + reasons + draft written to review queue,
     `status=pending_review`.
3. **Review panel**: lists pending items with job detail, match reasons,
   editable draft. Actions: approve / reject / edit-then-approve. Every
   approve/reject also records a fit label (approved→fit=1,
   rejected-with-reason→fit=0) into `labels`, feeding the eval dataset.
4. **Evals**: `run_eval.py` replays the match node over
   `evals/dataset.jsonl` (target 40–60 human-labeled jobs), reports
   precision / recall / F1 and a confusion matrix, writes results with
   timestamp + git commit to `evals/results/`.

## Knowledge base

Sources: resume (txt export), the three project READMEs (pulled from
GitHub), `preferences.md` (user-written: direction, hours, commute,
pay floor). Chunked by section headers. Embedded with
`all-MiniLM-L6-v2` locally; stored in Chroma. Re-ingest is a manual
command (`offerpilot ingest`).

## Error handling

- Collector failure for one company: log, skip, continue the batch.
- LLM structured-output validation failure: retry twice, then mark job
  `error` with the raw output saved for debugging.
- Playwright: on captcha/challenge/login wall → abort that company, mark
  `blocked` in DB, never retry automatically.
- Graph runner is resumable: jobs are processed by status, so a crash
  mid-batch loses nothing.
- Daily spend cap checked before each LLM call; exceeding it stops the
  runner with a clear message.

## Testing

- Collectors: pytest against saved JSON/HTML fixtures (no network).
- Graph nodes: mock LLM client; assert routing (gate thresholds, tool-call
  paths, validation-retry behavior).
- Review panel: API tests for approve/reject/label writes.
- End-to-end smoke: 3 real jobs through the full pipeline with the real
  LLM (manual, documented in README).

## Milestones (2–3 weeks)

1. Store + collectors (Greenhouse/Lever/Ashby/HN) + config. **Cuttable: none.**
2. KB ingest/retrieve + match node + gate + structured outputs.
3. Draft node + review panel + labels.
4. Evals + README with demo GIF + polish.
5. Stretch (cuttable if time runs out): Playwright careers-page tool —
   the graph works without it; `ats: none` companies are simply skipped.

## Resume phrasing target (honest, after completion)

“Built OfferPilot, a human-in-the-loop job-search agent (LangGraph +
DeepSeek): a research→match→draft graph with autonomous tool use
(ATS APIs, Playwright), retrieval over a personal knowledge base, Pydantic
structured outputs, and a labeled eval set (precision/recall tracked
across prompt and retrieval changes). All outreach gated behind a local
human-approval panel.”
