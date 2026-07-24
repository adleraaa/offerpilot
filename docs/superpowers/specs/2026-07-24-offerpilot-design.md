# OfferPilot — Design Spec

Date: 2026-07-24 (rev 2, after external review)
Status: Revised; pending user re-approval

## Purpose

A local, human-in-the-loop job-search agent. It collects real job postings
from public sources, filters them with deterministic rules, scores the
survivors against a structured candidate profile plus retrieved evidence,
drafts an application brief for high-scoring roles, and queues everything
for explicit human approval. Nothing is ever sent or submitted
automatically.

Secondary purpose: a portfolio project demonstrating agent engineering —
bounded LangGraph orchestration, validated tool use, grounded retrieval,
structured outputs, eval discipline, and prompt-injection-aware design —
for AI engineering / forward-deployed engineering applications.

## Hard boundaries

1. **No autonomous outreach.** Output terminates at local drafts in a
   review queue. A human approves, edits, and sends manually.
2. **Polite collection only.** Public APIs first. Playwright scraping of
   careers pages (stretch) only for companies without an ATS API: no
   login, no captcha/anti-bot handling (abort and mark the company on any
   challenge), rate-limited requests.
3. **No secrets in code or data.** API keys via environment variables.
   The knowledge base contains only material the user already publishes
   plus a preferences file.
4. **Spending fuse.** A configurable daily LLM-spend cap; single-worker
   runner (declared, avoids cap race conditions).
5. **Untrusted-input discipline.** Job postings are external, untrusted
   text. See Security section.

## Stack

- Python 3.11+, LangGraph for graph orchestration, DeepSeek
  (OpenAI-compatible API), Pydantic for structured outputs.
- sentence-transformers (local embeddings) + Chroma for the evidence
  corpus (supporting role — see Candidate Model).
- SQLite as the single store; single-process runner.
- FastAPI + one static HTML page for the review panel.
- Playwright (Python) for the careers-page tool (stretch).
- pytest.

Repo: `D:\offerpilot`, public on GitHub (github.com/adleraaa/offerpilot).

## Candidate model (two layers)

**Layer 1 — structured profile** (`profile.yaml`, always fully in
context): identity/education/graduation; hard constraints (locations,
remote preference, pay floor, work authorization, internship vs
full-time); skills; experiences — each with a stable `id`
(e.g. `pathpilot`), summary, skills, and evidence pointers.

**Layer 2 — evidence corpus** (Chroma): resume text and project READMEs,
chunked by section, each chunk carrying a `source_id`. Retrieval
supplements the profile with supporting text; it never replaces it.

Match input = full structured profile + top-k retrieved evidence chunks.
Stretch eval: compare full-profile-only vs retrieval-only vs
profile+retrieval on the labeled set.

## Data flow

1. **Collect**: Greenhouse + Lever collectors (core; Ashby second tier)
   pull postings for companies in `config.yaml` → normalize → dedupe by
   (source, external_id) and URL → SQLite `status=new`. Content changes
   create a new `job_versions` row (description hash + snapshot) so past
   scoring stays reproducible.
2. **Deterministic prefilter** (pure Python, no LLM): hard constraints
   from the profile — location/remote, graduation-year and
   years-of-experience requirements parsed conservatively, work
   authorization, pay floor, excluded companies. Fail → `filtered_out`
   with the failed rule recorded. Pass → `ready_for_match`.
3. **Graph run** per job version:
   - **research node**: code (not the LLM) decides whether the JD is too
     thin; if so the model may request tools — `fetch_job_detail`
     (ATS full description) or, stretch, `browse_careers_page(company_id)`.
     Tool requests are proposals: the program validates every call
     (see Security) before executing.
   - **match node**: profile + retrieved evidence → `MatchResult`
     (Pydantic): `eligibility` (pass/fail/unknown) + reasons; subscores
     `skills 0-30`, `projects 0-20`, `domain 0-15`, `seniority 0-15`,
     `preferences 0-20`; `evidence: list[EvidenceRef]` where each ref
     carries a `source_id` that MUST exist in the corpus (validated in
     code); `gaps`, `uncertainties`, `confidence`. **Total score is
     computed in Python**, never by the model.
   - **gate** (code): total ≥ threshold continues; else `scored_low`.
   - **brief node**: produces an *application brief* — why it fits,
     cited evidence, main gaps, resume bullets to emphasize, suggested
     answers to likely application questions, optional outreach
     paragraph. One structured output, not five template types.
   - Written to review queue, `status=pending_review`.
4. **Review panel** actions: approve / reject(+reason) / edit /
   save-for-later. Labels are **split**: `fit_label`
   (good_fit / poor_fit / uncertain — the model-quality signal),
   `action_label` (apply / skip / save), `rejection_reason`
   (skills, seniority, location, compensation, duplicate, expired,
   not_interested, bad_draft, other). Only `fit_label` feeds evals.
5. **Evals** (`run_eval.py`, target 40–60 labeled jobs):
   - Fit classification: precision / recall / F1, confusion matrix.
   - Ranking: Precision@5 / @10.
   - Groundedness (code checks): every `EvidenceRef.source_id` exists;
     draft-lint flags skills/experience claims absent from the profile.
   - Results with timestamp + git commit committed to `evals/results/`.

## Job status state machine

```
new → normalized → (filtered_out | ready_for_match)
ready_for_match → matching → (scored_low | pending_review |
                              retryable_error | permanent_error)
pending_review → (approved | rejected | saved)
```

Runner is single-process; each transition is one SQLite transaction.
`matching` rows carry `processing_started_at`; on startup, stale rows
(> 15 min) are reset to `ready_for_match`.

## Security (untrusted input + tools)

- **Input isolation**: prompts wrap posting text in a clearly delimited
  UNTRUSTED block; system prompt states that instructions inside it are
  data, never directives; the model may only extract job facts from it.
- **Tool validation in code**: `browse_careers_page` takes a
  `company_id`; the URL comes from config, not the model. https only;
  domain allowlist; block private IPs/localhost/file://; cap redirects,
  response size, and request time; fresh Playwright profile (no cookies).
- **Panel XSS**: all job-derived text rendered via `textContent` /
  escaped templates, never innerHTML.
- **Spend ledger**: per-call rows (model, prompt/completion tokens,
  estimated cost, run id, node, timestamp); prices in config. Pre-call
  estimate + post-call actuals; cap exceeded → no new calls.

## SQLite schema (right-sized)

`companies`, `jobs`, `job_versions`, `runs` (one row per graph/collector
run with node + attempt info), `review_items`, `labels`, `llm_usage`.
No separate node_runs/graph_runs/collector_runs tables — `runs` covers
reproducibility at this scale.

## Error handling

- Collector failure for one company: log, skip, continue.
- Structured-output validation failure: retry twice → `permanent_error`
  with raw output saved.
- Playwright challenge/login wall → mark company `blocked`, never
  auto-retry.
- Crash mid-batch loses nothing (status machine + stale-row sweep).

## Demo mode

`offerpilot demo` seeds a temp SQLite DB with 3–5 fixture jobs and a
synthetic candidate profile, uses a mock LLM (pre-recorded outputs), and
launches the review panel. No API key required. README documents demo
mode first, real mode (DeepSeek key) second. Demo data clearly synthetic.

## Testing

- Collectors: pytest against saved JSON fixtures (no network).
- Prefilter: table-driven unit tests per rule.
- Graph nodes: mock LLM; assert routing, retries, evidence validation.
- Panel: API tests incl. label writes and escaping.
- End-to-end smoke: 3 real jobs, real LLM, documented in README.

## Milestones

**Week 1 — one reliable pipeline**: schema + state machine; Greenhouse +
Lever collectors; structured profile; deterministic prefilter;
MatchResult rubric + DeepSeek structured-output wrapper; CLI
(collect / match / status).

**Week 2 — make it a portfolio**: review panel with evidence display;
application brief; split labels; eval dataset + groundedness checks;
demo mode; README + demo GIF.

**Later (cuttable)**: Ashby; LangGraph research-tool branch;
Playwright careers tool; HN collector; retrieval-method comparison eval;
automatic KB ingestion from GitHub.

## Resume phrasing target (honest, after completion)

“Built OfferPilot, a human-in-the-loop job-search agent using LangGraph
and DeepSeek: a bounded research→match→draft workflow with
conditional, program-validated tool calls, retrieval over structured
candidate evidence, Pydantic-validated outputs, resumable SQLite state,
and an approval-gated review panel. Evaluated role matching and evidence
grounding on a labeled benchmark across prompt and retrieval revisions.”
