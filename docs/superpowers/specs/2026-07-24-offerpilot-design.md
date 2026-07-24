# OfferPilot — Design Spec

Date: 2026-07-24 (rev 3, after second external review)
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
   pull postings for companies in `config.yaml`, normalize **in memory**
   (normalization is synchronous and deterministic — it is not a stored
   state), validate, dedupe by (source, external_id) and URL → SQLite
   `status=new`. Content changes create a new `job_versions` row
   (description hash + snapshot) so past scoring stays reproducible.
2. **Deterministic prefilter** (pure Python, no LLM): each hard-constraint
   rule returns a three-state `FilterResult`:
   `outcome: pass | fail | unknown`, plus `rule`, `extracted_value`,
   `reason` — all persisted. **Principle: only definite violations
   filter a job out; unparseable postings pass through as `unknown`.**
   Rules: location/remote, graduation-year window, years-of-experience,
   work authorization, pay floor, excluded companies. Any `fail` →
   `filtered_out` (failed rule recorded); otherwise `ready_for_match`.
3. **Graph run** per job version. **The core-MVP graph begins at the
   match node** using the collector-provided description:
   `match → gate → brief → pending_review`. The conditional research
   branch below is a later extension that adds nodes without changing
   the match/gate/brief interfaces.
   - **research node (extension, not MVP)**: code (not the LLM) decides
     whether the JD is too thin; if so the model may request tools —
     `fetch_job_detail` (ATS full description) or, stretch,
     `browse_careers_page(company_id)`. Tool requests are proposals:
     the program validates every call (see Security) before executing.
   - **match node**: profile + retrieved evidence → `MatchResult`
     (Pydantic): `eligibility` (pass/fail/unknown) + reasons; subscores
     `skills 0-30`, `projects 0-20`, `domain 0-15`, `seniority 0-15`,
     `preferences 0-20`; `evidence: list[EvidenceRef]` where each ref
     carries a `source_id` that MUST exist in the corpus (validated in
     code); `gaps`, `uncertainties`, `confidence`. **Total score is
     computed in Python**, never by the model.
   - **gate** (code): `eligibility == fail` → `eligibility_failed`;
     total < threshold → `scored_low`; otherwise continue.
     `eligibility == unknown` continues, but the review panel must show
     a prominent “Eligibility unresolved” banner — unknown is never
     silently treated as pass.
   - **brief node**: produces an *application brief* — why it fits,
     cited evidence, main gaps, resume bullets to emphasize,
     evidence-grounded talking points for common application themes
     (why this role / relevant project / main strength / gap to
     address — marked generic unless actual application questions were
     collected), optional outreach paragraph. One structured output.
   - Written to review queue, `status=pending_review`.
4. **Review panel** actions: approve / reject(+reason) / edit /
   save-for-later. Labels are **split**: `fit_label`
   (good_fit / poor_fit / uncertain), `action_label`
   (apply / skip / save), `rejection_reason` (skills, seniority,
   location, compensation, duplicate, expired, not_interested,
   bad_draft, other).
   **Label provenance**: every label row records `label_source` —
   `review_feedback` (given while model score/reasons/brief were
   visible; subject to anchoring bias) or `blind_eval` (given in a
   separate labeling view that shows only job + profile summary and
   hides all model output). **Formal eval metrics use `blind_eval`
   labels only**; review_feedback labels are auxiliary signal. The
   blind labeling view ships in Week 2 with the eval dataset.
5. **Evals** (`run_eval.py`, target 40–60 blind-labeled jobs):
   - Fit classification: precision / recall / F1, confusion matrix.
   - Ranking: Precision@5 / @10.
   - Groundedness — **automated heuristics** (not full fact-checking):
     every `EvidenceRef.source_id` exists; unknown skill/entity
     detection; numeric-claim and unsupported-proper-noun flags.
     Complemented by a **manual audit**: ~20 sampled briefs scored for
     unsupported-claim rate (severity: minor / material).
   - Results with timestamp + git commit committed to `evals/results/`.

## Job status state machine

```
new → (filtered_out | ready_for_match)
ready_for_match → matching → (eligibility_failed | scored_low |
                              pending_review | retryable_error |
                              permanent_error)
pending_review → (approved | rejected | saved)
```

(Normalization happens in memory before insert; there is no
`normalized` state.)

Runner is single-process; each transition is one SQLite transaction.
`matching` rows carry `processing_started_at`; on startup, stale rows
(> 15 min) are reset to `ready_for_match`.

**SQLite concurrency**: WAL mode with a busy timeout — the FastAPI
panel and the runner may access the DB concurrently. Writes use short
transactions; the runner **never holds a transaction open across an
LLM or network call** (read state → commit → call LLM → new
transaction → re-verify state → write result → commit).

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

`companies`, `jobs`, `job_versions`, `runs`, `run_steps`,
`review_items`, `labels`, `llm_usage`.

- `runs`: one row per graph or collector run — id, run_type,
  job_version_id, started_at, completed_at, status, git_commit,
  config_hash.
- `run_steps`: one row per node attempt — run_id, node, attempt,
  timestamps, status, input/output JSON, error. This is what makes
  multi-attempt traces, eval reproduction, and the demo UI trace view
  possible.

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
application brief; split labels; blind labeling view + eval dataset +
groundedness checks; demo mode; README + demo GIF.

**Later (cuttable)**: Ashby; LangGraph research-tool branch;
Playwright careers tool; HN collector; retrieval-method comparison eval;
automatic KB ingestion from GitHub.

## Resume phrasing targets (honest, staged)

**After core MVP** (no research tools yet — do not mention tool calls):
“Built OfferPilot, a human-in-the-loop job-matching agent using LangGraph
and DeepSeek, with deterministic eligibility filtering, structured
candidate evidence, Pydantic-validated rubric scoring, resumable SQLite
state, and an approval-gated review panel. Evaluated fit classification,
ranking quality, and evidence grounding on a blind-labeled benchmark.”

**After the research-tool extension ships**, upgrade to:
“…a bounded research→match→draft workflow with conditional,
program-validated tool calls (ATS APIs, Playwright)…”
