# OfferPilot

[![tests](https://github.com/adleraaa/offerpilot/actions/workflows/ci.yml/badge.svg)](https://github.com/adleraaa/offerpilot/actions/workflows/ci.yml)

A local, human-in-the-loop job-search agent. It pulls postings from public ATS
APIs, drops the ones that violate hard constraints using deterministic rules,
scores the rest against a structured candidate profile with one LLM call, and
stops at a review queue in SQLite. Nothing is ever sent or submitted to an
employer. There is no outreach code in this repo: the only outbound network
calls are the collectors' GETs to the two ATS APIs and the scoring call to the
configured LLM endpoint.

```
collect  ->  prefilter  ->  match (LLM)  ->  gate  ->  pending_review
             (6 rules,      (one node, schema   (Python)   (terminal)
              no LLM)       and grounding
                            checked)
```

## Collect

`src/offerpilot/collectors/`. Two collectors: Greenhouse
(`boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true`) and Lever
(`api.lever.co/v0/postings/{slug}?mode=json`). Each is a `fetch` and a `parse`,
and `parse` returns Pydantic `NormalizedJob` values. Greenhouse content is HTML,
so it is unescaped and tag-stripped; Lever supplies `descriptionPlain`.

Identity in SQLite is `unique(source, external_id)` on `jobs`. A canonicalized
URL (tracking params stripped, trailing slash and case normalized) is stored
alongside it. Content lives in `job_versions` as an immutable snapshot keyed by
a sha256 of title, location and description, so re-collecting unchanged text
adds nothing and an edited posting starts a fresh version at status `new`.

## Prefilter

`src/offerpilot/prefilter.py`. Six rules, pure Python, no model:
`years_of_experience`, `work_authorization`, `location`, `excluded_company`,
`graduation_window`, `pay_floor`. Each returns a `FilterResult` with one of
three outcomes, pass, fail or unknown, plus the rule name, the matched text and
a reason. All six are persisted to `filter_results`. Only a fail filters a job
out. Anything the rules cannot parse becomes unknown and goes on to scoring,
because a missed filter costs one LLM call while a wrong filter silently loses
a job.

The rules are written around how postings actually phrase things. The years rule
matches a requirement stated on either side of the number, then discards the
match if it is negated, if the surrounding clause marks it as preferred or nice
to have, or if the years belong to vesting or tenure rather than experience. The
authorization rule fails on a clearance requirement unless the text negates it,
and treats a refusal to sponsor as a fail only when the profile says sponsorship
is needed. The graduation rule scans every graduation phrase rather than the
first, and an including window beats an excluding one, so a posting that names a
class year and then widens it stays in. The pay rule takes the highest explicit
figure in the posting, hourly or annualized, so a low aside cannot sink a range
that clears the floor. 26 snippets taken from real postings are checked as a
regression net in `tests/test_prefilter.py`.

## Match: the only LLM node

`src/offerpilot/graph.py` and `src/offerpilot/llm.py`. One scoring step per job
version, and no other model call anywhere in the pipeline.

The prompt (`prompts.py`) puts the trusted profile JSON first and wraps the
posting in an `<untrusted_job_posting>` block. Before interpolation,
`graph._sanitize` strips the opening and closing forms of that delimiter out of
the job text, so a posting cannot close the block early.

The response is requested as a JSON object at temperature 0 and validated with
`MatchResult.model_validate_json`. A reply that fails is re-requested, up to 3
attempts inside one `structured()` call, and then becomes a permanent error.
The re-request is a repair turn, not a re-roll: the rejected reply and the
reason it was rejected are appended to the conversation, so the model is told
what to fix. The reason is collapsed to one line and clipped to 300
characters, because it can quote model-written text back into a trusted user
turn. `MatchResult` bounds each subscore (skills 0-30, project 0-20,
domain 0-15, seniority 0-15, preference 0-20) and a model validator rejects
`eligibility="fail"` unless the model also returned the posting excerpt it is
failing on.

Grounding check: `make_evidence_validator` compares every
`evidence[].source_id` against `Profile.experience_ids()`, the ids declared in
`profile.yaml`. It is handed to the client as `structured(validate=...)`, so an
id outside that set is rejected and re-asked like a schema violation; a model
that keeps citing invented ids exhausts the 3 attempts, the version goes to
`permanent_error`, and nothing reaches the queue. `run_match_for_version` then
runs that same validator again on whatever the client handed back, before the
result can reach the review queue: the repair turn depends on the client
honouring `validate`, the gate does not. The same validator also refuses a
result that cites nothing at all once it scores at or above the review
threshold — an uncited recommendation cannot reach a human. Below the
threshold, and on an eligibility fail, an empty evidence list is allowed: there
the posting itself is the evidence, and `MatchResult` already demands an
excerpt from it. That check is in code in `graph.py`. The prompt asks for the
same thing, but asking is not enforcement: a model that ignores the instruction
still cannot get an invented id past this check. The total score is summed in
Python (`models.total_score`), never read off the model.

HTTP 429, 5xx, timeouts and connection failures are retryable: the version goes
back to `ready_for_match` until `match.max_auto_retries` attempts are used, then
`permanent_error`. HTTP 401 and 403 raise `AuthLLMError`, which is neither: a
rejected key fails every queued job identically, so `cmd_match` aborts the whole
batch after the first one and the version is handed back to `ready_for_match`
with its attempt count unspent. Everything else is permanent on the first
occurrence. Each attempt writes a `run_steps` row with the prompt and either the
output or the error, and each API response writes token counts and an estimated
cost to `llm_usage`. Every run also records the git commit and a hash of the
effective config, so a result can be traced back to the code and settings that
produced it. Before every attempt the client sums today's `llm_usage` and raises
`SpendCapExceeded` at or above `llm.daily_spend_cap_usd`, which stops the run
and leaves the version resumable. The daily boundary is SQLite `date('now')`,
so the cap resets at UTC midnight, not local midnight.

## Gate and terminal state

Also plain Python, at the end of `run_match_for_version`. It re-reads the
version's status after the call and abandons the write if anything else moved
it. Then `eligibility == "fail"` goes to `eligibility_failed`, a total below
`match.score_threshold` goes to `scored_low`, and anything else inserts a
`review_items` row holding the full validated `MatchResult` and the computed
total, with the version set to `pending_review`.

`pending_review` is where the pipeline ends. `store/db.py` declares
`pending_review -> approved | rejected | saved`, but no code here performs that
transition, so approving today means reading `review_items` out of SQLite
yourself. Every status change goes through `db.set_status`, which refuses any
transition not listed in `ALLOWED_TRANSITIONS`.

## Why the pipeline is fixed

A job posting is text written by a stranger, fetched over the network, and put
into a prompt. Treating that as a prompt-injection surface rather than as
content is what sets the shape of the system:

1. Control flow is plain Python function calls. The model picks no route, no
   tool and no next step. It fills in one schema and returns.
2. Every model output is schema-checked before anything is written.
3. Every score that reaches a human must cite evidence ids, and each id must
   exist in `profile.yaml`, a file the model never writes.
4. The arithmetic that decides whether a job reaches a human runs in Python.
5. Spend is capped from the usage ledger in the same database.

The worst a hostile posting gets is a schema-shaped answer citing an invented
evidence id, and that is the case the grounding check rejects, re-asks, and
turns into `permanent_error` if the model will not correct it.

## Running it

Python 3.11 or newer.

```
pip install -e .
cp config.example.yaml config.yaml
cp profile.example.yaml profile.yaml
```

Edit `config.yaml` for the company list, model, spend cap and score threshold,
and `profile.yaml` for constraints and experiences. The experience ids in
`profile.yaml` are the evidence vocabulary the grounding check enforces.

```
python -m offerpilot collect
python -m offerpilot status
python -m offerpilot match
python -m offerpilot retry
```

`collect` fetches every company in `config.yaml`, upserts, prefilters and sets
status; a failure on one company or one job is printed and skipped rather than
fatal. `status` prints counts of `job_versions` by status. `match` scores
everything sitting at `ready_for_match`, and refuses to start if `profile.yaml`
is missing (so it cannot spend money scoring the synthetic example profile) or
if `DEEPSEEK_API_KEY` is unset. `retry` sweeps versions stuck in `matching` for
over 15 minutes back to `ready_for_match`, re-prefilters versions orphaned at
`new` by a crash between the insert and the prefilter, and resets
`permanent_error` rows to `ready_for_match` with a zeroed attempt count. That
orphan sweep also runs at the start of `collect`, so a crashed run self-heals.

Every subcommand takes `--db` (default `data/offerpilot.db`), `--config`
(default `config.yaml`), `--profile` (default `profile.yaml`) and `--limit`
(process at most N jobs, on `collect` and `match`). The model is reached with
the OpenAI SDK pointed at `llm.base_url`; the example config uses DeepSeek. The
key is read from the `DEEPSEEK_API_KEY` environment variable only, never from a
file.

## Tests

```
pip install -e ".[dev]"
python -m pytest -q
```

147 tests, all passing, and CI runs them on every push. They need no network and
no API key: collectors are tested by parsing recorded payloads from
`tests/fixtures/`, and the LLM client is tested against a fake SDK object.

## Storage

One SQLite file. Written by the current code: `jobs`, `job_versions`,
`filter_results`, `companies`, `labels`, `runs`, `run_steps`, `review_items`,
`llm_usage`. `db.migrate` adds new columns to an existing database rather than
requiring a rebuild. Created but not written yet: `review_items.brief_json`
(`db.save_brief` exists; nothing calls it, because the brief node does not).

## What is not built

- No LangGraph. `langgraph` is declared in `pyproject.toml` but nothing imports
  it; `graph.py` is ordinary function calls.
- No retrieval. No embeddings, no vector store, no Chroma or
  sentence-transformers. Evidence is the structured profile and nothing else.
- No review panel or web UI. Nothing serves the queue, and there is no approve,
  reject or edit path. The label vocabulary and the queries a reviewer would
  need exist (`labels.py`, `db.get_review_queue`, `db.get_blind_candidates`),
  but nothing drives them.
- No application brief node.
- No eval runner. The small blind-labeled evaluation set described in the design
  spec has not been assembled or run, so there are no fit, ranking or
  groundedness numbers.
- No demo mode. `match` needs a real key; `collect`, `status` and `retry` do not.
- No Ashby collector and no Playwright careers-page collector. Greenhouse and
  Lever are the only sources.
- The match node has never been run against a real API key. Every test of it
  uses a stub, so the prompt has not been checked against real model output and
  the cost of a real run is unmeasured.

Smaller known gaps, all visible in the code: `cmd_retry` still zeroes
`attempt_count` with raw SQL after going through `set_status`; the grounding
check binds evidence to the score threshold, so a low-scoring result may still
cite nothing; and `save_edited_brief` is written against a column no node fills.

## Docs

`docs/superpowers/specs/2026-07-24-offerpilot-design.md` is the design spec and
describes a larger system than the one above. It carries a status banner saying
what is actually built. `docs/superpowers/plans/` holds the implementation
plans, including the remaining Week 2 work.
