# OfferPilot

[![tests](https://github.com/adleraaa/offerpilot/actions/workflows/ci.yml/badge.svg)](https://github.com/adleraaa/offerpilot/actions/workflows/ci.yml)

A local, human-in-the-loop job-search agent. It pulls postings from public ATS
APIs, drops the ones that violate hard constraints using deterministic rules,
scores the rest against a structured candidate profile with one LLM call,
writes a short application brief for the ones that clear the threshold, and
stops at a review queue in SQLite. Nothing is ever sent or submitted to an
employer. There is no outreach code in this repo: the only outbound network
calls are the collectors' GETs to the two ATS APIs and the two calls to the
configured LLM endpoint — scoring, and the brief for jobs that reach review.

```
collect -> prefilter -> match (LLM) -> gate -> brief (LLM) -> persist
           (6 rules,    (schema and    (code)  (only above    (pending_review,
            no LLM)      grounding              threshold)     terminal)
                         checked)
```

The four boxes after `collect` are a compiled LangGraph `StateGraph`
(`graph.build_match_graph`): nodes `match`, `brief` and `persist`, with the
gate as a conditional edge. The topology is fixed at import time and the model
never picks an edge.

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

## Match: the scoring node

`src/offerpilot/graph.py` and `src/offerpilot/llm.py`. One scoring step per job
version. The brief below is the only other model call in the pipeline, and it
runs only for the jobs the gate sends to review.

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
`permanent_error`, and nothing reaches the queue. The `persist` node then runs
that same validator again on whatever the client handed back, before the
result can reach the review queue: the repair turn depends on the client
honouring `validate`, the gate does not. The routing gate checks it too, so a
result that is about to be rejected never buys the model a brief written off
invented evidence. The same validator also refuses a
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

Costing is `llm.price_usage`, and the pre-call estimate and the ledger row both
go through it so the two cannot drift. It prices three things the flat
per-token rate this project started with got wrong. Cached and uncached prompt
tokens are billed separately, from `prompt_cache_hit_tokens` and
`prompt_cache_miss_tokens` when the endpoint reports them: a cache hit costs
about a thirtieth of a miss, and since every job in a batch resends the same
system prompt and the same profile, almost all of them are hits after the first
call. Any prompt token the split does not account for is charged at the miss
rate, because a ledger that under-charges is a fuse that never blows. The rate
doubles inside DeepSeek's peak windows, 01:00-04:00 and 06:00-10:00 UTC. And
the row is keyed on the model the server said it served, not the one that was
requested -- `deepseek-chat` is a legacy alias that comes back as
`deepseek-v4-flash`, so recording the request would have priced the wrong model
and filed a name that never ran. A served model missing from the price table
falls back to the configured one rather than raising, so an unrecognised model
costs an approximate number, not a dead batch.

The rates live in `config.yaml` under `llm.prices`, keyed by model, and are
taken from <https://api-docs.deepseek.com/quick_start/pricing/> as checked on
2026-08-20 against a live probe. They move; the shipped numbers are a starting
point to re-check, not a guarantee.

## Brief: the second LLM node

`src/offerpilot/brief.py`. When the gate routes a job to review, the graph
makes one more call and stores an `ApplicationBrief` in
`review_items.brief_json`: why it fits, the gaps, which resume bullets to
emphasize, four themed talking points and an optional outreach paragraph. It
is written for the human doing the applying, and nothing sends it anywhere.

The brief prompt is built like the match prompt — profile first, posting last
and delimited — and the match result is interpolated too. The model-written
parts of that result (`gaps`, `uncertainties`) are sanitized as well, because
they were written while reading the posting. `make_brief_validator` holds the
brief to the match node's grounding rule: every `evidence_source_id` must
exist in `profile.yaml`, and no talking point may claim to be tailored to an
employer's application questions, because none are ever collected.

The brief is a nice-to-have and the graph treats it as one: if the call fails,
the `brief` node logs a `brief_failed` step and the job still reaches
`pending_review` with its match intact. `run_match_for_version` makes the call
only when its caller asks for it — `match` asks, from `brief.enabled` in
`config.yaml`, default true.

## Gate and terminal state

Still plain Python, and still no model involvement: the gate is a conditional
edge (`graph._gate`, which writes nothing), and the write happens in the
`persist` node. `persist` re-reads the version's status and abandons the write
if anything else moved it. Then `eligibility == "fail"` goes to
`eligibility_failed`, a total below
`match.score_threshold` goes to `scored_low`, and anything else inserts a
`review_items` row holding the full validated `MatchResult` and the computed
total, with the version set to `pending_review`.

`pending_review` is where the automated pipeline ends and the human takes over.
`store/db.py` declares `pending_review -> approved | rejected | saved`, and the
review panel below is the only thing that performs those transitions. Every
status change goes through `db.set_status`, which refuses any transition not
listed in `ALLOWED_TRANSITIONS`.

## Review panel

`python -m offerpilot panel` serves `src/offerpilot/panel/` on
`127.0.0.1:8000` (`panel.host` / `panel.port` in `config.yaml`). The queue is
ordered by total score; opening a job shows the subscores, the cited evidence,
the gaps and uncertainties, the brief and the full posting text, and links out
to the original. An `eligibility` of `unknown` gets a banner saying so, because
unknown is not a pass.

Approve, save for later and reject are the only three moves, and a reject needs
a reason from the fixed vocabulary in `labels.py`: the API answers 422 without
one, and 409 rather than 500 if the job was already decided in another tab.
Neither refusal writes anything. Every decision that does go through writes a
row in `labels` with `label_source='review_feedback'`, which is what keeps
panel decisions separable from blind eval labels later.

The brief is editable. An edit is validated against the `ApplicationBrief`
schema and stored in `review_items.edited_brief_json`, next to — never over —
the model's original, so the two stay comparable.

There is no auth, no CORS and no session, and `serve` binds the loopback
interface. That is the design for a single-user local tool that can approve a
job: not reachable off the machine, so there is nothing to authenticate. The
bind is enforced, not assumed — `serve` refuses to start on any host that is
not loopback, including a `panel.host` of `0.0.0.0` in `config.yaml`, and the
tests assert on the address actually handed to uvicorn rather than on the
parameter's default. If you need the panel from another machine, tunnel to
`127.0.0.1`; do not widen the bind without adding auth first.

Job text stays untrusted all the way to the screen. The API returns it
verbatim as JSON, and `panel/static/panel.js` builds every node with
`textContent`; the markup-injecting DOM properties are banned outright in that
file, and a test greps the source for them. The posting's own URL gets the
same treatment, because it arrives inside the ATS payload and
`canonicalize_url` does not constrain its scheme: the detail view renders a
link only for `http`/`https` and falls back to plain text otherwise, so a
`javascript:` URL has nothing to run in.

## Why the pipeline is fixed

A job posting is text written by a stranger, fetched over the network, and put
into a prompt. Treating that as a prompt-injection surface rather than as
content is what sets the shape of the system:

1. Control flow is a graph whose topology is fixed in code, and every edge —
   including the gate — is decided by Python. The model picks no route, no
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
python -m offerpilot panel
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
`panel` serves the review panel described above and blocks until you stop it;
it makes no LLM calls, so it needs no API key.

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

199 tests, all passing, and CI runs them on every push. They need no network and
no API key: collectors are tested by parsing recorded payloads from
`tests/fixtures/`, the LLM client is tested against a fake SDK object, and the
panel is driven in-process with FastAPI's `TestClient`.

## Storage

One SQLite file. Written by the current code: `jobs`, `job_versions`,
`filter_results`, `companies`, `labels`, `runs`, `run_steps`, `review_items`,
`llm_usage`. `db.migrate` adds new columns to an existing database rather than
requiring a rebuild. `labels`, `review_items.edited_brief_json` and
`review_items.edited_at` are written by the review panel and by nothing else.

## What is not built

- No retrieval. No embeddings, no vector store, no Chroma or
  sentence-transformers. Evidence is the structured profile and nothing else.
- No blind labeling view. `db.get_blind_candidates` and the `blind_eval` label
  source exist, and the panel's nav links to `/blind`, but that page is not
  built: the link answers 404 today.
- No eval runner. The small blind-labeled evaluation set described in the design
  spec has not been assembled or run, so there are no fit, ranking or
  groundedness numbers.
- No demo mode. `match` needs a real key; `collect`, `status`, `retry` and
  `panel` do not.
- No Ashby collector and no Playwright careers-page collector. Greenhouse and
  Lever are the only sources.
- Neither LLM node has ever been run against a real API key. Every test of the
  match and brief nodes uses a stub, so neither prompt has been checked against
  real model output and the cost of a real run is unmeasured. One standalone
  probe on 2026-08-20 did call the API to read back a real `usage` object --
  that is where the pricing above comes from -- but it did not go through this
  code and scored nothing.

Smaller known gaps, all visible in the code: `cmd_retry` still zeroes
`attempt_count` with raw SQL after going through `set_status`; and the grounding
check binds evidence to the score threshold, so a low-scoring result may still
cite nothing.

## Docs

`docs/superpowers/specs/2026-07-24-offerpilot-design.md` is the design spec and
describes a larger system than the one above. It carries a status banner saying
what is actually built. `docs/superpowers/plans/` holds the implementation
plans, including the remaining Week 2 work.
