# OfferPilot

[![tests](https://github.com/adleraaa/offerpilot/actions/workflows/ci.yml/badge.svg)](https://github.com/adleraaa/offerpilot/actions/workflows/ci.yml)

A local, human-in-the-loop job-search agent. It pulls postings from public ATS
APIs, drops the ones that violate hard constraints using deterministic rules,
scores the rest against a structured candidate profile with one LLM call,
writes a short application brief for the ones that clear the threshold, and
stops at a review queue in SQLite that a person works through by hand.

**Nothing is ever sent or submitted to an employer. The output terminates at
local drafts in a review queue.** There is no outreach code in this repo: the
only outbound network calls are the collectors' GETs to the two ATS APIs and
the two calls to the configured LLM endpoint — scoring, and the brief for jobs
that reach review.

## Try it (no API key)

Python 3.11 or newer.

```
pip install -e .
python -m offerpilot demo
```

That is the whole pipeline with no setup: no `config.yaml`, no `profile.yaml`,
no API key and no network call. It seeds a throwaway SQLite database in a temp
directory from the five synthetic postings and the synthetic profile in
`demo/`, runs them through the real prefilter and the real compiled graph with
a `MockLLM` replaying `demo/recorded_outputs.json`, and serves the review panel
on <http://127.0.0.1:8000>. Ctrl-C stops it; the temp database is disposable.

The fixtures cover the three states a clean run ends on: two jobs reach
`pending_review` (one of them with eligibility `unknown`, so the panel's
unresolved-eligibility banner has something to show), one is
`eligibility_failed`, and two are `filtered_out` by the deterministic rules —
one below the pay floor, one on the exclusion list. Those two have no recorded
output at all, and `MockLLM` raises `KeyError` rather than inventing one, so
the demo cannot quietly stop dropping them before the model call. `MockLLM`
also calls the same `validate` callback the real client does, so the grounding
check is armed on this path too. The blind labeling view is at
<http://127.0.0.1:8000/blind> and works on the same seeded data.

The other three states a run can end on are not on this path: no fixture scores
below the threshold, and a recorded reply cannot fail the way a live one can,
so `scored_low`, `retryable_error` and `permanent_error` are exercised in
`tests/test_graph.py` rather than in the demo.

> **TODO — screenshots.** This README ships without images. Two are worth
> adding once someone can point a browser at the demo: `docs/images/panel.png`
> (the review panel with a queue item open, evidence and brief visible), which
> belongs just below this section, and `docs/images/blind.png` (the blind
> labeling view), which belongs in **Blind labeling**. Create `docs/images/`,
> drop the PNGs in, and add the two image lines — not before, because a broken
> image is worse than no image.

## How it works

```
collect -> prefilter (6 deterministic rules) -> match (LLM) -> gate
        -> brief (LLM) -> review queue -> labels -> eval
```

Everything from `match` to the queue is a compiled LangGraph `StateGraph`
(`graph.build_match_graph`): nodes `match`, `brief` and `persist`, with the
gate as a conditional edge. The topology is fixed at import time and the model
never picks an edge.

**Filtering is conservative.** Only a definite violation removes a job.
Every prefilter rule returns pass, fail or unknown, and anything the rules
cannot parse becomes unknown and goes on to scoring, because a missed filter
costs one LLM call while a wrong filter silently loses a job. The match model
is held to the same standard: it may only report `eligibility="fail"` if it
also returns the posting excerpt it is failing on.

**The score is arithmetic, not an opinion.** The model returns five bounded
subscores; `models.total_score` sums them in Python, and the gate compares that
sum against `match.score_threshold` from `config.yaml`. No number the model
writes is ever used as the total, and no model output decides a route.

**Job text is untrusted input, all the way to the screen.** Postings are
wrapped in an `<untrusted_job_posting>` block whose delimiter is stripped out
of the job text first, model output is schema-checked before anything is
written, every cited evidence id must exist in `profile.yaml`, and the panel
renders every job-derived and model-derived string with `textContent`.

**Spend is capped from a ledger in the same database.** Every API response
writes token counts and an estimated cost to `llm_usage`, and every call sums
today's rows first and refuses to start at or above `llm.daily_spend_cap_usd`.
The daily boundary is SQLite `date('now')`, so the cap resets at UTC midnight,
not local midnight.

**Work is resumable.** State lives on `job_versions` as a status, and every
change goes through `db.set_status`, which refuses any transition not in
`ALLOWED_TRANSITIONS`. A crash mid-run leaves a job at a status a later run can
pick up; no transaction is ever held open across a network call.

### Collect

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

### Prefilter

`src/offerpilot/prefilter.py`. Six rules, pure Python, no model:
`years_of_experience`, `work_authorization`, `location`, `excluded_company`,
`graduation_window`, `pay_floor`. Each returns a `FilterResult` with one of
three outcomes, pass, fail or unknown, plus the rule name, the matched text and
a reason. All six are persisted to `filter_results`. Only a fail filters a job
out.

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

### Match: the scoring node

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
invented evidence. The same validator also refuses a result that cites nothing
at all once it scores at or above the review threshold — an uncited
recommendation cannot reach a human. Below the threshold, and on an eligibility
fail, an empty evidence list is allowed: there the posting itself is the
evidence, and `MatchResult` already demands an excerpt from it. That check is in
code in `graph.py`. The prompt asks for the same thing, but asking is not
enforcement: a model that ignores the instruction still cannot get an invented
id past this check. The total score is summed in Python
(`models.total_score`), never read off the model.

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
and leaves the version resumable.

Costing is `llm.price_usage`, and the pre-call estimate and the ledger row both
go through it so the two cannot drift. It prices three things the flat
per-token rate this project started with got wrong. Cached and uncached prompt
tokens are billed separately, from `prompt_cache_hit_tokens` and
`prompt_cache_miss_tokens` when the endpoint reports them: a cache hit costs
about a thirtieth of a miss. Resending the same system prompt and the same
profile for every job in a batch is what makes hits possible at all, but they
are a minority of the bill, not most of it — across the six calls in the smoke
run below, the ledger billed 3,072 of 14,043 prompt tokens (21.9%) at the hit
rate, in 1,024-token blocks, and charged three of the six calls entirely at the
miss rate. Any prompt token the split does not account for is charged at the
miss rate, because a ledger that under-charges is a fuse that never blows. The
rate
doubles inside DeepSeek's peak windows, 01:00-04:00 and 06:00-10:00 UTC. And
the row is keyed on the model the server said it served, not the one that was
requested — `deepseek-chat` is a legacy alias that comes back as
`deepseek-v4-flash`, so recording the request would have priced the wrong model
and filed a name that never ran. A served model missing from the price table
falls back to the configured one rather than raising, so an unrecognised model
costs an approximate number, not a dead batch.

The rates live in `config.yaml` under `llm.prices`, keyed by model, and are
taken from <https://api-docs.deepseek.com/quick_start/pricing/> as checked on
2026-08-20 against a live probe. They move; the shipped numbers are a starting
point to re-check, not a guarantee.

### Brief: the second LLM node

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

### Gate and terminal state

Still plain Python, and still no model involvement: the gate is a conditional
edge (`graph._gate`, which writes nothing), and the write happens in the
`persist` node. `persist` re-reads the version's status and abandons the write
if anything else moved it. Then `eligibility == "fail"` goes to
`eligibility_failed`, a total below `match.score_threshold` goes to
`scored_low`, and anything else inserts a `review_items` row holding the full
validated `MatchResult` and the computed total, with the version set to
`pending_review`.

`pending_review` is where the automated pipeline ends and the human takes over.
`store/db.py` declares `pending_review -> approved | rejected | saved`, and the
review panel below is the only thing that performs those transitions.

### Review panel

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

Binding loopback is not the whole story, though, because a page already open in
your browser can reach loopback too: any site can point a hostname it controls
at `127.0.0.1` and script requests to it, and those arrive on the loopback
interface, from your own browser, with no auth to fail. The Host header is what
separates them from your own tab, so the app checks it: `127.0.0.1`,
`localhost` and whatever host `serve` was told to bind are allowed, and
anything else gets a 400. The check is a few lines of middleware rather than
Starlette's `TrustedHostMiddleware`, which strips the port at the *first*
colon and so reads the `Host: [::1]:8000` a browser sends to an IPv6 loopback
bind as the literal `[` — a supported `panel.host` that 400s its own page.
Brackets and port are parsed off properly here and the hostname is matched
whole, so `[::2]` is still refused by a `::1` bind. Tests drive requests
through the app `serve` builds, not just the arguments it hands uvicorn. FastAPI's interactive docs are off for the same
reason: `/docs` and `/redoc` are script tags pointing at a CDN, loaded into the
same origin as the page that approves jobs, and this app's only API consumer is
the two static pages next to it, so `docs_url`, `redoc_url` and `openapi_url`
are all `None` and a test asserts `/docs` is a 404.

The two pages carry a light and a dark palette, both of which are checked
rather than eyed. Every colour in `panel/static/style.css` is a custom property
defined on `:root`, the `prefers-color-scheme: dark` block redefines those
tokens and nothing else, and a test parses the stylesheet and computes the WCAG
contrast of each pair — 4.5:1 for body text, 3:1 for the eligibility banner —
in both themes. That test exists because the file used to declare
`color-scheme: light dark`, define only the light half, and leave `body` with no
background: on a dark-themed OS the browser painted its dark canvas under
near-black text, 1.08:1, and the queue and the posting body were invisible.

Job text stays untrusted all the way to the screen. The API returns it
verbatim as JSON, and `panel/static/panel.js` builds every node with
`textContent`; the markup-injecting DOM properties are banned outright in that
file, and a test greps the source for them. The posting's own URL gets the
same treatment, and it is guarded twice. `canonicalize_url` refuses any scheme
outside `http`/`https` at the collector boundary, so a `javascript:` or `data:`
URL never reaches the database from a real posting. The detail view then
renders a link only for `http`/`https` and falls back to plain text otherwise.
The second guard is not redundant: demo fixtures construct `NormalizedJob`
directly and never pass through `canonicalize_url`, so the render site is the
only thing standing between a hand-written `canonical_url` and an anchor.

### Blind labeling

`/blind`, served by the same process, is the review panel with the model
subtracted. It shows one job posting and a summary of your own profile, and
that is all it is sent: `GET /api/blind/next` carries no score, no subscores,
no eligibility verdict, no evidence, no brief and not even the job's status.
The guarantee is about the payload, not about what the page chooses to draw,
so a test asserts the response's exact set of keys rather than searching it
for field names it already knows about — a denylist of eight words would pass
a score shipped as `priority`, and this is the one place where a leak nobody
notices quietly invalidates every number the eval produces.

The reason is that these labels are the eval's ground truth. A label written
after seeing an 82/100 is partly a label about the 82, and an eval scored
against it measures agreement with itself. So the blind page writes
`label_source='blind_eval'`, the review panel writes
`label_source='review_feedback'`, and `evaluate.py` reads only the first kind.
A blind label writes a row in `labels` and moves nothing: the job's status is
the pipeline's business, and labelling must not be able to change what the
pipeline does.

Each job version takes exactly one blind label; a second is a 409 and the
page disables the verdict buttons until the first one lands. Ground truth
that contradicts itself is worse than missing ground truth, and since the
eval reads every `blind_eval` row it finds, a double-click would otherwise
score the model against both answers.

Candidates come from `db.get_blind_candidates`, which spans every job version
in the database — including the ones the prefilter threw out at `filtered_out`
and the ones that never reached a human. That is deliberate: if the blind
queue were the review queue, the prefilter's false negatives would be
invisible to the eval by construction. The header shows progress against the
40–60 labels the design spec asks for before the numbers mean anything.

### Storage

One SQLite file. Written by the current code: `jobs`, `job_versions`,
`filter_results`, `companies`, `labels`, `runs`, `run_steps`, `review_items`,
`llm_usage`. `db.migrate` adds new columns to an existing database rather than
requiring a rebuild. `labels`, `review_items.edited_brief_json` and
`review_items.edited_at` are written by the review panel and by nothing else.

### Why the pipeline is fixed

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

## Real mode

Real mode is the same pipeline against real postings and a real model. It needs
two config files and one environment variable.

```
cp config.example.yaml config.yaml
cp profile.example.yaml profile.yaml
```

Edit `config.yaml` for the company list, model, spend cap and score threshold,
and `profile.yaml` for your constraints and experiences. The experience ids in
`profile.yaml` are the evidence vocabulary the grounding check enforces. Both
files are gitignored; only the `.example.` copies are committed.

The model is reached with the OpenAI SDK pointed at `llm.base_url`, and the
example config uses DeepSeek. The key is read from the `DEEPSEEK_API_KEY`
environment variable only, never from a file and never from `config.yaml`.

```
export DEEPSEEK_API_KEY=...
python -m offerpilot collect
python -m offerpilot status
python -m offerpilot match
python -m offerpilot panel
```

`collect` fetches every company in `config.yaml`, upserts, prefilters and sets
status; a failure on one company or one job is printed and skipped rather than
fatal. `status` prints counts of `job_versions` by status. `match` scores
everything sitting at `ready_for_match`, and refuses to start if `profile.yaml`
is missing (so it cannot spend money scoring the synthetic example profile) or
if `DEEPSEEK_API_KEY` is unset. `panel` serves the review panel and the blind
labeling page and blocks until you stop it; it makes no LLM calls, so it needs
no API key.

```
python -m offerpilot retry
```

`retry` sweeps versions stuck in `matching` for over 15 minutes back to
`ready_for_match`, re-prefilters versions orphaned at `new` by a crash between
the insert and the prefilter, and resets both error states — `permanent_error`
and `retryable_error` — to `ready_for_match` with a zeroed attempt count.
`retryable_error` is normally a state the graph writes and leaves in the same
call, but the two transitions are separate commits, so a batch killed in
between parks a row there; before it was swept here, nothing could move it
again. That orphan sweep also runs at the start of `collect`, so a crashed run
self-heals.

Every subcommand takes `--db` (default `data/offerpilot.db`), `--config`
(default `config.yaml`), `--profile` (default `profile.yaml`) and `--limit`
(process at most N jobs, on `collect` and `match`). `demo` is the exception: it
reads no config, no profile and no `--db`, because it brings its own.

### Smoke run, 2026-08-20

The pipeline has been run end to end against the live Greenhouse API and the
live DeepSeek API once, on a throwaway copy of the database. Numbers below are
measured, not estimated.

**Collect.** 213 postings from two Greenhouse boards in 4.4s, no errors. The
deterministic prefilter dropped 52 of them before any model call: 32 failed
years-of-experience, 16 location and 9 work authorization, five of them
failing two rules at once — which is why those three add to 57, not 52.
`pay_floor` parsed nine hourly figures — five ranges (`$40 to $55/hr`,
`$60-80/hr`, …) and four single rates (`$75 per hour`, `$50/hour`, …) — and
passed all nine, correctly: none fell below the configured floor.
`graduation_window` returned `unknown` on all 213, also correctly — no posting
in this corpus states a class-year window, so there was nothing to decide on.

**Match and brief.** 5 jobs scored, 6 model calls (5 match, 1 brief).

| | |
|---|---|
| Model served | `deepseek-v4-flash` |
| Prompt tokens | 14,043 — 3,072 (21.9%) billed at the cache-hit rate, in 1,024-token blocks; 3 of the 6 calls billed entirely at the miss rate |
| Completion tokens | 15,950 |
| Cost | $0.0130 total, ~$0.0026 per job |
| Schema repair turns | 0 — every call validated on the first attempt |
| Invented `source_id`s | 0 — the grounding check never had to fire |
| Outcomes | 4 `eligibility_failed`, 1 `pending_review` with a brief |

Two things worth knowing before running a full batch:

- **Completion tokens exceed prompt tokens** (15,950 vs 14,043). This model
  emits reasoning, and it is billed as output at 3× the cache-miss input rate,
  so cost is driven by the reply rather than by the posting. Estimating from
  prompt size alone understates it by roughly an order of magnitude.
- **A full batch will hit the spend cap.** 161 of the 213 postings cleared the
  prefilter and 5 were scored, so 156 were left at `ready_for_match`. At
  $0.0026 per job those cost about $0.41, while `config.example.yaml` caps
  the day at $2.00 and the author's local cap was $0.10 — which would have
  stopped the run around job 38. That is the fuse doing its job, but set the
  cap deliberately rather than discovering it mid-run.

The four `eligibility_failed` verdicts were correct: two Tier-1 consulting
roles asking for years of post-graduation experience, one role in Doha, and one
outside the profile's constraints. Each cited the posting excerpt that decided
it, as the schema requires.

## Evaluation

The eval scores the pipeline against **a small blind-labeled evaluation set**:
40–60 jobs you label yourself in the blind view, with every model output
hidden. It is a sanity check on one person's search, sized so the numbers are
readable rather than statistically strong.

Labels are split by provenance and only one kind counts. A label given in the
review panel (`label_source='review_feedback'`) was given after seeing the
model's score, reasoning and brief, so it is anchored to them and scoring
against it would partly measure the model's agreement with itself. Formal
metrics therefore read `label_source='blind_eval'` rows only; panel labels are
kept as auxiliary signal.

```
python run_eval.py
```

`python -m offerpilot eval` is the same code path — `run_eval.py` is a shim
that hands its arguments to that branch, so the spec's name and the subcommand
enforce the same guards and neither needs an API key. Those guards are that
`eval` refuses a `--db` that does not exist and refuses a missing
`profile.yaml`: an eval over an empty database reports zero labels and
all-None metrics, and one run against the synthetic example profile marks
every cited evidence id ungrounded, so both would otherwise produce a result
file that reads like a finding rather than like a typo.

What it measures is the pipeline's decision, not the model's. The decision rule
is fixed: `predicted_good_fit = (eligibility != "fail") and (total_score >=
threshold)`, and `filtered_out`, `eligibility_failed` and `scored_low` all
count as a predicted "no" — so a job the prefilter dropped is a prediction, and
prefilter false negatives land in the numbers instead of being invisible by
construction. They are also reported on their own, because they are the subset
no model ever saw. The result file records precision, recall, F1 and the
confusion matrix over the good_fit / poor_fit labels (`uncertain` is excluded
and counted separately), Precision@5 and @10 over the score ranking, the
prefilter false-negative count, and groundedness flag counts over the briefs.
It also records the git commit, the database path, and the profile it scored
against — path, hash, and the experience ids groundedness is defined over — so
a committed artifact can be checked against its inputs later.

Each run writes `evals/results/eval-<timestamp>.json`, which is committed.
`evals/dataset/README.md` explains why the dataset is produced in place rather
than shipped as a file of copied postings. **There are no numbers to report
yet:** no blind labels have been collected, so `evals/results/` is empty.

## What is not built

- No retrieval. No embeddings, no vector store, no Chroma or
  sentence-transformers. Evidence is the structured profile and nothing else.
- No eval numbers. The harness is built and reads the `blind_eval` labels, but
  the labeled set has not been assembled, so there are no fit, ranking or
  groundedness numbers to report and `evals/results/` is empty. The 40–60
  target the blind page displays is a target, not a count that has been
  reached. The groundedness checks are heuristics — unknown evidence ids,
  numbers and capitalised tokens that appear in neither the profile nor the
  posting — so they flag lines worth reading, not lines that are false.
- No recorded *real* model outputs. `demo` exists and needs no key, but the
  outputs it replays from `demo/recorded_outputs.json` were written by hand to
  exercise each terminal state, not captured from a live model.
- No sustained real-model run. Both LLM nodes have now been exercised against
  the live API once (see **Smoke run, 2026-08-20**), but on 5 jobs, so prompt
  behaviour across a full batch and across job types is still largely unproven. One standalone
  probe on 2026-08-20 did call the API to read back a real `usage` object —
  that is where the pricing above comes from — but it did not go through this
  code and scored nothing. Only `match` needs a real key; `collect`, `status`,
  `retry`, `panel`, `eval` and `demo` do not.
- No Ashby collector and no Playwright careers-page collector. Greenhouse and
  Lever are the only sources.
- No live run of the Lever collector. It has unit tests over recorded payloads,
  but `config.yaml` has only ever listed Greenhouse boards, so its `fetch` has
  never been pointed at `api.lever.co` — every job in every database this
  project has produced came from Greenhouse. Its `parse` is exercised by
  `tests/test_collectors.py`; its network path is not.
- No research or tool-calling branch. The graph has three nodes and no tools;
  the model is never given one to call.
- No screenshots in this README. See the TODO above.

Smaller known gaps, all visible in the code: `cmd_retry` still zeroes
`attempt_count` with raw SQL after going through `set_status`; and the grounding
check binds evidence to the score threshold, so a low-scoring result may still
cite nothing.

## Project layout

| Path | What lives there |
|---|---|
| `src/offerpilot/collectors/` | Greenhouse and Lever fetch + parse |
| `src/offerpilot/prefilter.py` | The six deterministic rules |
| `src/offerpilot/graph.py` | The compiled `StateGraph`, gate and persist |
| `src/offerpilot/brief.py` | `ApplicationBrief` and the brief node |
| `src/offerpilot/llm.py` | Client, repair turn, pricing, spend cap |
| `src/offerpilot/prompts.py` | Match and brief prompts |
| `src/offerpilot/models.py` | `MatchResult`, `NormalizedJob`, scoring |
| `src/offerpilot/profile.py` | `profile.yaml` loading and evidence ids |
| `src/offerpilot/labels.py` | Label vocabularies and validation |
| `src/offerpilot/store/db.py` | Schema, status machine, all queries |
| `src/offerpilot/panel/` | FastAPI app plus the two static pages |
| `src/offerpilot/evaluate.py` | Metrics and groundedness heuristics |
| `src/offerpilot/demo.py` | `MockLLM` and the seeded demo run |
| `src/offerpilot/cli.py` | The seven subcommands |
| `run_eval.py` | Shim for `offerpilot eval` |
| `demo/` | Synthetic postings, profile and recorded outputs |
| `evals/` | Dataset notes and committed result files |
| `tests/` | The suite, including recorded API fixtures |
| `config.example.yaml` | Companies, model, prices, thresholds |
| `profile.example.yaml` | Profile shape, with synthetic content |
| `docs/superpowers/` | Design spec and implementation plans |

## Running the tests

```
pip install -e ".[dev]"
python -m pytest -q
```

308 tests, all passing, and CI runs them on every push. They need no network and
no API key: collectors are tested by parsing recorded payloads from
`tests/fixtures/`, the LLM client is tested against a fake SDK object, the panel
is driven in-process with FastAPI's `TestClient`, and the graph is exercised
with stub clients that return schema-valid objects.

## Docs

`docs/superpowers/specs/2026-07-24-offerpilot-design.md` is the design spec. It
was frozen before implementation and describes a larger system than the one
above, so read it as a record of intent, not as a description of the code.

It carries a build-status banner, and **that banner is itself out of date**: it
predates the Week 2 work and still lists the LangGraph orchestration, the brief
node, the review panel and the blind-labeled evaluation harness as not built.
All four exist and are described above. The spec is frozen — scope changes take
a new revision, not an edit — so this README, not the banner, is the current
account of what runs. What the banner still gets right is the rest of **What is
not built** above: no retrieval and no eval numbers. Its claim that neither LLM
node has run against a real key is also out of date — see **Smoke run,
2026-08-20**.

`docs/superpowers/plans/` holds the implementation plans, including the
remaining Week 2 work.
