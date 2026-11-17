"""Pipeline v2 (PRV 201 M2 design): graduation year and location stop being
hard filters.

v1 drops a job the moment the prefilter finds a graduation window or an onsite
location that excludes the candidate. The August eval traced 6 of its 7 missed
good jobs to the graduation year, so v2 passes those two findings to the match
model as warnings instead, and the gate removes a job only for a quoted
location or work-authorization conflict. v1 stays the default so the two can
be run side by side on the same labeled jobs.
"""

import pytest

from offerpilot import graph, prefilter
from offerpilot.models import EvidenceRef, FilterResult, MatchResult, NormalizedJob
from offerpilot.profile import load_profile
from offerpilot.store import db


def _fr(rule, outcome, value=None):
    return FilterResult(outcome=outcome, rule=rule, extracted_value=value,
                        reason="r")


# --- prefilter.decide ------------------------------------------------------

def test_v1_still_drops_a_graduation_window_fail():
    results = [_fr("graduation_window", "fail"), _fr("pay_floor", "pass")]
    assert prefilter.decide(results) == "filtered_out"


def test_v2_lets_a_graduation_window_fail_through_as_a_warning():
    results = [_fr("graduation_window", "fail"), _fr("pay_floor", "pass")]
    assert prefilter.decide(results, version="v2") == "ready_for_match"


def test_v2_lets_a_location_fail_through_as_a_warning():
    assert prefilter.decide([_fr("location", "fail")],
                            version="v2") == "ready_for_match"


@pytest.mark.parametrize("rule", ["years_of_experience", "work_authorization",
                                  "excluded_company", "pay_floor"])
def test_v2_keeps_every_other_rule_a_hard_filter(rule):
    results = [_fr("graduation_window", "fail"), _fr(rule, "fail")]
    assert prefilter.decide(results, version="v2") == "filtered_out"


def test_unknown_pipeline_version_is_refused():
    with pytest.raises(ValueError):
        prefilter.decide([], version="v3")


def test_prefilter_warnings_lists_only_soft_fails():
    results = [_fr("graduation_window", "fail", "Class of 2026"),
               _fr("location", "pass"), _fr("pay_floor", "unknown")]
    assert prefilter.warnings(results) == [results[0]]


# --- MatchResult -----------------------------------------------------------

def _match(eligibility="pass", conflict=None, score=(25, 18, 12, 12, 18)):
    s, p, d, se, pr = score
    # A fail has to quote the posting (see the quote check below), so the
    # excerpt is a sentence the `env` posting really contains.
    return MatchResult(
        eligibility=eligibility,
        eligibility_evidence_excerpt=("Open to the Class of 2026"
                                      if eligibility == "fail" else None),
        eligibility_conflict_type=conflict,
        skills_score=s, project_score=p, domain_score=d, seniority_score=se,
        preference_score=pr,
        evidence=[EvidenceRef(source_id="sample_project", supporting_text="x")],
        confidence=0.7)


def test_conflict_type_defaults_to_none():
    assert _match().eligibility_conflict_type is None


def test_conflict_type_rejects_an_unknown_value():
    with pytest.raises(ValueError):
        _match(eligibility="fail", conflict="vibes")


# --- graph gate --------------------------------------------------------------

class _Stub:
    def __init__(self, result):
        self.result, self.prompts = result, []

    def structured(self, *, node, run_id, system, user, schema, validate=None):
        self.prompts.append((system, user))
        if validate is not None:
            validate(self.result)
        return self.result


@pytest.fixture()
def env(tmp_path):
    conn = db.connect(str(tmp_path / "t.db"))
    db.init_schema(conn)
    job = NormalizedJob(source="greenhouse", external_id="1", company_id="c",
                        title="SWE Intern", location="Austin, TX",
                        url="https://x.co/1", canonical_url="https://x.co/1",
                        description_text="Open to the Class of 2026. Onsite.")
    _, vid = db.upsert_job(conn, job)
    return conn, vid, load_profile("profile.example.yaml")


def _ready(conn, vid, results=()):
    db.record_filter_results(conn, vid, list(results))
    db.set_status(conn, vid, "ready_for_match")
    return conn.execute("SELECT * FROM job_versions WHERE id=?",
                        (vid,)).fetchone()


def _run(conn, row, profile, match, version):
    stub = _Stub(match)
    status = graph.run_match_for_version(
        conn, stub, profile, row, threshold=60, max_auto_retries=3,
        pipeline_version=version)
    return status, stub


def test_v1_gate_drops_any_eligibility_fail(env):
    conn, vid, profile = env
    row = _ready(conn, vid)
    status, _ = _run(conn, row, profile,
                     _match("fail", "graduation_year"), "v1")
    assert status == "eligibility_failed"


def test_v2_gate_sends_a_graduation_conflict_to_review(env):
    conn, vid, profile = env
    row = _ready(conn, vid)
    status, _ = _run(conn, row, profile,
                     _match("fail", "graduation_year"), "v2")
    assert status == "pending_review"


def test_v2_gate_treats_an_untyped_fail_as_a_warning(env):
    conn, vid, profile = env
    row = _ready(conn, vid)
    status, _ = _run(conn, row, profile, _match("fail", None), "v2")
    assert status == "pending_review"


@pytest.mark.parametrize("conflict", ["location", "work_authorization"])
def test_v2_gate_still_drops_a_quoted_hard_conflict(env, conflict):
    conn, vid, profile = env
    row = _ready(conn, vid)
    status, _ = _run(conn, row, profile, _match("fail", conflict), "v2")
    assert status == "eligibility_failed"


def test_v2_soft_fail_with_a_low_score_is_scored_low(env):
    conn, vid, profile = env
    row = _ready(conn, vid)
    status, _ = _run(conn, row, profile,
                     _match("fail", "graduation_year", (5, 5, 5, 5, 5)), "v2")
    assert status == "scored_low"


# --- quote check -------------------------------------------------------------
# M2: Python checks that every quote really appears in the posting. In v2 a
# fail is either a removal (location, work authorization) or a warning the
# student reads, so the sentence behind it has to be the posting's own.
# The `env` posting: title "SWE Intern", location "Austin, TX", description
# "Open to the Class of 2026. Onsite."

def _fail(quote, conflict="graduation_year"):
    return MatchResult(
        eligibility="fail", eligibility_evidence_excerpt=quote,
        eligibility_conflict_type=conflict,
        skills_score=25, project_score=18, domain_score=12,
        seniority_score=12, preference_score=18,
        evidence=[EvidenceRef(source_id="sample_project", supporting_text="x")],
        confidence=0.7)


def _validator(env, version="v2"):
    conn, vid, profile = env
    row = _ready(conn, vid)
    return graph.make_evidence_validator(profile, 60, version, job_row=row)


def test_v2_rejects_a_fail_whose_quote_is_not_in_the_posting(env):
    with pytest.raises(ValueError, match="word for word"):
        _validator(env)(_fail("Must graduate between Dec 2026 and June 2027"))


def test_v2_accepts_a_fail_quoted_from_the_description(env):
    _validator(env)(_fail("Open to the Class of 2026"))


def test_v2_quote_check_ignores_case_and_spacing(env):
    _validator(env)(_fail("open to   the class\nof 2026."))


def test_v2_quote_check_accepts_a_quote_from_the_location_field(env):
    _validator(env)(_fail("Austin, TX", conflict="location"))


def test_v2_quote_check_folds_typographic_quotes_and_dashes(tmp_path):
    conn = db.connect(str(tmp_path / "q.db"))
    db.init_schema(conn)
    job = NormalizedJob(
        source="greenhouse", external_id="2", company_id="c",
        title="SWE Intern", location="Austin, TX", url="https://x.co/2",
        canonical_url="https://x.co/2",
        description_text="You’ll work on‑site – no remote option.")
    _, vid = db.upsert_job(conn, job)
    row = _ready(conn, vid)
    profile = load_profile("profile.example.yaml")
    validate = graph.make_evidence_validator(profile, 60, "v2", job_row=row)
    validate(_fail("You'll work on-site - no remote option.", "location"))


# The next four come from replaying the 25 real "fail" answers recorded on
# 2026-08-20 through this check. Ten were refused, and most of those were
# honest quotes in a shape the first version did not allow for: the labelled
# "Location: ..." line of the prompt, or two sentences joined by an ellipsis.

def test_the_quote_check_reads_the_posting_exactly_as_the_prompt_shows_it(env):
    conn, vid, profile = env
    row = _ready(conn, vid)
    _, user = graph.build_prompts(row, profile, "v2")
    shown = user.split("<untrusted_job_posting>")[1].split(
        "</untrusted_job_posting>")[0]
    assert graph.quote_in_posting(shown, row)


def test_v2_quote_check_accepts_the_labelled_location_line(env):
    _validator(env)(_fail("Location: Austin, TX", conflict="location"))


def test_v2_quote_check_accepts_quotes_joined_by_an_ellipsis(env):
    validate = _validator(env)
    validate(_fail('"Location: Austin, TX" ... "Onsite."', "location"))
    validate(_fail("Open to the Class of 2026 … Onsite."))


def test_v2_quote_check_rejects_an_ellipsis_quote_with_an_invented_part(env):
    with pytest.raises(ValueError, match="word for word"):
        _validator(env)(_fail(
            "Open to the Class of 2026 ... relocation to Tokyo required"))


def test_v2_quote_check_rejects_a_quote_that_is_only_punctuation(env):
    with pytest.raises(ValueError, match="word for word"):
        _validator(env)(_fail('"."'))


def test_v1_does_not_check_the_quote(env):
    _validator(env, "v1")(_fail("nothing like this is in the posting"))


def test_v2_does_not_check_an_excerpt_on_a_pass(env):
    passing = _match().model_copy(
        update={"eligibility_evidence_excerpt": "not in the posting"})
    _validator(env)(passing)


class _DropsValidate:
    """A client that ignores `validate`: only the graph's own re-check is left."""

    def __init__(self, result):
        self.result = result

    def structured(self, *, node, run_id, system, user, schema, validate=None):
        return self.result


def test_v2_parks_an_invented_quote_even_if_the_client_skips_validation(env):
    conn, vid, profile = env
    row = _ready(conn, vid)
    status = graph.run_match_for_version(
        conn, _DropsValidate(_fail("Graduating in 2031 only")), profile, row,
        threshold=60, max_auto_retries=3, pipeline_version="v2")
    assert status == "permanent_error"
    assert conn.execute(
        "SELECT COUNT(*) n FROM review_items").fetchone()["n"] == 0


def test_v2_does_not_remove_a_job_on_a_location_quote_it_cannot_find(env):
    conn, vid, profile = env
    row = _ready(conn, vid)
    status = graph.run_match_for_version(
        conn, _DropsValidate(_fail("This role is onsite in Tokyo", "location")),
        profile, row, threshold=60, max_auto_retries=3, pipeline_version="v2")
    assert status == "permanent_error"


# --- prompts -----------------------------------------------------------------

def test_v2_prompt_carries_the_prefilter_warnings(env):
    conn, vid, profile = env
    row = _ready(conn, vid, [_fr("graduation_window", "fail",
                                 "Open to the Class of 2026")])
    _, stub = _run(conn, row, profile, _match(), "v2")
    system, user = stub.prompts[0]
    assert "PREFILTER WARNINGS" in user
    assert "Open to the Class of 2026" in user
    assert "graduation" in system.lower()


def test_v1_prompt_has_no_warning_block(env):
    conn, vid, profile = env
    row = _ready(conn, vid, [_fr("graduation_window", "fail",
                                 "Open to the Class of 2026")])
    _, stub = _run(conn, row, profile, _match(), "v1")
    _, user = stub.prompts[0]
    assert "PREFILTER WARNINGS" not in user


# --- cli ---------------------------------------------------------------------

def test_collect_uses_the_configured_pipeline_version(conn, monkeypatch):
    from offerpilot import cli
    job = NormalizedJob(
        source="greenhouse", external_id="9", company_id="acme",
        title="SWE Intern", location="Remote",
        url="https://x.co/9", canonical_url="https://x.co/9",
        description_text="Open to students graduating in 2026.")
    monkeypatch.setattr(cli, "_collect_company", lambda company: [job])
    profile = load_profile("profile.example.yaml")
    cfg = {"companies": [{"id": "acme", "ats": "greenhouse",
                          "ats_slug": "acme"}],
           "pipeline": {"version": "v2"}}
    cli.cmd_collect(conn, cfg, profile)
    statuses = [r["status"] for r in conn.execute(
        "SELECT status FROM job_versions")]
    rules = {r["rule"]: r["outcome"] for r in conn.execute(
        "SELECT rule, outcome FROM filter_results")}
    assert rules["graduation_window"] == "fail"
    assert statuses == ["ready_for_match"]
