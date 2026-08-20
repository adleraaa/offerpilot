import json
import os

import pytest
import yaml

from offerpilot import cli
from offerpilot.evaluate import (
    classification_metrics, groundedness_flags, precision_at_k,
    predicted_positive, profile_fingerprint, run_eval,
)
from offerpilot.store import db

from tests.conftest import _make_job


def test_status_to_prediction_mapping_matches_spec():
    for s in ("pending_review", "approved", "rejected", "saved"):
        assert predicted_positive(s) is True
    for s in ("filtered_out", "eligibility_failed", "scored_low"):
        assert predicted_positive(s) is False
    for s in ("new", "ready_for_match", "matching", "retryable_error",
              "permanent_error"):
        assert predicted_positive(s) is None


def test_classification_metrics_on_a_hand_checked_confusion_matrix():
    # 2 TP, 1 FP, 1 FN, 1 TN
    pairs = [(True, True), (True, True), (True, False), (False, True),
             (False, False)]
    m = classification_metrics(pairs)
    assert m["tp"] == 2 and m["fp"] == 1 and m["fn"] == 1 and m["tn"] == 1
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)


def test_classification_metrics_handles_empty_and_zero_denominators():
    m = classification_metrics([])
    assert m["precision"] is None and m["recall"] is None and m["f1"] is None
    only_tn = classification_metrics([(False, False)])
    assert only_tn["precision"] is None and only_tn["recall"] is None


def test_precision_at_k_uses_rank_order_and_shrinks_when_short():
    assert precision_at_k([True, True, False, True], 2) == pytest.approx(1.0)
    assert precision_at_k([True, False, False, False], 4) == pytest.approx(0.25)
    assert precision_at_k([True], 5) == pytest.approx(1.0)
    assert precision_at_k([], 5) is None


def test_groundedness_flags_unknown_source_id(profile):
    brief = {"why_it_fits": "x", "cited_evidence": [
        {"source_id": "ghost", "section": "", "supporting_text": "y"}],
        "main_gaps": [], "resume_bullets_to_emphasize": [],
        "talking_points": [], "outreach_paragraph": None}
    flags = groundedness_flags(brief, profile, "job text")
    assert flags["unknown_source_ids"] == ["ghost"]


def test_groundedness_flags_numbers_absent_from_profile_and_posting(profile):
    brief = {"why_it_fits": "Shipped to 40000 users.", "cited_evidence": [],
             "main_gaps": [], "resume_bullets_to_emphasize": [],
             "talking_points": [], "outreach_paragraph": None}
    flags = groundedness_flags(brief, profile, "We build agent tooling.")
    assert "40000" in flags["unsupported_numbers"]


def test_groundedness_ignores_numbers_present_in_the_posting(profile):
    brief = {"why_it_fits": "Matches the 2029 graduation window.",
             "cited_evidence": [], "main_gaps": [],
             "resume_bullets_to_emphasize": [], "talking_points": [],
             "outreach_paragraph": None}
    flags = groundedness_flags(brief, profile, "Class of 2029 welcome.")
    assert flags["unsupported_numbers"] == []


def test_groundedness_flags_proper_nouns_from_nowhere(profile):
    brief = {"why_it_fits": "Used Kubernetes at Netflix.", "cited_evidence": [],
             "main_gaps": [], "resume_bullets_to_emphasize": [],
             "talking_points": [], "outreach_paragraph": None}
    flags = groundedness_flags(brief, profile, "We build agent tooling.")
    assert "Netflix" in flags["unsupported_proper_nouns"]


def test_groundedness_reads_talking_point_evidence_ids_too(profile):
    """A forged id hidden in a talking point is still a forged id."""
    brief = {"why_it_fits": "x", "cited_evidence": [], "main_gaps": [],
             "resume_bullets_to_emphasize": [],
             "talking_points": [{"theme": "relevant_project", "point": "p",
                                 "evidence_source_id": "phantom",
                                 "generic": True}],
             "outreach_paragraph": None}
    flags = groundedness_flags(brief, profile, "job text")
    assert flags["unknown_source_ids"] == ["phantom"]
    assert flags["flag_count"] >= 1


def test_run_eval_counts_prefilter_false_negatives(conn, tmp_path, profile):
    # A job the prefilter dropped that the human blind-labeled as a good fit.
    _, dropped = db.upsert_job(conn, _make_job("1"))
    db.set_status(conn, dropped, "filtered_out")
    db.record_label(conn, dropped, label_source="blind_eval",
                    fit_label="good_fit")
    # A job that reached review and the human agrees with.
    _, kept = db.upsert_job(conn, _make_job("2"))
    db.set_status(conn, kept, "ready_for_match")
    db.set_status(conn, kept, "matching")
    conn.execute("INSERT INTO review_items(job_version_id, match_json, "
                 "total_score) VALUES(?,?,?)",
                 (kept, '{"eligibility":"pass"}', 80))
    conn.commit()
    db.set_status(conn, kept, "pending_review")
    db.record_label(conn, kept, label_source="blind_eval", fit_label="good_fit")

    out = run_eval(conn, profile, results_dir=str(tmp_path / "results"),
                   precision_at=[5])
    assert out["prefilter_false_negatives"] == 1
    assert out["classification"]["tp"] == 1
    assert out["classification"]["fn"] == 1
    assert out["labels"]["blind_labeled"] == 2


def test_run_eval_ignores_review_feedback_labels(conn, tmp_path, profile):
    _, vid = db.upsert_job(conn, _make_job("1"))
    db.set_status(conn, vid, "filtered_out")
    db.record_label(conn, vid, label_source="review_feedback",
                    fit_label="good_fit")
    out = run_eval(conn, profile, results_dir=str(tmp_path / "r"),
                   precision_at=[5])
    assert out["labels"]["blind_labeled"] == 0
    assert out["classification"]["tp"] == 0


def test_run_eval_excludes_uncertain_and_undecided_labels(conn, tmp_path,
                                                          profile):
    """Only decided statuses and decided labels enter the confusion matrix."""
    _, unsure = db.upsert_job(conn, _make_job("1"))
    db.set_status(conn, unsure, "filtered_out")
    db.record_label(conn, unsure, label_source="blind_eval",
                    fit_label="uncertain")
    _, midflight = db.upsert_job(conn, _make_job("2"))
    db.set_status(conn, midflight, "ready_for_match")
    db.record_label(conn, midflight, label_source="blind_eval",
                    fit_label="poor_fit")

    out = run_eval(conn, profile, results_dir=str(tmp_path / "r"),
                   precision_at=[5])
    assert out["labels"]["blind_labeled"] == 2
    assert out["labels"]["uncertain_excluded"] == 1
    assert out["labels"]["undecided_excluded"] == 1
    assert out["labels"]["scored"] == 0
    assert out["classification"]["n"] == 0


def test_run_eval_ranks_by_score_for_precision_at_k(conn, tmp_path, profile):
    """precision@k must read the model's own ordering, not insertion order."""
    def reviewed(ext, score, fit):
        _, vid = db.upsert_job(conn, _make_job(ext))
        db.set_status(conn, vid, "ready_for_match")
        db.set_status(conn, vid, "matching")
        conn.execute("INSERT INTO review_items(job_version_id, match_json, "
                     "total_score) VALUES(?,?,?)",
                     (vid, '{"eligibility":"pass"}', score))
        conn.commit()
        db.set_status(conn, vid, "pending_review")
        db.record_label(conn, vid, label_source="blind_eval", fit_label=fit)

    reviewed("1", 61, "poor_fit")     # inserted first, ranked last
    reviewed("2", 95, "good_fit")

    out = run_eval(conn, profile, results_dir=str(tmp_path / "r"),
                   precision_at=[1, 2])
    assert out["ranking"]["precision_at_1"] == pytest.approx(1.0)
    assert out["ranking"]["precision_at_2"] == pytest.approx(0.5)


def test_run_eval_flags_an_ungrounded_brief(conn, tmp_path, profile):
    _, vid = db.upsert_job(conn, _make_job("1"))
    db.set_status(conn, vid, "ready_for_match")
    db.set_status(conn, vid, "matching")
    conn.execute("INSERT INTO review_items(job_version_id, match_json, "
                 "total_score) VALUES(?,?,?)",
                 (vid, '{"eligibility":"pass"}', 80))
    conn.commit()
    db.set_status(conn, vid, "pending_review")
    db.save_brief(conn, vid, json.dumps(
        {"why_it_fits": "Shipped to 40000 users at Netflix.",
         "cited_evidence": [{"source_id": "ghost", "section": "",
                             "supporting_text": "y"}],
         "main_gaps": [], "resume_bullets_to_emphasize": [],
         "talking_points": [], "outreach_paragraph": None}))
    db.record_label(conn, vid, label_source="blind_eval", fit_label="good_fit")

    out = run_eval(conn, profile, results_dir=str(tmp_path / "r"),
                   precision_at=[5])
    g = out["groundedness"]
    assert g["briefs_checked"] == 1
    assert g["briefs_with_flags"] == 1
    assert g["unknown_source_ids"] == 1
    assert g["unsupported_numbers"] == 1
    assert g["unsupported_proper_nouns"] >= 1


def test_run_eval_writes_a_timestamped_result_with_the_git_commit(conn,
                                                                  tmp_path,
                                                                  profile):
    results = tmp_path / "results"
    out = run_eval(conn, profile, results_dir=str(results), precision_at=[5])
    files = list(results.glob("eval-*.json"))
    assert len(files) == 1
    saved = json.loads(files[0].read_text(encoding="utf-8"))
    assert saved["git_commit"] == out["git_commit"]
    assert "generated_at" in saved


def test_evaluate_reuses_the_single_git_commit_helper():
    """One implementation of provenance, not a private copy per module."""
    from offerpilot import config, evaluate
    assert evaluate.git_commit is config.git_commit


def test_cli_eval_command_writes_a_result_naming_its_inputs(tmp_path):
    results = tmp_path / "results"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(
        {"match": {"score_threshold": 60, "max_auto_retries": 3},
         "eval": {"results_dir": str(results), "precision_at": [5]}}),
        encoding="utf-8")
    # The database has to exist: `eval` refuses to create one, because an
    # eval over an empty database reports nothing and looks like a finding.
    db_path = tmp_path / "e.db"
    db.init_schema(db.connect(str(db_path)))
    cli.main(["eval", "--db", str(db_path), "--config", str(cfg_path),
              "--profile", "profile.example.yaml"])
    files = list(results.glob("eval-*.json"))
    assert len(files) == 1
    saved = json.loads(files[0].read_text(encoding="utf-8"))
    # Explicitly asking for the example profile is allowed; the result says so.
    assert saved["profile"]["path"] == "profile.example.yaml"
    assert os.path.samefile(saved["database"], db_path)


def test_run_eval_records_the_profile_it_scored_against(conn, tmp_path,
                                                        profile):
    """The artifact is committed as evidence, so it must name its inputs.

    Run against the wrong profile, every cited id in every brief counts as
    unknown and the groundedness numbers are garbage that looks exactly like
    a finding. Without provenance in the file, the two are indistinguishable
    once the warning has scrolled off the terminal.
    """
    results = tmp_path / "r"
    out = run_eval(conn, profile, results_dir=str(results), precision_at=[5],
                   profile_path="profile.yaml")
    saved = json.loads(next(results.glob("eval-*.json")).read_text("utf-8"))
    assert saved["profile"] == out["profile"]
    assert saved["profile"]["path"] == "profile.yaml"
    assert saved["profile"]["experience_ids"] == ["pathpilot"]
    assert len(saved["profile"]["sha256_16"]) == 16


def test_profile_fingerprint_tells_the_example_profile_apart(profile):
    from offerpilot.profile import load_profile
    example = profile_fingerprint(load_profile("profile.example.yaml"))
    assert example["sha256_16"] != profile_fingerprint(profile)["sha256_16"]
    assert example["experience_ids"] == ["sample_automation", "sample_project"]


def test_run_eval_records_the_database_it_read(conn, tmp_path, profile):
    out = run_eval(conn, profile, results_dir=str(tmp_path / "r"),
                   precision_at=[5])
    assert os.path.samefile(out["database"], tmp_path / "t.db")


def test_cli_eval_refuses_a_database_that_does_not_exist(tmp_path):
    """Zero labels and all-None metrics look like a finding, not a typo."""
    results = tmp_path / "results"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(
        {"match": {"score_threshold": 60, "max_auto_retries": 3},
         "eval": {"results_dir": str(results), "precision_at": [5]}}),
        encoding="utf-8")
    missing = tmp_path / "nope" / "absent.db"
    with pytest.raises(SystemExit):
        cli.main(["eval", "--db", str(missing), "--config", str(cfg_path),
                  "--profile", "profile.example.yaml"])
    assert not missing.exists()          # and it did not create one either
    assert not results.exists()


def test_run_eval_shim_and_cli_eval_guard_the_database_identically(tmp_path):
    """README calls these the same command; the guard has to agree."""
    import run_eval as shim
    with pytest.raises(SystemExit):
        shim.main(["--db", str(tmp_path / "absent.db"),
                   "--config", "config.example.yaml",
                   "--profile", "profile.example.yaml"])


def test_cli_eval_refuses_the_synthetic_example_profile_fallback(tmp_path):
    """Silently scoring against profile.example.yaml produces junk metrics."""
    results = tmp_path / "results"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(
        {"match": {"score_threshold": 60, "max_auto_retries": 3},
         "eval": {"results_dir": str(results), "precision_at": [5]}}),
        encoding="utf-8")
    db_path = tmp_path / "e.db"
    db.init_schema(db.connect(str(db_path)))
    with pytest.raises(SystemExit):
        cli.main(["eval", "--db", str(db_path), "--config", str(cfg_path),
                  "--profile", str(tmp_path / "no-such-profile.yaml")])
    assert not results.exists()


# --- regression net built from the first real brief (smoke run, 2026-08-20) ---

REAL_BRIEF_PROSE = (
    "This role aligns with my CS education at Stevens, my hands-on experience "
    "building LLM-powered AI tools, and the flexible remote schedule I need. "
    "Built OfferPilot, a human-in-the-loop job-search agent with rubric "
    "scoring. Developed PathPilot AI, a Next.js app that generates course "
    "plans. Created browser automation tools with strict safety boundaries."
)


def test_sentence_initial_verbs_are_not_proper_nouns(profile):
    """The first real brief flagged 'Created'/'Developed' -- all false."""
    brief = {"why_it_fits": REAL_BRIEF_PROSE, "cited_evidence": [],
             "main_gaps": [], "resume_bullets_to_emphasize": [],
             "talking_points": [], "outreach_paragraph": None}
    flags = groundedness_flags(brief, profile, "Stevens CS internship.")
    for false_positive in ("Created", "Developed", "Built", "This"):
        assert false_positive not in flags["unsupported_proper_nouns"]


def test_bullet_initial_verbs_are_not_proper_nouns(profile):
    """Resume bullets each start a new line, so each start is sentence-initial."""
    brief = {"why_it_fits": "", "cited_evidence": [], "main_gaps": [],
             "resume_bullets_to_emphasize": [
                 "Created a batch OCR pipeline",
                 "Deployed a Next.js app to Vercel",
                 "Led a two-person project"],
             "talking_points": [], "outreach_paragraph": None}
    flags = groundedness_flags(brief, profile, "job text")
    for w in ("Created", "Deployed", "Led"):
        assert w not in flags["unsupported_proper_nouns"]


def test_hyphenated_compound_of_known_words_is_not_flagged(profile):
    brief = {"why_it_fits": "I build LLM-powered tools.", "cited_evidence": [],
             "main_gaps": [], "resume_bullets_to_emphasize": [],
             "talking_points": [], "outreach_paragraph": None}
    flags = groundedness_flags(brief, profile, "We use LLM tooling.")
    assert "LLM-powered" not in flags["unsupported_proper_nouns"]


def test_a_genuine_mid_sentence_proper_noun_is_still_flagged(profile):
    """The heuristic must not be defanged into uselessness."""
    brief = {"why_it_fits": "I shipped this while working at Netflix.",
             "cited_evidence": [], "main_gaps": [],
             "resume_bullets_to_emphasize": [], "talking_points": [],
             "outreach_paragraph": None}
    flags = groundedness_flags(brief, profile, "We build agent tooling.")
    assert "Netflix" in flags["unsupported_proper_nouns"]


def test_the_real_brief_flags_only_genuinely_unsupported_names(profile):
    """The first real brief was 3-for-3 false positives; none may return.

    The `profile` fixture is deliberately minimal (one experience, one skill),
    so names it really does not contain -- OfferPilot, Next.js -- SHOULD be
    flagged. Asserting flag_count == 0 here would only prove the heuristic had
    been defanged.
    """
    brief = {"why_it_fits": REAL_BRIEF_PROSE, "cited_evidence": [],
             "main_gaps": [], "resume_bullets_to_emphasize": [],
             "talking_points": [], "outreach_paragraph": None}
    flags = groundedness_flags(brief, profile,
                               "Stevens CS part-time internship, remote.")
    flagged = set(flags["unsupported_proper_nouns"])
    assert flagged & {"OfferPilot", "Next.js"}, "heuristic went blind"
    assert not (flagged & {"Created", "Developed", "Built", "This",
                           "LLM-powered", "Stevens"}), flagged
