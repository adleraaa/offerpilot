"""The second blind evaluation set (PRV 201 M3 / iCNS poster).

The August set was 40 labels written in two batches, picked from jobs the
pipeline had already processed. This one is sampled from everything collected,
labeled in a spreadsheet by two people who never see a model output, and only
the labels both agree on (or settle together) become `blind_eval` rows.
"""

import pytest

from offerpilot import evalset
from offerpilot.models import NormalizedJob
from offerpilot.profile import load_profile
from offerpilot.store import db


def _add(conn, ext, title, company="acme"):
    job = NormalizedJob(source="greenhouse", external_id=str(ext),
                        company_id=company, title=title, location="Remote",
                        url=f"https://x.co/{ext}",
                        canonical_url=f"https://x.co/{ext}",
                        description_text=f"Posting {ext}.")
    _, vid = db.upsert_job(conn, job)
    return vid


@pytest.mark.parametrize("title,expected", [
    ("Software Engineer Intern (Summer 2027)", True),
    ("2027 Summer Internship - Data", True),
    ("Software Engineer, New Grad", True),
    ("Early Career Analyst", True),
    ("Campus AI Research Engineer", True),
    ("University Graduate - Backend", True),
    ("Senior Staff Engineer", False),
    ("Internal Tools Engineer", False),
    ("International Tax Manager", False),
])
def test_student_level_title(title, expected):
    assert evalset.is_student_level(title) is expected


def test_sample_is_stratified_and_reproducible(conn):
    student = [_add(conn, i, f"SWE Intern {i}") for i in range(10)]
    other = [_add(conn, 100 + i, f"Senior Engineer {i}") for i in range(10)]
    a = evalset.sample_version_ids(conn, n_student=4, n_other=3, seed=7)
    b = evalset.sample_version_ids(conn, n_student=4, n_other=3, seed=7)
    assert a == b
    assert len(a) == 7
    assert len(set(a) & set(student)) == 4
    assert len(set(a) & set(other)) == 3


def test_sample_takes_everything_when_a_stratum_is_short(conn):
    student = [_add(conn, i, f"SWE Intern {i}") for i in range(2)]
    ids = evalset.sample_version_ids(conn, n_student=5, n_other=0, seed=1)
    assert sorted(ids) == sorted(student)


def test_export_copies_only_the_sample_and_resets_status(conn, tmp_path):
    keep = _add(conn, 1, "SWE Intern")
    _add(conn, 2, "Senior Engineer")
    db.set_status(conn, keep, "filtered_out")
    out = evalset.export_eval_db(conn, str(tmp_path / "eval.db"), [keep])
    rows = out.execute("SELECT id, status, title FROM job_versions").fetchall()
    assert [(r["id"], r["status"], r["title"]) for r in rows] == [
        (keep, "new", "SWE Intern")]
    assert out.execute("SELECT COUNT(*) n FROM filter_results"
                       ).fetchone()["n"] == 0


def test_label_sheet_round_trip(conn, tmp_path):
    ids = [_add(conn, i, f"SWE Intern {i}") for i in range(3)]
    profile = load_profile("profile.example.yaml")
    path = tmp_path / "A.xlsx"
    evalset.write_label_sheet(conn, str(path), profile, seed=3)

    import openpyxl
    wb = openpyxl.load_workbook(path)
    ws = wb["Jobs"]
    header = [c.value for c in ws[1]]
    assert "label" in header
    # Blind: nothing the pipeline produced may be on the sheet.
    assert not {"status", "score", "eligibility"} & set(header)
    label_col = header.index("label") + 1
    for row, label in zip(range(2, 5), ["good_fit", "poor_fit", "uncertain"]):
        ws.cell(row=row, column=label_col, value=label)
    wb.save(path)

    labels = evalset.read_label_sheet(str(path))
    assert sorted(labels) == sorted(ids)
    assert sorted(labels.values()) == ["good_fit", "poor_fit", "uncertain"]


def test_reading_a_sheet_rejects_an_unknown_label(conn, tmp_path):
    _add(conn, 1, "SWE Intern")
    path = tmp_path / "A.xlsx"
    evalset.write_label_sheet(conn, str(path), load_profile(
        "profile.example.yaml"), seed=1)
    import openpyxl
    wb = openpyxl.load_workbook(path)
    ws = wb["Jobs"]
    col = [c.value for c in ws[1]].index("label") + 1
    ws.cell(row=2, column=col, value="maybe")
    wb.save(path)
    with pytest.raises(ValueError):
        evalset.read_label_sheet(str(path))


def test_blank_labels_are_left_out():
    assert evalset.normalize_label(None) is None
    assert evalset.normalize_label("  ") is None
    assert evalset.normalize_label(" Good_Fit ") == "good_fit"


def test_cohen_kappa_perfect_and_chance():
    a = {1: "good_fit", 2: "poor_fit", 3: "good_fit", 4: "poor_fit"}
    assert evalset.cohen_kappa(a, dict(a)) == pytest.approx(1.0)
    b = {1: "good_fit", 2: "good_fit", 3: "poor_fit", 4: "poor_fit"}
    assert evalset.cohen_kappa(a, b) == pytest.approx(0.0)


def test_merge_keeps_agreements_and_lists_disagreements():
    a = {1: "good_fit", 2: "poor_fit", 3: "good_fit", 4: "poor_fit"}
    b = {1: "good_fit", 2: "good_fit", 3: "good_fit"}
    agreed, disagreed = evalset.merge_labels(a, b)
    assert agreed == {1: "good_fit", 3: "good_fit"}
    assert disagreed == [2]


def test_record_consensus_writes_blind_eval_rows(conn):
    vid = _add(conn, 1, "SWE Intern")
    n = evalset.record_consensus(conn, {vid: "good_fit"}, note="A+B agree")
    assert n == 1
    rows = db.get_labels(conn, label_source="blind_eval")
    assert [(r["job_version_id"], r["fit_label"], r["notes"]) for r in rows] == [
        (vid, "good_fit", "A+B agree")]


def test_record_consensus_refuses_a_second_label(conn):
    vid = _add(conn, 1, "SWE Intern")
    evalset.record_consensus(conn, {vid: "good_fit"}, note="x")
    with pytest.raises(ValueError):
        evalset.record_consensus(conn, {vid: "poor_fit"}, note="y")
