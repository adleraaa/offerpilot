"""Build, hand out and collect the second blind evaluation set.

Flow:
1. `sample_version_ids` draws a stratified random sample from every collected
   job version, whatever its status: student-level titles in one stratum,
   everything else in the other. A plain random draw from a corpus that is
   mostly senior roles would hold a handful of good fits, and recall over a
   handful means nothing.
2. `export_eval_db` copies just those versions into a fresh database, reset to
   status `new` with no filter results, so the comparison can run each
   pipeline version from scratch on identical jobs.
3. `write_label_sheet` writes a spreadsheet per labeler: the candidate profile,
   the labeling rule, and the jobs in a shuffled order. Nothing the pipeline
   produced is on it, which is what makes the labels blind.
4. `read_label_sheet`, `cohen_kappa`, `merge_labels` and `record_consensus`
   bring the sheets back. Only agreed (or jointly settled) labels become
   `blind_eval` rows.
"""

import random
import re
import sqlite3

from offerpilot.labels import FIT_LABELS
from offerpilot.store import db

_STUDENT = re.compile(
    r"\b(?:intern|internship|co-?op|new\s+grad(?:uate)?|university\s+grad\w*|"
    r"early\s+career|campus|apprentice\w*)\b", re.I)


def is_student_level(title: str) -> bool:
    return bool(_STUDENT.search(title or ""))


def sample_version_ids(conn, *, n_student: int, n_other: int,
                       seed: int) -> list[int]:
    rows = conn.execute("SELECT id, title FROM job_versions ORDER BY id"
                        ).fetchall()
    student = [r["id"] for r in rows if is_student_level(r["title"])]
    other = [r["id"] for r in rows if not is_student_level(r["title"])]
    rng = random.Random(seed)
    picked = (rng.sample(student, min(n_student, len(student)))
              + rng.sample(other, min(n_other, len(other))))
    return sorted(picked)


def export_eval_db(src, dst_path: str, version_ids) -> sqlite3.Connection:
    out = db.connect(dst_path)
    db.init_schema(out)
    ids = sorted(set(version_ids))
    marks = ",".join("?" * len(ids))
    versions = src.execute(
        f"SELECT * FROM job_versions WHERE id IN ({marks}) ORDER BY id",
        ids).fetchall()
    job_ids = sorted({v["job_id"] for v in versions})
    jmarks = ",".join("?" * len(job_ids))
    jobs = src.execute(f"SELECT * FROM jobs WHERE id IN ({jmarks})",
                       job_ids).fetchall()
    companies = src.execute(
        f"SELECT * FROM companies WHERE id IN (SELECT company_id FROM jobs "
        f"WHERE id IN ({jmarks}))", job_ids).fetchall()
    out.executemany("INSERT INTO companies(id, name) VALUES(?,?)",
                    [(c["id"], c["name"]) for c in companies])
    out.executemany(
        "INSERT INTO jobs(id, source, external_id, company_id, canonical_url, "
        "first_seen_at, last_seen_at, active) VALUES(?,?,?,?,?,?,?,?)",
        [(j["id"], j["source"], j["external_id"], j["company_id"],
          j["canonical_url"], j["first_seen_at"], j["last_seen_at"],
          j["active"]) for j in jobs])
    # Ids are kept, so a label, a sheet row and a source row all name the same
    # job. Status goes back to `new`: the comparison prefilters from scratch.
    out.executemany(
        "INSERT INTO job_versions(id, job_id, content_hash, title, location, "
        "url, description_text, posted_at, collected_at, status) "
        "VALUES(?,?,?,?,?,?,?,?,?, 'new')",
        [(v["id"], v["job_id"], v["content_hash"], v["title"], v["location"],
          v["url"], v["description_text"], v["posted_at"], v["collected_at"])
         for v in versions])
    out.commit()
    return out


SHEET_COLUMNS = ["job_id", "company", "title", "location", "posting_url",
                 "description", "label", "note"]

INSTRUCTIONS = [
    "How to label (takes about 1-2 minutes per job)",
    "",
    "You are judging each job for ONE student: the profile on the 'Profile' tab.",
    "Read the title and description (or open the posting_url), then set 'label':",
    "  good_fit  - this student could realistically apply and would want to:",
    "              eligible, roughly the right level, related to their skills.",
    "  poor_fit  - clearly not for this student (too senior, unrelated field,",
    "              requires something they do not have).",
    "  uncertain - you really cannot tell. Use it rarely.",
    "",
    "Rules:",
    "- Work alone. Do not compare answers until both sheets are done.",
    "- Do not look up what OfferPilot said about a job. The labels must be",
    "  blind, or the evaluation measures the model agreeing with itself.",
    "- Treat graduation-year wording ('Class of 2027') as information, not as",
    "  an automatic no; decide whether the student would still want to apply.",
    "- 'note' is optional: one short reason helps when we settle disagreements.",
]


def _profile_rows(profile):
    p = profile.model_dump()
    rows = [("Name", p["identity"].get("name")),
            ("Education", p["identity"].get("education")),
            ("Graduation", p["identity"].get("graduation"))]
    c = p["constraints"]
    rows += [("Locations", ", ".join(c.get("locations") or [])),
             ("Remote OK", str(c.get("remote_ok"))),
             ("Pay floor ($/hr)", str(c.get("pay_floor_hourly_usd"))),
             ("Work authorization", c.get("work_authorization")),
             ("Looking for", ", ".join(c.get("employment_types") or []))]
    for group, items in (p.get("skills") or {}).items():
        rows.append((f"Skills: {group}", ", ".join(items or [])))
    for e in p.get("experiences") or []:
        rows.append((f"Project: {e.get('title')}", e.get("summary")))
    return rows


def write_label_sheet(conn, path: str, profile, *, seed: int) -> int:
    import openpyxl
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation

    rows = conn.execute(
        "SELECT jv.id, j.company_id, c.name AS company, jv.title, jv.location, "
        "jv.url, jv.description_text FROM job_versions jv "
        "JOIN jobs j ON j.id = jv.job_id "
        "LEFT JOIN companies c ON c.id = j.company_id ORDER BY jv.id"
    ).fetchall()
    rows = list(rows)
    random.Random(seed).shuffle(rows)

    wb = openpyxl.Workbook()
    ws_i = wb.active
    ws_i.title = "Instructions"
    for line in INSTRUCTIONS:
        ws_i.append([line])
    ws_i["A1"].font = Font(bold=True, size=13)
    ws_i.column_dimensions["A"].width = 90

    ws_p = wb.create_sheet("Profile")
    for k, v in _profile_rows(profile):
        ws_p.append([k, v])
    ws_p.column_dimensions["A"].width = 26
    ws_p.column_dimensions["B"].width = 100

    ws = wb.create_sheet("Jobs")
    ws.append(SHEET_COLUMNS)
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for r in rows:
        ws.append([r["id"], r["company"] or r["company_id"], r["title"],
                   r["location"], r["url"],
                   (r["description_text"] or "")[:32000], None, None])
    dv = DataValidation(type="list", formula1='"good_fit,poor_fit,uncertain"',
                        allow_blank=True)
    ws.add_data_validation(dv)
    label_col = openpyxl.utils.get_column_letter(SHEET_COLUMNS.index("label") + 1)
    dv.add(f"{label_col}2:{label_col}{len(rows) + 1}")
    widths = {"A": 9, "B": 16, "C": 40, "D": 22, "E": 30, "F": 80, "G": 12,
              "H": 30}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=False, vertical="top")
    ws.freeze_panes = "C2"
    wb.active = wb.index(ws_i)
    wb.save(path)
    return len(rows)


def normalize_label(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def read_label_sheet(path: str) -> dict[int, str]:
    """{job_version_id: fit_label} for every row with a label filled in."""
    import openpyxl
    ws = openpyxl.load_workbook(path, read_only=True)["Jobs"]
    rows = ws.iter_rows(values_only=True)
    header = [str(h) for h in next(rows)]
    id_i, label_i = header.index("job_id"), header.index("label")
    out = {}
    for row in rows:
        if row[id_i] is None:
            continue
        label = normalize_label(row[label_i])
        if label is None:
            continue
        if label not in FIT_LABELS:
            raise ValueError(f"job {row[id_i]}: unknown label {row[label_i]!r}; "
                             f"use one of {sorted(FIT_LABELS)}")
        out[int(row[id_i])] = label
    return out


def cohen_kappa(a: dict, b: dict):
    """Agreement beyond chance over the items both labelers labeled."""
    common = sorted(set(a) & set(b))
    if not common:
        return None
    n = len(common)
    cats = sorted({a[k] for k in common} | {b[k] for k in common})
    po = sum(a[k] == b[k] for k in common) / n
    pe = sum((sum(a[k] == c for k in common) / n)
             * (sum(b[k] == c for k in common) / n) for c in cats)
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def merge_labels(a: dict, b: dict):
    """(agreed {id: label}, sorted ids both labeled differently)."""
    common = set(a) & set(b)
    agreed = {k: a[k] for k in sorted(common) if a[k] == b[k]}
    disagreed = sorted(k for k in common if a[k] != b[k])
    return agreed, disagreed


def record_consensus(conn, labels: dict, *, note: str) -> int:
    for vid in labels:
        if db.get_labels(conn, version_id=vid, label_source="blind_eval"):
            raise ValueError(f"job version {vid} already has a blind label")
    for vid, label in labels.items():
        if label not in FIT_LABELS:
            raise ValueError(f"unknown label {label!r}")
        db.record_label(conn, vid, label_source="blind_eval", fit_label=label,
                        notes=note)
    return len(labels)
