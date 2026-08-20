"""Eval harness. Formal metrics use blind_eval labels only (spec section 5).

Two rules shape this file.

The first is that the decision being measured is the pipeline's, not the match
node's. `predicted_good_fit` is defined by the spec over the model's output,
but it is evaluated here end-to-end over `job_versions.status`, so a job the
prefilter threw out counts as a prediction of "no" and shows up as a false
negative when the human disagreed. Scoring only the jobs that survived the
prefilter would make the prefilter's mistakes invisible by construction, and
those are the expensive ones -- a job that never reached the model is a job
the user never saw.

The second is that only `blind_eval` labels count. A label written in the
review panel was written after seeing an 82/100, so it is partly a label about
the 82; scoring against it measures the system's agreement with itself. Those
rows are kept as auxiliary signal and excluded here.
"""

import json
import os
import re
from datetime import datetime, timezone

from offerpilot.config import git_commit
from offerpilot.store import db

# Spec section 5, verbatim:
#   predicted_good_fit = (eligibility != "fail") and (total_score >= threshold)
# ...evaluated end-to-end over status, so prefilter mistakes stay visible.
# `eligibility_failed` and `scored_low` are the two ways that expression goes
# false inside the graph; `filtered_out` is the way it never got asked.
PREDICTED_POSITIVE_STATUSES = frozenset({
    "pending_review", "approved", "rejected", "saved"})
PREDICTED_NEGATIVE_STATUSES = frozenset({
    "filtered_out", "eligibility_failed", "scored_low"})

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_PROPER_NOUN = re.compile(r"\b[A-Z][A-Za-z0-9.+#-]{2,}\b")
_STOPWORDS = frozenset({
    "The", "This", "That", "These", "Those", "They", "There", "Their", "Then",
    "Your", "You", "With", "While", "When", "Where", "What", "Which", "And",
    "But", "For", "From", "Into", "Both", "Also", "Because", "Built",
    "Building", "Shipped", "Used", "Using", "Worked", "Working", "Strong",
    "Matches", "Match", "Role", "Team", "Company", "Candidate", "Experience",
    "Gap", "Gaps", "Comfortable", "Turning", "Small"})


def predicted_positive(status: str):
    """True / False / None, where None means "the pipeline has not decided".

    Undecided versions (`new`, `ready_for_match`, `matching`, and both error
    states) are excluded from the metrics and counted separately: an errored
    job is a reliability number, and folding it into precision as a "no" would
    let a crashing pipeline look conservative.
    """
    if status in PREDICTED_POSITIVE_STATUSES:
        return True
    if status in PREDICTED_NEGATIVE_STATUSES:
        return False
    return None


def classification_metrics(pairs) -> dict:
    """`pairs` is a list of (predicted, actual) booleans."""
    tp = sum(1 for p, a in pairs if p and a)
    fp = sum(1 for p, a in pairs if p and not a)
    fn = sum(1 for p, a in pairs if not p and a)
    tn = sum(1 for p, a in pairs if not p and not a)
    # None, not 0.0: "no positive predictions" is an undefined precision, and
    # reporting it as zero would read as "every prediction was wrong".
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision,
            "recall": recall, "f1": f1, "n": len(pairs)}


def precision_at_k(ranked_actuals, k: int):
    """ranked_actuals: actual-positive booleans, best-predicted first.

    Divided by the length of the slice, not by k: with 3 labeled jobs,
    precision@10 over 3 hits is 1.0, not 0.3.
    """
    if not ranked_actuals:
        return None
    top = ranked_actuals[:k]
    return sum(1 for a in top if a) / len(top)


def _profile_text(profile) -> str:
    return json.dumps(profile.model_dump(), ensure_ascii=False, default=str)


def groundedness_flags(brief: dict, profile, job_text: str) -> dict:
    """Automated heuristics, not fact-checking (spec section 5).

    These are three cheap questions with objective answers: does every cited
    id exist, does every number in the prose appear in the profile or the
    posting, does every capitalised token. A flag means "a human should read
    this line", not "this is false" -- a brief that says "shipped it to real
    users" is unflagged and could still be wrong, and "Python" appearing in
    the profile does not make "expert in Python" true. Reported as a count of
    briefs worth reading, which is what the number is good for.
    """
    valid_ids = profile.experience_ids()
    cited = [e.get("source_id") for e in brief.get("cited_evidence") or []]
    # Talking points carry their own id, and a forgery hidden in one is just
    # as ungrounded as a forgery in cited_evidence.
    cited += [tp.get("evidence_source_id")
              for tp in brief.get("talking_points") or []]
    unknown_ids = sorted({c for c in cited if c and c not in valid_ids})

    prose_parts = [brief.get("why_it_fits") or "",
                   brief.get("outreach_paragraph") or ""]
    prose_parts += brief.get("main_gaps") or []
    prose_parts += brief.get("resume_bullets_to_emphasize") or []
    prose_parts += [tp.get("point", "")
                    for tp in brief.get("talking_points") or []]
    prose = " ".join(prose_parts)

    haystack = _profile_text(profile) + " " + job_text
    haystack_numbers = {n.replace(",", "") for n in _NUMBER.findall(haystack)}
    unsupported_numbers = sorted({
        n.replace(",", "") for n in _NUMBER.findall(prose)
        if n.replace(",", "") not in haystack_numbers})

    known_skills = set()
    for group in profile.skills.model_dump().values():
        known_skills.update(str(s) for s in (group or []))
    haystack_lower = haystack.lower()
    unsupported_proper_nouns = sorted({
        w for w in _PROPER_NOUN.findall(prose)
        if w not in _STOPWORDS
        and w not in known_skills
        and w.lower() not in haystack_lower})

    return {"unknown_source_ids": unknown_ids,
            "unsupported_numbers": unsupported_numbers,
            "unsupported_proper_nouns": unsupported_proper_nouns,
            "flag_count": (len(unknown_ids) + len(unsupported_numbers)
                           + len(unsupported_proper_nouns))}


def run_eval(conn, profile, *, results_dir: str = "evals/results",
             precision_at=(5, 10)) -> dict:
    blind = {}
    for row in db.get_labels(conn, label_source="blind_eval"):
        # The blind view refuses a second label per version, so this is one
        # row per key in practice; last-write-wins is the safe reading if a
        # database predating that rule still has two.
        blind[row["job_version_id"]] = row["fit_label"]

    pairs, ranked = [], []
    uncertain = undecided = prefilter_false_negatives = 0
    groundedness = {"briefs_checked": 0, "briefs_with_flags": 0,
                    "unknown_source_ids": 0, "unsupported_numbers": 0,
                    "unsupported_proper_nouns": 0}

    for version_id, fit in blind.items():
        row = conn.execute(
            "SELECT jv.status, jv.title, jv.description_text, j.company_id "
            "FROM job_versions jv JOIN jobs j ON j.id = jv.job_id "
            "WHERE jv.id=?", (version_id,)).fetchone()
        if row is None:
            continue
        if fit == "uncertain":
            uncertain += 1
            continue
        pred = predicted_positive(row["status"])
        if pred is None:
            undecided += 1
            continue
        actual = fit == "good_fit"
        # Reported on its own as well as inside `fn`: this is the subset of
        # false negatives no LLM ever saw, and the only one fixable in
        # `prefilter.py` rather than in a prompt.
        if row["status"] == "filtered_out" and actual:
            prefilter_false_negatives += 1
        pairs.append((pred, actual))
        item = db.get_review_item(conn, version_id)
        # Never-scored jobs sort below every scored one rather than above.
        ranked.append((item["total_score"] if item is not None else -1, actual))
        if item is not None:
            brief_json = item["edited_brief_json"] or item["brief_json"]
            if brief_json:
                # Title and company are part of the posting the brief was
                # written about, so a brief naming them is grounded.
                job_text = " ".join(filter(None, (
                    row["title"], row["company_id"], row["description_text"])))
                flags = groundedness_flags(json.loads(brief_json), profile,
                                           job_text)
                groundedness["briefs_checked"] += 1
                if flags["flag_count"]:
                    groundedness["briefs_with_flags"] += 1
                for key in ("unknown_source_ids", "unsupported_numbers",
                            "unsupported_proper_nouns"):
                    groundedness[key] += len(flags[key])

    ranked.sort(key=lambda t: t[0], reverse=True)
    ranked_actuals = [a for _, a in ranked]

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "labels": {"blind_labeled": len(blind), "uncertain_excluded": uncertain,
                   "undecided_excluded": undecided, "scored": len(pairs)},
        "classification": classification_metrics(pairs),
        "ranking": {f"precision_at_{k}": precision_at_k(ranked_actuals, k)
                    for k in precision_at},
        "prefilter_false_negatives": prefilter_false_negatives,
        "groundedness": groundedness,
        "note": ("Formal metrics use blind_eval labels only; review_feedback "
                 "labels are auxiliary signal (spec section 4)."),
    }
    os.makedirs(results_dir, exist_ok=True)
    stamp = result["generated_at"].replace(":", "").replace("-", "")
    path = os.path.join(results_dir, f"eval-{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    result["written_to"] = path
    return result
