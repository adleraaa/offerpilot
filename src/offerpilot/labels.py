"""Human-label vocabularies shared by the review panel, the blind labeling
view, and the eval harness.

The vocabularies are the single source of truth: `db.record_label` validates
`label_source` against `LABEL_SOURCES`, and the HTTP layer validates the rest
through `LabelInput`.
"""

from typing import Literal, Optional

from pydantic import BaseModel

FitLabel = Literal["good_fit", "poor_fit", "uncertain"]
ActionLabel = Literal["apply", "skip", "save"]
RejectionReason = Literal[
    "skills", "seniority", "location", "compensation", "duplicate",
    "expired", "not_interested", "bad_draft", "other",
]
LabelSource = Literal["review_feedback", "blind_eval"]

FIT_LABELS = frozenset({"good_fit", "poor_fit", "uncertain"})
ACTION_LABELS = frozenset({"apply", "skip", "save"})
REJECTION_REASONS = frozenset({
    "skills", "seniority", "location", "compensation", "duplicate",
    "expired", "not_interested", "bad_draft", "other"})
LABEL_SOURCES = frozenset({"review_feedback", "blind_eval"})


class LabelInput(BaseModel):
    fit_label: Optional[FitLabel] = None
    action_label: Optional[ActionLabel] = None
    rejection_reason: Optional[RejectionReason] = None
    notes: Optional[str] = None
