# Eval dataset

The eval set is **not** a file of copied job postings — redistributing employer
job text is not ours to do. It is produced in place:

1. `offerpilot collect` populates `job_versions` from public ATS APIs.
2. `offerpilot panel` -> **Blind labeling** shows job + profile with every model
   output hidden, and writes `labels` rows with `label_source='blind_eval'`.
3. Target 40-60 labeled jobs, drawn from *all* statuses including
   `filtered_out`, so prefilter false negatives are measurable.
4. `python run_eval.py` scores the pipeline end-to-end and writes
   `evals/results/eval-<timestamp>.json`, which **is** committed.

Labels given in the review panel (`label_source='review_feedback'`) are
recorded but excluded from formal metrics: the reviewer saw the model's score
and reasoning first, so those labels are anchored.
