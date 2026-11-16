"""Bring two labelers' sheets back into the eval database.

    python scripts/import_labels.py --db data/eval-2026-11.db \
        labeling/label_A.xlsx labeling/label_B.xlsx [--resolved resolved.xlsx]

Prints Cohen's kappa, records every label both sheets agree on as a
`blind_eval` row, and writes `disagreements.xlsx` next to the first sheet:
fill in its `label` column together and pass it back with --resolved.
"""

import argparse
import os

from offerpilot import evalset
from offerpilot.store import db


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("sheets", nargs=2)
    p.add_argument("--resolved")
    a = p.parse_args(argv)
    conn = db.connect(a.db)
    la, lb = (evalset.read_label_sheet(s) for s in a.sheets)
    total = conn.execute("SELECT COUNT(*) n FROM job_versions").fetchone()["n"]
    kappa = evalset.cohen_kappa(la, lb)
    agreed, disagreed = evalset.merge_labels(la, lb)
    print(f"sheet A labeled {len(la)}/{total}, sheet B {len(lb)}/{total}")
    print(f"agree on {len(agreed)}, disagree on {len(disagreed)}, "
          f"Cohen's kappa = {kappa if kappa is None else round(kappa, 3)}")

    if a.resolved:
        resolved = evalset.read_label_sheet(a.resolved)
        missing = sorted(set(disagreed) - set(resolved))
        if missing:
            raise SystemExit(f"--resolved has no label for {missing}")
        n = evalset.record_consensus(
            conn, {k: resolved[k] for k in disagreed},
            note="resolved together after independent disagreement")
        print(f"recorded {n} resolved labels")
        return

    n = evalset.record_consensus(conn, agreed,
                                 note="two independent labelers agreed")
    print(f"recorded {n} agreed labels as blind_eval")
    if disagreed:
        import openpyxl
        out = os.path.join(os.path.dirname(a.sheets[0]), "disagreements.xlsx")
        wb = openpyxl.load_workbook(a.sheets[0])
        ws = wb["Jobs"]
        header = [c.value for c in ws[1]]
        id_col = header.index("job_id")
        label_col = header.index("label") + 1
        keep = set(disagreed)
        for r in range(ws.max_row, 1, -1):
            vid = ws.cell(row=r, column=id_col + 1).value
            if vid not in keep:
                ws.delete_rows(r)
            else:
                ws.cell(row=r, column=label_col, value=None)
        ws.cell(row=1, column=len(header) + 1, value="A said")
        ws.cell(row=1, column=len(header) + 2, value="B said")
        for r in range(2, ws.max_row + 1):
            vid = ws.cell(row=r, column=id_col + 1).value
            ws.cell(row=r, column=len(header) + 1, value=la[vid])
            ws.cell(row=r, column=len(header) + 2, value=lb[vid])
        wb.save(out)
        print(f"wrote {out}: settle these {len(disagreed)} together, then "
              f"rerun with --resolved {out}")


if __name__ == "__main__":
    main()
