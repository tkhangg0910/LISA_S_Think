"""Regenerate matrix.csv / matrix.md / matrix.xlsx from the measured metrics.

Single source of truth: the (-R) columns are read out of runs/<exp>/metrics.json, never
typed by hand. Re-run this after every new checkpoint is evaluated so the three files
cannot drift apart.

    python scripts/build_matrix.py
"""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# published baselines, transcribed from the source table
BASE = {
    ("Seg", "Dice"):    (0.269, "0.685/0.638", "0.531/0.420", 0.025, 0.3790),
    ("Seg", "IoU"):     (0.205, "0.560/0.515", "0.413/0.325", 0.014, 0.2338),
    ("MTA", "Dice"):    (0.196, "0.848/0.794", "0.518/0.664", 0.000, 0.4659),
    ("MTA", "IoU"):     (0.138, "0.739/0.676", "0.397/0.539", 0.000, 0.3037),
    ("GCA", "Dice"):    (0.497, "0.687/0.658", "0.666/0.111", 0.076, 0.3924),
    ("GCA", "IoU"):     (0.400, "0.558/0.530", "0.538/0.081", 0.042, 0.2441),
    ("Koedam", "Dice"): (0.114, "0.519/0.461", "0.408/0.485", 0.000, 0.2571),
    ("Koedam", "IoU"):  (0.077, "0.383/0.339", "0.302/0.354", 0.000, 0.1475),
}
ORDER = [("Seg", "Dice"), ("Seg", "IoU"), ("MTA", "Dice"), ("MTA", "IoU"),
         ("GCA", "Dice"), ("GCA", "IoU"), ("Koedam", "Dice"), ("Koedam", "IoU")]
KEY = {"Seg": "OVERALL", "MTA": "MTA", "GCA": "GCA", "Koedam": "Koedam"}

# our runs: (-R) column -> {size: experiment dir}
OURS = {
    # reasoning fine-tunes: Q4 -> Q1 boxes + A2 + presence + A3 + A4 + three [SEG]
    "LISA (7B/13B) (-R)":   {"7B": "brainM-q1",       "13B": "brain13B-q1"},
    "LISA++ (7B/13B) (-R)": {"7B": "brainPlus7B-q1",  "13B": None},  # no public LISA++ 13B
    # non-reasoning fine-tunes, same recipe: Q4 -> A4 + three [SEG], nothing else
    "LISA (7B/13B) ours-NR": {"7B": "brainNR7B-a4seg", "13B": "brainNR13B-a4seg"},
}
NOT_TRAINED = "xxx"
NO_CHECKPOINT = "n/a"


def load(exp):
    if exp is None:
        return None
    p = os.path.join(ROOT, "runs", exp, "metrics.json")
    return json.load(open(p)) if os.path.exists(p) else None


def cell(col, region, metric):
    """'0.7419/0.7607' — one side per model size, in the source table's format."""
    out = []
    for size in ("7B", "13B"):
        exp = OURS[col][size]
        if exp is None:
            out.append(NO_CHECKPOINT)
            continue
        m = load(exp)
        if m is None:
            out.append(NOT_TRAINED)
            continue
        r = m[KEY[region]]["present"]
        out.append(f"{r['DICE' if metric == 'Dice' else 'gIoU']:.4f}")
    return "/".join(out)


def main():
    header = ["Task", "Metric", "", "G-Dino + SAM-Med2D", "LISA (7B/13B)",
              "LISA++ (7B/13B)", "LISA (7B/13B) ours-NR", "LISA (7B/13B) (-R)",
              "LISA++ (7B/13B) (-R)", "LLaVA-G-7B", "Our"]
    rows = []
    for region, metric in ORDER:
        gdino, lisa, lisapp, llavag, our = BASE[(region, metric)]
        rows.append(["Grounding Segmentation", region, metric, f"{gdino:.4f}",
                     lisa, lisapp,
                     cell("LISA (7B/13B) ours-NR", region, metric),
                     cell("LISA (7B/13B) (-R)", region, metric),
                     cell("LISA++ (7B/13B) (-R)", region, metric),
                     f"{llavag:.4f}", f"{our:.4f}"])

    with open(os.path.join(ROOT, "matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("wrote matrix.csv")

    trained = {c: [s for s in ("7B", "13B") if load(OURS[c][s])] for c in OURS}
    for c, s in trained.items():
        print(f"  {c:<24} filled: {s or '(none yet)'}")
    return header, rows


if __name__ == "__main__":
    main()
