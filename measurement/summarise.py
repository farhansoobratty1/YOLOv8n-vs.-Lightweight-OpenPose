"""Average the runs in results/raw_runs.csv into one row per model and clip.

    python -m measurement.summarise
    python -m measurement.summarise --ground-truth clip1=2 clip2=8 clip3=6

Ground truth is your hand count of strikes per clip (defaults are the counts from the essay,
Table 2). "accuracy_pct" follows the essay: average strikes detected / ground truth * 100.
Note that this ratio can reach or exceed 100% if a model over-counts, so the signed
"count_error" (average detected minus ground truth) is reported next to it.
"""
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

DEFAULT_GROUND_TRUTH = {"clip1": 2, "clip2": 8, "clip3": 6}


def parse_ground_truth(items):
    if not items:
        return dict(DEFAULT_GROUND_TRUTH)
    return {name: float(value) for name, value in (item.split("=") for item in items)}


def sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=Path("results/raw_runs.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/summary.csv"))
    parser.add_argument("--ground-truth", nargs="*", help="e.g. clip1=2 clip2=8 clip3=6")
    args = parser.parse_args()
    truth = parse_ground_truth(args.ground_truth)

    groups = defaultdict(list)
    with open(args.raw, newline="") as handle:
        for row in csv.DictReader(handle):
            groups[(row["model"], row["clip"])].append(row)

    header = ["model", "clip", "runs", "avg_cpu_pct", "std_dev_cpu", "avg_fps", "std_dev_fps",
              "avg_strikes_detected", "ground_truth_strikes", "accuracy_pct", "count_error"]
    rows = []
    for (model, clip), runs in sorted(groups.items()):
        cpu = [float(r["avg_cpu_pct"]) for r in runs]
        fps = [float(r["fps"]) for r in runs]
        strikes = [float(r["strikes"]) for r in runs]
        mean_strikes = statistics.mean(strikes)
        gt = truth.get(Path(clip).stem)
        accuracy = round(100 * mean_strikes / gt, 1) if gt else ""
        error = round(mean_strikes - gt, 2) if gt else ""
        rows.append([model, clip, len(runs), round(statistics.mean(cpu), 2), round(sample_std(cpu), 2),
                     round(statistics.mean(fps), 2), round(sample_std(fps), 2),
                     round(mean_strikes, 2), gt if gt else "", accuracy, error])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)

    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    for line in [header] + rows:
        print("  ".join(str(x).ljust(w) for x, w in zip(line, widths)))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
