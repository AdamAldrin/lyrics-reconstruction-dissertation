from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiment_utils import PROJECT_ROOT, clean_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare evaluation summaries across runs.")
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Specific run IDs to compare. If omitted, all runs with summaries are included.",
    )
    return parser.parse_args()


def collect_summary_files(run_ids: list[str] | None) -> list[Path]:
    runs_root = PROJECT_ROOT / "outputs" / "runs"
    if run_ids:
        return [runs_root / run_id / "evaluation_summary.csv" for run_id in run_ids]
    return sorted(runs_root.glob("*/evaluation_summary.csv"))


def main() -> None:
    args = parse_args()
    summary_files = [path for path in collect_summary_files(args.runs) if path.exists()]
    if not summary_files:
        raise SystemExit("No evaluation_summary.csv files found to compare.")

    frames = []
    for path in summary_files:
        df = pd.read_csv(path)
        if "run_id" not in df.columns:
            df["run_id"] = clean_text(path.parent.name)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    output_path = PROJECT_ROOT / "outputs" / "evaluation" / "run_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    print(f"Saved run comparison to: {output_path}")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
