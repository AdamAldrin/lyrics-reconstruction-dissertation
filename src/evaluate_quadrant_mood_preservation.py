from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from experiment_utils import PROJECT_ROOT
from run_quadrant_mood_classifier import MOOD_TO_QUADRANT, add_quadrant_labels, make_pipeline, normalize_text_series


DEFAULT_RUN_IDS = [
    "01-baseline-lycon-final",
    "02-structured-final",
    "03-structured-freq-final",
    "04-baseline-raw-vocab-final",
    "06-baseline-top20-final",
    "07-baseline-top100-final",
    "08-minimal-control-final",
    "09-strong-control-final",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 4-mood quadrant preservation for generated lyrics.")
    parser.add_argument("--classifier-data", default="data/processed/mood_classifier/mood_classifier_dataset.csv")
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--run-ids", nargs="*", default=DEFAULT_RUN_IDS)
    parser.add_argument("--output-dir", default="outputs/mood_preservation/4_mood_quadrant")
    parser.add_argument("--max-features", type=int, default=100000)
    parser.add_argument("--ngram-max", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--logreg-c", type=float, default=3.0)
    return parser.parse_args()


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def find_generated_text_column(df: pd.DataFrame) -> str:
    clean_columns = [column for column in df.columns if column.endswith("_output_clean")]
    if len(clean_columns) == 1:
        return clean_columns[0]
    if len(clean_columns) > 1:
        raise ValueError(f"Multiple generated output clean columns found: {clean_columns}")

    output_columns = [column for column in df.columns if column.endswith("_output")]
    if len(output_columns) == 1:
        return output_columns[0]
    raise ValueError(f"Could not identify generated output text column. Candidate columns: {output_columns}")


def train_final_classifier(
    classifier_data_path: Path,
    max_features: int,
    ngram_max: int,
    logreg_c: float,
):
    train_df = add_quadrant_labels(pd.read_csv(classifier_data_path))
    model = make_pipeline(max_features=max_features, ngram_max=ngram_max, logreg_c=logreg_c)
    model.fit(normalize_text_series(train_df["lyrics"]), train_df["quadrant_label"])
    return model, train_df


def evaluate_run(run_path: Path, model) -> pd.DataFrame:
    generated_path = run_path / "generated_outputs.csv"
    if not generated_path.exists():
        raise FileNotFoundError(f"Missing generated outputs file: {generated_path}")

    df = pd.read_csv(generated_path)
    generated_text_column = find_generated_text_column(df)
    df["original_quadrant"] = df["mood_label"].map(MOOD_TO_QUADRANT)
    if df["original_quadrant"].isna().any():
        missing = sorted(df.loc[df["original_quadrant"].isna(), "mood_label"].dropna().unique())
        raise ValueError(f"Missing quadrant mapping for run {run_path.name}: {missing}")

    predictions = model.predict(normalize_text_series(df[generated_text_column]))
    result_df = df[
        [
            "run_id",
            "song_id",
            "title",
            "artist",
            "genre",
            "valence",
            "arousal",
            "mood_label",
            "original_quadrant",
        ]
    ].copy()
    result_df["generated_text_column"] = generated_text_column
    result_df["predicted_generated_quadrant"] = predictions
    result_df["quadrant_match"] = result_df["original_quadrant"] == result_df["predicted_generated_quadrant"]
    return result_df


def summarize_by_run(per_song_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run_id, run_df in per_song_df.groupby("run_id", sort=True):
        row = {
            "run_id": run_id,
            "num_songs": len(run_df),
            "quadrant_match_rate": run_df["quadrant_match"].mean(),
        }
        for quadrant in sorted(MOOD_TO_QUADRANT.values()):
            quadrant_df = run_df[run_df["original_quadrant"] == quadrant]
            row[f"{quadrant}_num_songs"] = len(quadrant_df)
            row[f"{quadrant}_match_rate"] = quadrant_df["quadrant_match"].mean() if len(quadrant_df) else pd.NA
        rows.append(row)
    return pd.DataFrame(rows).sort_values("quadrant_match_rate", ascending=False)


def save_reports(per_song_df: pd.DataFrame, output_dir: Path) -> None:
    labels = sorted(MOOD_TO_QUADRANT.values())
    for run_id, run_df in per_song_df.groupby("run_id", sort=True):
        run_dir = output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        report_df = pd.DataFrame(
            classification_report(
                run_df["original_quadrant"],
                run_df["predicted_generated_quadrant"],
                labels=labels,
                output_dict=True,
                zero_division=0,
            )
        ).transpose().reset_index().rename(columns={"index": "label"})
        confusion_df = pd.DataFrame(
            confusion_matrix(
                run_df["original_quadrant"],
                run_df["predicted_generated_quadrant"],
                labels=labels,
            ),
            index=labels,
            columns=labels,
        )
        confusion_df.index.name = "original_quadrant"
        report_df.to_csv(run_dir / "classification_report.csv", index=False)
        confusion_df.to_csv(run_dir / "confusion_matrix.csv")


def main() -> None:
    args = parse_args()
    classifier_data_path = resolve_path(args.classifier_data)
    runs_dir = resolve_path(args.runs_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Training final 4-mood quadrant classifier on non-reconstruction Music4All lyrics.", flush=True)
    model, train_df = train_final_classifier(
        classifier_data_path=classifier_data_path,
        max_features=args.max_features,
        ngram_max=args.ngram_max,
        logreg_c=args.logreg_c,
    )
    print(f"Training rows: {len(train_df)}", flush=True)

    per_run_frames = []
    for run_id in args.run_ids:
        print(f"Evaluating generated lyrics for {run_id}", flush=True)
        per_run_frames.append(evaluate_run(runs_dir / run_id, model))

    per_song_df = pd.concat(per_run_frames, ignore_index=True)
    summary_df = summarize_by_run(per_song_df)

    per_song_df.to_csv(output_dir / "quadrant_mood_preservation_per_song.csv", index=False)
    summary_df.to_csv(output_dir / "quadrant_mood_preservation_summary.csv", index=False)
    save_reports(per_song_df, output_dir)

    print(f"\nSaved 4-mood quadrant preservation outputs to: {output_dir}", flush=True)
    print("\nRun summary:")
    print(summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
