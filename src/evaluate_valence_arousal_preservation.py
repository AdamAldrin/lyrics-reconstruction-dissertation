from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from experiment_utils import PROJECT_ROOT
from run_valence_arousal_classifiers import (
    add_binary_labels,
    make_pipeline,
    normalize_text_series,
)


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
    parser = argparse.ArgumentParser(description="Evaluate binary valence/arousal preservation for generated lyrics.")
    parser.add_argument("--classifier-data", default="data/processed/mood_classifier/mood_classifier_dataset.csv")
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--run-ids", nargs="*", default=DEFAULT_RUN_IDS)
    parser.add_argument("--output-dir", default="outputs/mood_preservation/binary_valence_arousal")
    parser.add_argument("--max-features", type=int, default=100000)
    parser.add_argument("--ngram-max", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--logreg-c", type=float, default=1.0)
    parser.add_argument("--class-weight-mode", choices=["balanced", "none"], default="balanced")
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


def train_final_classifiers(
    classifier_data_path: Path,
    max_features: int,
    ngram_max: int,
    logreg_c: float,
    class_weight_mode: str,
):
    train_df = add_binary_labels(pd.read_csv(classifier_data_path))
    valence_model = make_pipeline(max_features, ngram_max, logreg_c, class_weight_mode)
    arousal_model = make_pipeline(max_features, ngram_max, logreg_c, class_weight_mode)
    valence_model.fit(normalize_text_series(train_df["lyrics"]), train_df["valence_label"])
    arousal_model.fit(normalize_text_series(train_df["lyrics"]), train_df["arousal_label"])
    return valence_model, arousal_model, train_df


def evaluate_run(run_path: Path, valence_model, arousal_model) -> pd.DataFrame:
    generated_path = run_path / "generated_outputs.csv"
    if not generated_path.exists():
        raise FileNotFoundError(f"Missing generated outputs file: {generated_path}")

    df = add_binary_labels(pd.read_csv(generated_path))
    generated_text_column = find_generated_text_column(df)
    generated_text = normalize_text_series(df[generated_text_column])
    predicted_valence = valence_model.predict(generated_text)
    predicted_arousal = arousal_model.predict(generated_text)
    predicted_quadrant = pd.Series(predicted_valence) + "_" + pd.Series(predicted_arousal)

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
            "valence_label",
            "arousal_label",
            "quadrant_from_binary_labels",
        ]
    ].copy()
    result_df = result_df.rename(
        columns={
            "valence_label": "original_valence_label",
            "arousal_label": "original_arousal_label",
            "quadrant_from_binary_labels": "original_binary_quadrant",
        }
    )
    result_df["generated_text_column"] = generated_text_column
    result_df["predicted_generated_valence_label"] = predicted_valence
    result_df["predicted_generated_arousal_label"] = predicted_arousal
    result_df["predicted_generated_binary_quadrant"] = predicted_quadrant.to_numpy()
    result_df["valence_match"] = (
        result_df["original_valence_label"] == result_df["predicted_generated_valence_label"]
    )
    result_df["arousal_match"] = (
        result_df["original_arousal_label"] == result_df["predicted_generated_arousal_label"]
    )
    result_df["both_match"] = result_df["valence_match"] & result_df["arousal_match"]
    return result_df


def summarize_by_run(per_song_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run_id, run_df in per_song_df.groupby("run_id", sort=True):
        rows.append(
            {
                "run_id": run_id,
                "num_songs": len(run_df),
                "valence_match_rate": run_df["valence_match"].mean(),
                "arousal_match_rate": run_df["arousal_match"].mean(),
                "both_match_rate": run_df["both_match"].mean(),
                "original_positive_num_songs": int((run_df["original_valence_label"] == "positive").sum()),
                "positive_valence_match_rate": run_df.loc[
                    run_df["original_valence_label"] == "positive", "valence_match"
                ].mean(),
                "original_negative_num_songs": int((run_df["original_valence_label"] == "negative").sum()),
                "negative_valence_match_rate": run_df.loc[
                    run_df["original_valence_label"] == "negative", "valence_match"
                ].mean(),
                "original_high_num_songs": int((run_df["original_arousal_label"] == "high").sum()),
                "high_arousal_match_rate": run_df.loc[
                    run_df["original_arousal_label"] == "high", "arousal_match"
                ].mean(),
                "original_low_num_songs": int((run_df["original_arousal_label"] == "low").sum()),
                "low_arousal_match_rate": run_df.loc[
                    run_df["original_arousal_label"] == "low", "arousal_match"
                ].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["both_match_rate", "valence_match_rate", "arousal_match_rate"], ascending=False)


def save_reports(per_song_df: pd.DataFrame, output_dir: Path) -> None:
    for run_id, run_df in per_song_df.groupby("run_id", sort=True):
        run_dir = output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        for name, actual_col, predicted_col in [
            ("valence", "original_valence_label", "predicted_generated_valence_label"),
            ("arousal", "original_arousal_label", "predicted_generated_arousal_label"),
            ("binary_quadrant", "original_binary_quadrant", "predicted_generated_binary_quadrant"),
        ]:
            labels = sorted(run_df[actual_col].unique())
            report_df = pd.DataFrame(
                classification_report(
                    run_df[actual_col],
                    run_df[predicted_col],
                    labels=labels,
                    output_dict=True,
                    zero_division=0,
                )
            ).transpose().reset_index().rename(columns={"index": "label"})
            confusion_df = pd.DataFrame(
                confusion_matrix(run_df[actual_col], run_df[predicted_col], labels=labels),
                index=labels,
                columns=labels,
            )
            confusion_df.index.name = f"actual_{name}"
            report_df.to_csv(run_dir / f"{name}_classification_report.csv", index=False)
            confusion_df.to_csv(run_dir / f"{name}_confusion_matrix.csv")


def main() -> None:
    args = parse_args()
    classifier_data_path = resolve_path(args.classifier_data)
    runs_dir = resolve_path(args.runs_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Training final binary valence/arousal classifiers on non-reconstruction Music4All lyrics.", flush=True)
    valence_model, arousal_model, train_df = train_final_classifiers(
        classifier_data_path=classifier_data_path,
        max_features=args.max_features,
        ngram_max=args.ngram_max,
        logreg_c=args.logreg_c,
        class_weight_mode=args.class_weight_mode,
    )
    print(f"Training rows: {len(train_df)}", flush=True)

    per_run_frames = []
    for run_id in args.run_ids:
        print(f"Evaluating generated lyrics for {run_id}", flush=True)
        per_run_frames.append(evaluate_run(runs_dir / run_id, valence_model, arousal_model))

    per_song_df = pd.concat(per_run_frames, ignore_index=True)
    summary_df = summarize_by_run(per_song_df)
    per_song_df.to_csv(output_dir / "valence_arousal_preservation_per_song.csv", index=False)
    summary_df.to_csv(output_dir / "valence_arousal_preservation_summary.csv", index=False)
    save_reports(per_song_df, output_dir)

    print(f"\nSaved binary valence/arousal preservation outputs to: {output_dir}", flush=True)
    print("\nRun summary:")
    print(summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
