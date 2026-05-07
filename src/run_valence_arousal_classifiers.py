from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from experiment_utils import PROJECT_ROOT


DEFAULT_SEEDS = list(range(1, 11))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run binary valence and arousal lyric classifier experiments.")
    parser.add_argument("--data-dir", default="data/processed/mood_classifier")
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output-dir", default="outputs/mood_classifier_experiments/binary_valence_arousal_tfidf")
    parser.add_argument("--max-features", type=int, default=100000)
    parser.add_argument("--ngram-max", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--logreg-c", type=float, default=1.0)
    parser.add_argument("--class-weight-mode", choices=["balanced", "none"], default="balanced")
    parser.add_argument(
        "--boundary-margin",
        type=float,
        default=0.0,
        help="If > 0, remove training songs with valence or arousal within this distance of 0.5.",
    )
    return parser.parse_args()


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_text_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str)


def add_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["valence_label"] = np.where(df["valence"] >= 0.5, "positive", "negative")
    df["arousal_label"] = np.where(df["arousal"] >= 0.5, "high", "low")
    df["quadrant_from_binary_labels"] = df["valence_label"] + "_" + df["arousal_label"]
    return df


def filter_boundary_ambiguous_training_rows(train_df: pd.DataFrame, margin: float) -> pd.DataFrame:
    if margin <= 0:
        return train_df
    keep_mask = (train_df["valence"].sub(0.5).abs() >= margin) & (train_df["arousal"].sub(0.5).abs() >= margin)
    return train_df.loc[keep_mask].reset_index(drop=True)


def make_pipeline(max_features: int, ngram_max: int, logreg_c: float, class_weight_mode: str) -> Pipeline:
    class_weight = None if class_weight_mode == "none" else "balanced"
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, ngram_max),
                    max_features=max_features,
                    min_df=3,
                    max_df=0.9,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight=class_weight,
                    solver="liblinear",
                    C=logreg_c,
                ),
            ),
        ]
    )


def load_split(data_dir: Path, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_dir = data_dir / "splits"
    train_path = split_dir / f"seed_{seed}_train.csv"
    test_path = split_dir / f"seed_{seed}_test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing split files for seed {seed}: {train_path}, {test_path}")
    return add_binary_labels(pd.read_csv(train_path)), add_binary_labels(pd.read_csv(test_path))


def train_predict_binary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_column: str,
    max_features: int,
    ngram_max: int,
    logreg_c: float,
    class_weight_mode: str,
) -> tuple[np.ndarray, Pipeline]:
    model = make_pipeline(max_features, ngram_max, logreg_c, class_weight_mode)
    model.fit(normalize_text_series(train_df["lyrics"]), train_df[label_column])
    predictions = model.predict(normalize_text_series(test_df["lyrics"]))
    return predictions, model


def metric_row(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, prefix: str) -> dict:
    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_macro_f1": f1_score(y_true, y_pred, average="macro"),
        f"{prefix}_weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def save_binary_report(seed_dir: Path, name: str, y_true: pd.Series, y_pred: np.ndarray) -> None:
    labels = sorted(pd.Series(y_true).unique())
    report_df = pd.DataFrame(
        classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    ).transpose().reset_index().rename(columns={"index": "label"})
    confusion_df = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels)
    confusion_df.index.name = f"actual_{name}"
    report_df.to_csv(seed_dir / f"{name}_classification_report.csv", index=False)
    confusion_df.to_csv(seed_dir / f"{name}_confusion_matrix.csv")


def run_seed(
    data_dir: Path,
    output_dir: Path,
    seed: int,
    max_features: int,
    ngram_max: int,
    logreg_c: float,
    class_weight_mode: str,
    boundary_margin: float,
) -> dict:
    train_df, test_df = load_split(data_dir, seed)
    train_df = filter_boundary_ambiguous_training_rows(train_df, boundary_margin)
    valence_pred, _ = train_predict_binary(
        train_df, test_df, "valence_label", max_features, ngram_max, logreg_c, class_weight_mode
    )
    arousal_pred, _ = train_predict_binary(
        train_df, test_df, "arousal_label", max_features, ngram_max, logreg_c, class_weight_mode
    )

    predicted_quadrants = pd.Series(valence_pred) + "_" + pd.Series(arousal_pred)
    actual_quadrants = test_df["quadrant_from_binary_labels"]

    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    save_binary_report(seed_dir, "valence", test_df["valence_label"], valence_pred)
    save_binary_report(seed_dir, "arousal", test_df["arousal_label"], arousal_pred)
    save_binary_report(seed_dir, "combined_quadrant", actual_quadrants, predicted_quadrants.to_numpy())

    predictions_df = test_df[["song_id", "title", "artist", "mood_label", "valence", "arousal"]].copy()
    predictions_df["actual_valence_label"] = test_df["valence_label"]
    predictions_df["predicted_valence_label"] = valence_pred
    predictions_df["actual_arousal_label"] = test_df["arousal_label"]
    predictions_df["predicted_arousal_label"] = arousal_pred
    predictions_df["actual_quadrant"] = actual_quadrants
    predictions_df["predicted_quadrant"] = predicted_quadrants.to_numpy()
    predictions_df["valence_correct"] = predictions_df["actual_valence_label"] == predictions_df["predicted_valence_label"]
    predictions_df["arousal_correct"] = predictions_df["actual_arousal_label"] == predictions_df["predicted_arousal_label"]
    predictions_df["both_correct"] = predictions_df["actual_quadrant"] == predictions_df["predicted_quadrant"]
    predictions_df.to_csv(seed_dir / "test_predictions.csv", index=False)

    metrics = {
        "seed": seed,
        "max_features": max_features,
        "ngram_max": ngram_max,
        "logreg_c": logreg_c,
        "class_weight_mode": class_weight_mode,
        "boundary_margin": boundary_margin,
        "num_train": len(train_df),
        "num_test": len(test_df),
    }
    metrics.update(metric_row(test_df["valence_label"], valence_pred, "valence"))
    metrics.update(metric_row(test_df["arousal_label"], arousal_pred, "arousal"))
    metrics.update(metric_row(actual_quadrants, predicted_quadrants.to_numpy(), "combined_quadrant"))
    return metrics


def aggregate_metrics(split_metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "valence_accuracy",
        "valence_macro_f1",
        "valence_weighted_f1",
        "arousal_accuracy",
        "arousal_macro_f1",
        "arousal_weighted_f1",
        "combined_quadrant_accuracy",
        "combined_quadrant_macro_f1",
        "combined_quadrant_weighted_f1",
    ]
    rows = []
    for metric in metric_cols:
        values = split_metrics_df[metric]
        rows.append(
            {
                "metric": metric,
                "mean": values.mean(),
                "std": values.std(ddof=1),
                "min": values.min(),
                "max": values.max(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    data_dir = resolve_path(args.data_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in args.seeds:
        print(f"Running binary valence/arousal classifiers seed {seed}", flush=True)
        rows.append(
            run_seed(
                data_dir=data_dir,
                output_dir=output_dir,
                seed=seed,
                max_features=args.max_features,
                ngram_max=args.ngram_max,
                logreg_c=args.logreg_c,
                class_weight_mode=args.class_weight_mode,
                boundary_margin=args.boundary_margin,
            )
        )

    split_metrics_df = pd.DataFrame(rows)
    aggregate_df = aggregate_metrics(split_metrics_df)
    split_metrics_df.to_csv(output_dir / "split_metrics.csv", index=False)
    aggregate_df.to_csv(output_dir / "aggregate_metrics.csv", index=False)

    print(f"\nSaved binary valence/arousal classifier outputs to: {output_dir}")
    print("\nSplit metrics:")
    print(split_metrics_df.to_string(index=False))
    print("\nAggregate metrics:")
    print(aggregate_df.to_string(index=False))


if __name__ == "__main__":
    main()
