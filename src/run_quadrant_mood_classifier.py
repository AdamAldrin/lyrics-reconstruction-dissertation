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
MOOD_TO_QUADRANT = {
    "pleased": "positive_high",
    "happy": "positive_high",
    "excited": "positive_high",
    "relaxed": "positive_low",
    "peaceful": "positive_low",
    "calm": "positive_low",
    "annoying": "negative_high",
    "angry": "negative_high",
    "nervous": "negative_high",
    "sad": "negative_low",
    "bored": "negative_low",
    "sleepy": "negative_low",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated 4-mood quadrant lyric classifier experiments.")
    parser.add_argument("--data-dir", default="data/processed/mood_classifier")
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output-dir", default="outputs/mood_classifier_experiments/4_mood_quadrant_tfidf")
    parser.add_argument("--max-features", type=int, default=100000)
    parser.add_argument("--ngram-max", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--logreg-c", type=float, default=3.0)
    parser.add_argument(
        "--class-weight-mode",
        choices=["balanced", "none", "mild"],
        default="balanced",
        help="Class weighting strategy for logistic regression.",
    )
    parser.add_argument(
        "--balance-train",
        action="store_true",
        help="Undersample each training split to the same number of songs per quadrant.",
    )
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


def add_quadrant_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["quadrant_label"] = df["mood_label"].map(MOOD_TO_QUADRANT)
    if df["quadrant_label"].isna().any():
        missing = sorted(df.loc[df["quadrant_label"].isna(), "mood_label"].dropna().unique())
        raise ValueError(f"Missing quadrant mapping for mood labels: {missing}")
    return df


def class_weight_value(mode: str) -> str | dict[str, float] | None:
    if mode == "none":
        return None
    if mode == "mild":
        return {
            "negative_high": 1.0,
            "positive_high": 1.0,
            "negative_low": 1.5,
            "positive_low": 3.0,
        }
    return "balanced"


def make_pipeline(max_features: int, ngram_max: int, logreg_c: float, class_weight_mode: str = "balanced") -> Pipeline:
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
                    class_weight=class_weight_value(class_weight_mode),
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
    return add_quadrant_labels(pd.read_csv(train_path)), add_quadrant_labels(pd.read_csv(test_path))


def balance_training_quadrants(train_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    min_count = train_df["quadrant_label"].value_counts().min()
    balanced_parts = []
    for _, quadrant_df in train_df.groupby("quadrant_label", sort=True):
        balanced_parts.append(quadrant_df.sample(n=min_count, random_state=seed))
    return pd.concat(balanced_parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def filter_boundary_ambiguous_training_rows(train_df: pd.DataFrame, margin: float) -> pd.DataFrame:
    if margin <= 0:
        return train_df
    keep_mask = (train_df["valence"].sub(0.5).abs() >= margin) & (train_df["arousal"].sub(0.5).abs() >= margin)
    return train_df.loc[keep_mask].reset_index(drop=True)


def run_seed(
    data_dir: Path,
    output_dir: Path,
    seed: int,
    max_features: int,
    ngram_max: int,
    logreg_c: float,
    class_weight_mode: str,
    balance_train: bool,
    boundary_margin: float,
) -> tuple[dict, pd.DataFrame]:
    train_df, test_df = load_split(data_dir, seed)
    train_df = filter_boundary_ambiguous_training_rows(train_df, boundary_margin)
    if balance_train:
        train_df = balance_training_quadrants(train_df, seed)
    model = make_pipeline(max_features, ngram_max, logreg_c, class_weight_mode)
    model.fit(normalize_text_series(train_df["lyrics"]), train_df["quadrant_label"])
    predictions = model.predict(normalize_text_series(test_df["lyrics"]))

    labels = sorted(test_df["quadrant_label"].unique())
    report_df = pd.DataFrame(
        classification_report(test_df["quadrant_label"], predictions, output_dict=True, zero_division=0)
    ).transpose().reset_index().rename(columns={"index": "label"})
    confusion_df = pd.DataFrame(
        confusion_matrix(test_df["quadrant_label"], predictions, labels=labels),
        index=labels,
        columns=labels,
    )
    confusion_df.index.name = "actual_quadrant"

    predictions_df = test_df[["song_id", "title", "artist", "mood_label", "quadrant_label"]].copy()
    predictions_df["predicted_quadrant"] = predictions
    predictions_df["correct"] = predictions_df["quadrant_label"] == predictions_df["predicted_quadrant"]

    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(seed_dir / "classification_report.csv", index=False)
    confusion_df.to_csv(seed_dir / "confusion_matrix.csv")
    predictions_df.to_csv(seed_dir / "test_predictions.csv", index=False)

    metrics = {
        "seed": seed,
        "max_features": max_features,
        "ngram_max": ngram_max,
        "logreg_c": logreg_c,
        "class_weight_mode": class_weight_mode,
        "balance_train": balance_train,
        "boundary_margin": boundary_margin,
        "num_train": len(train_df),
        "num_test": len(test_df),
        "num_classes": len(labels),
        "accuracy": accuracy_score(test_df["quadrant_label"], predictions),
        "macro_f1": f1_score(test_df["quadrant_label"], predictions, average="macro"),
        "weighted_f1": f1_score(test_df["quadrant_label"], predictions, average="weighted"),
    }
    return metrics, confusion_df


def aggregate_metrics(split_metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["accuracy", "macro_f1", "weighted_f1"]
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


def mean_confusion_matrix(confusion_matrices: list[pd.DataFrame]) -> pd.DataFrame:
    labels = confusion_matrices[0].index
    stacked = np.stack([matrix.loc[labels, labels].to_numpy(dtype=float) for matrix in confusion_matrices])
    mean_matrix = stacked.mean(axis=0)
    mean_df = pd.DataFrame(mean_matrix, index=labels, columns=labels)
    mean_df.index.name = "actual_quadrant"
    return mean_df


def main() -> None:
    args = parse_args()
    data_dir = resolve_path(args.data_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    confusion_matrices = []
    for seed in args.seeds:
        print(f"Running 4-mood quadrant classifier seed {seed}", flush=True)
        metrics, confusion_df = run_seed(
            data_dir=data_dir,
            output_dir=output_dir,
            seed=seed,
            max_features=args.max_features,
            ngram_max=args.ngram_max,
            logreg_c=args.logreg_c,
            class_weight_mode=args.class_weight_mode,
            balance_train=args.balance_train,
            boundary_margin=args.boundary_margin,
        )
        rows.append(metrics)
        confusion_matrices.append(confusion_df)

    split_metrics_df = pd.DataFrame(rows)
    aggregate_df = aggregate_metrics(split_metrics_df)
    mean_confusion_df = mean_confusion_matrix(confusion_matrices)

    split_metrics_df.to_csv(output_dir / "split_metrics.csv", index=False)
    aggregate_df.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    mean_confusion_df.to_csv(output_dir / "mean_confusion_matrix.csv")

    print(f"\nSaved 4-mood quadrant classifier outputs to: {output_dir}")
    print("\nSplit metrics:")
    print(split_metrics_df.to_string(index=False))
    print("\nAggregate metrics:")
    print(aggregate_df.to_string(index=False))


if __name__ == "__main__":
    main()
