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
    parser = argparse.ArgumentParser(description="Run hierarchical 12-mood lyric classifier experiments.")
    parser.add_argument("--data-dir", default="data/processed/mood_classifier")
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output-dir", default="outputs/mood_classifier_experiments/12_mood_hierarchical_tfidf")
    parser.add_argument("--max-features", type=int, default=100000)
    parser.add_argument("--ngram-max", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--logreg-c", type=float, default=3.0)
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


def make_pipeline(max_features: int, ngram_max: int, logreg_c: float) -> Pipeline:
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
                    class_weight="balanced",
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


def train_quadrant_model(train_df: pd.DataFrame, max_features: int, ngram_max: int, logreg_c: float) -> Pipeline:
    model = make_pipeline(max_features, ngram_max, logreg_c)
    model.fit(normalize_text_series(train_df["lyrics"]), train_df["quadrant_label"])
    return model


def train_within_quadrant_models(
    train_df: pd.DataFrame,
    max_features: int,
    ngram_max: int,
    logreg_c: float,
) -> dict[str, Pipeline]:
    models = {}
    for quadrant in sorted(train_df["quadrant_label"].unique()):
        quadrant_df = train_df[train_df["quadrant_label"] == quadrant]
        model = make_pipeline(max_features, ngram_max, logreg_c)
        model.fit(normalize_text_series(quadrant_df["lyrics"]), quadrant_df["mood_label"])
        models[quadrant] = model
    return models


def predict_hierarchical(
    test_df: pd.DataFrame,
    quadrant_model: Pipeline,
    within_models: dict[str, Pipeline],
) -> tuple[np.ndarray, np.ndarray]:
    predicted_quadrants = quadrant_model.predict(normalize_text_series(test_df["lyrics"]))
    predicted_moods = []
    for quadrant in predicted_quadrants:
        row_text = normalize_text_series(test_df.iloc[[len(predicted_moods)]]["lyrics"])
        predicted_moods.append(within_models[quadrant].predict(row_text)[0])
    return predicted_quadrants, np.array(predicted_moods)


def predict_oracle_within_quadrant(test_df: pd.DataFrame, within_models: dict[str, Pipeline]) -> np.ndarray:
    predictions = []
    for quadrant in test_df["quadrant_label"]:
        row_text = normalize_text_series(test_df.iloc[[len(predictions)]]["lyrics"])
        predictions.append(within_models[quadrant].predict(row_text)[0])
    return np.array(predictions)


def metric_row(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, prefix: str) -> dict:
    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_macro_f1": f1_score(y_true, y_pred, average="macro"),
        f"{prefix}_weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def save_reports(
    output_dir: Path,
    seed: int,
    test_df: pd.DataFrame,
    predicted_quadrants: np.ndarray,
    predicted_moods: np.ndarray,
    oracle_moods: np.ndarray,
) -> None:
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    mood_labels = sorted(test_df["mood_label"].unique())
    quadrant_labels = sorted(test_df["quadrant_label"].unique())

    mood_report = pd.DataFrame(
        classification_report(test_df["mood_label"], predicted_moods, output_dict=True, zero_division=0)
    ).transpose().reset_index().rename(columns={"index": "label"})
    quadrant_report = pd.DataFrame(
        classification_report(test_df["quadrant_label"], predicted_quadrants, output_dict=True, zero_division=0)
    ).transpose().reset_index().rename(columns={"index": "label"})

    mood_confusion = pd.DataFrame(
        confusion_matrix(test_df["mood_label"], predicted_moods, labels=mood_labels),
        index=mood_labels,
        columns=mood_labels,
    )
    mood_confusion.index.name = "actual_mood"
    quadrant_confusion = pd.DataFrame(
        confusion_matrix(test_df["quadrant_label"], predicted_quadrants, labels=quadrant_labels),
        index=quadrant_labels,
        columns=quadrant_labels,
    )
    quadrant_confusion.index.name = "actual_quadrant"

    predictions_df = test_df[["song_id", "title", "artist", "mood_label", "quadrant_label"]].copy()
    predictions_df["predicted_quadrant"] = predicted_quadrants
    predictions_df["predicted_mood"] = predicted_moods
    predictions_df["oracle_quadrant_predicted_mood"] = oracle_moods
    predictions_df["quadrant_correct"] = predictions_df["quadrant_label"] == predictions_df["predicted_quadrant"]
    predictions_df["mood_correct"] = predictions_df["mood_label"] == predictions_df["predicted_mood"]
    predictions_df["oracle_within_quadrant_correct"] = (
        predictions_df["mood_label"] == predictions_df["oracle_quadrant_predicted_mood"]
    )

    mood_report.to_csv(seed_dir / "mood_classification_report.csv", index=False)
    quadrant_report.to_csv(seed_dir / "quadrant_classification_report.csv", index=False)
    mood_confusion.to_csv(seed_dir / "mood_confusion_matrix.csv")
    quadrant_confusion.to_csv(seed_dir / "quadrant_confusion_matrix.csv")
    predictions_df.to_csv(seed_dir / "test_predictions.csv", index=False)


def run_seed(
    data_dir: Path,
    output_dir: Path,
    seed: int,
    max_features: int,
    ngram_max: int,
    logreg_c: float,
) -> dict:
    train_df, test_df = load_split(data_dir, seed)
    quadrant_model = train_quadrant_model(train_df, max_features, ngram_max, logreg_c)
    within_models = train_within_quadrant_models(train_df, max_features, ngram_max, logreg_c)

    predicted_quadrants, predicted_moods = predict_hierarchical(test_df, quadrant_model, within_models)
    oracle_moods = predict_oracle_within_quadrant(test_df, within_models)

    save_reports(output_dir, seed, test_df, predicted_quadrants, predicted_moods, oracle_moods)

    metrics = {
        "seed": seed,
        "max_features": max_features,
        "ngram_max": ngram_max,
        "logreg_c": logreg_c,
        "num_train": len(train_df),
        "num_test": len(test_df),
        "num_classes": test_df["mood_label"].nunique(),
        "num_quadrants": test_df["quadrant_label"].nunique(),
    }
    metrics.update(metric_row(test_df["quadrant_label"], predicted_quadrants, "quadrant"))
    metrics.update(metric_row(test_df["mood_label"], predicted_moods, "hierarchical_mood"))
    metrics.update(metric_row(test_df["mood_label"], oracle_moods, "oracle_within_quadrant_mood"))
    return metrics


def aggregate_metrics(split_metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "quadrant_accuracy",
        "quadrant_macro_f1",
        "quadrant_weighted_f1",
        "hierarchical_mood_accuracy",
        "hierarchical_mood_macro_f1",
        "hierarchical_mood_weighted_f1",
        "oracle_within_quadrant_mood_accuracy",
        "oracle_within_quadrant_mood_macro_f1",
        "oracle_within_quadrant_mood_weighted_f1",
    ]
    rows = []
    for metric in metric_cols:
        values = split_metrics_df[metric]
        rows.append(
            {
                "metric": metric,
                "mean": values.mean(),
                "std": values.std(ddof=1) if len(values) > 1 else 0.0,
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
        print(f"Running hierarchical mood classifier seed {seed}", flush=True)
        rows.append(
            run_seed(
                data_dir=data_dir,
                output_dir=output_dir,
                seed=seed,
                max_features=args.max_features,
                ngram_max=args.ngram_max,
                logreg_c=args.logreg_c,
            )
        )

    split_metrics_df = pd.DataFrame(rows)
    aggregate_df = aggregate_metrics(split_metrics_df)
    split_metrics_df.to_csv(output_dir / "split_metrics.csv", index=False)
    aggregate_df.to_csv(output_dir / "aggregate_metrics.csv", index=False)

    print(f"\nSaved hierarchical mood classifier outputs to: {output_dir}")
    print("\nSplit metrics:")
    print(split_metrics_df.to_string(index=False))
    print("\nAggregate metrics:")
    print(aggregate_df.to_string(index=False))


if __name__ == "__main__":
    main()
