from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from experiment_utils import PROJECT_ROOT


DEFAULT_SEEDS = list(range(1, 11))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated 12-mood lyric classifier experiments.")
    parser.add_argument(
        "--data-dir",
        default="data/processed/mood_classifier",
        help="Prepared mood classifier dataset directory.",
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS, help="Split seeds to evaluate.")
    parser.add_argument(
        "--model",
        choices=["logreg", "linear_svc"],
        default="logreg",
        help="Classifier to train on the TF-IDF lyric representation.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for experiment outputs.",
    )
    parser.add_argument("--max-features", type=int, default=30000, help="Maximum TF-IDF vocabulary size.")
    parser.add_argument("--ngram-max", type=int, choices=[1, 2, 3], default=2, help="Maximum word n-gram length.")
    parser.add_argument("--logreg-c", type=float, default=1.0, help="Inverse regularisation strength for logistic regression.")
    return parser.parse_args()


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_text_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str)


def default_output_dir(model_name: str) -> Path:
    model_slug = "linear_svc" if model_name == "linear_svc" else "logreg"
    return PROJECT_ROOT / "outputs" / "mood_classifier_experiments" / f"12_mood_{model_slug}_tfidf"


def make_classifier(model_name: str, logreg_c: float):
    if model_name == "linear_svc":
        return LinearSVC(
            class_weight="balanced",
            max_iter=5000,
            random_state=0,
        )

    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        C=logreg_c,
    )


def make_pipeline(model_name: str, max_features: int, ngram_max: int, logreg_c: float) -> Pipeline:
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
                make_classifier(model_name, logreg_c),
            ),
        ]
    )


def run_single_seed(
    data_dir: Path,
    output_dir: Path,
    seed: int,
    model_name: str,
    max_features: int = 30000,
    ngram_max: int = 2,
    logreg_c: float = 1.0,
    save_artifacts: bool = True,
) -> tuple[dict, pd.DataFrame]:
    train_path = data_dir / "splits" / f"seed_{seed}_train.csv"
    test_path = data_dir / "splits" / f"seed_{seed}_test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing train/test split for seed {seed}: {train_path}, {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    pipeline = make_pipeline(model_name, max_features, ngram_max, logreg_c)
    pipeline.fit(normalize_text_series(train_df["lyrics"]), train_df["mood_label"])
    predictions = pipeline.predict(normalize_text_series(test_df["lyrics"]))

    labels = sorted(pd.concat([train_df["mood_label"], test_df["mood_label"]]).unique())
    report = classification_report(test_df["mood_label"], predictions, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"})
    report_df.insert(0, "seed", seed)

    matrix = confusion_matrix(test_df["mood_label"], predictions, labels=labels)
    confusion_df = pd.DataFrame(matrix, index=labels, columns=labels)
    confusion_df.index.name = "actual_mood"

    test_predictions_df = test_df[["song_id", "title", "artist", "mood_label"]].copy()
    test_predictions_df["predicted_mood"] = predictions
    test_predictions_df["correct"] = test_predictions_df["mood_label"] == test_predictions_df["predicted_mood"]

    if save_artifacts:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(seed_dir / "classification_report.csv", index=False)
        confusion_df.to_csv(seed_dir / "confusion_matrix.csv")
        test_predictions_df.to_csv(seed_dir / "test_predictions.csv", index=False)

    metrics = {
        "seed": seed,
        "model": model_name,
        "max_features": max_features,
        "ngram_max": ngram_max,
        "logreg_c": logreg_c if model_name == "logreg" else np.nan,
        "num_train": len(train_df),
        "num_test": len(test_df),
        "num_classes": len(labels),
        "accuracy": accuracy_score(test_df["mood_label"], predictions),
        "macro_f1": f1_score(test_df["mood_label"], predictions, average="macro"),
        "weighted_f1": f1_score(test_df["mood_label"], predictions, average="weighted"),
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
    mean_df.index.name = "actual_mood"
    return mean_df


def main() -> None:
    args = parse_args()
    data_dir = resolve_path(args.data_dir)
    output_dir = resolve_path(args.output_dir) if args.output_dir else default_output_dir(args.model)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_metrics = []
    confusion_matrices = []
    for seed in args.seeds:
        metrics, confusion_df = run_single_seed(
            data_dir=data_dir,
            output_dir=output_dir,
            seed=seed,
            model_name=args.model,
            max_features=args.max_features,
            ngram_max=args.ngram_max,
            logreg_c=args.logreg_c,
        )
        split_metrics.append(metrics)
        confusion_matrices.append(confusion_df)

    split_metrics_df = pd.DataFrame(split_metrics)
    aggregate_df = aggregate_metrics(split_metrics_df)
    mean_confusion_df = mean_confusion_matrix(confusion_matrices)

    split_metrics_df.to_csv(output_dir / "split_metrics.csv", index=False)
    aggregate_df.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    mean_confusion_df.to_csv(output_dir / "mean_confusion_matrix.csv")

    print(f"Saved mood classifier experiment outputs to: {output_dir}")
    print("\nSplit metrics:")
    print(split_metrics_df.to_string(index=False))
    print("\nAggregate metrics:")
    print(aggregate_df.to_string(index=False))


if __name__ == "__main__":
    main()
