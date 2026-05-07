from __future__ import annotations

import argparse
from itertools import product

import pandas as pd

from run_valence_arousal_classifiers import (
    aggregate_metrics,
    load_split,
    resolve_path,
    train_predict_binary,
)


DEFAULT_SEEDS = [1, 2, 3]
DEFAULT_MAX_FEATURES = [60000, 100000]
DEFAULT_NGRAM_MAX = [2, 3]
DEFAULT_LOGREG_C = [0.3, 1.0, 3.0]
DEFAULT_CLASS_WEIGHT_MODES = ["none", "balanced"]
TARGETS = {
    "valence": "valence_label",
    "arousal": "arousal_label",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune binary valence and arousal lyric classifiers separately.")
    parser.add_argument("--data-dir", default="data/processed/mood_classifier")
    parser.add_argument("--output-dir", default="outputs/mood_classifier_experiments/binary_valence_arousal_tfidf_tuning")
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--max-features", nargs="*", type=int, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--ngram-max", nargs="*", type=int, choices=[1, 2, 3], default=DEFAULT_NGRAM_MAX)
    parser.add_argument("--logreg-c", nargs="*", type=float, default=DEFAULT_LOGREG_C)
    parser.add_argument("--class-weight-mode", nargs="*", choices=DEFAULT_CLASS_WEIGHT_MODES, default=DEFAULT_CLASS_WEIGHT_MODES)
    parser.add_argument("--targets", nargs="*", choices=list(TARGETS), default=list(TARGETS))
    return parser.parse_args()


def binary_metrics(y_true: pd.Series, y_pred, prefix: str) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_macro_f1": f1_score(y_true, y_pred, average="macro"),
        f"{prefix}_weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def tune_target(args: argparse.Namespace, target_name: str, label_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = resolve_path(args.data_dir)
    configs = list(product(args.max_features, args.ngram_max, args.logreg_c, args.class_weight_mode))
    summary_rows = []
    split_rows = []

    for config_index, (max_features, ngram_max, logreg_c, class_weight_mode) in enumerate(configs, start=1):
        config_name = f"{target_name}_features_{max_features}_ngram_{ngram_max}_c_{logreg_c:g}_weight_{class_weight_mode}"
        print(f"[{target_name} {config_index}/{len(configs)}] {config_name}", flush=True)

        config_split_rows = []
        for seed in args.seeds:
            train_df, test_df = load_split(data_dir, seed)
            predictions, _ = train_predict_binary(
                train_df=train_df,
                test_df=test_df,
                label_column=label_column,
                max_features=max_features,
                ngram_max=ngram_max,
                logreg_c=logreg_c,
                class_weight_mode=class_weight_mode,
            )
            metrics = {
                "target": target_name,
                "config": config_name,
                "seed": seed,
                "max_features": max_features,
                "ngram_max": ngram_max,
                "logreg_c": logreg_c,
                "class_weight_mode": class_weight_mode,
                "num_train": len(train_df),
                "num_test": len(test_df),
            }
            metrics.update(binary_metrics(test_df[label_column], predictions, target_name))
            config_split_rows.append(metrics)
            split_rows.append(metrics)

        config_split_df = pd.DataFrame(config_split_rows)
        metric_columns = [f"{target_name}_accuracy", f"{target_name}_macro_f1", f"{target_name}_weighted_f1"]
        row = {
            "target": target_name,
            "config": config_name,
            "max_features": max_features,
            "ngram_max": ngram_max,
            "logreg_c": logreg_c,
            "class_weight_mode": class_weight_mode,
        }
        for metric in metric_columns:
            values = config_split_df[metric]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1)
            row[f"{metric}_min"] = values.min()
            row[f"{metric}_max"] = values.max()
        summary_rows.append(row)

    split_df = pd.DataFrame(split_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        [f"{target_name}_macro_f1_mean", f"{target_name}_accuracy_mean", f"{target_name}_weighted_f1_mean"],
        ascending=False,
    )
    return summary_df, split_df


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    all_splits = []
    for target_name in args.targets:
        label_column = TARGETS[target_name]
        summary_df, split_df = tune_target(args, target_name, label_column)
        summary_df.to_csv(output_dir / f"{target_name}_tuning_summary.csv", index=False)
        split_df.to_csv(output_dir / f"{target_name}_split_metrics.csv", index=False)
        all_summaries.append(summary_df)
        all_splits.append(split_df)

        print(f"\nTop {target_name} configurations by macro F1:")
        print(summary_df.head(8).to_string(index=False))

    pd.concat(all_summaries, ignore_index=True).to_csv(output_dir / "tuning_summary.csv", index=False)
    pd.concat(all_splits, ignore_index=True).to_csv(output_dir / "all_split_metrics.csv", index=False)
    print(f"\nSaved binary valence/arousal tuning outputs to: {output_dir}")


if __name__ == "__main__":
    main()
