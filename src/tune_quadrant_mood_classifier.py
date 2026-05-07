from __future__ import annotations

import argparse
from itertools import product

import pandas as pd

from run_quadrant_mood_classifier import aggregate_metrics, resolve_path, run_seed


DEFAULT_SEEDS = [1, 2, 3]
DEFAULT_MAX_FEATURES = [30000, 60000, 100000]
DEFAULT_NGRAM_MAX = [1, 2, 3]
DEFAULT_LOGREG_C = [0.3, 1.0, 3.0, 10.0]
DEFAULT_CLASS_WEIGHT_MODES = ["none", "balanced", "mild"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune TF-IDF logistic regression for the 4-mood quadrant classifier.")
    parser.add_argument("--data-dir", default="data/processed/mood_classifier")
    parser.add_argument("--output-dir", default="outputs/mood_classifier_experiments/4_mood_quadrant_tfidf_tuning")
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--max-features", nargs="*", type=int, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--ngram-max", nargs="*", type=int, choices=[1, 2, 3], default=DEFAULT_NGRAM_MAX)
    parser.add_argument("--logreg-c", nargs="*", type=float, default=DEFAULT_LOGREG_C)
    parser.add_argument("--class-weight-mode", nargs="*", choices=DEFAULT_CLASS_WEIGHT_MODES, default=DEFAULT_CLASS_WEIGHT_MODES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = resolve_path(args.data_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = list(product(args.max_features, args.ngram_max, args.logreg_c, args.class_weight_mode))
    config_rows = []
    all_split_rows = []

    for config_index, (max_features, ngram_max, logreg_c, class_weight_mode) in enumerate(configs, start=1):
        config_name = f"features_{max_features}_ngram_{ngram_max}_c_{logreg_c:g}_weight_{class_weight_mode}"
        print(f"[{config_index}/{len(configs)}] {config_name}", flush=True)

        split_rows = []
        config_output_dir = output_dir / config_name
        for seed in args.seeds:
            metrics, _ = run_seed(
                data_dir=data_dir,
                output_dir=config_output_dir,
                seed=seed,
                max_features=max_features,
                ngram_max=ngram_max,
                logreg_c=logreg_c,
                class_weight_mode=class_weight_mode,
                balance_train=False,
            )
            metrics["config"] = config_name
            split_rows.append(metrics)
            all_split_rows.append(metrics)

        split_df = pd.DataFrame(split_rows)
        aggregate_df = aggregate_metrics(split_df)
        metric_lookup = aggregate_df.set_index("metric")
        config_rows.append(
            {
                "config": config_name,
                "max_features": max_features,
                "ngram_max": ngram_max,
                "logreg_c": logreg_c,
                "class_weight_mode": class_weight_mode,
                "accuracy_mean": metric_lookup.loc["accuracy", "mean"],
                "macro_f1_mean": metric_lookup.loc["macro_f1", "mean"],
                "weighted_f1_mean": metric_lookup.loc["weighted_f1", "mean"],
                "accuracy_std": metric_lookup.loc["accuracy", "std"],
                "macro_f1_std": metric_lookup.loc["macro_f1", "std"],
                "weighted_f1_std": metric_lookup.loc["weighted_f1", "std"],
            }
        )

    all_split_df = pd.DataFrame(all_split_rows)
    summary_df = pd.DataFrame(config_rows).sort_values(
        ["macro_f1_mean", "accuracy_mean", "weighted_f1_mean"],
        ascending=False,
    )

    all_split_df.to_csv(output_dir / "all_split_metrics.csv", index=False)
    summary_df.to_csv(output_dir / "tuning_summary.csv", index=False)

    print(f"\nSaved 4-mood tuning outputs to: {output_dir}")
    print("\nTop configurations by macro F1:")
    print(summary_df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
