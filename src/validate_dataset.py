from __future__ import annotations

import argparse

import pandas as pd

from experiment_utils import get_run_dir, get_run_id, load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate prepared or prompt datasets.")
    parser.add_argument("--config", default=None, help="Path to experiment JSON config.")
    parser.add_argument("--run-id", default=None, help="Run identifier to validate run-specific prompt dataset.")
    parser.add_argument(
        "--dataset",
        choices=["sample", "prompt"],
        default="sample",
        help="Which dataset layer to validate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_id = get_run_id(config, args.run_id)

    if args.dataset == "sample":
        dataset_path = resolve_path(config["dataset"]["output_file"])
        required_columns = [
            "song_id",
            "title",
            "artist",
            "genre",
            "valence",
            "arousal",
            "vocab_words",
            "vocab_freq",
            "reference_lyrics",
        ]
    else:
        dataset_path = get_run_dir(run_id) / "prompt_dataset.csv"
        required_columns = ["song_id", "title", "artist", "genre", "valence", "arousal", "mood_label"]
        required_columns += [f"{variant['name']}_prompt" for variant in config["prompt_variants"]]

    df = pd.read_csv(dataset_path)
    missing = [column for column in required_columns if column not in df.columns]

    print(f"Validating: {dataset_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    if missing:
        print(f"Missing required columns: {missing}")
        raise SystemExit(1)

    print("No required columns are missing.")
    print("\nNull counts:")
    print(df[required_columns].isna().sum().to_string())

    if "mood_label" in df.columns:
        print("\nMood distribution:")
        print(df["mood_label"].value_counts(dropna=False).to_string())

    if "genre" in df.columns:
        print("\nTop genres:")
        print(df["genre"].value_counts(dropna=False).head(10).to_string())


if __name__ == "__main__":
    main()
