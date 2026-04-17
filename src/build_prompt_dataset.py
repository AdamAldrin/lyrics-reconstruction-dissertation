from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from experiment_utils import (
    clean_text,
    get_prompt_variants,
    get_run_dir,
    get_run_id,
    load_config,
    load_template,
    render_template,
    resolve_path,
    write_json,
)


def center_if_unit_scale(value: float) -> float:
    if pd.isna(value):
        return float("nan")
    if 0.0 <= value <= 1.0:
        return (value * 2.0) - 1.0
    return value


def compute_theta(valence: float, arousal: float) -> float:
    x = center_if_unit_scale(valence)
    y = center_if_unit_scale(arousal)
    theta = math.atan2(y, x)
    if theta < 0:
        theta += 2 * math.pi
    return theta


def mood_from_valence_arousal(valence: float, arousal: float) -> str:
    theta = compute_theta(valence, arousal)
    if 0 <= theta < math.pi / 2:
        return "positive and energetic"
    if math.pi / 2 <= theta < math.pi:
        return "tense and emotional"
    if math.pi <= theta < 3 * math.pi / 2:
        return "sad and reflective"
    return "positive and calm"


def build_frequency_block(vocab_freq: str) -> str:
    vocab_freq = clean_text(vocab_freq)
    if not vocab_freq:
        return ""
    return f"\n\nPrioritize words roughly according to these frequencies:\n{vocab_freq}"


def build_prompt_dataset(config: dict, run_id: str) -> tuple[pd.DataFrame, Path]:
    input_file = resolve_path(config["dataset"]["output_file"])
    output_file = resolve_path(config["prompt_dataset"]["output_file"])
    run_dir = get_run_dir(run_id)

    df = pd.read_csv(input_file).copy()

    required_columns = ["song_id", "title", "artist", "genre", "valence", "arousal", "vocab_words", "vocab_freq"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for column in ["song_id", "title", "artist", "genre", "reference_lyrics", "tags", "release", "vocab_words", "vocab_freq"]:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].apply(clean_text)

    df["valence"] = pd.to_numeric(df["valence"], errors="coerce")
    df["arousal"] = pd.to_numeric(df["arousal"], errors="coerce")
    df = df.dropna(subset=["valence", "arousal"]).copy()

    df["theta"] = df.apply(lambda row: compute_theta(row["valence"], row["arousal"]), axis=1)
    df["mood_label"] = df.apply(lambda row: mood_from_valence_arousal(row["valence"], row["arousal"]), axis=1)

    prompt_variants = get_prompt_variants(config)
    prompt_metadata = []

    for variant in prompt_variants:
        name = clean_text(variant["name"])
        template_file = variant["template_file"]
        vocab_words_column = clean_text(variant.get("vocab_words_column", "vocab_words")) or "vocab_words"
        vocab_freq_column = clean_text(variant.get("vocab_freq_column", ""))
        template_text = load_template(template_file)
        prompt_col = f"{name}_prompt"
        template_version = Path(template_file).stem

        if vocab_words_column not in df.columns:
            raise ValueError(f"Prompt variant '{name}' expects missing column '{vocab_words_column}'.")
        if vocab_freq_column and vocab_freq_column not in df.columns:
            raise ValueError(f"Prompt variant '{name}' expects missing column '{vocab_freq_column}'.")

        def render_row(row: pd.Series) -> str:
            vocab_words = clean_text(row.get(vocab_words_column, ""))
            vocab_freq = clean_text(row.get(vocab_freq_column, "")) if vocab_freq_column else ""
            mapping = {
                "song_id": clean_text(row.get("song_id", "")),
                "title": clean_text(row.get("title", "")),
                "artist": clean_text(row.get("artist", "")),
                "genre": clean_text(row.get("genre", "")),
                "valence": row.get("valence", ""),
                "arousal": row.get("arousal", ""),
                "theta": row.get("theta", ""),
                "mood_label": clean_text(row.get("mood_label", "")),
                "tags": clean_text(row.get("tags", "")),
                "release": clean_text(row.get("release", "")),
                "vocab_words": vocab_words,
                "vocab_freq": vocab_freq,
                "vocab_frequency_block": build_frequency_block(vocab_freq),
            }
            return render_template(template_text, mapping)

        df[prompt_col] = df.apply(render_row, axis=1)
        prompt_metadata.append(
            {
                "name": name,
                "template_file": template_file,
                "template_version": template_version,
                "vocab_words_column": vocab_words_column,
                "vocab_freq_column": vocab_freq_column,
                "requires_structure": bool(variant.get("requires_structure", False)),
            }
        )

    df["run_id"] = run_id

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    run_output_file = run_dir / "prompt_dataset.csv"
    df.to_csv(run_output_file, index=False)

    write_json(run_dir / "prompt_variants.json", {"prompt_variants": prompt_metadata})
    write_json(run_dir / "config_snapshot.json", config)

    return df, run_output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build prompt dataset from prepared song metadata.")
    parser.add_argument("--config", default=None, help="Path to experiment JSON config.")
    parser.add_argument("--run-id", default=None, help="Reusable run identifier for saving outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_id = get_run_id(config, args.run_id)
    df, run_output_file = build_prompt_dataset(config, run_id)

    print(f"Saved prompt dataset to: {run_output_file}")
    print(f"Run ID: {run_id}")
    print("\nMood distribution:")
    print(df["mood_label"].value_counts(dropna=False).to_string())

    preview_cols = ["song_id", "title", "artist", "genre", "valence", "arousal", "theta", "mood_label"]
    print("\nSample preview:")
    print(df[preview_cols].head(10).to_string(index=False))

    for variant in get_prompt_variants(config):
        prompt_col = f"{clean_text(variant['name'])}_prompt"
        if len(df) > 0 and prompt_col in df.columns:
            print(f"\nSample {variant['name']} prompt:")
            print(df.iloc[0][prompt_col])


if __name__ == "__main__":
    main()
