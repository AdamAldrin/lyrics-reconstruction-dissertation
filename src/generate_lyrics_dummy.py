# src/generate_lyrics.py

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "prompt_dataset.csv"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "generated" / "generated_outputs.csv"

MAX_ROWS = None  # keep None to use all rows from prompt_dataset.csv


def generate_text_dry_run(prompt: str, label: str) -> str:
    """
    Fake generation function for pipeline testing.
    Does not call any API.
    """
    prompt_preview = prompt[:200].replace("\n", " ")
    return f"[DRY RUN {label}] {prompt_preview}..."


def load_existing_outputs(output_file: Path) -> pd.DataFrame:
    if output_file.exists():
        return pd.read_csv(output_file)
    return pd.DataFrame()


def build_done_key_set(existing_df: pd.DataFrame) -> set:
    required = {"title", "artist"}
    if existing_df.empty or not required.issubset(existing_df.columns):
        return set()
    return set(zip(existing_df["title"], existing_df["artist"]))


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Prompt dataset not found:\n{INPUT_FILE}\n\n"
            "Run src/build_prompt_dataset.py first."
        )

    df = pd.read_csv(INPUT_FILE).copy()
    if MAX_ROWS is not None:
        df = df.head(MAX_ROWS)

    required_columns = [
        "title",
        "artist",
        "genre",
        "valence",
        "arousal",
        "mood_label",
        "bow_keywords",
        "reproduction_prompt",
        "extension_prompt",
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in prompt dataset: {missing}")

    existing_df = load_existing_outputs(OUTPUT_FILE)
    done_keys = build_done_key_set(existing_df)

    new_rows = []

    for idx, row in df.iterrows():
        key = (row["title"], row["artist"])
        if key in done_keys:
            print(f"Skipping existing row: {row['title']} - {row['artist']}")
            continue

        print(f"Processing row {idx + 1}/{len(df)}: {row['title']} - {row['artist']}")

        reproduction_output = generate_text_dry_run(
            prompt=row["reproduction_prompt"],
            label="REPRODUCTION",
        )
        extension_output = generate_text_dry_run(
            prompt=row["extension_prompt"],
            label="EXTENSION",
        )

        result = {
            "title": row["title"],
            "artist": row["artist"],
            "genre": row["genre"],
            "valence": row["valence"],
            "arousal": row["arousal"],
            "mood_label": row["mood_label"],
            "bow_keywords": row["bow_keywords"],
            "model_name": "DRY_RUN",
            "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "reproduction_prompt_version": "lycon_repro_v1",
            "extension_prompt_version": "structured_extension_v1",
            "reproduction_prompt": row["reproduction_prompt"],
            "extension_prompt": row["extension_prompt"],
            "reproduction_output": reproduction_output,
            "extension_output": extension_output,
        }

        new_rows.append(result)

        combined_df = pd.concat(
            [existing_df, pd.DataFrame(new_rows)],
            ignore_index=True
        )
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved dry-run generated outputs to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()