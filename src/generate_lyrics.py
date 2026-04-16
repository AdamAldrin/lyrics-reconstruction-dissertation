# src/generate_lyrics.py

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

INPUT_FILE = Path("data/processed/prompt_dataset.csv")
OUTPUT_FILE = Path("outputs/generated/generated_outputs.csv")

MODEL_NAME = "gpt-4o-mini"   # use "gpt-4o" for closer paper reproduction
MAX_ROWS = None                # set to None to run all rows
SLEEP_BETWEEN_CALLS = 1.0
MAX_RETRIES = 3


def generate_text(client: OpenAI, prompt: str, model: str = MODEL_NAME, max_retries: int = MAX_RETRIES) -> str:
    """
    Call the Responses API with retries.
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
            )
            return response.output_text.strip()
        except Exception as e:
            last_error = e
            wait_time = 2 ** attempt
            print(f"Generation failed (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(wait_time)

    raise last_error


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
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Set it in your environment first.")

    client = OpenAI(api_key=api_key)

    df = pd.read_csv(INPUT_FILE).copy()
    if MAX_ROWS is not None:
        df = df.head(MAX_ROWS)

    required_columns = [
    "song_id",
    "title",
    "artist",
    "genre",
    "valence",
    "arousal",
    "mood_label",
    "bow_keywords",
    "reference_lyrics",
    "tags",
    "release",
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

        print(f"\nProcessing row {idx + 1}/{len(df)}: {row['title']} - {row['artist']}")

        reproduction_output = ""
        extension_output = ""

        try:
            reproduction_output = generate_text(
                client=client,
                prompt=row["reproduction_prompt"],
                model=MODEL_NAME,
            )
        except Exception as e:
            print(f"Reproduction prompt failed for row {idx}: {e}")

        time.sleep(SLEEP_BETWEEN_CALLS)

        try:
            extension_output = generate_text(
                client=client,
                prompt=row["extension_prompt"],
                model=MODEL_NAME,
            )
        except Exception as e:
            print(f"Extension prompt failed for row {idx}: {e}")

        time.sleep(SLEEP_BETWEEN_CALLS)

        result = {
            "song_id": row.get("song_id", ""),
            "title": row["title"],
            "artist": row["artist"],
            "genre": row["genre"],
            "valence": row["valence"],
            "arousal": row["arousal"],
            "mood_label": row["mood_label"],
            "bow_keywords": row.get("bow_keywords", ""),
            "reference_lyrics": row.get("reference_lyrics", ""),
            "tags": row.get("tags", ""),
            "release": row.get("release", ""),
            "model_name": MODEL_NAME,
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

    print(f"\nSaved generated outputs to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()