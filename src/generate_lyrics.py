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
OUTPUT_FILE = Path(f"outputs/generated/generated_outputs_{MODEL_NAME}.csv")

MODEL_NAME = "gpt-4o-mini"   # use "gpt-4o" for closer paper reproduction
MAX_ROWS = None              # set to None to run all rows
SLEEP_BETWEEN_CALLS = 1.0
MAX_RETRIES = 3


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def generate_text(
    client: OpenAI,
    prompt: str,
    model: str = MODEL_NAME,
    max_retries: int = MAX_RETRIES,
) -> str:
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
    """
    Prefer song_id if present. Fall back to (title, artist).
    """
    if existing_df.empty:
        return set()

    if "song_id" in existing_df.columns:
        song_ids = existing_df["song_id"].fillna("").astype(str).str.strip()
        return set(song_ids[song_ids != ""])

    required = {"title", "artist"}
    if required.issubset(existing_df.columns):
        return set(
            zip(
                existing_df["title"].fillna("").astype(str).str.strip(),
                existing_df["artist"].fillna("").astype(str).str.strip(),
            )
        )

    return set()


def is_done(row: pd.Series, done_keys: set) -> bool:
    song_id = clean_text(row.get("song_id", ""))
    if song_id and song_id in done_keys:
        return True

    key = (clean_text(row.get("title", "")), clean_text(row.get("artist", "")))
    return key in done_keys


def get_optional_col(row: pd.Series, col_name: str) -> str:
    return clean_text(row.get(col_name, ""))


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
        if is_done(row, done_keys):
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
            "song_id": get_optional_col(row, "song_id"),
            "title": clean_text(row["title"]),
            "artist": clean_text(row["artist"]),
            "genre": clean_text(row["genre"]),
            "valence": row["valence"],
            "arousal": row["arousal"],
            "theta": row["theta"] if "theta" in row.index else "",
            "mood_label": clean_text(row["mood_label"]),

            # vocabulary views
            "bow_all_words": get_optional_col(row, "bow_all_words"),
            "bow_all_freq": get_optional_col(row, "bow_all_freq"),
            "bow_keywords_words": get_optional_col(row, "bow_keywords_words"),
            "bow_keywords_freq": get_optional_col(row, "bow_keywords_freq"),
            "bow_keywords": get_optional_col(row, "bow_keywords"),  # backward compatibility

            # metadata / references
            "reference_lyrics": get_optional_col(row, "reference_lyrics"),
            "tags": get_optional_col(row, "tags"),
            "release": get_optional_col(row, "release"),

            # run metadata
            "model_name": MODEL_NAME,
            "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "reproduction_prompt_version": "lycon_repro_v2",
            "extension_prompt_version": "structured_extension_v2",

            # prompts
            "reproduction_prompt": clean_text(row["reproduction_prompt"]),
            "extension_prompt": clean_text(row["extension_prompt"]),

            # outputs
            "reproduction_output": reproduction_output,
            "extension_output": extension_output,
        }

        new_rows.append(result)

        combined_df = pd.concat(
            [existing_df, pd.DataFrame(new_rows)],
            ignore_index=True,
        )
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved generated outputs to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()