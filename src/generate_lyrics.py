from __future__ import annotations

import argparse
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from experiment_utils import (
    clean_text,
    get_prompt_variants,
    get_run_dir,
    get_run_id,
    load_config,
    resolve_path,
)


load_dotenv()


BOILERPLATE_PREFIX_PATTERNS = [
    r"^sure[!,.:\s].*$",
    r"^here (?:are|is) the lyrics.*$",
    r"^below (?:are|is) the lyrics.*$",
    r"^lyrics[:\s]*$",
    r"^title[:\s].*$",
]
BOILERPLATE_EXACT_LINES = {
    "---",
    "here are the lyrics:",
    "here is the lyrics:",
    "i hope you enjoy it!",
    "i hope you enjoy!",
}


def generate_text(
    client: OpenAI,
    prompt: str,
    *,
    model: str,
    max_retries: int,
) -> str:
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.responses.create(model=model, input=prompt)
            return response.output_text.strip()
        except Exception as exc:
            last_error = exc
            wait_time = 2 ** attempt
            print(f"Generation failed (attempt {attempt + 1}/{max_retries}): {exc}")
            time.sleep(wait_time)
    raise last_error


def generate_text_dry_run(prompt: str, *, label: str) -> str:
    preview = clean_text(prompt).replace("\n", " ")[:240]
    return f"[DRY RUN {label}] {preview}..."


def clean_generated_lyrics(text: str) -> tuple[str, bool]:
    cleaned = clean_text(text)
    if not cleaned:
        return "", False

    original = cleaned
    lines = cleaned.splitlines()

    # Remove obvious wrapper lines at the start, such as "Here are the lyrics".
    while lines:
        stripped = lines[0].strip()
        lowered = stripped.lower()
        if not stripped:
            lines.pop(0)
            continue
        if lowered in BOILERPLATE_EXACT_LINES or any(re.match(pattern, lowered) for pattern in BOILERPLATE_PREFIX_PATTERNS):
            lines.pop(0)
            continue
        break

    # Remove lightweight wrapper lines at the end, such as closing remarks.
    while lines:
        stripped = lines[-1].strip()
        lowered = stripped.lower()
        if not stripped:
            lines.pop()
            continue
        if lowered in BOILERPLATE_EXACT_LINES or lowered.startswith("i hope you enjoy"):
            lines.pop()
            continue
        break

    cleaned = "\n".join(lines).strip()
    return cleaned, cleaned != original


def load_existing_outputs(output_file: Path) -> pd.DataFrame:
    if output_file.exists():
        return pd.read_csv(output_file)
    return pd.DataFrame()


def build_done_key_set(existing_df: pd.DataFrame) -> set[str]:
    if existing_df.empty or "song_id" not in existing_df.columns:
        return set()
    song_ids = existing_df["song_id"].fillna("").astype(str).str.strip()
    return set(song_ids[song_ids != ""])


def generate_outputs(config: dict, run_id: str) -> tuple[pd.DataFrame, Path]:
    prompt_variants = get_prompt_variants(config)
    generation_config = config["generation"]
    run_dir = get_run_dir(run_id)

    input_file = run_dir / "prompt_dataset.csv"
    if not input_file.exists():
        input_file = resolve_path(config["prompt_dataset"]["output_file"])

    output_file = run_dir / "generated_outputs.csv"
    model_name = clean_text(generation_config.get("model_name", "gpt-4o-mini")) or "gpt-4o-mini"
    max_rows = generation_config.get("max_rows")
    sleep_between_calls = float(generation_config.get("sleep_between_calls", 1.0))
    max_retries = int(generation_config.get("max_retries", 3))
    dry_run = bool(generation_config.get("dry_run", False))

    if dry_run:
        client = None
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found. Set it in your environment first or enable dry_run.")
        client = OpenAI(api_key=api_key)

    df = pd.read_csv(input_file).copy()
    if max_rows is not None:
        df = df.head(int(max_rows))

    required_columns = ["song_id", "title", "artist", "genre", "valence", "arousal", "mood_label"]
    required_columns += [f"{clean_text(variant['name'])}_prompt" for variant in prompt_variants]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in prompt dataset: {missing}")

    existing_df = load_existing_outputs(output_file)
    done_keys = build_done_key_set(existing_df)
    new_rows = []

    for idx, row in df.iterrows():
        song_id = clean_text(row.get("song_id", ""))
        if song_id in done_keys:
            print(f"Skipping existing row: {row['title']} - {row['artist']}")
            continue

        print(f"\nProcessing row {idx + 1}/{len(df)}: {row['title']} - {row['artist']}")

        result = {
            "run_id": run_id,
            "song_id": song_id,
            "title": clean_text(row["title"]),
            "artist": clean_text(row["artist"]),
            "genre": clean_text(row["genre"]),
            "valence": row["valence"],
            "arousal": row["arousal"],
            "theta": row.get("theta", ""),
            "mood_label": clean_text(row["mood_label"]),
            "tags": clean_text(row.get("tags", "")),
            "release": clean_text(row.get("release", "")),
            "reference_lyrics": clean_text(row.get("reference_lyrics", "")),
            "vocab_strategy": clean_text(row.get("vocab_strategy", "")),
            "vocab_words": clean_text(row.get("vocab_words", "")),
            "vocab_freq": clean_text(row.get("vocab_freq", "")),
            "vocab_words_raw": clean_text(row.get("vocab_words_raw", "")),
            "vocab_freq_raw": clean_text(row.get("vocab_freq_raw", "")),
            "vocab_words_content": clean_text(row.get("vocab_words_content", "")),
            "vocab_freq_content": clean_text(row.get("vocab_freq_content", "")),
            "model_name": model_name,
            "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

        for variant in prompt_variants:
            name = clean_text(variant["name"])
            prompt_col = f"{name}_prompt"
            output_col = f"{name}_output"
            prompt_text = clean_text(row[prompt_col])
            result[prompt_col] = prompt_text
            result[f"{name}_prompt_version"] = Path(variant["template_file"]).stem

            try:
                if dry_run:
                    output_text = generate_text_dry_run(prompt_text, label=name.upper())
                else:
                    output_text = generate_text(
                        client,
                        prompt_text,
                        model=model_name,
                        max_retries=max_retries,
                    )
            except Exception as exc:
                print(f"{name} prompt failed for row {idx}: {exc}")
                output_text = ""

            cleaned_output_text, output_was_cleaned = clean_generated_lyrics(output_text)
            result[output_col] = output_text
            result[f"{name}_output_clean"] = cleaned_output_text
            result[f"{name}_output_was_cleaned"] = output_was_cleaned
            time.sleep(sleep_between_calls)

        new_rows.append(result)
        combined_df = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(output_file, index=False)

    combined_df = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_file, index=False)
    return combined_df, output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate lyrics from the prompt dataset.")
    parser.add_argument("--config", default=None, help="Path to experiment JSON config.")
    parser.add_argument("--run-id", default=None, help="Reusable run identifier for saving outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_id = get_run_id(config, args.run_id)
    generated_df, output_file = generate_outputs(config, run_id)

    print(f"\nSaved generated outputs to: {output_file}")
    print(f"Rows: {len(generated_df)}")
    print(f"Run ID: {run_id}")


if __name__ == "__main__":
    main()
