# src/build_prompt_dataset.py

import math
from pathlib import Path
import pandas as pd

INPUT_FILE = Path("data/samples/music4all_sample_with_prompt_inputs.csv")
OUTPUT_FILE = Path("data/processed/prompt_dataset.csv")


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def normalize_vocab(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def center_if_unit_scale(value: float) -> float:
    """
    If value appears to be in [0,1], map it to [-1,1].
    Otherwise leave it unchanged.
    """
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
    elif math.pi / 2 <= theta < math.pi:
        return "tense and emotional"
    elif math.pi <= theta < 3 * math.pi / 2:
        return "sad and reflective"
    else:
        return "positive and calm"


def pick_column(df: pd.DataFrame, candidates: list[str], default: str = "") -> str:
    for col in candidates:
        if col in df.columns:
            return col
    return default


def build_reproduction_prompt(row: pd.Series, vocab_col: str) -> str:
    mood = mood_from_valence_arousal(row["valence"], row["arousal"])
    vocab = normalize_vocab(row[vocab_col]) if vocab_col else ""

    return (
        f'Compose {row["genre"]} lyrics, in a style reminiscent of {row["artist"]} '
        f'which represents a {mood} mood under the title of "{row["title"]}" '
        f'using the following vocabulary {vocab}.'
    ).strip()


def build_extension_prompt(row: pd.Series, vocab_words_col: str, vocab_freq_col: str) -> str:
    mood = mood_from_valence_arousal(row["valence"], row["arousal"])

    vocab_words = normalize_vocab(row[vocab_words_col]) if vocab_words_col else ""
    vocab_freq = normalize_vocab(row[vocab_freq_col]) if vocab_freq_col else ""

    if vocab_freq:
        vocab_block = f"""Use the following vocabulary naturally:
{vocab_words}

Prioritize words roughly according to these frequencies:
{vocab_freq}"""
    else:
        vocab_block = f"""Use the following vocabulary naturally:
{vocab_words}"""

    return f"""Compose {row["genre"]} lyrics, in a style reminiscent of {row["artist"]}, which represents a {mood} mood under the title of "{row["title"]}".

{vocab_block}

Additional requirements:
- Keep the lyrics coherent and emotionally consistent.
- Structure the lyrics into Verse 1, Chorus, Verse 2, Chorus.
- Make the chorus memorable and central to the song's emotion.
- Do not output explanations or notes, only the lyrics.""".strip()


def main():
    df = pd.read_csv(INPUT_FILE).copy()

    required_columns = [
        "song_id",
        "title",
        "artist",
        "genre",
        "valence",
        "arousal",
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Pick best available vocabulary columns
    reproduction_vocab_col = pick_column(
        df,
        ["bow_all_words", "bow_keywords_words", "bow_keywords"]
    )
    extension_vocab_words_col = pick_column(
        df,
        ["bow_keywords_words", "bow_all_words", "bow_keywords"]
    )
    extension_vocab_freq_col = pick_column(
        df,
        ["bow_keywords_freq", "bow_all_freq"]
    )

    if not reproduction_vocab_col:
        raise ValueError(
            "No usable vocabulary column found. Expected one of: "
            "bow_all_words, bow_keywords_words, bow_keywords"
        )

    # Ensure optional columns exist
    optional_text_columns = ["reference_lyrics", "tags", "release"]
    for col in optional_text_columns:
        if col not in df.columns:
            df[col] = ""

    text_columns = [
        "song_id",
        "title",
        "artist",
        "genre",
        "reference_lyrics",
        "tags",
        "release",
    ]
    for col in text_columns:
        df[col] = df[col].apply(clean_text)

    # Clean chosen vocab columns if present
    for col in {reproduction_vocab_col, extension_vocab_words_col, extension_vocab_freq_col}:
        if col:
            df[col] = df[col].apply(normalize_vocab)

    df["valence"] = pd.to_numeric(df["valence"], errors="coerce")
    df["arousal"] = pd.to_numeric(df["arousal"], errors="coerce")

    df = df.dropna(subset=["valence", "arousal"]).copy()

    df["theta"] = df.apply(
        lambda row: compute_theta(row["valence"], row["arousal"]),
        axis=1,
    )
    df["mood_label"] = df.apply(
        lambda row: mood_from_valence_arousal(row["valence"], row["arousal"]),
        axis=1,
    )

    df["reproduction_prompt"] = df.apply(
        lambda row: build_reproduction_prompt(row, reproduction_vocab_col),
        axis=1,
    )
    df["extension_prompt"] = df.apply(
        lambda row: build_extension_prompt(row, extension_vocab_words_col, extension_vocab_freq_col),
        axis=1,
    )

    output_columns = [
        "song_id",
        "title",
        "artist",
        "genre",
        "valence",
        "arousal",
        "theta",
        "mood_label",
    ]

    for col in [
        "bow_all_words",
        "bow_all_freq",
        "bow_keywords_words",
        "bow_keywords_freq",
        "bow_keywords",
        "reference_lyrics",
        "tags",
        "release",
    ]:
        if col in df.columns:
            output_columns.append(col)

    output_columns += [
        "reproduction_prompt",
        "extension_prompt",
    ]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df[output_columns].to_csv(OUTPUT_FILE, index=False)

    print(f"Saved prompt dataset to: {OUTPUT_FILE}")
    print("\nUsing columns:")
    print(f"  reproduction vocab: {reproduction_vocab_col}")
    print(f"  extension vocab words: {extension_vocab_words_col}")
    print(f"  extension vocab freq: {extension_vocab_freq_col or '[none]'}")

    print("\nMood distribution:")
    print(df["mood_label"].value_counts(dropna=False).to_string())

    print("\nSample preview:")
    preview_cols = [
        "song_id",
        "title",
        "artist",
        "genre",
        "valence",
        "arousal",
        "theta",
        "mood_label",
    ]
    print(df[preview_cols].head(10).to_string(index=False))

    if len(df) > 0:
        print("\nSample reproduction prompt:")
        print(df.iloc[0]["reproduction_prompt"])

        print("\nSample extension prompt:")
        print(df.iloc[0]["extension_prompt"])


if __name__ == "__main__":
    main()