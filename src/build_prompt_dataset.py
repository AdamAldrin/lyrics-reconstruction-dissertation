# src/build_prompt_dataset.py

import math
import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/samples/music4all_sample_with_prompt_inputs.csv")
OUTPUT_FILE = Path("data/processed/prompt_dataset.csv")


def mood_from_valence_arousal(valence: float, arousal: float) -> str:
    """
    Map valence/arousal to a mood label using the angle theta in [0, 2pi).
    """
    theta = math.atan2(arousal, valence)
    if theta < 0:
        theta += 2 * math.pi

    if 0 <= theta < math.pi / 2:
        return "positive and energetic"
    elif math.pi / 2 <= theta < math.pi:
        return "tense and emotional"
    elif math.pi <= theta < 3 * math.pi / 2:
        return "sad and reflective"
    else:
        return "positive and calm"


def normalize_bow_keywords(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_reproduction_prompt(row: pd.Series) -> str:
    mood = mood_from_valence_arousal(row["valence"], row["arousal"])
    vocab = normalize_bow_keywords(row["bow_keywords"])

    return (
        f'Compose {row["genre"]} lyrics, in a style reminiscent of {row["artist"]} '
        f'which represents a {mood} mood under the title of "{row["title"]}" '
        f'using the following vocabulary: {vocab}.'
    ).strip()


def build_extension_prompt(row: pd.Series) -> str:
    mood = mood_from_valence_arousal(row["valence"], row["arousal"])
    vocab = normalize_bow_keywords(row["bow_keywords"])

    return f"""Write original song lyrics with the following requirements.

Title: "{row['title']}"
Artist style inspiration: {row['artist']}
Genre: {row['genre']}
Mood: {mood}

Constraints:
1. Use the provided vocabulary naturally and make sure the important words appear clearly.
2. Keep the lyrics coherent and emotionally consistent.
3. Structure the lyrics into:
   - Verse 1
   - Chorus
   - Verse 2
   - Chorus
4. Make the chorus memorable and central to the song's emotion.
5. Keep the style consistent with the genre and artist inspiration.
6. Do not output explanations or notes, only the lyrics.

Vocabulary:
{vocab}""".strip()


def main():
    df = pd.read_csv(INPUT_FILE).copy()

    required_columns = [
        "song_id",
        "title",
        "artist",
        "genre",
        "valence",
        "arousal",
        "bow_keywords",
        "reference_lyrics",
        "tags",
        "release",
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["song_id"] = df["song_id"].apply(clean_text)
    df["title"] = df["title"].apply(clean_text)
    df["artist"] = df["artist"].apply(clean_text)
    df["genre"] = df["genre"].apply(clean_text)
    df["reference_lyrics"] = df["reference_lyrics"].apply(clean_text)
    df["tags"] = df["tags"].apply(clean_text)
    df["release"] = df["release"].apply(clean_text)
    df["bow_keywords"] = df["bow_keywords"].apply(normalize_bow_keywords)

    df["valence"] = pd.to_numeric(df["valence"], errors="coerce")
    df["arousal"] = pd.to_numeric(df["arousal"], errors="coerce")

    df = df.dropna(subset=["valence", "arousal"])

    df["mood_label"] = df.apply(
        lambda row: mood_from_valence_arousal(row["valence"], row["arousal"]),
        axis=1,
    )

    df["reproduction_prompt"] = df.apply(build_reproduction_prompt, axis=1)
    df["extension_prompt"] = df.apply(build_extension_prompt, axis=1)

    output_columns = [
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

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df[output_columns].to_csv(OUTPUT_FILE, index=False)

    print(f"Saved prompt dataset to: {OUTPUT_FILE}")
    print("\nColumns:")
    print(df[output_columns].columns.tolist())
    print("\nSample preview:")
    preview_cols = [
        "song_id",
        "title",
        "artist",
        "genre",
        "valence",
        "arousal",
        "mood_label",
        "release",
    ]
    print(df[preview_cols].head().to_string(index=False))


if __name__ == "__main__":
    main()