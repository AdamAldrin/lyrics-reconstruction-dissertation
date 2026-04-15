# src/build_prompt_dataset.py

import math
import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/samples/lycon_sample_with_prompt_inputs.csv")
OUTPUT_FILE = Path("data/processed/prompt_dataset.csv")


def mood_from_valence_arousal(valence: float, arousal: float) -> str:
    """
    Map valence/arousal to a mood label using the angle theta in [0, 2pi).

    Note:
    The LyCon paper defines mood categories from valence-arousal angle ranges.
    This implementation uses four angle sectors. If you want exact replication,
    align the boundaries with Figure 1 from the paper. :contentReference[oaicite:1]{index=1}
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
    """
    Ensure the vocabulary field becomes a clean string.

    Assumes the input column already contains keywords sorted by descending
    frequency upstream, which is what the paper describes for the vocabulary
    list. Verify this in your dataset construction. :contentReference[oaicite:2]{index=2}
    """
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_reproduction_prompt(row: pd.Series) -> str:
    """
    Prompt close to the LyCon paper formulation.
    """
    mood = mood_from_valence_arousal(row["valence"], row["arousal"])
    vocab = normalize_bow_keywords(row["bow_keywords"])

    return (
        f'Compose {row["genre"]} lyrics, in a style reminiscent of {row["artist"]} '
        f'which represents a {mood} mood under the title of "{row["title"]}" '
        f'using the following vocabulary: {vocab}.'
    ).strip()


def build_extension_prompt(row: pd.Series) -> str:
    """
    Your extension prompt: adds structure and stricter constraints.
    """
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

    required_columns = ["title", "artist", "genre", "valence", "arousal", "bow_keywords"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["bow_keywords"] = df["bow_keywords"].apply(normalize_bow_keywords)
    df["mood_label"] = df.apply(
        lambda row: mood_from_valence_arousal(row["valence"], row["arousal"]),
        axis=1,
    )

    df["reproduction_prompt"] = df.apply(build_reproduction_prompt, axis=1)
    df["extension_prompt"] = df.apply(build_extension_prompt, axis=1)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved prompt dataset to: {OUTPUT_FILE}")
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nSample preview:")
    preview_cols = ["title", "artist", "genre", "valence", "arousal", "mood_label"]
    print(df[preview_cols].head().to_string(index=False))


if __name__ == "__main__":
    main()