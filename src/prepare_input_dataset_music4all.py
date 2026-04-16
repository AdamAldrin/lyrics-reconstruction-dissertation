# src/prepare_input_dataset_music4all.py

from pathlib import Path
from collections import Counter
import re
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "music4all"

INFO_FILE = DATASET_ROOT / "id_information.csv"
LANG_FILE = DATASET_ROOT / "id_lang.csv"
METADATA_FILE = DATASET_ROOT / "id_metadata.csv"
TAGS_FILE = DATASET_ROOT / "id_tags.csv"
GENRES_FILE = DATASET_ROOT / "id_genres.csv"   # optional if present
LYRICS_DIR = DATASET_ROOT / "lyrics"

OUTPUT_FILE = PROJECT_ROOT / "data" / "samples" / "music4all_sample_with_prompt_inputs.csv"

MAX_ROWS = 100
TOP_K_BOW = 50
ENGLISH_ONLY = True


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"\b[a-zA-Z']+\b", text.lower())


def load_lyrics(song_id: str) -> str:
    lyric_file = LYRICS_DIR / f"{song_id}.txt"
    if not lyric_file.exists():
        return ""
    return lyric_file.read_text(encoding="utf-8", errors="ignore").strip()


def choose_primary_tag(tag_string: str) -> str:
    text = clean_text(tag_string)
    if not text:
        return ""
    return text.split(",")[0].strip()


def extract_bow_keywords(lyrics: str, top_k: int = 50) -> str:
    stopwords = {
        "the", "a", "an", "and", "or", "but", "i", "you", "me", "my", "we", "our",
        "your", "to", "of", "in", "on", "for", "with", "at", "by", "is", "it",
        "this", "that", "be", "am", "are", "was", "were", "so", "if", "not",
        "oh", "yeah", "la", "na", "ooh", "ah"
    }

    tokens = [t for t in tokenize(lyrics) if t not in stopwords and len(t) > 1]
    counts = Counter(tokens)
    keywords = [word for word, _ in counts.most_common(top_k)]
    return " ".join(keywords)


def build_dataframe() -> pd.DataFrame:
    info_df = read_tsv(INFO_FILE).rename(columns={
        "id": "song_id",
        "artist": "artist",
        "song": "title",
    })

    lang_df = read_tsv(LANG_FILE).rename(columns={"id": "song_id"})
    metadata_df = read_tsv(METADATA_FILE).rename(columns={"id": "song_id"})
    tags_df = read_tsv(TAGS_FILE).rename(columns={"id": "song_id"})

    df = info_df.merge(lang_df, on="song_id", how="left")
    df = df.merge(metadata_df, on="song_id", how="left")
    df = df.merge(tags_df, on="song_id", how="left")

    if GENRES_FILE.exists():
        genres_df = read_tsv(GENRES_FILE).rename(columns={"id": "song_id"})
        df = df.merge(genres_df, on="song_id", how="left")

    df["title"] = df["title"].apply(clean_text)
    df["artist"] = df["artist"].apply(clean_text)

    # Prefer id_genres.csv if available, otherwise use first tag
    if "genres" in df.columns:
        df["genre"] = df["genres"].apply(choose_primary_tag)
    else:
        df["genre"] = df["tags"].apply(choose_primary_tag)

    df["reference_lyrics"] = df["song_id"].astype(str).apply(load_lyrics)
    df["bow_keywords"] = df["reference_lyrics"].apply(lambda x: extract_bow_keywords(x, TOP_K_BOW))

    df["valence"] = pd.to_numeric(df["valence"], errors="coerce")
    df["arousal"] = pd.to_numeric(df["energy"], errors="coerce")  # proxy for arousal

    out_df = df[[
        "song_id",
        "title",
        "artist",
        "genre",
        "valence",
        "arousal",
        "tags",
        "reference_lyrics",
        "bow_keywords",
        "lang",
        "spotify_id",
        "release",
    ]].copy()

    if ENGLISH_ONLY:
        out_df = out_df[out_df["lang"] == "en"]

    out_df = out_df[
        (out_df["title"].str.len() > 0) &
        (out_df["artist"].str.len() > 0) &
        (out_df["genre"].str.len() > 0) &
        (out_df["reference_lyrics"].str.len() > 0) &
        (out_df["bow_keywords"].str.len() > 0) &
        (pd.notna(out_df["valence"])) &
        (pd.notna(out_df["arousal"]))
    ]

    out_df = out_df.drop_duplicates(subset=["song_id"]).reset_index(drop=True)

    if MAX_ROWS is not None:
        out_df = out_df.head(MAX_ROWS)

    return out_df


def main():
    df = build_dataframe()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved cleaned Music4All input to: {OUTPUT_FILE}")
    print(f"Rows: {len(df)}")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()