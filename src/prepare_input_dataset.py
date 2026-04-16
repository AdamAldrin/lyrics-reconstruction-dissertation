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

MAX_ROWS = 5
TOP_K_BOW = 50
ENGLISH_ONLY = True
RANDOM_SEED = 42

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "i", "you", "me", "my", "we", "our",
    "your", "yours", "to", "of", "in", "on", "for", "with", "at", "by", "is", "it",
    "its", "this", "that", "these", "those", "be", "am", "are", "was", "were",
    "so", "if", "not", "as", "from", "into", "than", "then", "too", "very",
    "oh", "yeah", "la", "na", "ooh", "ah", "hey", "woo",

    # contractions / helper words
    "im", "ive", "ill", "id", "youre", "youve", "youll", "youd",
    "were", "weve", "well", "wed", "theyre", "theyve", "theyll", "theyd",
    "hes", "shes", "thats", "theres", "whats", "heres",
    "dont", "didnt", "doesnt", "cant", "couldnt", "wouldnt", "shouldnt",
    "isnt", "arent", "wasnt", "werent", "wont", "hadnt", "hasnt", "havent",

    # common lyric filler
    "got", "get", "gets", "getting", "just", "like", "know", "see", "go", "gone",
    "going", "come", "comes", "coming", "back", "away", "right", "never", "ever",
    "still", "cause", "make", "makes", "made", "take", "takes", "want", "wanna",
    "need", "say", "says", "said", "let", "lets", "one", "two", "thing", "things"
}


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def normalize_text_for_tokens(text: str) -> str:
    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub(r"[']", "", text)   # it's -> its, won't -> wont
    return text


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    text = normalize_text_for_tokens(text)
    return re.findall(r"\b[a-z]+\b", text)


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


def normalize_tag_list(tag_string: str, max_tags: int = 5) -> str:
    text = clean_text(tag_string)
    if not text:
        return ""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return ", ".join(parts[:max_tags])


def deduplicate_lines(text: str) -> str:
    if not text:
        return ""
    seen = set()
    kept = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line not in seen:
            seen.add(line)
            kept.append(line)
    return "\n".join(kept)


def get_all_tokens(text: str, dedup_lines: bool = True) -> list[str]:
    if not text:
        return []
    if dedup_lines:
        text = deduplicate_lines(text)
    return tokenize(text)


def get_keyword_tokens(text: str, dedup_lines: bool = True) -> list[str]:
    tokens = get_all_tokens(text, dedup_lines=dedup_lines)
    return [t for t in tokens if t not in STOPWORDS]


def words_string_from_tokens(tokens: list[str], top_k: int) -> str:
    counts = Counter(tokens)
    words = [word for word, _ in counts.most_common(top_k)]
    return " ".join(words)


def freq_string_from_tokens(tokens: list[str], top_k: int) -> str:
    counts = Counter(tokens)
    items = [f"{word}:{count}" for word, count in counts.most_common(top_k)]
    return " ".join(items)


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

    for col in ["title", "artist", "tags", "lang", "spotify_id", "release"]:
        if col not in df.columns:
            df[col] = ""

    df["title"] = df["title"].apply(clean_text)
    df["artist"] = df["artist"].apply(clean_text)
    df["tags"] = df["tags"].apply(clean_text)

    if "genres" in df.columns:
        df["genre"] = df["genres"].apply(normalize_tag_list)
        df["genre"] = df["genre"].replace("", pd.NA)
        df["genre"] = df["genre"].fillna(df["tags"].apply(choose_primary_tag))
    else:
        df["genre"] = df["tags"].apply(choose_primary_tag)

    df["reference_lyrics"] = df["song_id"].astype(str).apply(load_lyrics)
    df["reference_lyrics"] = df["reference_lyrics"].str.replace(r"\r\n?", "\n", regex=True)

    df["all_tokens"] = df["reference_lyrics"].apply(get_all_tokens)
    df["keyword_tokens"] = df["reference_lyrics"].apply(get_keyword_tokens)

    df["bow_all_words"] = df["all_tokens"].apply(lambda x: words_string_from_tokens(x, TOP_K_BOW))
    df["bow_all_freq"] = df["all_tokens"].apply(lambda x: freq_string_from_tokens(x, TOP_K_BOW))
    df["bow_keywords_words"] = df["keyword_tokens"].apply(lambda x: words_string_from_tokens(x, TOP_K_BOW))
    df["bow_keywords_freq"] = df["keyword_tokens"].apply(lambda x: freq_string_from_tokens(x, TOP_K_BOW))

    # keep compatibility with older downstream code
    df["bow_keywords"] = df["bow_keywords_words"]

    df["valence"] = pd.to_numeric(df.get("valence"), errors="coerce").round(3)
    df["arousal"] = pd.to_numeric(df.get("energy"), errors="coerce").round(3)

    out_df = df[[
        "song_id",
        "title",
        "artist",
        "genre",
        "valence",
        "arousal",
        "tags",
        "reference_lyrics",
        "bow_all_words",
        "bow_all_freq",
        "bow_keywords_words",
        "bow_keywords_freq",
        "bow_keywords",
        "lang",
        "spotify_id",
        "release",
    ]].copy()

    if ENGLISH_ONLY:
        out_df["lang"] = out_df["lang"].astype(str).str.strip().str.lower()
        out_df = out_df[out_df["lang"] == "en"]

    out_df = out_df[
        (out_df["title"].str.len() > 0) &
        (out_df["artist"].str.len() > 0) &
        (out_df["genre"].str.len() > 0) &
        (out_df["reference_lyrics"].str.len() > 0) &
        (out_df["bow_all_words"].str.len() > 0) &
        (out_df["bow_all_freq"].str.len() > 0) &
        (out_df["bow_keywords_words"].str.len() > 0) &
        (out_df["bow_keywords_freq"].str.len() > 0) &
        (pd.notna(out_df["valence"])) &
        (pd.notna(out_df["arousal"]))
    ]

    out_df = out_df.drop_duplicates(subset=["song_id"]).reset_index(drop=True)

    if MAX_ROWS is not None and len(out_df) > MAX_ROWS:
        out_df = out_df.sample(n=MAX_ROWS, random_state=RANDOM_SEED).reset_index(drop=True)

    return out_df


def main():
    df = build_dataframe()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved cleaned Music4All input to: {OUTPUT_FILE}")
    print(f"Rows: {len(df)}")
    print("Columns:", list(df.columns))
    print(
        df[[
            "song_id",
            "title",
            "artist",
            "bow_all_words",
            "bow_all_freq",
            "bow_keywords_words",
            "bow_keywords_freq",
        ]].head(3).to_string(index=False)
    )


if __name__ == "__main__":
    main()