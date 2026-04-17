from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from experiment_utils import clean_text, load_config, resolve_path


DEFAULT_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "i", "you", "me", "my", "we", "our",
    "your", "yours", "to", "of", "in", "on", "for", "with", "at", "by", "is", "it",
    "its", "this", "that", "these", "those", "be", "am", "are", "was", "were",
    "so", "if", "not", "as", "from", "into", "than", "then", "too", "very",
    "oh", "yeah", "la", "na", "ooh", "ah", "hey", "woo",
    "im", "ive", "ill", "id", "youre", "youve", "youll", "youd",
    "were", "weve", "well", "wed", "theyre", "theyve", "theyll", "theyd",
    "hes", "shes", "thats", "theres", "whats", "heres",
    "dont", "didnt", "doesnt", "cant", "couldnt", "wouldnt", "shouldnt",
    "isnt", "arent", "wasnt", "werent", "wont", "hadnt", "hasnt", "havent",
    "got", "get", "gets", "getting", "just", "like", "know", "see", "go", "gone",
    "going", "come", "comes", "coming", "back", "away", "right", "never", "ever",
    "still", "cause", "make", "makes", "made", "take", "takes", "want", "wanna",
    "need", "say", "says", "said", "let", "lets", "one", "two", "thing", "things",
    "time", "day", "night", "baby",
}


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def normalize_text_for_tokens(text: str) -> str:
    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub(r"[']", "", text)
    return text


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"\b[a-z]+\b", normalize_text_for_tokens(text))


def choose_primary_tag(tag_string: str) -> str:
    text = clean_text(tag_string)
    if not text:
        return ""
    return text.split(",")[0].strip()


def normalize_tag_list(tag_string: str, max_tags: int = 5) -> str:
    text = clean_text(tag_string)
    if not text:
        return ""
    parts = [part.strip() for part in text.split(",") if part.strip()]
    return ", ".join(parts[:max_tags])


def deduplicate_lines(text: str) -> str:
    if not text:
        return ""

    seen = set()
    kept = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line in seen:
            continue
        seen.add(line)
        kept.append(line)
    return "\n".join(kept)


def load_lyrics(song_id: str, lyrics_dir: Path) -> str:
    lyric_file = lyrics_dir / f"{song_id}.txt"
    if not lyric_file.exists():
        return ""
    return lyric_file.read_text(encoding="utf-8", errors="ignore").strip()


def extract_tokens(
    text: str,
    *,
    dedup_lines: bool,
    remove_stopwords: bool,
    min_token_length: int,
    stopwords: set[str],
) -> list[str]:
    if not text:
        return []

    if dedup_lines:
        text = deduplicate_lines(text)

    tokens = tokenize(text)
    if remove_stopwords:
        tokens = [token for token in tokens if token not in stopwords]

    if min_token_length > 1:
        tokens = [token for token in tokens if len(token) >= min_token_length]

    return tokens


def words_string_from_tokens(tokens: list[str], top_k: int) -> str:
    counts = Counter(tokens)
    words = [word for word, _ in counts.most_common(top_k)]
    return " ".join(words)


def freq_string_from_tokens(tokens: list[str], top_k: int) -> str:
    counts = Counter(tokens)
    items = [f"{word}:{count}" for word, count in counts.most_common(top_k)]
    return " ".join(items)


def build_dataframe(config: dict) -> pd.DataFrame:
    dataset_config = config["dataset"]
    dataset_root = resolve_path(dataset_config["input_root"])

    info_file = dataset_root / "id_information.csv"
    lang_file = dataset_root / "id_lang.csv"
    metadata_file = dataset_root / "id_metadata.csv"
    tags_file = dataset_root / "id_tags.csv"
    genres_file = dataset_root / "id_genres.csv"
    lyrics_dir = dataset_root / "lyrics"

    info_df = read_tsv(info_file).rename(columns={"id": "song_id", "song": "title"})
    lang_df = read_tsv(lang_file).rename(columns={"id": "song_id"})
    metadata_df = read_tsv(metadata_file).rename(columns={"id": "song_id"})
    tags_df = read_tsv(tags_file).rename(columns={"id": "song_id"})

    df = info_df.merge(lang_df, on="song_id", how="left")
    df = df.merge(metadata_df, on="song_id", how="left")
    df = df.merge(tags_df, on="song_id", how="left")

    if genres_file.exists():
        genres_df = read_tsv(genres_file).rename(columns={"id": "song_id"})
        df = df.merge(genres_df, on="song_id", how="left")

    for column in ["title", "artist", "tags", "lang", "spotify_id", "release"]:
        if column not in df.columns:
            df[column] = ""

    df["title"] = df["title"].apply(clean_text)
    df["artist"] = df["artist"].apply(clean_text)
    df["tags"] = df["tags"].apply(clean_text)

    if "genres" in df.columns:
        df["genre"] = df["genres"].apply(normalize_tag_list)
        df["genre"] = df["genre"].replace("", pd.NA)
        df["genre"] = df["genre"].fillna(df["tags"].apply(choose_primary_tag))
    else:
        df["genre"] = df["tags"].apply(choose_primary_tag)

    df["valence"] = pd.to_numeric(df.get("valence"), errors="coerce").round(3)
    df["arousal"] = pd.to_numeric(df.get("energy"), errors="coerce").round(3)

    if dataset_config.get("english_only", True):
        df["lang"] = df["lang"].astype(str).str.strip().str.lower()
        df = df[df["lang"] == "en"]

    df = df[
        (df["title"].str.len() > 0)
        & (df["artist"].str.len() > 0)
        & (df["genre"].str.len() > 0)
        & pd.notna(df["valence"])
        & pd.notna(df["arousal"])
    ].copy()

    max_rows = dataset_config.get("max_rows")
    candidate_multiplier = int(dataset_config.get("candidate_multiplier", 20))
    if max_rows is not None:
        candidate_rows = int(max_rows) * max(1, candidate_multiplier)
        if len(df) > candidate_rows:
            df = df.sample(n=candidate_rows, random_state=int(dataset_config.get("random_seed", 42)))
            df = df.reset_index(drop=True)

    df["reference_lyrics"] = df["song_id"].astype(str).apply(
        lambda song_id: load_lyrics(song_id, lyrics_dir)
    )
    df["reference_lyrics"] = df["reference_lyrics"].str.replace(r"\r\n?", "\n", regex=True)

    dedup_lines = bool(dataset_config.get("deduplicate_lines", True))
    min_token_length = int(dataset_config.get("min_token_length", 3))
    top_k_vocab = int(dataset_config.get("top_k_vocab", 50))

    df["raw_tokens"] = df["reference_lyrics"].apply(
        lambda text: extract_tokens(
            text,
            dedup_lines=dedup_lines,
            remove_stopwords=False,
            min_token_length=1,
            stopwords=DEFAULT_STOPWORDS,
        )
    )
    df["content_tokens"] = df["reference_lyrics"].apply(
        lambda text: extract_tokens(
            text,
            dedup_lines=dedup_lines,
            remove_stopwords=True,
            min_token_length=min_token_length,
            stopwords=DEFAULT_STOPWORDS,
        )
    )

    df["vocab_words_raw"] = df["raw_tokens"].apply(lambda tokens: words_string_from_tokens(tokens, top_k_vocab))
    df["vocab_freq_raw"] = df["raw_tokens"].apply(lambda tokens: freq_string_from_tokens(tokens, top_k_vocab))
    df["vocab_words_content"] = df["content_tokens"].apply(
        lambda tokens: words_string_from_tokens(tokens, top_k_vocab)
    )
    df["vocab_freq_content"] = df["content_tokens"].apply(
        lambda tokens: freq_string_from_tokens(tokens, top_k_vocab)
    )

    primary_vocab_strategy = clean_text(dataset_config.get("primary_vocab_strategy", "content")) or "content"
    if primary_vocab_strategy not in {"content", "raw"}:
        raise ValueError("primary_vocab_strategy must be 'content' or 'raw'.")

    df["vocab_strategy"] = primary_vocab_strategy
    df["vocab_words"] = df[f"vocab_words_{primary_vocab_strategy}"]
    df["vocab_freq"] = df[f"vocab_freq_{primary_vocab_strategy}"]

    out_df = df[
        [
            "song_id",
            "title",
            "artist",
            "genre",
            "valence",
            "arousal",
            "tags",
            "reference_lyrics",
            "vocab_strategy",
            "vocab_words",
            "vocab_freq",
            "vocab_words_raw",
            "vocab_freq_raw",
            "vocab_words_content",
            "vocab_freq_content",
            "lang",
            "spotify_id",
            "release",
        ]
    ].copy()

    out_df = out_df[
        (out_df["reference_lyrics"].str.len() > 0)
        & (out_df["vocab_words"].str.len() > 0)
        & (out_df["vocab_freq"].str.len() > 0)
    ]

    out_df = out_df.drop_duplicates(subset=["song_id"]).reset_index(drop=True)

    if max_rows is not None and len(out_df) > int(max_rows):
        out_df = out_df.sample(n=int(max_rows), random_state=int(dataset_config.get("random_seed", 42)))
        out_df = out_df.reset_index(drop=True)

    return out_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the Music4All input dataset.")
    parser.add_argument("--config", default=None, help="Path to experiment JSON config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_file = resolve_path(config["dataset"]["output_file"])

    df = build_dataframe(config)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"Saved cleaned Music4All input to: {output_file}")
    print(f"Rows: {len(df)}")
    print("Columns:", list(df.columns))
    preview_cols = [
        "song_id",
        "title",
        "artist",
        "genre",
        "valence",
        "arousal",
        "vocab_strategy",
        "vocab_words",
        "vocab_freq",
    ]
    print(df[preview_cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
