from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from build_prompt_dataset import mood_from_valence_arousal
from experiment_utils import PROJECT_ROOT, clean_text


DEFAULT_SEEDS = list(range(1, 11))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare held-out Music4All data and repeated train/test splits for mood classification."
    )
    parser.add_argument("--input-root", default="data/raw/music4all", help="Music4All root directory.")
    parser.add_argument(
        "--exclude-run",
        default="01-baseline-lycon-final",
        help="Run whose prompt_dataset.csv defines the reconstruction songs to exclude.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size.")
    parser.add_argument("--max-songs", type=int, default=0, help="Optional maximum songs after filtering. 0 means all.")
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS, help="Random seeds for repeated splits.")
    parser.add_argument(
        "--output-dir",
        default="data/processed/mood_classifier",
        help="Directory for prepared classifier dataset and split files.",
    )
    return parser.parse_args()


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def load_lyrics(song_id: str, lyrics_dir: Path) -> str:
    lyric_file = lyrics_dir / f"{song_id}.txt"
    if not lyric_file.exists():
        return ""
    return lyric_file.read_text(encoding="utf-8", errors="ignore").strip()


def get_excluded_song_ids(run_id: str) -> set[str]:
    prompt_path = PROJECT_ROOT / "outputs" / "runs" / run_id / "prompt_dataset.csv"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Cannot find exclusion prompt dataset: {prompt_path}")
    df = pd.read_csv(prompt_path, usecols=["song_id"])
    return set(df["song_id"].astype(str))


def build_dataset(input_root: Path, excluded_song_ids: set[str], max_songs: int) -> pd.DataFrame:
    lyrics_dir = input_root / "lyrics"
    info_df = read_tsv(input_root / "id_information.csv").rename(columns={"id": "song_id", "song": "title"})
    lang_df = read_tsv(input_root / "id_lang.csv").rename(columns={"id": "song_id"})
    metadata_df = read_tsv(input_root / "id_metadata.csv").rename(columns={"id": "song_id"})
    tags_df = read_tsv(input_root / "id_tags.csv").rename(columns={"id": "song_id"})

    df = info_df.merge(lang_df, on="song_id", how="left")
    df = df.merge(metadata_df, on="song_id", how="left")
    df = df.merge(tags_df, on="song_id", how="left")

    genres_file = input_root / "id_genres.csv"
    if genres_file.exists():
        genres_df = read_tsv(genres_file).rename(columns={"id": "song_id"})
        df = df.merge(genres_df, on="song_id", how="left")

    df["song_id"] = df["song_id"].astype(str)
    df["title"] = df["title"].apply(clean_text)
    df["artist"] = df["artist"].apply(clean_text)
    df["lang"] = df["lang"].astype(str).str.strip().str.lower()
    df["valence"] = pd.to_numeric(df.get("valence"), errors="coerce")
    df["arousal"] = pd.to_numeric(df.get("energy"), errors="coerce")

    if "genres" in df.columns:
        df["genre"] = df["genres"].apply(clean_text)
    else:
        df["genre"] = ""
    df["tags"] = df["tags"].apply(clean_text)

    df = df[
        (df["lang"] == "en")
        & (~df["song_id"].isin(excluded_song_ids))
        & (df["title"].str.len() > 0)
        & (df["artist"].str.len() > 0)
        & pd.notna(df["valence"])
        & pd.notna(df["arousal"])
    ].copy()

    df["lyrics"] = df["song_id"].apply(lambda song_id: load_lyrics(song_id, lyrics_dir))
    df["lyrics"] = df["lyrics"].str.replace(r"\r\n?", "\n", regex=True).fillna("")
    df = df[df["lyrics"].str.strip().str.len() > 0].copy()
    df["mood_label"] = df.apply(lambda row: mood_from_valence_arousal(row["valence"], row["arousal"]), axis=1)
    df = df.drop_duplicates(subset=["song_id"]).reset_index(drop=True)

    if max_songs and len(df) > max_songs:
        df = df.sample(n=max_songs, random_state=42).reset_index(drop=True)

    return df[
        [
            "song_id",
            "title",
            "artist",
            "genre",
            "tags",
            "valence",
            "arousal",
            "mood_label",
            "lyrics",
        ]
    ]


def write_splits(df: pd.DataFrame, output_dir: Path, seeds: list[int], test_size: float) -> pd.DataFrame:
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for seed in seeds:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=df["mood_label"],
        )
        train_path = split_dir / f"seed_{seed}_train.csv"
        test_path = split_dir / f"seed_{seed}_test.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        summary_rows.append(
            {
                "seed": seed,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_file": str(train_path.relative_to(PROJECT_ROOT)),
                "test_file": str(test_path.relative_to(PROJECT_ROOT)),
            }
        )

    return pd.DataFrame(summary_rows)


def main() -> None:
    args = parse_args()
    input_root = resolve_path(args.input_root)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    excluded_song_ids = get_excluded_song_ids(args.exclude_run)
    df = build_dataset(input_root, excluded_song_ids, args.max_songs)
    class_distribution = df["mood_label"].value_counts().rename_axis("mood_label").reset_index(name="count")
    split_summary = write_splits(df, output_dir, args.seeds, args.test_size)

    df.to_csv(output_dir / "mood_classifier_dataset.csv", index=False)
    pd.DataFrame({"excluded_song_id": sorted(excluded_song_ids)}).to_csv(
        output_dir / "excluded_reconstruction_song_ids.csv",
        index=False,
    )
    class_distribution.to_csv(output_dir / "class_distribution.csv", index=False)
    split_summary.to_csv(output_dir / "split_summary.csv", index=False)

    print(f"Saved mood-classifier dataset to: {output_dir / 'mood_classifier_dataset.csv'}")
    print(f"Rows: {len(df)}")
    print(f"Excluded reconstruction songs: {len(excluded_song_ids)}")
    print("\nClass distribution:")
    print(class_distribution.to_string(index=False))
    print("\nSplit summary:")
    print(split_summary.to_string(index=False))


if __name__ == "__main__":
    main()
