from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from build_prompt_dataset import mood_from_valence_arousal
from experiment_utils import PROJECT_ROOT, clean_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lyric mood classifier on held-out Music4All lyrics.")
    parser.add_argument("--input-root", default="data/raw/music4all", help="Music4All root directory.")
    parser.add_argument(
        "--exclude-run",
        default="01-baseline-lycon-final",
        help="Run whose prompt_dataset.csv defines reconstruction song IDs to exclude.",
    )
    parser.add_argument("--max-songs", type=int, default=20000, help="Maximum songs to use for classifier training.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", default="outputs/mood_classifier", help="Directory for classifier outputs.")
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


def normalize_for_classifier(text: str) -> str:
    text = clean_text(text).lower()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^a-z'\n ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_excluded_song_ids(run_id: str) -> set[str]:
    prompt_path = PROJECT_ROOT / "outputs" / "runs" / run_id / "prompt_dataset.csv"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Cannot find exclusion prompt dataset: {prompt_path}")
    df = pd.read_csv(prompt_path, usecols=["song_id"])
    return set(df["song_id"].astype(str))


def build_classifier_dataset(
    input_root: Path,
    *,
    exclude_song_ids: set[str],
    max_songs: int,
    random_seed: int,
) -> pd.DataFrame:
    lyrics_dir = input_root / "lyrics"
    info_df = read_tsv(input_root / "id_information.csv").rename(columns={"id": "song_id", "song": "title"})
    lang_df = read_tsv(input_root / "id_lang.csv").rename(columns={"id": "song_id"})
    metadata_df = read_tsv(input_root / "id_metadata.csv").rename(columns={"id": "song_id"})

    df = info_df.merge(lang_df, on="song_id", how="left")
    df = df.merge(metadata_df, on="song_id", how="left")

    df["song_id"] = df["song_id"].astype(str)
    df["lang"] = df["lang"].astype(str).str.strip().str.lower()
    df["valence"] = pd.to_numeric(df.get("valence"), errors="coerce")
    df["arousal"] = pd.to_numeric(df.get("energy"), errors="coerce")

    df = df[
        (df["lang"] == "en")
        & (~df["song_id"].isin(exclude_song_ids))
        & pd.notna(df["valence"])
        & pd.notna(df["arousal"])
    ].copy()

    if max_songs and len(df) > max_songs:
        df = df.sample(n=max_songs, random_state=random_seed).copy()

    df["lyrics"] = df["song_id"].apply(lambda song_id: load_lyrics(song_id, lyrics_dir))
    df["lyrics_clean"] = df["lyrics"].apply(normalize_for_classifier)
    df = df[df["lyrics_clean"].str.len() > 0].copy()
    df["mood_label"] = df.apply(lambda row: mood_from_valence_arousal(row["valence"], row["arousal"]), axis=1)
    df = df.drop_duplicates(subset=["song_id"]).reset_index(drop=True)

    return df[["song_id", "title", "artist", "valence", "arousal", "mood_label", "lyrics_clean"]]


def make_pipeline(random_seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    max_features=30000,
                    min_df=3,
                    max_df=0.9,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=random_seed,
                ),
            ),
        ]
    )


def train_and_evaluate(
    df: pd.DataFrame,
    *,
    test_size: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=df["mood_label"],
    )

    pipeline = make_pipeline(random_seed)
    pipeline.fit(train_df["lyrics_clean"], train_df["mood_label"])
    predictions = pipeline.predict(test_df["lyrics_clean"])

    metrics_df = pd.DataFrame(
        [
            {
                "num_songs": len(df),
                "num_train": len(train_df),
                "num_test": len(test_df),
                "num_classes": df["mood_label"].nunique(),
                "accuracy": accuracy_score(test_df["mood_label"], predictions),
                "macro_f1": f1_score(test_df["mood_label"], predictions, average="macro"),
                "weighted_f1": f1_score(test_df["mood_label"], predictions, average="weighted"),
            }
        ]
    )

    report = classification_report(test_df["mood_label"], predictions, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"})

    labels = sorted(df["mood_label"].unique())
    matrix = confusion_matrix(test_df["mood_label"], predictions, labels=labels)
    confusion_df = pd.DataFrame(matrix, index=labels, columns=labels)

    test_predictions_df = test_df[["song_id", "title", "artist", "mood_label"]].copy()
    test_predictions_df["predicted_mood"] = predictions
    test_predictions_df["correct"] = test_predictions_df["mood_label"] == test_predictions_df["predicted_mood"]

    return metrics_df, report_df, confusion_df, test_predictions_df


def main() -> None:
    args = parse_args()
    input_root = resolve_path(args.input_root)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    excluded_song_ids = get_excluded_song_ids(args.exclude_run)
    df = build_classifier_dataset(
        input_root,
        exclude_song_ids=excluded_song_ids,
        max_songs=args.max_songs,
        random_seed=args.random_seed,
    )

    if df["mood_label"].nunique() < 2:
        raise ValueError("Mood classifier dataset must contain at least two mood labels.")

    metrics_df, report_df, confusion_df, test_predictions_df = train_and_evaluate(
        df,
        test_size=args.test_size,
        random_seed=args.random_seed,
    )

    class_distribution_df = df["mood_label"].value_counts().rename_axis("mood_label").reset_index(name="count")

    df.drop(columns=["lyrics_clean"]).to_csv(output_dir / "classifier_dataset.csv", index=False)
    class_distribution_df.to_csv(output_dir / "class_distribution.csv", index=False)
    metrics_df.to_csv(output_dir / "classifier_metrics.csv", index=False)
    report_df.to_csv(output_dir / "classification_report.csv", index=False)
    confusion_df.to_csv(output_dir / "confusion_matrix.csv")
    test_predictions_df.to_csv(output_dir / "test_predictions.csv", index=False)

    print(f"Saved mood classifier outputs to: {output_dir}")
    print("\nClassifier metrics:")
    print(metrics_df.to_string(index=False))
    print("\nClass distribution:")
    print(class_distribution_df.to_string(index=False))


if __name__ == "__main__":
    main()
