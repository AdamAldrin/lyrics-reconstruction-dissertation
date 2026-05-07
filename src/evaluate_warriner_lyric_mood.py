from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd


DEFAULT_FINAL_RUNS = [
    "01-baseline-lycon-final",
    "02-structured-final",
    "03-structured-freq-final",
    "04-baseline-raw-vocab-final",
    "06-baseline-top20-final",
    "07-baseline-top100-final",
    "08-minimal-control-final",
    "09-strong-control-final",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def tokenize(text: str) -> list[str]:
    if pd.isna(text) or not str(text).strip():
        return []
    return re.findall(r"\b[a-zA-Z]+\b", str(text).lower())


def load_warriner_lexicon(path: Path) -> dict[str, tuple[float, float]]:
    df = pd.read_csv(path)
    required = {"Word", "V.Mean.Sum", "A.Mean.Sum"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Warriner file is missing required columns: {sorted(missing)}")

    lexicon: dict[str, tuple[float, float]] = {}
    for row in df[["Word", "V.Mean.Sum", "A.Mean.Sum"]].dropna().itertuples(index=False):
        word = str(row[0]).strip().lower()
        if word:
            lexicon[word] = (float(row[1]), float(row[2]))
    return lexicon


def lyric_affect(text: str, lexicon: dict[str, tuple[float, float]]) -> dict[str, float]:
    tokens = tokenize(text)
    matched = [lexicon[token] for token in tokens if token in lexicon]

    if not tokens or not matched:
        return {
            "token_count": float(len(tokens)),
            "matched_token_count": float(len(matched)),
            "warriner_coverage": 0.0,
            "valence": math.nan,
            "arousal": math.nan,
        }

    valence = sum(item[0] for item in matched) / len(matched)
    arousal = sum(item[1] for item in matched) / len(matched)
    return {
        "token_count": float(len(tokens)),
        "matched_token_count": float(len(matched)),
        "warriner_coverage": len(matched) / len(tokens),
        "valence": valence,
        "arousal": arousal,
    }


def find_output_column(df: pd.DataFrame) -> str:
    clean_columns = [col for col in df.columns if col.endswith("_output_clean")]
    if len(clean_columns) == 1:
        return clean_columns[0]
    if "generated_lyrics" in df.columns:
        return "generated_lyrics"
    raise ValueError(
        "Could not identify generated lyric column. Expected exactly one '*_output_clean' column."
    )


def evaluate_run(run_dir: Path, lexicon: dict[str, tuple[float, float]]) -> pd.DataFrame:
    generated_path = run_dir / "generated_outputs.csv"
    if not generated_path.exists():
        raise FileNotFoundError(f"Missing generated outputs: {generated_path}")

    df = pd.read_csv(generated_path)
    output_col = find_output_column(df)
    required = {"run_id", "song_id", "title", "artist", "genre", "reference_lyrics", output_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{generated_path} is missing columns: {sorted(missing)}")

    records: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        row_data = row._asdict()
        original = lyric_affect(str(row_data["reference_lyrics"]), lexicon)
        generated = lyric_affect(str(row_data[output_col]), lexicon)

        valence_diff = generated["valence"] - original["valence"]
        arousal_diff = generated["arousal"] - original["arousal"]
        valence_error = abs(valence_diff)
        arousal_error = abs(arousal_diff)
        va_distance = math.sqrt((valence_error**2) + (arousal_error**2))

        records.append(
            {
                "run_id": row_data["run_id"],
                "song_id": row_data["song_id"],
                "title": row_data["title"],
                "artist": row_data["artist"],
                "genre": row_data["genre"],
                "original_token_count": original["token_count"],
                "original_matched_token_count": original["matched_token_count"],
                "original_warriner_coverage": original["warriner_coverage"],
                "original_lyric_valence": original["valence"],
                "original_lyric_arousal": original["arousal"],
                "generated_token_count": generated["token_count"],
                "generated_matched_token_count": generated["matched_token_count"],
                "generated_warriner_coverage": generated["warriner_coverage"],
                "generated_lyric_valence": generated["valence"],
                "generated_lyric_arousal": generated["arousal"],
                "valence_diff_generated_minus_original": valence_diff,
                "arousal_diff_generated_minus_original": arousal_diff,
                "abs_valence_error": valence_error,
                "abs_arousal_error": arousal_error,
                "va_distance": va_distance,
            }
        )

    return pd.DataFrame(records)


def summarise(per_song: pd.DataFrame) -> pd.DataFrame:
    summary = (
        per_song.groupby("run_id", as_index=False)
        .agg(
            num_songs=("song_id", "count"),
            avg_original_warriner_coverage=("original_warriner_coverage", "mean"),
            avg_generated_warriner_coverage=("generated_warriner_coverage", "mean"),
            avg_original_lyric_valence=("original_lyric_valence", "mean"),
            avg_generated_lyric_valence=("generated_lyric_valence", "mean"),
            avg_original_lyric_arousal=("original_lyric_arousal", "mean"),
            avg_generated_lyric_arousal=("generated_lyric_arousal", "mean"),
            avg_abs_valence_error=("abs_valence_error", "mean"),
            avg_abs_arousal_error=("abs_arousal_error", "mean"),
            avg_va_distance=("va_distance", "mean"),
            median_va_distance=("va_distance", "median"),
        )
        .sort_values(["avg_va_distance", "avg_abs_valence_error", "avg_abs_arousal_error"])
    )
    return summary


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Evaluate lyric-derived valence/arousal preservation using Warriner word ratings."
    )
    parser.add_argument(
        "--warriner-csv",
        type=Path,
        default=Path.home() / "Downloads" / "Warriner_et_alemotratings.csv",
        help="Path to Warriner affective word ratings CSV.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=root / "outputs" / "runs",
        help="Directory containing final reconstruction run folders.",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        dest="run_ids",
        help="Run ID to evaluate. Can be passed multiple times. Defaults to all final runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs" / "mood_preservation" / "warriner_lyric_affect",
        help="Directory for per-song and summary CSV outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_ids = args.run_ids or DEFAULT_FINAL_RUNS

    lexicon = load_warriner_lexicon(args.warriner_csv)
    per_run_frames = []
    for run_id in run_ids:
        per_run_frames.append(evaluate_run(args.runs_dir / run_id, lexicon))

    per_song = pd.concat(per_run_frames, ignore_index=True)
    summary = summarise(per_song)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_song_path = args.output_dir / "warriner_lyric_affect_per_song.csv"
    summary_path = args.output_dir / "warriner_lyric_affect_summary.csv"
    per_song.to_csv(per_song_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Saved per-song Warriner lyric affect results to: {per_song_path}")
    print(f"Saved Warriner lyric affect summary to: {summary_path}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
