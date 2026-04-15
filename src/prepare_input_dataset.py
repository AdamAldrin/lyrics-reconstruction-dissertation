# src/prepare_input_dataset.py

from pathlib import Path
import ast
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "lycon" / "LYCONxMXMxM4A.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "samples" / "lycon_sample_with_prompt_inputs.csv"

MAX_ROWS = 3

TITLE_COL = "track_name_LYCON"
ARTIST_COL = "artist_name_LYCON"
GENRE_PRIMARY_COL = "genre_LYCON"
GENRE_FALLBACK_COL = "style_LYCON"
VALENCE_COL = "valence_LYCON"
AROUSAL_COL = "arousal_LYCON"
BOW_COL = "terms_LYCON"

REQUIRED_COLUMNS = [
    TITLE_COL,
    ARTIST_COL,
    GENRE_PRIMARY_COL,
    GENRE_FALLBACK_COL,
    VALENCE_COL,
    AROUSAL_COL,
    BOW_COL,
]


def parse_terms_column(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if not text:
        return ""

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            cleaned_terms = [str(x).strip() for x in parsed if str(x).strip()]
            return " ".join(cleaned_terms)
    except (ValueError, SyntaxError):
        pass

    return text


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def choose_genre(row: pd.Series) -> str:
    primary = clean_text(row.get(GENRE_PRIMARY_COL, ""))
    fallback = clean_text(row.get(GENRE_FALLBACK_COL, ""))

    if primary:
        return primary
    if fallback:
        return fallback
    return ""


def load_raw_dataframe(file_path: Path) -> pd.DataFrame:
    """
    Load the raw LyCon file.

    The dataset appears to be tab-separated, and it contains quoted multiline
    lyric fields, so we read it with sep='\\t'.
    """
    try:
        return pd.read_csv(file_path, sep="\t")
    except Exception as e:
        raise RuntimeError(
            f"Failed to read raw file as tab-separated: {file_path}\n"
            f"Error: {e}"
        ) from e


def main():
    if not RAW_INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Raw input file not found:\n{RAW_INPUT_FILE}\n\n"
            "Place LYCONxMXMxM4A.csv in data/raw/lycon/ or update RAW_INPUT_FILE."
        )

    df = load_raw_dataframe(RAW_INPUT_FILE)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n\n"
            f"Available columns are:\n{list(df.columns)}"
        )

    out_df = pd.DataFrame({
        "title": df[TITLE_COL].apply(clean_text),
        "artist": df[ARTIST_COL].apply(clean_text),
        "genre": df.apply(choose_genre, axis=1),
        "valence": pd.to_numeric(df[VALENCE_COL], errors="coerce"),
        "arousal": pd.to_numeric(df[AROUSAL_COL], errors="coerce"),
        "bow_keywords": df[BOW_COL].apply(parse_terms_column),
    })

    out_df = out_df.dropna(subset=["valence", "arousal"])

    out_df = out_df[
        (out_df["title"].str.len() > 0) &
        (out_df["artist"].str.len() > 0) &
        (out_df["genre"].str.len() > 0) &
        (out_df["bow_keywords"].str.len() > 0)
    ]

    out_df = out_df.drop_duplicates(subset=["title", "artist"]).reset_index(drop=True)

    if MAX_ROWS is not None:
        out_df = out_df.head(MAX_ROWS)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved cleaned sample input to: {OUTPUT_FILE}")
    print(f"Rows: {len(out_df)}")
    print("\nColumns:")
    print(out_df.columns.tolist())
    print("\nPreview:")
    print(out_df.head().to_string(index=False))


if __name__ == "__main__":
    main()