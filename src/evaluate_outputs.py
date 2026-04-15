# src/evaluate_outputs.py

import re
from collections import Counter
from pathlib import Path

import pandas as pd

INPUT_FILE = Path("outputs/generated/generated_outputs.csv")
OUTPUT_FILE = Path("outputs/evaluation/evaluation_summary.csv")


SECTION_HEADERS = {
    "verse", "chorus", "bridge", "intro", "outro", "pre-chorus", "hook", "refrain"
}


def tokenize(text: str) -> list[str]:
    if pd.isna(text) or not str(text).strip():
        return []
    return re.findall(r"\b\w+\b", str(text).lower())


def count_lines(text: str) -> int:
    if pd.isna(text) or not str(text).strip():
        return 0
    return sum(1 for line in str(text).splitlines() if line.strip())


def count_sections(text: str) -> int:
    if pd.isna(text) or not str(text).strip():
        return 0

    count = 0
    for line in str(text).splitlines():
        clean = line.strip().lower().replace("[", "").replace("]", "").replace(":", "")
        if clean in SECTION_HEADERS or any(clean.startswith(h) for h in SECTION_HEADERS):
            count += 1
    return count


def get_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def summarize_column(df: pd.DataFrame, text_col: str, label: str) -> dict:
    all_tokens = []
    all_bigrams = []
    all_trigrams = []

    word_counts = []
    line_counts = []
    section_counts = []

    for text in df[text_col].fillna(""):
        tokens = tokenize(text)
        all_tokens.extend(tokens)
        all_bigrams.extend(get_ngrams(tokens, 2))
        all_trigrams.extend(get_ngrams(tokens, 3))

        word_counts.append(len(tokens))
        line_counts.append(count_lines(text))
        section_counts.append(count_sections(text))

    return {
        "output_type": label,
        "num_songs": len(df),
        "avg_word_count": sum(word_counts) / len(word_counts) if word_counts else 0,
        "avg_line_count": sum(line_counts) / len(line_counts) if line_counts else 0,
        "avg_section_count": sum(section_counts) / len(section_counts) if section_counts else 0,
        "unique_unigrams": len(set(all_tokens)),
        "unique_bigrams": len(set(all_bigrams)),
        "unique_trigrams": len(set(all_trigrams)),
    }


def main():
    df = pd.read_csv(INPUT_FILE).copy()

    required_columns = ["reproduction_output", "extension_output"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required output columns: {missing}")

    summaries = [
        summarize_column(df, "reproduction_output", "reproduction"),
        summarize_column(df, "extension_output", "extension"),
    ]

    summary_df = pd.DataFrame(summaries)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved evaluation summary to: {OUTPUT_FILE}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()