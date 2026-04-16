# src/evaluate_outputs.py

from pathlib import Path
import re
from collections import Counter
import pandas as pd

INPUT_FILE = Path("outputs/generated/generated_outputs.csv")
SUMMARY_OUTPUT_FILE = Path("outputs/evaluation/evaluation_summary.csv")
PER_SONG_OUTPUT_FILE = Path("outputs/evaluation/per_song_evaluation.csv")


SECTION_HEADERS = {
    "verse", "chorus", "bridge", "intro", "outro", "pre-chorus", "hook", "refrain"
}

REQUIRED_EXTENSION_ORDER = ["verse 1", "chorus", "verse 2", "chorus"]


def tokenize(text: str) -> list[str]:
    if pd.isna(text) or not str(text).strip():
        return []
    return re.findall(r"\b\w+\b", str(text).lower())


def split_nonempty_lines(text: str) -> list[str]:
    if pd.isna(text) or not str(text).strip():
        return []
    return [line.strip() for line in str(text).splitlines() if line.strip()]


def count_lines(text: str) -> int:
    return len(split_nonempty_lines(text))


def normalize_section_line(line: str) -> str:
    return (
        line.strip()
        .lower()
        .replace("[", "")
        .replace("]", "")
        .replace(":", "")
    )


def extract_section_labels(text: str) -> list[str]:
    labels = []
    for line in split_nonempty_lines(text):
        clean = normalize_section_line(line)
        if clean in SECTION_HEADERS or any(clean.startswith(h) for h in SECTION_HEADERS):
            labels.append(clean)
    return labels


def count_sections(text: str) -> int:
    return len(extract_section_labels(text))


def get_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(tokens: list[str], n: int) -> float:
    ngrams = get_ngrams(tokens, n)
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def type_token_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def repeated_line_ratio(text: str) -> float:
    lines = [line.lower() for line in split_nonempty_lines(text)]
    if not lines:
        return 0.0
    counts = Counter(lines)
    repeated_instances = sum(count for _, count in counts.items() if count > 1)
    return repeated_instances / len(lines)


def parse_vocab_keywords(vocab: str) -> list[str]:
    if pd.isna(vocab) or not str(vocab).strip():
        return []
    return tokenize(str(vocab))


def keyword_coverage(text: str, vocab: str) -> float:
    text_vocab = set(tokenize(text))
    target_vocab = set(parse_vocab_keywords(vocab))
    if not target_vocab:
        return 0.0
    return len(text_vocab & target_vocab) / len(target_vocab)


def top_keyword_coverage(text: str, vocab: str, top_k: int = 10) -> float:
    text_vocab = set(tokenize(text))
    target_vocab = parse_vocab_keywords(vocab)[:top_k]
    if not target_vocab:
        return 0.0
    return len(set(target_vocab) & text_vocab) / len(set(target_vocab))


def keyword_repetition_ratio(text: str, vocab: str) -> float:
    tokens = tokenize(text)
    vocab_set = set(parse_vocab_keywords(vocab))
    if not tokens or not vocab_set:
        return 0.0

    keyword_tokens = [tok for tok in tokens if tok in vocab_set]
    if not keyword_tokens:
        return 0.0

    counts = Counter(keyword_tokens)
    repeated = sum(count for _, count in counts.items() if count > 1)
    return repeated / len(keyword_tokens)


def structure_score_extension(text: str) -> int:
    """
    Score from 0 to 4 based on whether the extension output includes:
    Verse 1, Chorus, Verse 2, Chorus
    """
    labels = extract_section_labels(text)
    normalized = [label.lower() for label in labels]

    score = 0
    for required in REQUIRED_EXTENSION_ORDER:
        if required in normalized:
            score += 1
    return score


def structure_order_match(text: str) -> int:
    """
    Returns 1 if Verse 1 -> Chorus -> Verse 2 -> Chorus appears in order, else 0.
    """
    labels = extract_section_labels(text)
    normalized = [label.lower() for label in labels]

    try:
        i1 = normalized.index("verse 1")
        i2 = normalized.index("chorus", i1 + 1)
        i3 = normalized.index("verse 2", i2 + 1)
        i4 = normalized.index("chorus", i3 + 1)
        return 1 if i1 < i2 < i3 < i4 else 0
    except ValueError:
        return 0


def jaccard_similarity(text_a: str, text_b: str) -> float:
    a = set(tokenize(text_a))
    b = set(tokenize(text_b))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def unigram_precision(candidate: str, reference: str) -> float:
    cand_tokens = tokenize(candidate)
    ref_vocab = set(tokenize(reference))
    if not cand_tokens:
        return 0.0
    return sum(1 for tok in cand_tokens if tok in ref_vocab) / len(cand_tokens)


def unigram_recall(candidate: str, reference: str) -> float:
    cand_vocab = set(tokenize(candidate))
    ref_vocab = set(tokenize(reference))
    if not ref_vocab:
        return 0.0
    return len(cand_vocab & ref_vocab) / len(ref_vocab)


def bigram_overlap(candidate: str, reference: str) -> float:
    cand_bigrams = set(get_ngrams(tokenize(candidate), 2))
    ref_bigrams = set(get_ngrams(tokenize(reference), 2))
    if not ref_bigrams:
        return 0.0
    return len(cand_bigrams & ref_bigrams) / len(ref_bigrams)


def length_ratio_vs_reference(candidate: str, reference: str) -> float:
    cand_len = len(tokenize(candidate))
    ref_len = len(tokenize(reference))
    if ref_len == 0:
        return 0.0
    return cand_len / ref_len


def evaluate_text(text: str, vocab: str, reference_text: str, output_type: str) -> dict:
    tokens = tokenize(text)
    all_bigrams = get_ngrams(tokens, 2)
    all_trigrams = get_ngrams(tokens, 3)

    return {
        "output_type": output_type,
        "word_count": len(tokens),
        "line_count": count_lines(text),
        "section_count": count_sections(text),
        "unique_unigrams": len(set(tokens)),
        "unique_bigrams": len(set(all_bigrams)),
        "unique_trigrams": len(set(all_trigrams)),
        "distinct_1": distinct_n(tokens, 1),
        "distinct_2": distinct_n(tokens, 2),
        "type_token_ratio": type_token_ratio(tokens),
        "repeated_line_ratio": repeated_line_ratio(text),
        "keyword_coverage": keyword_coverage(text, vocab),
        "top10_keyword_coverage": top_keyword_coverage(text, vocab, top_k=10),
        "keyword_repetition_ratio": keyword_repetition_ratio(text, vocab),
        "reference_jaccard": jaccard_similarity(text, reference_text),
        "reference_unigram_precision": unigram_precision(text, reference_text),
        "reference_unigram_recall": unigram_recall(text, reference_text),
        "reference_bigram_overlap": bigram_overlap(text, reference_text),
        "length_ratio_vs_reference": length_ratio_vs_reference(text, reference_text),
        "structure_score_extension": structure_score_extension(text) if output_type == "extension" else None,
        "structure_order_match": structure_order_match(text) if output_type == "extension" else None,
    }


def summarize_metrics(per_song_df: pd.DataFrame, output_type: str) -> dict:
    sub = per_song_df[per_song_df["output_type"] == output_type].copy()

    numeric_cols = [
        "word_count",
        "line_count",
        "section_count",
        "unique_unigrams",
        "unique_bigrams",
        "unique_trigrams",
        "distinct_1",
        "distinct_2",
        "type_token_ratio",
        "repeated_line_ratio",
        "keyword_coverage",
        "top10_keyword_coverage",
        "keyword_repetition_ratio",
        "reference_jaccard",
        "reference_unigram_precision",
        "reference_unigram_recall",
        "reference_bigram_overlap",
        "length_ratio_vs_reference",
    ]

    summary = {
        "output_type": output_type,
        "num_songs": len(sub),
    }

    for col in numeric_cols:
        summary[f"avg_{col}"] = sub[col].mean() if len(sub) else 0.0

    if output_type == "extension":
        summary["avg_structure_score_extension"] = sub["structure_score_extension"].mean() if len(sub) else 0.0
        summary["structure_order_match_rate"] = sub["structure_order_match"].mean() if len(sub) else 0.0

    return summary


def main():
    df = pd.read_csv(INPUT_FILE).copy()

    required_columns = [
        "song_id",
        "title",
        "artist",
        "genre",
        "valence",
        "arousal",
        "mood_label",
        "bow_keywords",
        "reference_lyrics",
        "reproduction_output",
        "extension_output",
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    per_song_rows = []

    for _, row in df.iterrows():
        base_info = {
            "song_id": row.get("song_id", ""),
            "title": row["title"],
            "artist": row["artist"],
            "genre": row.get("genre", ""),
            "valence": row.get("valence", ""),
            "arousal": row.get("arousal", ""),
            "mood_label": row.get("mood_label", ""),
            "bow_keywords": row.get("bow_keywords", ""),
            "reference_lyrics": row.get("reference_lyrics", ""),
            "tags": row.get("tags", ""),
            "release": row.get("release", ""),
        }

        repro_metrics = evaluate_text(
            text=row.get("reproduction_output", ""),
            vocab=row.get("bow_keywords", ""),
            reference_text=row.get("reference_lyrics", ""),
            output_type="reproduction",
        )

        ext_metrics = evaluate_text(
            text=row.get("extension_output", ""),
            vocab=row.get("bow_keywords", ""),
            reference_text=row.get("reference_lyrics", ""),
            output_type="extension",
        )

        per_song_rows.append({**base_info, **repro_metrics})
        per_song_rows.append({**base_info, **ext_metrics})

    per_song_df = pd.DataFrame(per_song_rows)

    summaries = [
        summarize_metrics(per_song_df, "reproduction"),
        summarize_metrics(per_song_df, "extension"),
    ]
    summary_df = pd.DataFrame(summaries)

    SUMMARY_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    per_song_df.to_csv(PER_SONG_OUTPUT_FILE, index=False)
    summary_df.to_csv(SUMMARY_OUTPUT_FILE, index=False)

    print(f"Saved per-song evaluation to: {PER_SONG_OUTPUT_FILE}")
    print(f"Saved evaluation summary to: {SUMMARY_OUTPUT_FILE}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()