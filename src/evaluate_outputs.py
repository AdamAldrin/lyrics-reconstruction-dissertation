from __future__ import annotations

import argparse
import re
from collections import Counter
from math import isnan
from pathlib import Path

import pandas as pd

from experiment_utils import clean_text, get_prompt_variants, get_run_dir, get_run_id, load_config


SECTION_HEADERS = {
    "verse",
    "chorus",
    "bridge",
    "intro",
    "outro",
    "pre-chorus",
    "hook",
    "refrain",
}
REQUIRED_EXTENSION_ORDER = ["verse 1", "chorus", "verse 2", "chorus"]


def tokenize(text: str) -> list[str]:
    if pd.isna(text) or not str(text).strip():
        return []
    return re.findall(r"\b[a-zA-Z]+\b", str(text).lower())


def split_nonempty_lines(text: str) -> list[str]:
    if pd.isna(text) or not str(text).strip():
        return []
    return [line.strip() for line in str(text).splitlines() if line.strip()]


def count_lines(text: str) -> int:
    return len(split_nonempty_lines(text))


def normalize_section_line(line: str) -> str:
    return line.strip().lower().replace("[", "").replace("]", "").replace(":", "")


def extract_section_labels(text: str) -> list[str]:
    labels = []
    for line in split_nonempty_lines(text):
        clean = normalize_section_line(line)
        if clean in SECTION_HEADERS or any(clean.startswith(header) for header in SECTION_HEADERS):
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


def parse_vocab_words(vocab: str) -> list[str]:
    text = clean_text(vocab)
    if not text:
        return []

    words = []
    for part in text.split():
        if ":" in part:
            words.append(part.split(":", 1)[0].strip().lower())
        else:
            words.extend(tokenize(part))
    return [word for word in words if word]


def parse_vocab_freq_map(vocab: str) -> dict[str, int]:
    text = clean_text(vocab)
    if not text:
        return {}

    freq_map: dict[str, int] = {}
    for part in text.split():
        if ":" not in part:
            continue
        word, count = part.split(":", 1)
        word = word.strip().lower()
        try:
            freq_map[word] = int(count)
        except ValueError:
            continue
    return freq_map


def keyword_coverage(text: str, vocab: str) -> float:
    text_vocab = set(tokenize(text))
    target_vocab = set(parse_vocab_words(vocab))
    if not target_vocab:
        return 0.0
    return len(text_vocab & target_vocab) / len(target_vocab)


def top_keyword_coverage(text: str, vocab: str, top_k: int) -> float:
    text_vocab = set(tokenize(text))
    target_vocab = parse_vocab_words(vocab)[:top_k]
    if not target_vocab:
        return 0.0
    return len(set(target_vocab) & text_vocab) / len(set(target_vocab))


def keyword_repetition_ratio(text: str, vocab: str) -> float:
    tokens = tokenize(text)
    vocab_set = set(parse_vocab_words(vocab))
    if not tokens or not vocab_set:
        return 0.0

    keyword_tokens = [token for token in tokens if token in vocab_set]
    if not keyword_tokens:
        return 0.0

    counts = Counter(keyword_tokens)
    repeated = sum(count for _, count in counts.items() if count > 1)
    return repeated / len(keyword_tokens)


def keyword_density(text: str, vocab: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    vocab_set = set(parse_vocab_words(vocab))
    return sum(1 for token in tokens if token in vocab_set) / len(tokens)


def weighted_keyword_coverage(text: str, vocab_freq: str, fallback_vocab: str) -> float:
    freq_map = parse_vocab_freq_map(vocab_freq)
    if not freq_map:
        return keyword_coverage(text, fallback_vocab)

    text_vocab = set(tokenize(text))
    total_weight = sum(freq_map.values())
    if total_weight == 0:
        return 0.0

    matched_weight = sum(weight for word, weight in freq_map.items() if word in text_vocab)
    return matched_weight / total_weight


def weighted_top_keyword_coverage(text: str, vocab_freq: str, top_k: int) -> float:
    freq_map = parse_vocab_freq_map(vocab_freq)
    if not freq_map:
        return 0.0

    ranked_items = sorted(freq_map.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    total_weight = sum(weight for _, weight in ranked_items)
    if total_weight == 0:
        return 0.0

    text_vocab = set(tokenize(text))
    matched_weight = sum(weight for word, weight in ranked_items if word in text_vocab)
    return matched_weight / total_weight


def rank_order_correlation(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return 0.0

    def rank(values: list[float]) -> list[float]:
        indexed = sorted(enumerate(values), key=lambda item: item[1])
        ranks = [0.0] * len(values)
        idx = 0
        while idx < len(indexed):
            end = idx
            while end + 1 < len(indexed) and indexed[end + 1][1] == indexed[idx][1]:
                end += 1
            avg_rank = (idx + end + 2) / 2.0
            for pos in range(idx, end + 1):
                ranks[indexed[pos][0]] = avg_rank
            idx = end + 1
        return ranks

    rank_a = rank(values_a)
    rank_b = rank(values_b)
    mean_a = sum(rank_a) / len(rank_a)
    mean_b = sum(rank_b) / len(rank_b)
    cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(rank_a, rank_b))
    var_a = sum((a - mean_a) ** 2 for a in rank_a)
    var_b = sum((b - mean_b) ** 2 for b in rank_b)
    if var_a == 0 or var_b == 0:
        return 0.0
    return cov / ((var_a ** 0.5) * (var_b ** 0.5))


def keyword_frequency_rank_correlation(text: str, vocab_freq: str, top_k: int) -> float:
    freq_map = parse_vocab_freq_map(vocab_freq)
    if not freq_map:
        return 0.0

    ranked_items = sorted(freq_map.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    if len(ranked_items) < 2:
        return 0.0

    generated_counts = Counter(tokenize(text))
    target_values = [weight for _, weight in ranked_items]
    generated_values = [generated_counts.get(word, 0) for word, _ in ranked_items]
    return rank_order_correlation(target_values, generated_values)


def structure_score(text: str) -> int:
    labels = [label.lower() for label in extract_section_labels(text)]
    score = 0
    for required in REQUIRED_EXTENSION_ORDER:
        if required in labels:
            score += 1
    return score


def structure_order_match(text: str) -> int:
    labels = [label.lower() for label in extract_section_labels(text)]
    try:
        i1 = labels.index("verse 1")
        i2 = labels.index("chorus", i1 + 1)
        i3 = labels.index("verse 2", i2 + 1)
        i4 = labels.index("chorus", i3 + 1)
        return int(i1 < i2 < i3 < i4)
    except ValueError:
        return 0


def jaccard_similarity(text_a: str, text_b: str) -> float:
    a = set(tokenize(text_a))
    b = set(tokenize(text_b))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def unigram_precision(candidate: str, reference: str) -> float:
    candidate_tokens = tokenize(candidate)
    reference_vocab = set(tokenize(reference))
    if not candidate_tokens:
        return 0.0
    return sum(1 for token in candidate_tokens if token in reference_vocab) / len(candidate_tokens)


def unigram_recall(candidate: str, reference: str) -> float:
    candidate_vocab = set(tokenize(candidate))
    reference_vocab = set(tokenize(reference))
    if not reference_vocab:
        return 0.0
    return len(candidate_vocab & reference_vocab) / len(reference_vocab)


def bigram_overlap(candidate: str, reference: str) -> float:
    candidate_bigrams = set(get_ngrams(tokenize(candidate), 2))
    reference_bigrams = set(get_ngrams(tokenize(reference), 2))
    if not reference_bigrams:
        return 0.0
    return len(candidate_bigrams & reference_bigrams) / len(reference_bigrams)


def length_ratio_vs_reference(candidate: str, reference: str) -> float:
    candidate_len = len(tokenize(candidate))
    reference_len = len(tokenize(reference))
    if reference_len == 0:
        return 0.0
    return candidate_len / reference_len


def title_token_coverage(text: str, title: str) -> float:
    title_tokens = set(tokenize(title))
    if not title_tokens:
        return 0.0
    text_tokens = set(tokenize(text))
    return len(title_tokens & text_tokens) / len(title_tokens)


def evaluate_text(
    text: str,
    *,
    vocab: str,
    vocab_freq: str,
    reference_text: str,
    title: str,
    top_keyword_k: int,
    requires_structure: bool,
) -> dict:
    tokens = tokenize(text)
    bigrams = get_ngrams(tokens, 2)
    trigrams = get_ngrams(tokens, 3)
    metrics = {
        "word_count": len(tokens),
        "line_count": count_lines(text),
        "section_count": count_sections(text),
        "unique_unigrams": len(set(tokens)),
        "unique_bigrams": len(set(bigrams)),
        "unique_trigrams": len(set(trigrams)),
        "distinct_1": distinct_n(tokens, 1),
        "distinct_2": distinct_n(tokens, 2),
        "type_token_ratio": type_token_ratio(tokens),
        "repeated_line_ratio": repeated_line_ratio(text),
        "keyword_coverage": keyword_coverage(text, vocab),
        "top_keyword_coverage": top_keyword_coverage(text, vocab, top_keyword_k),
        "weighted_keyword_coverage": weighted_keyword_coverage(text, vocab_freq, vocab),
        "weighted_top_keyword_coverage": weighted_top_keyword_coverage(text, vocab_freq, top_keyword_k),
        "keyword_frequency_rank_correlation": keyword_frequency_rank_correlation(text, vocab_freq, top_keyword_k),
        "keyword_repetition_ratio": keyword_repetition_ratio(text, vocab),
        "keyword_density": keyword_density(text, vocab),
        "title_token_coverage": title_token_coverage(text, title),
        "reference_jaccard": jaccard_similarity(text, reference_text),
        "reference_unigram_precision": unigram_precision(text, reference_text),
        "reference_unigram_recall": unigram_recall(text, reference_text),
        "reference_bigram_overlap": bigram_overlap(text, reference_text),
        "length_ratio_vs_reference": length_ratio_vs_reference(text, reference_text),
    }
    if requires_structure:
        metrics["structure_score"] = structure_score(text)
        metrics["structure_order_match"] = structure_order_match(text)
    else:
        metrics["structure_score"] = None
        metrics["structure_order_match"] = None
    return metrics


def summarize_metrics(per_song_df: pd.DataFrame) -> dict:
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
        "top_keyword_coverage",
        "weighted_keyword_coverage",
        "weighted_top_keyword_coverage",
        "keyword_frequency_rank_correlation",
        "keyword_repetition_ratio",
        "keyword_density",
        "title_token_coverage",
        "reference_jaccard",
        "reference_unigram_precision",
        "reference_unigram_recall",
        "reference_bigram_overlap",
        "length_ratio_vs_reference",
        "structure_score",
        "structure_order_match",
    ]
    summary = {
        "run_id": clean_text(per_song_df["run_id"].iloc[0]) if len(per_song_df) else "",
        "output_type": clean_text(per_song_df["output_type"].iloc[0]) if len(per_song_df) else "",
        "num_songs": len(per_song_df),
    }
    for column in numeric_cols:
        if column in per_song_df.columns:
            summary[f"avg_{column}"] = per_song_df[column].mean()
    return summary


def evaluate_outputs(config: dict, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    run_dir = get_run_dir(run_id)
    input_file = run_dir / "generated_outputs.csv"
    if not input_file.exists():
        raise FileNotFoundError(f"Generated outputs not found at {input_file}.")

    per_song_output_file = run_dir / "per_song_evaluation.csv"
    summary_output_file = run_dir / "evaluation_summary.csv"

    evaluation_config = config.get("evaluation", {})
    top_keyword_k = int(evaluation_config.get("top_keyword_k", 10))
    prompt_variants = get_prompt_variants(config)
    df = pd.read_csv(input_file).copy()

    per_song_rows = []
    for _, row in df.iterrows():
        base_info = {
            "run_id": clean_text(row.get("run_id", run_id)),
            "song_id": clean_text(row.get("song_id", "")),
            "title": clean_text(row.get("title", "")),
            "artist": clean_text(row.get("artist", "")),
            "genre": clean_text(row.get("genre", "")),
            "valence": row.get("valence", ""),
            "arousal": row.get("arousal", ""),
            "theta": row.get("theta", ""),
            "mood_label": clean_text(row.get("mood_label", "")),
            "reference_lyrics": clean_text(row.get("reference_lyrics", "")),
            "vocab_strategy": clean_text(row.get("vocab_strategy", "")),
            "vocab_words": clean_text(row.get("vocab_words", "")),
            "vocab_freq": clean_text(row.get("vocab_freq", "")),
            "model_name": clean_text(row.get("model_name", "")),
        }

        for variant in prompt_variants:
            name = clean_text(variant["name"])
            prompt_col = f"{name}_prompt"
            output_col = f"{name}_output"
            vocab_freq_column = clean_text(variant.get("vocab_freq_column", "")) or "vocab_freq"
            metrics = evaluate_text(
                clean_text(row.get(output_col, "")),
                vocab=clean_text(row.get(variant.get("vocab_words_column", "vocab_words"), "")),
                vocab_freq=clean_text(row.get(vocab_freq_column, "")),
                reference_text=clean_text(row.get("reference_lyrics", "")),
                title=clean_text(row.get("title", "")),
                top_keyword_k=top_keyword_k,
                requires_structure=bool(variant.get("requires_structure", False)),
            )
            per_song_rows.append(
                {
                    **base_info,
                    "output_type": name,
                    "prompt_version": Path(variant["template_file"]).stem,
                    "prompt_text": clean_text(row.get(prompt_col, "")),
                    "generated_text": clean_text(row.get(output_col, "")),
                    **metrics,
                }
            )

    per_song_df = pd.DataFrame(per_song_rows)
    summary_df = pd.DataFrame(
        [summarize_metrics(per_song_df[per_song_df["output_type"] == clean_text(variant["name"])]) for variant in prompt_variants]
    )

    per_song_output_file.parent.mkdir(parents=True, exist_ok=True)
    per_song_df.to_csv(per_song_output_file, index=False)
    summary_df.to_csv(summary_output_file, index=False)
    return per_song_df, summary_df, per_song_output_file, summary_output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated lyric outputs.")
    parser.add_argument("--config", default=None, help="Path to experiment JSON config.")
    parser.add_argument("--run-id", default=None, help="Reusable run identifier for saving outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_id = get_run_id(config, args.run_id)
    _, summary_df, per_song_output_file, summary_output_file = evaluate_outputs(config, run_id)

    print(f"Saved per-song evaluation to: {per_song_output_file}")
    print(f"Saved evaluation summary to: {summary_output_file}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
