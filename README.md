# Lyrics Reconstruction Dissertation

This repository contains the experimental pipeline for a dissertation project on lyric generation and reconstruction with large language models. The project uses song metadata, affective features, and vocabulary extracted from reference lyrics in the Music4All dataset to build prompts, generate new lyrics, and evaluate the results quantitatively.

At a high level, the workflow is:

1. Build a cleaned song sample from a source dataset.
2. Derive prompt inputs such as title, artist, genre, valence, arousal, and lyric vocabulary.
3. Construct multiple prompt variants for lyric generation.
4. Generate lyrics with an LLM.
5. Evaluate the generated outputs for structure, vocabulary usage, diversity, and similarity to the reference lyrics.

## Research Idea

The repo is set up around the question:

How well can an LLM generate lyrics that are consistent with a target song's metadata and emotional profile when guided by vocabulary extracted from the original lyrics?

The current implementation compares two prompt styles:

- `reproduction_prompt`: a lighter prompt focused on genre, artist style, mood, title, and vocabulary
- `extension_prompt`: a more constrained prompt that adds structural requirements such as `Verse 1`, `Chorus`, `Verse 2`, `Chorus`

## Data Source

The active pipeline in this repository uses only the `Music4All` dataset.

### Music4All

Raw Music4All files live under `data/raw/music4all/` and include:

- `id_information.csv`: song id, artist, song title, album name
- `id_metadata.csv`: Spotify-derived metadata such as `valence`, `energy`, `tempo`, `release`
- `id_tags.csv`: tag strings
- `id_genres.csv`: genre strings
- `id_lang.csv`: lyric language
- `lyrics/`: one lyric text file per song id

In the current scripts:

- `valence` is used directly from metadata
- `energy` is used as a proxy for arousal
- lyrics are tokenized to create bag-of-words vocabulary fields

## Repository Structure

```text
lyrics-reconstruction-dissertation/
├── data/
│   ├── raw/
│   │   └── music4all/
│   ├── samples/
│   └── processed/
├── outputs/
│   ├── evaluation/
│   ├── generated/
│   └── prompts/
├── src/
│   ├── prepare_input_dataset_music4all.py
│   ├── build_prompt_dataset.py
│   ├── generate_lyrics.py
│   └── evaluate_outputs.py
└── README.md
```

## Pipeline

### 1. Prepare an Input Dataset

[src/prepare_input_dataset_music4all.py](src/prepare_input_dataset_music4all.py) prepares the input dataset from Music4All.

It:

- reads and merges Music4All metadata tables
- load reference lyrics by `song_id`
- clean text and standardize columns
- build vocabulary summaries from lyrics
- retain song metadata needed for prompting
- export a sample CSV under `data/samples/`

The current output file is typically:

- `data/samples/music4all_sample_with_prompt_inputs.csv`

The preprocessing includes:

- language filtering, usually English only
- lyric tokenization
- stopword removal for keyword extraction
- repeated-line reduction before counting words
- random sampling using a fixed seed

### 2. Build the Prompt Dataset

[src/build_prompt_dataset.py](src/build_prompt_dataset.py) transforms the cleaned sample into a prompt-ready dataset.

It:

- validates required fields such as `title`, `artist`, `genre`, `valence`, and `arousal`
- converts valence/arousal into a polar angle `theta`
- maps that angle into a coarse mood label
- creates `reproduction_prompt`
- creates `extension_prompt`
- writes the final prompt dataset to `data/processed/prompt_dataset.csv`

The mood labels used are the 12 LyCon-inspired theta sectors:

- `pleased`
- `happy`
- `excited`
- `annoying`
- `angry`
- `nervous`
- `sad`
- `bored`
- `sleepy`
- `calm`
- `peaceful`
- `relaxed`

### 3. Generate Lyrics

[src/generate_lyrics.py](src/generate_lyrics.py) performs API-based generation using the OpenAI Python SDK.

The generation stage:

- reads `data/processed/prompt_dataset.csv`
- generates one output for each prompt type
- stores prompts, metadata, and outputs together
- writes results under `outputs/generated/`

The two output columns are:

- `reproduction_output`
- `extension_output`

### 4. Evaluate Outputs

[src/evaluate_outputs.py](src/evaluate_outputs.py) scores generated lyrics and writes evaluation tables.

It measures:

- word count and line count
- number of detected sections
- lexical diversity such as `distinct_1`, `distinct_2`, and type-token ratio
- repeated line ratio
- keyword coverage against the supplied vocabulary
- overlap with reference lyrics through unigram, bigram, and Jaccard-style metrics
- extension prompt structural compliance

Outputs are written to:

- `outputs/evaluation/per_song_evaluation.csv`
- `outputs/evaluation/evaluation_summary.csv`

## Expected Data Flow

The intended end-to-end flow is:

```text
Music4All raw files
  -> prepare_input_dataset_music4all.py
  -> data/samples/music4all_sample_with_prompt_inputs.csv
  -> build_prompt_dataset.py
  -> data/processed/prompt_dataset.csv
  -> generate_lyrics.py
  -> outputs/generated/*.csv
  -> evaluate_outputs.py
  -> outputs/evaluation/*.csv
```

## Setup

### Python

The repo does not currently define a populated `requirements.txt`, but the scripts depend on at least:

- `pandas`
- `python-dotenv`
- `openai`
- `bert-score` for semantic evaluation with BERTScore

A simple setup flow is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas python-dotenv openai bert-score
```

If you plan to use API generation, also set:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

The evaluation stage can also compute BERTScore against the reference lyrics. In `evaluation` config, this is controlled by:

- `enable_bertscore`
- `bertscore_lang`
- `bertscore_model_type`
- `bertscore_batch_size`

## Running the Project

### Build a sample dataset

```bash
python src/prepare_input_dataset_music4all.py
```

### Build the prompt dataset

```bash
python src/build_prompt_dataset.py
```

### Generate with the OpenAI API

```bash
python src/generate_lyrics.py
```

### Evaluate results

```bash
python src/evaluate_outputs.py
```

## Files of Interest

- [src/prepare_input_dataset_music4all.py](src/prepare_input_dataset_music4all.py): Music4All preprocessing pipeline
- [src/build_prompt_dataset.py](src/build_prompt_dataset.py): mood mapping and prompt construction
- [src/generate_lyrics.py](src/generate_lyrics.py): OpenAI generation runner
- [src/evaluate_outputs.py](src/evaluate_outputs.py): automatic evaluation metrics

## Current Outputs in the Repo

The repository already includes:

- sample input CSVs under `data/samples/`
- a processed prompt dataset under `data/processed/`

This makes it possible to inspect the intended experiment structure even before re-running the pipeline.

## Known Issues and Caveats

The repo is promising, but it is still a working dissertation codebase rather than a fully polished package. A few points are worth knowing before running everything end to end:

- `README.md` was previously empty, so the code has been the main source of truth.
- `requirements.txt` is currently empty and should eventually be filled in.
- There is some schema drift between scripts. For example, the sample dataset on disk uses fields such as `bow_keywords_base` and `bow_keywords_with_freq`, while `build_prompt_dataset.py` prefers columns like `bow_all_words`, `bow_keywords_words`, or `bow_keywords`.
- `src/generate_lyrics.py` currently defines `OUTPUT_FILE` using `MODEL_NAME` before `MODEL_NAME` is assigned, which will need a small fix before real API generation runs cleanly.
- `src/tempCodeRunnerFile.py` looks like an editor artifact rather than part of the core project.

## Dissertation Framing

From the current code and outputs, the repo appears to support a dissertation chapter or experiment comparing prompt formulations for lyric generation under controlled song conditions within Music4All. The core variables seem to be:

- source song metadata
- emotional conditioning via valence and arousal
- vocabulary constraints derived from reference lyrics
- prompt structure
- output evaluation metrics

In that sense, this is not just a lyric generator. It is an experimental framework for testing reconstruction-style prompting strategies.
