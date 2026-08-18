from __future__ import annotations

import argparse

from build_prompt_dataset import build_prompt_dataset
from evaluate_outputs import evaluate_outputs
from experiment_utils import get_run_id, load_config, resolve_path
from generate_lyrics import generate_outputs
from prepare_input_dataset_music4all import build_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full experiment pipeline.")
    parser.add_argument("--config", default=None, help="Path to experiment JSON config.")
    parser.add_argument("--run-id", default=None, help="Reusable run identifier for saving outputs.")
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Build datasets and prompts but skip model generation and downstream evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_id = get_run_id(config, args.run_id)

    df = build_dataframe(config)
    dataset_output = resolve_path(config["dataset"]["output_file"])
    dataset_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dataset_output, index=False)
    print(f"Prepared dataset rows: {len(df)} -> {dataset_output}")

    _, prompt_dataset_path = build_prompt_dataset(config, run_id)
    print(f"Prompt dataset ready at: {prompt_dataset_path}")

    if args.skip_generation:
        print(f"Skipping generation and evaluation for run: {run_id}")
        return

    _, generation_output = generate_outputs(config, run_id)
    print(f"Generated outputs saved to: {generation_output}")

    _, summary_df, _, summary_path = evaluate_outputs(config, run_id)
    print(f"Evaluation summary saved to: {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
