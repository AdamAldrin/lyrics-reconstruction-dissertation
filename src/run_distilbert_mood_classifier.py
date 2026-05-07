from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from experiment_utils import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for 12-mood lyric classification.")
    parser.add_argument("--data-dir", default="data/processed/mood_classifier")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--progress-steps", type=int, default=250)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    return parser.parse_args()


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LyricsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_to_id: dict[str, int]) -> None:
        self.texts = df["lyrics"].fillna("").astype(str).tolist()
        self.labels = df["mood_label"].map(label_to_id).astype(int).tolist()
        self.song_ids = df["song_id"].astype(str).tolist()
        self.titles = df["title"].fillna("").astype(str).tolist()
        self.artists = df["artist"].fillna("").astype(str).tolist()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict:
        return {
            "text": self.texts[index],
            "label": self.labels[index],
            "song_id": self.song_ids[index],
            "title": self.titles[index],
            "artist": self.artists[index],
        }


def make_collate_fn(tokenizer, max_length: int):
    def collate(batch: list[dict]) -> dict:
        encoded = tokenizer(
            [item["text"] for item in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        encoded["song_id"] = [item["song_id"] for item in batch]
        encoded["title"] = [item["title"] for item in batch]
        encoded["artist"] = [item["artist"] for item in batch]
        return encoded

    return collate


def load_split(data_dir: Path, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_dir = data_dir / "splits"
    train_path = split_dir / f"seed_{seed}_train.csv"
    test_path = split_dir / f"seed_{seed}_test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing split files for seed {seed}: {train_path}, {test_path}")
    return pd.read_csv(train_path), pd.read_csv(test_path)


def limit_rows(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or max_rows >= len(df):
        return df
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def class_weight_tensor(train_df: pd.DataFrame, labels: list[str], device: torch.device) -> torch.Tensor:
    counts = train_df["mood_label"].value_counts().reindex(labels).to_numpy(dtype=float)
    weights = counts.sum() / (len(labels) * counts)
    return torch.tensor(weights, dtype=torch.float, device=device)


def train_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    progress_steps: int,
) -> float:
    model.train()
    losses = []
    for step, batch in enumerate(loader, start=1):
        metadata_keys = {"song_id", "title", "artist"}
        model_batch = {key: value.to(device) for key, value in batch.items() if key not in metadata_keys}
        outputs = model(**model_batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        if progress_steps and step % progress_steps == 0:
            print(f"  step {step}/{len(loader)} loss={np.mean(losses[-progress_steps:]):.4f}", flush=True)
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device, id_to_label: dict[int, str]) -> tuple[dict, pd.DataFrame]:
    model.eval()
    y_true = []
    y_pred = []
    rows = []

    for batch in loader:
        metadata_keys = {"song_id", "title", "artist"}
        model_batch = {key: value.to(device) for key, value in batch.items() if key not in metadata_keys}
        outputs = model(**model_batch)
        predictions = outputs.logits.argmax(dim=-1).detach().cpu().tolist()
        labels = model_batch["labels"].detach().cpu().tolist()

        y_true.extend(labels)
        y_pred.extend(predictions)
        for song_id, title, artist, actual_id, predicted_id in zip(
            batch["song_id"],
            batch["title"],
            batch["artist"],
            labels,
            predictions,
        ):
            rows.append(
                {
                    "song_id": song_id,
                    "title": title,
                    "artist": artist,
                    "mood_label": id_to_label[actual_id],
                    "predicted_mood": id_to_label[predicted_id],
                    "correct": actual_id == predicted_id,
                }
            )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }
    return metrics, pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = resolve_path(args.data_dir)
    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "outputs" / "mood_classifier_experiments" / f"12_mood_distilbert_seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_split(data_dir, args.seed)
    train_df = limit_rows(train_df, args.max_train_samples, args.seed)
    test_df = limit_rows(test_df, args.max_test_samples, args.seed)

    labels = sorted(pd.concat([train_df["mood_label"], test_df["mood_label"]]).unique())
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    device = get_device()

    print(f"Device: {device}", flush=True)
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)} | Labels: {len(labels)}", flush=True)
    print(f"Loading model: {args.model_name}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    if not args.no_class_weights:
        weights = class_weight_tensor(train_df, labels, device)
        model.config.problem_type = "single_label_classification"
        original_forward = model.forward

        def weighted_forward(**kwargs):
            labels_tensor = kwargs.pop("labels", None)
            outputs = original_forward(**kwargs)
            if labels_tensor is not None:
                loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
                outputs.loss = loss_fn(outputs.logits.view(-1, len(labels)), labels_tensor.view(-1))
            return outputs

        model.forward = weighted_forward

    model.to(device)

    collate_fn = make_collate_fn(tokenizer, args.max_length)
    train_loader = DataLoader(
        LyricsDataset(train_df, label_to_id),
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        LyricsDataset(test_df, label_to_id),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history = []
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}", flush=True)
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, args.progress_steps)
        metrics, _ = evaluate(model, test_loader, device, id_to_label)
        row = {"epoch": epoch, "train_loss": train_loss, **metrics}
        history.append(row)
        print(
            f"  eval accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
            f"weighted_f1={metrics['weighted_f1']:.4f}",
            flush=True,
        )

    final_metrics, predictions_df = evaluate(model, test_loader, device, id_to_label)
    actual = predictions_df["mood_label"]
    predicted = predictions_df["predicted_mood"]
    report_df = pd.DataFrame(
        classification_report(actual, predicted, output_dict=True, zero_division=0)
    ).transpose().reset_index().rename(columns={"index": "label"})
    confusion_df = pd.DataFrame(confusion_matrix(actual, predicted, labels=labels), index=labels, columns=labels)
    confusion_df.index.name = "actual_mood"

    metrics_row = {
        "seed": args.seed,
        "model_name": args.model_name,
        "num_train": len(train_df),
        "num_test": len(test_df),
        "num_classes": len(labels),
        "epochs": args.epochs,
        "max_length": args.max_length,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "class_weights": not args.no_class_weights,
        **final_metrics,
    }

    pd.DataFrame([metrics_row]).to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    report_df.to_csv(output_dir / "classification_report.csv", index=False)
    confusion_df.to_csv(output_dir / "confusion_matrix.csv")
    predictions_df.to_csv(output_dir / "test_predictions.csv", index=False)
    with (output_dir / "label_mapping.json").open("w", encoding="utf-8") as handle:
        json.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, handle, indent=2)

    if args.save_model:
        model.save_pretrained(output_dir / "model")
        tokenizer.save_pretrained(output_dir / "model")

    print(f"\nSaved DistilBERT mood classifier outputs to: {output_dir}", flush=True)
    print(pd.DataFrame([metrics_row]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
