import os
import sys
import csv
import random
import argparse
from typing import Dict, List, Sequence

import torch  # type: ignore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils.utils as utils  
import utils.datasets as datasets  


class EvalSample:
    """Container for one test sample and its baseline prediction."""
    def __init__(self, sample_index: int, image: torch.Tensor, true_label: int, baseline_pred: int) -> None:
        self.sample_index = sample_index
        self.image = image
        self.true_label = true_label
        self.baseline_pred = baseline_pred


def get_dataset_label(dataset: torch.utils.data.Dataset, idx: int) -> int:
    """Read class label for an item, handling common dataset formats."""
    
    # helper
    def unwrap_dataset_index(dataset: torch.utils.data.Dataset, idx: int) -> tuple[torch.utils.data.Dataset, int]:
        """Resolve nested Subset indices down to the underlying base dataset."""
        if isinstance(dataset, torch.utils.data.Subset):
            return unwrap_dataset_index(dataset.dataset, dataset.indices[idx])
        return dataset, idx

    base_dataset, base_idx = unwrap_dataset_index(dataset, idx)
    if hasattr(base_dataset, "targets"):
        return int(base_dataset.targets[base_idx])
    if hasattr(base_dataset, "labels"):
        return int(base_dataset.labels[base_idx])
    _, label = dataset[idx]
    return int(label)


def fetch_memory_samples(dataset: torch.utils.data.Dataset, dataset_indices: Sequence[int]) -> torch.Tensor:
    """Fetch and stack memory images for the provided dataset indices."""
    images = [dataset[idx][0] for idx in dataset_indices]
    return torch.stack(images, dim=0)


def build_class_index_map(memory_dataset: torch.utils.data.Dataset) -> Dict[int, List[int]]:
    """Build lookup map: class label -> list of dataset indices."""
    class_to_indices: Dict[int, List[int]] = {}
    for idx in range(len(memory_dataset)):
        label = get_dataset_label(memory_dataset, idx)
        class_to_indices.setdefault(label, []).append(idx)
    return class_to_indices


def collect_eval_samples(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, baseline_memory: torch.Tensor, device: torch.device, max_samples: int) -> List[EvalSample]:
    """Collect up to `max_samples` from test split with baseline predictions."""
    collected: List[EvalSample] = []
    sample_index = 0

    with torch.no_grad():
        for images, labels in test_loader:
            for i in range(images.shape[0]):
                img = images[i].unsqueeze(0).to(device)
                label = int(labels[i].item())
                # One fixed memory batch defines the baseline label for later strategy comparisons.
                outputs, _ = model(img, baseline_memory, return_weights=True)
                baseline_pred = int(outputs.argmax(dim=1).item())

                collected.append(
                    EvalSample(
                        sample_index=sample_index,
                        image=images[i].detach().cpu(),
                        true_label=label,
                        baseline_pred=baseline_pred,
                    )
                )
                sample_index += 1
                if len(collected) >= max_samples:
                    return collected
    return collected


def build_memory_batch(strategy: str, sample: EvalSample, baseline_memory: torch.Tensor, class_to_dataset_indices: Dict[int, List[int]], memory_dataset: torch.utils.data.Dataset, memory_size: int, num_classes: int, rng: random.Random) -> torch.Tensor:
    """Construct memory tensor according to one of the 4 target strategies."""
    if strategy == "baseline_random":
        return baseline_memory.clone()
    if strategy == "same_class":
        candidates = class_to_dataset_indices[sample.true_label]
        chosen = [rng.choice(candidates) for _ in range(memory_size)]
        return fetch_memory_samples(memory_dataset, chosen)
    if strategy == "different_class":
        candidates = [lbl for lbl in range(num_classes) if lbl != sample.true_label]
        selected_label = rng.choice(candidates)
        candidates = class_to_dataset_indices[selected_label]
        chosen = [rng.choice(candidates) for _ in range(memory_size)]
        return fetch_memory_samples(memory_dataset, chosen)
    if strategy == "single_repeated":
        candidates = class_to_dataset_indices[sample.true_label]
        selected_idx = rng.choice(candidates)
        chosen = [selected_idx] * memory_size
        return fetch_memory_samples(memory_dataset, chosen)
    raise ValueError(f"Unknown strategy: {strategy}")


def compute_strategy_metrics(predictions: Dict[str, List[int]], true_labels: List[int]) -> List[Dict[str, float | str]]:
    """Compute aggregate metrics for each strategy."""
    strategies = list(predictions.keys())
    baseline_name = "baseline_random"
    baseline_preds = predictions[baseline_name]
    total = len(true_labels)

    metrics: List[Dict[str, float | str]] = []
    for strategy in strategies:
        preds = predictions[strategy]
        correct = sum(int(pred == target) for pred, target in zip(preds, true_labels))
        # changed / corrected / corrupted are all relative to baseline_random on the same samples.
        changed = sum(int(pred != base) for pred, base in zip(preds, baseline_preds))
        corrected = sum(
            int(base != target and pred == target)
            for pred, base, target in zip(preds, baseline_preds, true_labels)
        )
        corrupted = sum(
            int(base == target and pred != target)
            for pred, base, target in zip(preds, baseline_preds, true_labels)
        )

        metrics.append(
            {
                "strategy": strategy,
                "accuracy": 100.0 * correct / total,
                "percent_changed": 100.0 * changed / total,
                "percent_corrected": 100.0 * corrected / total,
                "percent_corrupted": 100.0 * corrupted / total,
            }
        )
    return metrics


def print_summary_table(rows: List[Dict[str, float | str]]) -> None:
    """Print a compact summary table in the console."""
    # Fixed widths keep numeric columns aligned for quick visual scan.
    headers = [
        "strategy",
        "accuracy",
        "percent_changed",
        "percent_corrected",
        "percent_corrupted",
    ]
    col_widths = {
        "strategy": 18,
        "accuracy": 10,
        "percent_changed": 16,
        "percent_corrected": 18,
        "percent_corrupted": 18,
    }

    header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    separator = "-+-".join("-" * col_widths[h] for h in headers)
    print(header_line)
    print(separator)
    for row in rows:
        print(
            " | ".join(
                [
                    str(row["strategy"]).ljust(col_widths["strategy"]),
                    f"{row['accuracy']:.2f}".rjust(col_widths["accuracy"]),
                    f"{row['percent_changed']:.2f}".rjust(col_widths["percent_changed"]),
                    f"{row['percent_corrected']:.2f}".rjust(col_widths["percent_corrected"]),
                    f"{row['percent_corrupted']:.2f}".rjust(col_widths["percent_corrupted"]),
                ]
            )
        )


def run_evaluation(args: argparse.Namespace) -> None:
    """Execute memory strategy evaluation for one checkpoint."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.path, map_location=device)
    model = utils.get_model(checkpoint["model_name"], checkpoint["num_classes"], model_type=checkpoint["modality"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    if args.dataset != checkpoint["dataset_name"]:
        raise ValueError(f"Dataset mismatch. checkpoint dataset={checkpoint['dataset_name']}, --dataset={args.dataset}")
    load_dataset = getattr(datasets, f"get_{args.dataset}")
    _, _, test_loader, mem_loader = load_dataset(
        args.dir_dataset,
        batch_size_train=128,
        batch_size_test=args.batch_size_test,
        batch_size_memory=checkpoint["mem_examples"],
        size_train=checkpoint["train_examples"],
    )

    # Same default memory batch as training-style loaders; baseline_random clones this tensor.
    baseline_memory, _ = next(iter(mem_loader))
    baseline_memory = baseline_memory.to(device)

    samples = collect_eval_samples(
        model=model,
        test_loader=test_loader,
        baseline_memory=baseline_memory,
        device=device,
        max_samples=args.num_samples,
    )
    if not samples:
        raise RuntimeError("No test samples were collected.")

    memory_dataset = mem_loader.dataset
    class_to_indices = build_class_index_map(memory_dataset)
    for class_id, idxs in class_to_indices.items():
        if not idxs:
            raise RuntimeError(f"No memory samples available for class {class_id}.")

    strategies = ["baseline_random", "same_class", "different_class", "single_repeated"]
    rng = random.Random(args.seed)
    memory_size = int(checkpoint["mem_examples"])
    num_classes = int(checkpoint["num_classes"])

    true_labels = [sample.true_label for sample in samples]
    predictions: Dict[str, List[int]] = {name: [] for name in strategies}

    with torch.no_grad():
        for sample in samples:
            img_batch = sample.image.unsqueeze(0).to(device)
            for strategy in strategies:
                memory_batch = build_memory_batch(
                    strategy=strategy,
                    sample=sample,
                    baseline_memory=baseline_memory,
                    class_to_dataset_indices=class_to_indices,
                    memory_dataset=memory_dataset,
                    memory_size=memory_size,
                    num_classes=num_classes,
                    rng=rng,
                ).to(device)
                outputs, _ = model(img_batch, memory_batch, return_weights=True)
                pred = int(outputs.argmax(dim=1).item())
                predictions[strategy].append(pred)

    metrics = compute_strategy_metrics(predictions=predictions, true_labels=true_labels)

    checkpoint_name = os.path.splitext(os.path.basename(args.path))[0]
    out_root = os.path.join(args.output_dir, args.dataset, checkpoint["model_name"])
    os.makedirs(out_root, exist_ok=True)
    csv_path = os.path.join(out_root, f"{checkpoint_name}_memory_strategy_metrics.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "strategy",
                "accuracy",
                "percent_changed",
                "percent_corrected",
                "percent_corrupted",
            ],
        )
        writer.writeheader()
        writer.writerows(metrics)

    print(f"Evaluated {len(samples)} test samples.")
    print(f"Saved metrics CSV: {csv_path}")
    print_summary_table(metrics)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate memory strategies for one checkpoint.")
    parser.add_argument("--path", required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--dataset", required=True, help="Dataset name: SVHN, CIFAR10, CINIC10")
    parser.add_argument("--dir_dataset", default="datasets/", help="Dataset directory")
    parser.add_argument("--output_dir", default="paper/results/memory_strategy_eval", help="Output root dir")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of test samples to evaluate")
    parser.add_argument("--batch_size_test", type=int, default=64, help="Test batch size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for memory sampling")
    return parser.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())
