import os
import sys
import csv
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence

import torch  # type: ignore
import torchvision  # type: ignore
import numpy as np
from PIL import Image  # type: ignore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils.utils as utils  # noqa: E402
import utils.datasets as datasets  # noqa: E402


@dataclass
class SelectedSample:
    """Container for a single selected test sample used in interventions."""

    sample_index: int
    image: torch.Tensor
    true_label: int
    baseline_pred: int


class MemoryStrategy:
    """Base interface for pluggable memory construction strategies."""

    name = "base"

    def build(
        self,
        sample: SelectedSample,
        baseline_memory: torch.Tensor,
        class_to_dataset_indices: Dict[int, List[int]],
        memory_dataset: torch.utils.data.Dataset,
        memory_size: int,
        num_classes: int,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Build a memory tensor and return associated dataset indices."""
        raise NotImplementedError


def unwrap_dataset_index(
    dataset: torch.utils.data.Dataset, idx: int
) -> Tuple[torch.utils.data.Dataset, int]:
    """Resolve nested Subset indices down to the underlying base dataset."""
    if isinstance(dataset, torch.utils.data.Subset):
        return unwrap_dataset_index(dataset.dataset, dataset.indices[idx])
    return dataset, idx


def get_dataset_label(dataset: torch.utils.data.Dataset, idx: int) -> int:
    """Read the class label for an item, handling different dataset formats."""
    base_dataset, base_idx = unwrap_dataset_index(dataset, idx)
    if hasattr(base_dataset, "targets"):
        return int(base_dataset.targets[base_idx])
    if hasattr(base_dataset, "labels"):
        return int(base_dataset.labels[base_idx])
    _, label = dataset[idx]
    return int(label)


def fetch_memory_samples(
    dataset: torch.utils.data.Dataset, dataset_indices: Sequence[int]
) -> torch.Tensor:
    """Fetch and stack a list of images into a memory tensor."""
    images = [dataset[idx][0] for idx in dataset_indices]
    return torch.stack(images, dim=0)


def choose_another_label(true_label: int, num_classes: int, rng: random.Random) -> int:
    """Pick a random label that differs from the provided true label."""
    candidates = [lbl for lbl in range(num_classes) if lbl != true_label]
    return rng.choice(candidates)


class BaselineRandomStrategy(MemoryStrategy):
    """Default memory set (baseline)."""

    name = "baseline_random"

    def build(
        self,
        sample: SelectedSample,
        baseline_memory: torch.Tensor,
        class_to_dataset_indices: Dict[int, List[int]],
        memory_dataset: torch.utils.data.Dataset,
        memory_size: int,
        num_classes: int,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Return baseline memory and placeholder indices when not tracked."""
        del sample, class_to_dataset_indices, memory_dataset, memory_size, num_classes
        return baseline_memory.clone(), [-1] * baseline_memory.shape[0]


class SameClassStrategy(MemoryStrategy):
    """Construct memory by sampling images only from the sample's true class."""

    name = "same_class"

    def __init__(self, rng: random.Random):
        self.rng = rng

    def build(
        self,
        sample: SelectedSample,
        baseline_memory: torch.Tensor,
        class_to_dataset_indices: Dict[int, List[int]],
        memory_dataset: torch.utils.data.Dataset,
        memory_size: int,
        num_classes: int,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Build memory from true-label class examples."""
        del baseline_memory, num_classes
        candidates = class_to_dataset_indices[sample.true_label]
        chosen = [self.rng.choice(candidates) for _ in range(memory_size)]
        return fetch_memory_samples(memory_dataset, chosen), chosen


class DifferentClassStrategy(MemoryStrategy):
    """Construct memory from one class that is different from true label."""

    name = "different_class"

    def __init__(self, rng: random.Random):
        self.rng = rng

    def build(
        self,
        sample: SelectedSample,
        baseline_memory: torch.Tensor,
        class_to_dataset_indices: Dict[int, List[int]],
        memory_dataset: torch.utils.data.Dataset,
        memory_size: int,
        num_classes: int,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Build memory from a randomly chosen non-true class."""
        del baseline_memory
        chosen_label = choose_another_label(sample.true_label, num_classes, self.rng)
        candidates = class_to_dataset_indices[chosen_label]
        chosen = [self.rng.choice(candidates) for _ in range(memory_size)]
        return fetch_memory_samples(memory_dataset, chosen), chosen


class RepeatedSingleSampleStrategy(MemoryStrategy):
    """Construct degenerate memory set by repeating one sample K times."""

    name = "single_repeated"

    def __init__(self, rng: random.Random, repeat_k: int):
        """Initialize strategy with RNG and repetition count."""
        self.rng = rng
        self.repeat_k = repeat_k

    def build(
        self,
        sample: SelectedSample,
        baseline_memory: torch.Tensor,
        class_to_dataset_indices: Dict[int, List[int]],
        memory_dataset: torch.utils.data.Dataset,
        memory_size: int,
        num_classes: int,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Sample one true-class item and replicate it K times."""
        del baseline_memory, memory_size, num_classes
        candidates = class_to_dataset_indices[sample.true_label]
        dataset_index = self.rng.choice(candidates)
        chosen = [dataset_index] * self.repeat_k
        return fetch_memory_samples(memory_dataset, chosen), chosen


class RandomSamplesStrategy(MemoryStrategy):
    """Construct memory from random dataset indices irrespective of class."""

    name = "random_samples"

    def __init__(self, rng: random.Random):
        self.rng = rng

    def build(
        self,
        sample: SelectedSample,
        baseline_memory: torch.Tensor,
        class_to_dataset_indices: Dict[int, List[int]],
        memory_dataset: torch.utils.data.Dataset,
        memory_size: int,
        num_classes: int,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Build memory by uniformly sampling indices from memory dataset."""
        del sample, baseline_memory, class_to_dataset_indices, num_classes
        chosen = [self.rng.randrange(len(memory_dataset)) for _ in range(memory_size)]
        return fetch_memory_samples(memory_dataset, chosen), chosen


class VerySmallMemoryStrategy(MemoryStrategy):
    """Construct a tiny memory bank (size 1 or 2) for stress testing."""

    name = "very_small_memory"

    def __init__(self, rng: random.Random, small_size: int):
        """Initialize strategy with RNG and very small target size."""
        self.rng = rng
        self.small_size = small_size

    def build(
        self,
        sample: SelectedSample,
        baseline_memory: torch.Tensor,
        class_to_dataset_indices: Dict[int, List[int]],
        memory_dataset: torch.utils.data.Dataset,
        memory_size: int,
        num_classes: int,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Build a memory tensor with only 1-2 random samples."""
        del sample, baseline_memory, class_to_dataset_indices, memory_size, num_classes
        chosen = [self.rng.randrange(len(memory_dataset)) for _ in range(self.small_size)]
        return fetch_memory_samples(memory_dataset, chosen), chosen


def tensor_to_uint8_img(image_tensor: torch.Tensor, undo_normalize_fn) -> np.ndarray:
    """Convert normalized tensor image to uint8 HWC image for saving."""
    # Undo dataset normalization before exporting to PNG/JPG.
    image = undo_normalize_fn(image_tensor.detach().cpu())
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return (image * 255.0).astype(np.uint8)


def save_image(image_tensor: torch.Tensor, path: str, undo_normalize_fn) -> None:
    """Save one tensor image to disk after undoing normalization."""
    # Convert tensor to displayable RGB image and write file.
    img_uint8 = tensor_to_uint8_img(image_tensor, undo_normalize_fn)
    Image.fromarray(img_uint8).save(path)


def topk_indices(values: torch.Tensor, k: int) -> List[int]:
    """Return top k index positions from a 1D tensor."""
    # Guard against requesting more elements than available.
    k = min(k, values.numel())
    if k == 0:
        return []
    _, idx = torch.topk(values, k=k, dim=0)
    return idx.detach().cpu().tolist()


def collect_selected_samples(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    baseline_memory: torch.Tensor,
    device: torch.device,
) -> List[SelectedSample]:
    """Collect exactly 2 correct and 2 misclassified test samples."""
    selected_correct: List[SelectedSample] = []
    selected_wrong: List[SelectedSample] = []
    sample_index = 0

    with torch.no_grad():
        # Scan test set until we have the required 2+2 samples.
        for images, labels in test_loader:
            for i in range(images.shape[0]):
                # Evaluate one test image using the baseline memory.
                image = images[i].unsqueeze(0).to(device)
                label = int(labels[i].item())
                output = model(image, baseline_memory.to(device))
                pred = int(output.argmax(dim=1).item())

                # Cache metadata to reuse this sample across all strategies.
                current = SelectedSample(
                    sample_index=sample_index,
                    image=images[i].detach().cpu(),
                    true_label=label,
                    baseline_pred=pred,
                )
                # Keep first 2 correct and first 2 incorrect examples.
                if pred == label and len(selected_correct) < 2:
                    selected_correct.append(current)
                elif pred != label and len(selected_wrong) < 2:
                    selected_wrong.append(current)

                sample_index += 1
                # Stop early once quota is met.
                if len(selected_correct) == 2 and len(selected_wrong) == 2:
                    return selected_correct + selected_wrong
    return selected_correct + selected_wrong


def build_class_index_map(memory_dataset: torch.utils.data.Dataset) -> Dict[int, List[int]]:
    """Build a label -> dataset indices map for memory sample lookup."""
    class_to_indices: Dict[int, List[int]] = {}
    for idx in range(len(memory_dataset)):
        label = get_dataset_label(memory_dataset, idx)
        class_to_indices.setdefault(label, []).append(idx)
    return class_to_indices


def run_analysis(args: argparse.Namespace) -> None:
    """Execute the full controlled-memory analysis pipeline."""
    # load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.path, map_location=device)
    model = utils.get_model(
        checkpoint["model_name"], checkpoint["num_classes"], model_type=checkpoint["modality"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # load dataset
    dataset_name = args.dataset if args.dataset else checkpoint["dataset_name"]
    if dataset_name != checkpoint["dataset_name"]:
        raise ValueError(
            f"Dataset mismatch. checkpoint dataset={checkpoint['dataset_name']}, --dataset={dataset_name}"
        )

    load_dataset = getattr(datasets, f"get_{dataset_name}")
    undo_normalization = getattr(datasets, f"undo_normalization_{dataset_name}")

    _, _, test_loader, mem_loader = load_dataset(
        args.dir_dataset,
        batch_size_train=128,
        batch_size_test=args.batch_size_test,
        batch_size_memory=checkpoint["mem_examples"],
        size_train=checkpoint["train_examples"],
    )
    # Retrieve one default memory batch for baseline selection.
    baseline_memory, _ = next(iter(mem_loader))
    baseline_memory = baseline_memory.to(device)

    # select target samples
    selected_samples = collect_selected_samples(model, test_loader, baseline_memory, device)
    if len(selected_samples) < 4:
        raise RuntimeError(
            f"Could not find 4 selected samples (2 correct + 2 wrong). Found {len(selected_samples)}."
        )

    # Build lookup maps used by manual memory construction strategies.
    memory_dataset = mem_loader.dataset
    class_to_indices = build_class_index_map(memory_dataset)
    rng = random.Random(args.seed)
    for class_id, idxs in class_to_indices.items():
        if not idxs:
            raise RuntimeError(f"No samples available for class {class_id} in memory dataset.")

    # prepare outputs
    out_root = os.path.join(args.output_dir, dataset_name, checkpoint["model_name"])
    images_dir = os.path.join(out_root, "images")
    os.makedirs(images_dir, exist_ok=True)
    csv_path = os.path.join(out_root, "analysis_results.csv")

    # Register pluggable memory intervention strategies.
    memory_size = int(checkpoint["mem_examples"])
    strategies: List[MemoryStrategy] = [
        BaselineRandomStrategy(),
        SameClassStrategy(rng),
        DifferentClassStrategy(rng),
        RepeatedSingleSampleStrategy(rng, repeat_k=args.repeat_k),
        RandomSamplesStrategy(rng),
        VerySmallMemoryStrategy(rng, small_size=args.small_memory_size),
    ]

    # run interventions
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "sample_index",
                "true_label",
                "predicted_label",
                "memory_type",
                "top5_memory_positions",
                "top5_memory_dataset_indices",
            ],
        )
        writer.writeheader()

        with torch.no_grad():
            for sample in selected_samples:
                # Save the current test input image once.
                input_img_path = os.path.join(images_dir, f"sample_{sample.sample_index}_input.png")
                save_image(sample.image, input_img_path, undo_normalization)

                img_batch = sample.image.unsqueeze(0).to(device)
                for strategy in strategies:
                    # Manually build memory tensor according to strategy.
                    memory_batch, dataset_indices = strategy.build(
                        sample=sample,
                        baseline_memory=baseline_memory.detach().cpu(),
                        class_to_dataset_indices=class_to_indices,
                        memory_dataset=memory_dataset,
                        memory_size=memory_size,
                        num_classes=int(checkpoint["num_classes"]),
                    )
                    memory_batch = memory_batch.to(device)

                    # Run model with explicit memory tensor and collect read weights.
                    outputs, rw = model(img_batch, memory_batch, return_weights=True)
                    pred_label = int(outputs.argmax(dim=1).item())
                    weights = rw[0]

                    # Find most influential memory entries by read weight.
                    top5_pos = topk_indices(weights, 5)
                    top3_pos = topk_indices(weights, 3)

                    # Map memory positions back to dataset indices for interpretability.
                    top5_dataset = []
                    for pos in top5_pos:
                        if 0 <= pos < len(dataset_indices):
                            top5_dataset.append(dataset_indices[pos])
                        else:
                            top5_dataset.append(-1)

                    # Log one row per (sample, strategy) to CSV.
                    writer.writerow(
                        {
                            "sample_index": sample.sample_index,
                            "true_label": sample.true_label,
                            "predicted_label": pred_label,
                            "memory_type": strategy.name,
                            "top5_memory_positions": ",".join(map(str, top5_pos)),
                            "top5_memory_dataset_indices": ",".join(map(str, top5_dataset)),
                        }
                    )

                    # Save the top-3 influential memory images for this run.
                    for rank, mem_pos in enumerate(top3_pos, start=1):
                        mem_img = memory_batch[mem_pos].detach().cpu()
                        out_path = os.path.join(
                            images_dir,
                            f"sample_{sample.sample_index}_{strategy.name}_top{rank}_mempos_{mem_pos}.png",
                        )
                        save_image(mem_img, out_path, undo_normalization)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved images directory: {images_dir}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the analysis script."""
    parser = argparse.ArgumentParser(description="Analyze memory interventions for BaselineMemory.")
    parser.add_argument("--path", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--dataset", required=True, help="Dataset name: SVHN, CIFAR10, CINIC10")
    parser.add_argument("--dir_dataset", default="datasets/", help="Dataset directory")
    parser.add_argument("--output_dir", default="paper/images/memory_variant_analysis", help="Output root dir")
    parser.add_argument("--batch_size_test", type=int, default=64, help="Test batch size for sample search")
    parser.add_argument("--repeat_k", type=int, default=100, help="K for repeated single memory sample strategy")
    parser.add_argument(
        "--small_memory_size",
        type=int,
        default=2,
        choices=[1, 2],
        help="Small memory size for degenerate strategy",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for memory sampling")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(args)
