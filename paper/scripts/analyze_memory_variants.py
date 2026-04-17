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
    sample_index: int
    image: torch.Tensor
    true_label: int
    baseline_pred: int


class MemoryStrategy:
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
        raise NotImplementedError


def unwrap_dataset_index(
    dataset: torch.utils.data.Dataset, idx: int
) -> Tuple[torch.utils.data.Dataset, int]:
    if isinstance(dataset, torch.utils.data.Subset):
        return unwrap_dataset_index(dataset.dataset, dataset.indices[idx])
    return dataset, idx


def get_dataset_label(dataset: torch.utils.data.Dataset, idx: int) -> int:
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
    images = [dataset[idx][0] for idx in dataset_indices]
    return torch.stack(images, dim=0)


def choose_another_label(true_label: int, num_classes: int, rng: random.Random) -> int:
    candidates = [lbl for lbl in range(num_classes) if lbl != true_label]
    return rng.choice(candidates)


class BaselineRandomStrategy(MemoryStrategy):
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
        del sample, class_to_dataset_indices, memory_dataset, memory_size, num_classes
        return baseline_memory.clone(), [-1] * baseline_memory.shape[0]


class SameClassStrategy(MemoryStrategy):
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
        del baseline_memory, num_classes
        candidates = class_to_dataset_indices[sample.true_label]
        chosen = [self.rng.choice(candidates) for _ in range(memory_size)]
        return fetch_memory_samples(memory_dataset, chosen), chosen


class DifferentClassStrategy(MemoryStrategy):
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
        del baseline_memory
        chosen_label = choose_another_label(sample.true_label, num_classes, self.rng)
        candidates = class_to_dataset_indices[chosen_label]
        chosen = [self.rng.choice(candidates) for _ in range(memory_size)]
        return fetch_memory_samples(memory_dataset, chosen), chosen


class RepeatedSingleSampleStrategy(MemoryStrategy):
    name = "single_repeated"

    def __init__(self, rng: random.Random, repeat_k: int):
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
        del baseline_memory, memory_size, num_classes
        candidates = class_to_dataset_indices[sample.true_label]
        dataset_index = self.rng.choice(candidates)
        chosen = [dataset_index] * self.repeat_k
        return fetch_memory_samples(memory_dataset, chosen), chosen


class RandomSamplesStrategy(MemoryStrategy):
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
        del sample, baseline_memory, class_to_dataset_indices, num_classes
        chosen = [self.rng.randrange(len(memory_dataset)) for _ in range(memory_size)]
        return fetch_memory_samples(memory_dataset, chosen), chosen


class VerySmallMemoryStrategy(MemoryStrategy):
    name = "very_small_memory"

    def __init__(self, rng: random.Random, small_size: int):
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
        del sample, baseline_memory, class_to_dataset_indices, memory_size, num_classes
        chosen = [self.rng.randrange(len(memory_dataset)) for _ in range(self.small_size)]
        return fetch_memory_samples(memory_dataset, chosen), chosen


def tensor_to_uint8_img(image_tensor: torch.Tensor, undo_normalize_fn) -> np.ndarray:
    image = undo_normalize_fn(image_tensor.detach().cpu())
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return (image * 255.0).astype(np.uint8)


def save_image(image_tensor: torch.Tensor, path: str, undo_normalize_fn) -> None:
    img_uint8 = tensor_to_uint8_img(image_tensor, undo_normalize_fn)
    Image.fromarray(img_uint8).save(path)


def topk_indices(values: torch.Tensor, k: int) -> List[int]:
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
    selected_correct: List[SelectedSample] = []
    selected_wrong: List[SelectedSample] = []
    sample_index = 0

    with torch.no_grad():
        for images, labels in test_loader:
            for i in range(images.shape[0]):
                image = images[i].unsqueeze(0).to(device)
                label = int(labels[i].item())
                output = model(image, baseline_memory.to(device))
                pred = int(output.argmax(dim=1).item())

                current = SelectedSample(
                    sample_index=sample_index,
                    image=images[i].detach().cpu(),
                    true_label=label,
                    baseline_pred=pred,
                )
                if pred == label and len(selected_correct) < 2:
                    selected_correct.append(current)
                elif pred != label and len(selected_wrong) < 2:
                    selected_wrong.append(current)

                sample_index += 1
                if len(selected_correct) == 2 and len(selected_wrong) == 2:
                    return selected_correct + selected_wrong
    return selected_correct + selected_wrong


def build_class_index_map(memory_dataset: torch.utils.data.Dataset) -> Dict[int, List[int]]:
    class_to_indices: Dict[int, List[int]] = {}
    for idx in range(len(memory_dataset)):
        label = get_dataset_label(memory_dataset, idx)
        class_to_indices.setdefault(label, []).append(idx)
    return class_to_indices


def run_analysis(args: argparse.Namespace) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.path, map_location=device)
    model = utils.get_model(
        checkpoint["model_name"], checkpoint["num_classes"], model_type=checkpoint["modality"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

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
    baseline_memory, _ = next(iter(mem_loader))
    baseline_memory = baseline_memory.to(device)

    selected_samples = collect_selected_samples(model, test_loader, baseline_memory, device)
    if len(selected_samples) < 4:
        raise RuntimeError(
            f"Could not find 4 selected samples (2 correct + 2 wrong). Found {len(selected_samples)}."
        )

    memory_dataset = mem_loader.dataset
    class_to_indices = build_class_index_map(memory_dataset)
    rng = random.Random(args.seed)
    for class_id, idxs in class_to_indices.items():
        if not idxs:
            raise RuntimeError(f"No samples available for class {class_id} in memory dataset.")

    out_root = os.path.join(args.output_dir, dataset_name, checkpoint["model_name"])
    images_dir = os.path.join(out_root, "images")
    os.makedirs(images_dir, exist_ok=True)
    csv_path = os.path.join(out_root, "analysis_results.csv")

    memory_size = int(checkpoint["mem_examples"])
    strategies: List[MemoryStrategy] = [
        BaselineRandomStrategy(),
        SameClassStrategy(rng),
        DifferentClassStrategy(rng),
        RepeatedSingleSampleStrategy(rng, repeat_k=args.repeat_k),
        RandomSamplesStrategy(rng),
        VerySmallMemoryStrategy(rng, small_size=args.small_memory_size),
    ]

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
                input_img_path = os.path.join(images_dir, f"sample_{sample.sample_index}_input.png")
                save_image(sample.image, input_img_path, undo_normalization)

                img_batch = sample.image.unsqueeze(0).to(device)
                for strategy in strategies:
                    memory_batch, dataset_indices = strategy.build(
                        sample=sample,
                        baseline_memory=baseline_memory.detach().cpu(),
                        class_to_dataset_indices=class_to_indices,
                        memory_dataset=memory_dataset,
                        memory_size=memory_size,
                        num_classes=int(checkpoint["num_classes"]),
                    )
                    memory_batch = memory_batch.to(device)

                    outputs, rw = model(img_batch, memory_batch, return_weights=True)
                    pred_label = int(outputs.argmax(dim=1).item())
                    weights = rw[0]
                    top5_pos = topk_indices(weights, 5)
                    top3_pos = topk_indices(weights, 3)

                    top5_dataset = []
                    for pos in top5_pos:
                        if 0 <= pos < len(dataset_indices):
                            top5_dataset.append(dataset_indices[pos])
                        else:
                            top5_dataset.append(-1)

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
