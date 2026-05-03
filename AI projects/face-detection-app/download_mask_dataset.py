"""Download a public face-mask dataset and convert YOLO boxes into ImageFolder crops."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from huggingface_hub import snapshot_download
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare a mask / no-mask dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mask_dataset"),
        help="Directory where Mask and NoMask folders will be created.",
    )
    parser.add_argument(
        "--dataset-repo",
        type=str,
        default="hlydecker/face-masks",
        help="Public Hugging Face dataset repo in YOLO format.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/_downloads/face_masks_repo"),
        help="Local directory used for the downloaded dataset repo.",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=0,
        help="Optional max number of crops to save per class. Use 0 for all.",
    )
    parser.add_argument(
        "--mask-class-id",
        type=int,
        default=None,
        help="Override YOLO class id for the Mask class.",
    )
    parser.add_argument(
        "--nomask-class-id",
        type=int,
        default=None,
        help="Override YOLO class id for the NoMask class.",
    )
    parser.add_argument(
        "--min-crop-size",
        type=int,
        default=24,
        help="Skip crops smaller than this many pixels in width or height.",
    )
    return parser.parse_args()


def ensure_dirs(root: Path) -> tuple[Path, Path]:
    mask_dir = root / "Mask"
    nomask_dir = root / "NoMask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    nomask_dir.mkdir(parents=True, exist_ok=True)
    return mask_dir, nomask_dir


def parse_class_mapping(repo_root: Path) -> dict[int, str]:
    """Best-effort parser for YOLO yaml names."""
    yaml_files = list(repo_root.glob("*.yaml"))
    if not yaml_files:
        return {}

    text = yaml_files[0].read_text(encoding="utf-8", errors="ignore")
    mapping: dict[int, str] = {}

    dict_matches = re.findall(r"^\s*(\d+)\s*:\s*['\"]?([^'\"]+)['\"]?\s*$", text, flags=re.MULTILINE)
    for key, value in dict_matches:
        mapping[int(key)] = value.strip()
    if mapping:
        return mapping

    names_block = re.search(r"names\s*:\s*\[(.*?)\]", text, flags=re.DOTALL)
    if names_block:
        raw_items = names_block.group(1).split(",")
        for idx, item in enumerate(raw_items):
            value = item.strip().strip("'\"")
            if value:
                mapping[idx] = value
    return mapping


def resolve_class_ids(mapping: dict[int, str], mask_override: int | None, nomask_override: int | None) -> tuple[int, int]:
    if mask_override is not None and nomask_override is not None:
        return mask_override, nomask_override

    lowered = {idx: name.lower().replace("-", " ").replace("_", " ").strip() for idx, name in mapping.items()}
    mask_id = mask_override
    nomask_id = nomask_override

    for idx, name in lowered.items():
        if mask_id is None and name == "mask":
            mask_id = idx
        if nomask_id is None and name in {"no mask", "without mask", "nomask"}:
            nomask_id = idx

    if mask_id is None:
        mask_id = 0
    if nomask_id is None:
        nomask_id = 1
    return mask_id, nomask_id


def yolo_to_xyxy(cx: float, cy: float, bw: float, bh: float, width: int, height: int) -> tuple[int, int, int, int]:
    x_center = cx * width
    y_center = cy * height
    box_width = bw * width
    box_height = bh * height

    x1 = max(0, int(round(x_center - box_width / 2)))
    y1 = max(0, int(round(y_center - box_height / 2)))
    x2 = min(width, int(round(x_center + box_width / 2)))
    y2 = min(height, int(round(y_center + box_height / 2)))
    return x1, y1, x2, y2


def find_image_for_label(images_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    args = parse_args()
    mask_dir, nomask_dir = ensure_dirs(args.output_dir)

    repo_root = Path(
        snapshot_download(
            repo_id=args.dataset_repo,
            repo_type="dataset",
            local_dir=str(args.cache_dir),
            allow_patterns=["train/**", "valid/**", "val/**", "test/**", "*.yaml", "README.md"],
        )
    )

    class_mapping = parse_class_mapping(repo_root)
    mask_class_id, nomask_class_id = resolve_class_ids(class_mapping, args.mask_class_id, args.nomask_class_id)

    print(f"Downloaded dataset repo: {args.dataset_repo}")
    if class_mapping:
        print(f"Parsed class mapping: {class_mapping}")
    else:
        print("Could not parse class names from yaml, using default class ids.")
    print(f"Using class ids -> Mask: {mask_class_id}, NoMask: {nomask_class_id}")

    counts = {"Mask": 0, "NoMask": 0, "Skipped": 0}
    split_names = ("train", "valid", "val", "test")

    for split_name in split_names:
        split_dir = repo_root / split_name
        labels_dir = split_dir / "labels"
        images_dir = split_dir / "images"
        if not labels_dir.exists() or not images_dir.exists():
            continue

        print(f"Processing split: {split_name}")
        for label_file in sorted(labels_dir.glob("*.txt")):
            image_path = find_image_for_label(images_dir, label_file.stem)
            if image_path is None:
                counts["Skipped"] += 1
                continue

            with Image.open(image_path) as image:
                image = image.convert("RGB")
                width, height = image.size

                for line_index, line in enumerate(label_file.read_text(encoding="utf-8", errors="ignore").splitlines()):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        counts["Skipped"] += 1
                        continue

                    class_id = int(float(parts[0]))
                    if class_id == mask_class_id:
                        target_dir = mask_dir
                    elif class_id == nomask_class_id:
                        target_dir = nomask_dir
                    else:
                        counts["Skipped"] += 1
                        continue

                    class_name = target_dir.name
                    if args.limit_per_class and counts[class_name] >= args.limit_per_class:
                        continue

                    cx, cy, bw, bh = map(float, parts[1:])
                    x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, width, height)
                    if x2 <= x1 or y2 <= y1:
                        counts["Skipped"] += 1
                        continue

                    crop_w = x2 - x1
                    crop_h = y2 - y1
                    if crop_w < args.min_crop_size or crop_h < args.min_crop_size:
                        counts["Skipped"] += 1
                        continue

                    crop = image.crop((x1, y1, x2, y2))
                    output_path = target_dir / f"{split_name}_{label_file.stem}_{line_index:02d}.png"
                    crop.save(output_path)
                    counts[class_name] += 1

    print("Download and crop preparation complete.")
    print(f"Mask crops: {counts['Mask']}")
    print(f"NoMask crops: {counts['NoMask']}")
    print(f"Skipped items: {counts['Skipped']}")
    print(f"Saved dataset to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
