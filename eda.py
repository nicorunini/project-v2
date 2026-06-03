from argparse import ArgumentParser
from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(dataset_dir: Path):
    summary = {}
    total = 0
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        return summary, total

    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        files = [
            path for path in sorted(class_dir.iterdir())
            if path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        summary[class_dir.name] = files
        total += len(files)
    return summary, total


def print_summary(dataset_dir: Path, sample_count: int):
    summary, total = count_images(dataset_dir)
    if not summary:
        print(f"No dataset classes found in {dataset_dir}")
        return

    print(f"Dataset directory: {dataset_dir}")
    print(f"Total classes: {len(summary)}")
    print(f"Total images: {total}")
    print("\nClass distribution:")
    for label, files in summary.items():
        print(f"  {label}: {len(files)}")

    if sample_count > 0:
        print(f"\nSample file names per class (first {sample_count}):")
        for label, files in summary.items():
            print(f"\n[{label}]")
            for path in files[:sample_count]:
                print(f"  {path.name}")


def main():
    parser = ArgumentParser(description="Terminal-only dataset EDA for the eye-state dataset.")
    parser.add_argument(
        "--dataset-dir",
        default="datasets",
        help="Path to the dataset root directory containing class subfolders.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of sample file names to print per class.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    print_summary(dataset_dir, args.samples)


if __name__ == "__main__":
    main()
