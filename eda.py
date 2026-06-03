from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(dataset_dir: Path) -> Tuple[Dict[str, List[Path]], int]:
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


def format_table(rows: List[List[str]], headers: List[str]) -> str:
    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)]
    header_line = " | ".join(str(cell).ljust(width) for cell, width in zip(headers, widths))
    separator = "-+-".join("-" * width for width in widths)
    row_lines = [" | ".join(str(cell).ljust(width) for cell, width in zip(row, widths)) for row in rows]
    return "\n".join([header_line, separator, *row_lines])


def save_class_distribution_chart(summary: Dict[str, List[Path]], output_path: Path) -> None:
    labels = list(summary.keys())
    counts = [len(summary[label]) for label in labels]
    if not labels:
        return

    width = 900
    height = 500
    margin = 60
    bar_width = int((width - margin * 2) / max(len(labels), 1) * 0.7)
    spacing = int((width - margin * 2 - bar_width * len(labels)) / max(len(labels) - 1, 1))
    max_count = max(counts)
    chart_height = height - margin * 2

    image = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        title_font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
        title_font = font

    draw.text((margin, 12), "Class distribution", fill="#000000", font=title_font)
    draw.line((margin, margin, width - margin, margin), fill="#444444", width=2)

    for index, (label, count) in enumerate(zip(labels, counts)):
        x0 = margin + index * (bar_width + spacing)
        x1 = x0 + bar_width
        bar_height = 0 if max_count == 0 else int((count / max_count) * (chart_height - 40))
        y0 = height - margin
        y1 = y0 - bar_height
        draw.rectangle([x0, y1, x1, y0], fill="#4C72B0")
        text = str(count)
        text_width, text_height = draw.textsize(text, font=font)
        draw.text((x0 + (bar_width - text_width) / 2, y1 - text_height - 6), text, fill="#000000", font=font)
        label_text = label[:20]
        label_width, label_height = draw.textsize(label_text, font=font)
        draw.text((x0 + (bar_width - label_width) / 2, y0 + 6), label_text, fill="#000000", font=font)

    image.save(output_path)


def print_summary(dataset_dir: Path, sample_count: int, output_dir: Path) -> None:
    summary, total = count_images(dataset_dir)
    if not summary:
        print(f"No dataset classes found in {dataset_dir}")
        return

    print(f"Dataset directory : {dataset_dir}")
    print(f"Total classes     : {len(summary)}")
    print(f"Total images      : {total}")
    print(f"Average per class: {total / len(summary):.2f}")

    rows = []
    for label, files in summary.items():
        rows.append([label, str(len(files))])

    print("\nClass distribution table:")
    print(format_table(rows, ["Class", "Count"]))

    chart_path = output_dir / "eda_class_distribution.png"
    save_class_distribution_chart(summary, chart_path)
    print(f"\nSaved class distribution chart to: {chart_path}")

    if sample_count > 0:
        print(f"\nSample file names per class (first {sample_count}):")
        for label, files in summary.items():
            print(f"\n[{label}]")
            for path in files[:sample_count]:
                print(f"  {path.name}")


def main() -> None:
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
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where chart output files will be written.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_summary(dataset_dir, args.samples, output_dir)


if __name__ == "__main__":
    main()
