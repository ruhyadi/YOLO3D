"""
Generate KITTI trainval sets for training and validation.
usage:
python tools/kitti_trainval_sets.py \
    --label_path data/kitti/label_2 \
    --train_size 0.8 \
    --output_path data/kitti
"""

import argparse
from pathlib import Path


def main(label_dir: str, train_size: float, output_dir: str):
    """Split KITTI label files into train and val sets."""
    label_dir: Path = Path(label_dir)
    output_dir: Path = Path(output_dir)
    assert label_dir.exists(), f"Label dir {label_dir} not found"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all label files
    label_files = list(label_dir.glob("*.txt"))
    label_files = [f.stem for f in label_files]

    # Split train and val
    num_train = int(len(label_files) * train_size)
    train_files = label_files[:num_train]
    val_files = label_files[num_train:]

    # Write to files
    with open(output_dir / f"train_{int(train_size * 100)}.txt", "w") as f:
        f.write("\n".join(train_files))
    with open(output_dir / f"val_{int(train_size * 100)}.txt", "w") as f:
        f.write("\n".join(val_files))

    print(f"Train: {len(train_files)}")
    print(f"Val: {len(val_files)}")
    print(f"Train file: {output_dir / 'train.txt'}")
    print(f"Val file: {output_dir / 'val.txt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate KITTI trainval sets for training and validation."
    )
    parser.add_argument(
        "--label_path", type=str, help="Path to KITTI label_2 directory"
    )
    parser.add_argument("--train_size", type=float, default=0.8, help="Train size")
    parser.add_argument("--output_path", type=str, help="Path to save trainval files")
    args = parser.parse_args()
    main(args.label_path, args.train_size, args.output_path)
