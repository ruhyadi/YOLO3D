"""
Convert KITTI annotations to COCO format.
usage:
python tools/kitti_to_coco.py \
    --anns_dir data/kitti/label_2 \
    --json_path data/kitti/annotations.json
"""

import rootutils

ROOT = rootutils.autosetup()

import argparse
from typing import List

from src.schema import CocoSchema
from src.utils import get_logger

log = get_logger()

def main(
    anns_dir: str,
    json_path: str,
    categories: List[str] = ["car", "pedestrian", "cyclist"],
) -> None:
    """Convert KITTI annotations to COCO format."""
    log.info(f"Converting KITTI annotations to COCO format")
    
    coco = CocoSchema()
    coco.load_kitti(anns_dir, categories)

    coco.save_json(json_path)

    log.info(f"Saved COCO annotations to {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert KITTI annotations to COCO format."
    )
    parser.add_argument(
        "--anns_dir",
        type=str,
        help="Directory containing KITTI annotations",
        required=True,
    )
    parser.add_argument(
        "--json_path",
        type=str,
        help="Path to save COCO annotations",
        required=True,
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=["car", "pedestrian", "cyclist"],
        help="Categories to include in COCO annotations",
    )

    args = parser.parse_args()
    main(args.anns_dir, args.json_path, args.categories)
