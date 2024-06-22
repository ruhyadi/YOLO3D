"""Data module"""

"""Data module"""

import rootutils

ROOT = rootutils.autosetup()

from pathlib import Path
from typing import List

from torch.utils.data import Dataset

from src.utils import get_logger

log = get_logger()


class KittiData(Dataset):
    """KITTI dataloader."""

    def __init__(
        self,
        images_dir: str = "data/kitti/images_2",
        json_path: str = "data/kitti/annotations.json",
        calib_path: str = "assets/kitti_calib.json",
        categories: List[str] = ["car", "pedestrian", "cyclist"],
        n_bins: int = 2,
        bin_overlap: float = 0.5,
    ) -> None:
        """Initialize KITTI dataset."""
        super().__init__()
        self.images_dir = Path(images_dir)
        self.json_path = Path(json_path)
        self.calib_path = Path(calib_path)
        self.categories = categories
        self.n_bins = n_bins
        self.bin_overlap = bin_overlap
