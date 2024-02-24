"""Demo scripts."""

import rootutils

ROOT = rootutils.autosetup()

from typing import List

from src.utils.calib_utils import load_matrix_p2
from src.utils.logger import get_logger

log = get_logger()


class Demo:
    """Demo yolo3d."""

    def __init__(
        self,
        det_engine_path: str,
        multibin_engine_path: str,
        multibin_max_batch_size: int = 8,
        multibin_bin_size: int = 2,
        categories: List[str] = ["car", "pedestrian", "cyclist"],
        avg_dim_path: str = "assets/kitti_dimensions.json",
        calib_path: str = "assets/kitti_calib.txt",
        provider: str = "cpu",
    ) -> None:
        """Initialize demo."""
        self.det_engine_path = det_engine_path
        self.multibin_engine_path = multibin_engine_path
        self.multibin_max_batch_size = multibin_max_batch_size
        self.multibin_bin_size = multibin_bin_size
        self.categories = categories
        self.avg_dim_path = avg_dim_path
        self.calib_path = calib_path
        self.provider = provider
