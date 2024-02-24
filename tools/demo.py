"""Demo scripts."""

import rootutils

ROOT = rootutils.autosetup()

import json
from pathlib import Path
from typing import Dict, List

from src.utils.calib_utils import load_matrix_p2
from src.utils.logger import get_logger
from src.utils.plot_utils import Plot3DBox
from src.engine.yolo_onnx_engine import YoloOnnxEngine

log = get_logger()


class Demo:
    """Demo yolo3d."""

    def __init__(
        self,
        yolo_engine_path: str,
        yolo_categories: List[str],
        yolo_provider: str,
        yolo_end2end: bool,
        yolo_arch: str,
        yolo_pretrained: bool,
        yolo_max_det_end2end: int,

        multibin_engine_path: str,
        multibin_max_batch_size: int = 8,
        multibin_bin_size: int = 2,
        categories: List[str] = ["car", "pedestrian", "cyclist"],
        avg_dim_path: str = "assets/kitti_dimensions.json",
        calib_path: str = "assets/kitti_calib.txt",
        provider: str = "cpu",
        output_dir: str = "tmp/demo",
    ) -> None:
        """Initialize demo."""
        self.yolo_engine_path = yolo_engine_path
        self.yolo_categories = yolo_categories
        self.yolo_provider = yolo_provider
        self.yolo_end2end = yolo_end2end
        self.yolo_arch = yolo_arch
        self.yolo_pretrained = yolo_pretrained
        self.yolo_max_det_end2end = yolo_max_det_end2end

        self.multibin_engine_path = multibin_engine_path
        self.multibin_max_batch_size = multibin_max_batch_size
        self.multibin_bin_size = multibin_bin_size
        self.categories = categories
        self.avg_dim_path = avg_dim_path
        self.calib_path = calib_path
        self.provider = provider

        self.matrix_p2 = load_matrix_p2(calib_path)
        self.avg_dim = self.load_avg_dim(avg_dim_path)
        self.plotter = Plot3DBox(proj_matrix=self.matrix_p2, categories=self.categories)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup(self) -> None:
        """Setup YOLO and multibin engine."""
        log.info(f"Setup YOLO3D engine...")
        self.yolo_engine = YoloOnnxEngine(
            engine_path=self.yolo_engine_path,
            categories=self.yolo_categories,
            provider=self.yolo_provider,
            end2end=self.yolo_end2end,
            arch=self.yolo_arch,
            pretrained=self.yolo_pretrained,
            max_det_end2end=self.yolo_max_det_end2end,
        )

    def load_avg_dim(self, json_path: str) -> Dict[str, List[float]]:
        """
        Load average dimension from json file.
        Example output:
        {
            'pedestrian': [1.76, 0.66, 0.84],
            'cyclist': [1.73, 0.60, 1.76],
            'car': [1.52, 1.63, 3.89]
        }
        """
        with open(json_path, "r") as f:
            avg_dim: Dict[str, List[float]] = json.load(f)

        return {k.lower(): v for k, v in avg_dim.items()}
