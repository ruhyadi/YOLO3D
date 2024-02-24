"""YOLO3D onnx engine."""

import rootutils

ROOT = rootutils.autosetup()

import json
from typing import Dict, List

import numpy as np

from src.engine.multibin_onnx_engine import MultibinOnnxEngine
from src.engine.yolo_onnx_engine import YoloOnnxEngine
from src.schema.yolo_schema import YoloResultSchema
from src.utils.calib_utils import load_matrix_p2
from src.utils.logger import get_logger
from src.utils.plot_utils import Plot3DBox

log = get_logger()


class Yolo3dOnnxEngine:
    """YOLO3D onnx engine."""

    def __init__(
        self,
        yolo_engine_path: str,
        yolo_categories: List[str],
        yolo_end2end: bool,
        yolo_arch: str,
        yolo_max_det_end2end: int,
        multibin_engine_path: str,
        provider: str = "cpu",
        avg_dim_path: str = "assets/kitti_dimensions.json",
        calib_path: str = "assets/kitti_calib.txt",
    ) -> None:
        """Initialize demo."""
        self.yolo_engine_path = yolo_engine_path
        self.yolo_categories = yolo_categories
        self.yolo_end2end = yolo_end2end
        self.yolo_arch = yolo_arch
        self.yolo_max_det_end2end = yolo_max_det_end2end

        self.multibin_engine_path = multibin_engine_path
        self.avg_dim_path = avg_dim_path
        self.calib_path = calib_path
        self.provider = provider

        self.matrix_p2 = load_matrix_p2(calib_path)
        self.avg_dim = self.load_avg_dim(avg_dim_path)
        self.plotter = Plot3DBox(proj_matrix=self.matrix_p2, categories=self.categories)

    def setup(self) -> None:
        """Setup YOLO and multibin engine."""
        log.info(f"Setup YOLO3D ONNX engine...")
        self.yolo_engine = YoloOnnxEngine(
            engine_path=self.yolo_engine_path,
            categories=self.yolo_categories,
            provider=self.provider,
            end2end=self.yolo_end2end,
            arch=self.yolo_arch,
        )
        self.yolo_engine.setup()
        self.multibin_engine = MultibinOnnxEngine(
            enigne_path=self.multibin_engine_path,
            provider=self.provider,
        )
        self.multibin_engine.setup()

        log.info(f"YOLO3D ONNX engine is ready!")

    def predict(self, img: np.ndarray, conf: float = 0.25, nms: float = 0.45) -> None:
        """Predict."""
        yolo_results = self.yolo_engine.predict(img, conf, nms)
        cropped_imgs = self.preprocess_multibin(img, yolo_results)
        results = self.multibin_engine.predict(imgs=cropped_imgs)

        return results

    def preprocess_multibin(
        self, img: np.ndarray, yolo_results: List[YoloResultSchema]
    ) -> List[np.ndarray]:
        """Preprocess images for multibin engine."""
        results: List[np.ndarray] = []
        for box in yolo_results[0].boxes:  # single image
            x1, y1, x2, y2 = box
            cropped_img = img[y1:y2, x1:x2]
            results.append(cropped_img)

        return results

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
