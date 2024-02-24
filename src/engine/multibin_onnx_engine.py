"""Multibin ONNX Engine."""

import rootutils

ROOT = rootutils.autosetup()

from typing import List

import cv2
import numpy as np

from src.engine.onnx_engine import CommonOnnxEngine
from src.schema.multibin_schema import MultibinResultSchema
from src.utils.logger import get_logger

log = get_logger()


class MultibinOnnxEngine(CommonOnnxEngine):
    """Multibin ONNX engine."""

    def __init__(self, engine_path: str, provider: str = "cpu") -> None:
        """Initialize Multibin ONNX engine."""
        super().__init__(engine_path, provider)

    def predict(self, imgs: List[np.ndarray]) -> List[MultibinResultSchema]:
        """Predict cropped images."""
        imgs = self.preprocess_imgs(imgs)
        outputs = self.engine.run(None, {self.metadata[0].input_name: imgs})
        results = self.postprocess_outputs(outputs)

        return results

    def preprocess_imgs(
        self, imgs: List[np.ndarray], normalize: bool = True
    ) -> np.ndarray:
        """Preprocess images for multibin inference."""

        # resize images
        dst_h, dst_w = self.img_shape[2:]
        resized_imgs = np.zeros((len(imgs), dst_h, dst_w, 3), dtype=np.float32)
        for i, img in enumerate(imgs):
            img = cv2.resize(imgs, (dst_w, dst_h))
            resized_imgs[i] = img

        # normalize and transpose images
        resized_imgs = resized_imgs.transpose(0, 3, 1, 2)
        resized_imgs = resized_imgs / 255.0 if normalize else resized_imgs

        return resized_imgs

    def postprocess_outputs(
        self, outputs: List[np.ndarray]
    ) -> List[MultibinResultSchema]:
        """Postprocess outputs."""
        results: List[MultibinResultSchema] = []
        for output in outputs:
            results.append(
                MultibinResultSchema(
                    orientation=output[0],
                    confidence=output[1],
                    dimension=output[2],
                )
            )

        return results
