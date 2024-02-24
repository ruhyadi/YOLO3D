"""Yolo ONNX engine."""

import rootutils

ROOT = rootutils.autosetup()

from typing import List, Union

import cv2
import numpy as np

from src.engine.onnx_engine import CommonOnnxEngine
from src.schema.yolo_schema import YoloEnd2EndOutputSchema, YoloResultSchema
from src.utils.logger import get_logger
from src.utils.nms_utils import multiclass_nms

log = get_logger()


class YoloOnnxEngine(CommonOnnxEngine):
    """Yolo ONNX engine module."""

    def __init__(
        self,
        engine_path: str,
        categories: List[str],
        provider: str = "cpu",
        end2end: bool = False,
        arch: str = "yolox",
        pretrained: bool = False,
        max_det_end2end: int = 100,
    ) -> None:
        """Initialize YOLO ONNX engine."""
        assert arch in ["yolox"], "Invalid architecture"
        super().__init__(engine_path, provider)
        self.categories = categories
        self.end2end = end2end
        self.arch = arch
        self.pretrained = pretrained
        self.max_det_end2end = max_det_end2end

        self.normalize = False if self.arch == "yolox" else True

    def predict(
        self,
        imgs: Union[np.ndarray, List[np.ndarray]],
        conf: float = 0.25,
        nms: float = 0.45,
    ) -> List[YoloResultSchema]:
        """Detect objects in image(s)."""
        imgs, ratios, pads = self.preprocess_imgs(imgs, normalize=self.normalize)
        outputs = self.engine.run(None, {self.metadata[0].input_name: imgs})
        if self.end2end:
            outputs = self.postprocess_end2end(outputs, ratios, pads, conf)
        else:
            outputs = self.postprocess_nms(outputs, ratios, pads, conf, nms=nms)

        return outputs

    def preprocess_imgs(
        self,
        imgs: Union[np.ndarray, List[np.ndarray]],
        mode="center",
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Preprocess image(s) (batch) like resize, normalize, padding, etc.

        Args:
            imgs (Union[np.ndarray, List[np.ndarray]]): Image(s) to preprocess.
            mode (str, optional): Padding mode. Defaults to "center".
            normalize (bool, optional): Whether to normalize image(s). Defaults to True.

        Returns:
            np.ndarray: Preprocessed image(s) in size (B, C, H, W).
        """
        assert mode in ["center", "left"], "Invalid mode, must be 'center' or 'left'"
        if isinstance(imgs, np.ndarray):
            imgs = [imgs]

        # resize and pad
        dst_h, dst_w = self.img_shape
        resized_imgs = np.ones((len(imgs), dst_h, dst_w, 3), dtype=np.float32) * 114
        ratios = np.ones((len(imgs)), dtype=np.float32)
        pads = np.ones((len(imgs), 2), dtype=np.float32)
        for i, img in enumerate(imgs):
            src_h, src_w = img.shape[:2]
            ratio = min(dst_w / src_w, dst_h / src_h)
            resized_w, resized_h = int(src_w * ratio), int(src_h * ratio)
            dw, dh = (dst_w - resized_w) / 2, (dst_h - resized_h) / 2
            img = cv2.resize(img, (resized_w, resized_h))
            if mode == "center":
                top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
                left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
                img = cv2.copyMakeBorder(
                    img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=114
                )
                resized_imgs[i] = img
            elif mode == "left":
                resized_imgs[i][:resized_h, :resized_w, :] = img

            pads[i] = np.array([dw, dh], dtype=np.float32)
            ratios[i] = ratio

        # normalize
        resized_imgs = resized_imgs.transpose(0, 3, 1, 2)
        resized_imgs /= 255.0 if normalize else 1.0
        # resized_imgs = np.ascontiguousarray(resized_imgs).astype(np.float32)

        return resized_imgs, ratios, pads

    def postprocess_end2end(
        self,
        outputs: List[np.ndarray],
        ratios: np.ndarray,
        pads: np.ndarray,
        conf: float = 0.25,
    ) -> List[YoloResultSchema]:
        """Postprocess end2end ONNX engine."""

        # parsing ort run outputs
        run_outputs: List[YoloEnd2EndOutputSchema] = []
        for i in range((outputs[0].shape[0])):  # batch size
            num_det = int(outputs[0][i][0])
            boxes = outputs[1][i][:num_det]
            scores = outputs[2][i][:num_det]
            classes = outputs[3][i][:num_det]
            run_outputs.append(
                YoloEnd2EndOutputSchema(
                    num_det=num_det, boxes=boxes, scores=scores, classes=classes
                )
            )

        # scaling and filtering
        results: List[YoloResultSchema] = []
        for i, out in enumerate(run_outputs):
            # scale bbox to original image size
            out.boxes[:, 0::2] -= pads[i][0]
            out.boxes[:, 1::2] -= pads[i][1]
            out.boxes /= ratios[i]

            # filter by confidence
            mask = out.scores > conf
            out.boxes = out.boxes[mask].astype(np.int32)
            out.scores = out.scores[mask]
            out.classes = out.classes[mask]

            # filter by class
            mask = out.classes < len(self.categories)
            out.boxes = out.boxes[mask]
            out.scores = out.scores[mask]
            out.classes = out.classes[mask]

            results.append(
                YoloResultSchema(
                    categories=[self.categories[int(i)] for i in out.classes],
                    scores=out.scores,
                    boxes=out.boxes,
                )
            )

        return results

    def postprocess_nms(
        self,
        outputs: List[np.ndarray],
        ratios: np.ndarray,
        pads: np.ndarray,
        conf: float = 0.25,
        nms: float = 0.45,
    ) -> List[YoloResultSchema]:
        """Postprocess NMS ONNX engine."""
        outputs: np.ndarray = outputs[0]
        boxes = outputs[:, :, :4]
        scores = outputs[:, :, 4:5] * outputs[:, :, 5:]
        classes = outputs[:, :, 5:].argmax(axis=-1)

        # convert xywh to xyxy
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, :, 0] = boxes[:, :, 0] - boxes[:, :, 2] / 2
        boxes_xyxy[:, :, 1] = boxes[:, :, 1] - boxes[:, :, 3] / 2
        boxes_xyxy[:, :, 2] = boxes[:, :, 0] + boxes[:, :, 2] / 2
        boxes_xyxy[:, :, 3] = boxes[:, :, 1] + boxes[:, :, 3] / 2

        # scaling and filtering
        boxes_xyxy[:, :, 0] -= pads[:, 0, None]
        boxes_xyxy[:, :, 1] -= pads[:, 1, None]
        boxes_xyxy[:, :, 2] -= pads[:, 0, None]
        boxes_xyxy[:, :, 3] -= pads[:, 1, None]
        boxes_xyxy /= ratios[:, None, None]

        results: List[YoloResultSchema] = []
        for i in range(outputs.shape[0]):
            dets = multiclass_nms(boxes_xyxy[i], scores[i], nms=nms, conf=conf)
            if dets is None:
                results.append(YoloResultSchema())
                continue

            # filter confidence and class
            mask_score = dets[:, -2] > conf
            mask_class = dets[:, -1] < len(self.categories)
            dets = dets[mask_score & mask_class]

            class_ids = dets[:, -1].astype(np.int32).tolist()
            categories = [self.categories[int(i)] for i in class_ids]
            results.append(
                YoloResultSchema(
                    categories=categories,
                    scores=dets[:, -2].astype(np.float32).tolist(),
                    boxes=dets[:, :-2].astype(np.int32).tolist(),
                )
            )

        return results
