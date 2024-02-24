"""
Non maximum suppression utils with numpy.
"""

import rootutils

ROOT = rootutils.autosetup()

import numpy as np


def multiclass_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    nms: float = 0.45,
    conf: float = 0.25,
) -> np.ndarray:
    """
    Multiclass NMS with numpy.

    Args:
        boxes (np.ndarray): Bounding boxes with shape (N, 4). Format: (x1, y1, x2, y2).
        scores (np.ndarray): Scores with shape (N, num_classes).
        nms (float): NMS threshold.
        conf (float): Confidence threshold.

    Returns:
        np.ndarray: NMSed bounding boxes with shape (N, 6). Format: (x1, y1, x2, y2, score, class_id).
    """
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > conf
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms_numpy(valid_boxes, valid_scores, nms)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)

def nms_numpy(boxes: np.ndarray, scores: np.ndarray, nms: float = 0.45) -> list:
    """
    Non-maximum suppression with numpy.

    Args:
        boxes (np.ndarray): Bounding boxes with shape (N, 4). Format: (x1, y1, x2, y2).
        scores (np.ndarray): Scores with shape (N, 1).
        nms (float): NMS threshold.

    Returns:
        list: Indices of boxes to keep.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms)[0]
        order = order[inds + 1]

    return keep
