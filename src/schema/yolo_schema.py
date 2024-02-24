"""YOLO schema."""

import rootutils

ROOT = rootutils.autosetup()

from typing import List

import numpy as np
from pydantic import BaseModel, Field, validator


class YoloEnd2EndOutputSchema(BaseModel):
    """YOLO end-to-end output schema."""

    num_det: int = Field(..., example=20)
    boxes: np.ndarray = Field(..., example=[0, 0, 100, 100, 50, 50, 150, 150])
    scores: np.ndarray = Field(..., example=[0.9, 0.8])
    classes: np.ndarray = Field(..., example=[0, 1])

    class Config:
        arbitrary_types_allowed = True


class YoloResultSchema(BaseModel):
    """YOLO result schema."""

    boxes: List[List[int]] = Field([], example=[[0, 0, 100, 100], [50, 50, 150, 150]])
    scores: List[float] = Field([], example=[0.9, 0.8])
    categories: List[str] = Field([], example=["person", "car"])

    @validator("scores", pre=True)
    def scores_validator(cls, v):
        """Round scores to 2 decimal places."""
        return [round(x, 2) for x in v]

    @validator("boxes", pre=True)
    def clip_boxes(cls, v):
        """Clip boxes to image size."""
        return [[max(x, 1) for x in box] for box in v]
