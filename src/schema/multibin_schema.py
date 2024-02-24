"""Multibin schema."""

import rootutils

ROOT = rootutils.autosetup()

import numpy as np
from pydantic import BaseModel, Field


class MultibinResultSchema(BaseModel):
    """Multibin result schema."""

    orientation: np.ndarray = Field(np.array([]), example=[0.91, 0.39])
    confidence: np.ndarray = Field(np.array([]), example=[0.88])
    dimension: np.ndarray = Field(
        np.array([]), example=[-0.0066245, -0.01338733, -0.04965857]
    )

    class Config:
        arbitrary_types_allowed = True
