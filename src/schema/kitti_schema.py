"""KITTI schema module."""

import rootutils

ROOT = rootutils.autosetup()

import json
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field


class MultibinLabelSchema(BaseModel):
    """MultiBin label schema."""

    index: str = Field(..., example="042069")
    category: str = Field(..., example="Car")
    bbox: List[int] = Field(..., example=[0, 0, 100, 100])
    alpha: float = Field(..., example=1.5)
    dimensions: List[float] = Field(..., example=[1.2, 1.5, 1.2])
    orientation: List[List[float]] = Field(..., example=[[0.19, 0.98], [-0.19, -0.98]])
    confidence: List[float] = Field(..., example=[0.5, 0.5])


class KittiLabelSchema(BaseModel):
    """KITTI label schema."""

    type: Optional[str] = Field(
        None,
        example="Car",
        description="Describes the type of object: 'Car','Pedestrian', etc",
    )
    truncated: Optional[float] = Field(
        None,
        example=0.00,
        description="0 to 1, Truncated refers to the object leaving image boundaries",
    )
    occluded: Optional[int] = Field(
        None,
        example=0,
        description="indicating occlusion state: 0 = fully visible, 1 = partly occluded 2 largely occluded, 3 = unknown",
    )
    alpha: Optional[float] = Field(
        None,
        example=1.00,
        description="Observation angle of object, ranging [-pi..pi]",
    )
    bbox: Optional[List[float]] = Field(
        None,
        example=[50.00, 25.00, 25.00, 50.00],
        description="2D bounding box of object in the image (0-based index): contains left, top, right, bottom pixel coordinates",
    )
    dimensions: Optional[List[float]] = Field(
        None,
        example=[1.2, 1.5, 1.2],
        description="3D object dimensions: height, width, length (in meters)",
    )
    location: Optional[List[float]] = Field(
        None,
        example=[2.5, 4.5, 3.3],
        description="3D object location x,y,z in camera coordinates (in meters)",
    )
    rotation_y: Optional[float] = Field(
        None,
        example=0.75,
        description="Rotation ry around Y-axis in camera coordinates [-pi..pi]",
    )
    score: Optional[float] = Field(
        None,
        example=0.873,
        description="Only for results: Float",
    )

    def save(self, path: str, include_score: bool = True) -> None:
        """Save kitti label to txt file."""

        if not include_score:
            self.score = ""
        with open(path, "a") as f:
            if self.type is None:
                f.write("\n")
            else:
                f.write(
                    f"{self.type} {self.truncated} {self.occluded} {self.alpha} "
                    f"{(self.bbox[0])} {self.bbox[1]} {self.bbox[2]} {self.bbox[3]} "
                    f"{self.dimensions[0]} {self.dimensions[1]} {self.dimensions[2]} "
                    f"{self.location[0]} {self.location[1]} {self.location[2]} "
                    f"{self.rotation_y} {self.score}\n"
                )

    def load(self, label: list) -> None:
        """Load kitti label from line in txt file."""

        self.type = label[0]
        self.truncated = float(label[1])
        self.occluded = float(label[2])
        self.alpha = float(label[3])
        self.bbox = [
            int(float(label[4])),
            int(float(label[5])),
            int(float(label[6])),
            int(float(label[7])),
        ]
        self.dimensions = [float(label[8]), float(label[9]), float(label[10])]
        self.location = [float(label[11]), float(label[12]), float(label[13])]
        self.rotation_y = float(label[14])
        if len(label) == 16:
            self.score = float(label[15])


class KittiDimensionsAvgSchema(BaseModel):
    """kitti dimensions average schema."""

    car: List[float] = Field([], example=[1.52, 1.62, 3.88])
    pedestrian: List[float] = Field([], example=[0.84, 0.50, 1.76])
    cyclist: List[float] = Field([], example=[0.60, 1.76, 1.73])

    def __getitem__(self, category: str) -> List[float]:
        """Get kitti dimension average of category."""

        return self.__dict__[str(category).lower()]

    def load(self, labels_path: Union[str, Path], sets_path: Union[str, Path]) -> None:
        """Load kitti dimension average from labels file."""
        if isinstance(labels_path, str):
            labels_path = Path(labels_path)
        if isinstance(sets_path, str):
            sets_path = Path(sets_path)

        indexes = sets_path.read_text().splitlines()

        for i in indexes:
            label_path = labels_path / f"{i}.txt"
            lines = label_path.read_text().splitlines()
            for line in lines:
                line = line.split()
                category = str(line[0]).lower()
                if not category in self.model_fields:
                    continue
                dimension = [float(line[8]), float(line[9]), float(line[10])]
                self.__dict__[category].append(dimension)

    def generate_json(self, path: str) -> None:
        """Generate kitti dimension average json file."""
        for key, value in self.__dict__.items():
            if not value:
                continue
            self.__dict__[key] = np.mean(value, axis=0).tolist()

        # save kitti dimension average json file
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=4)
