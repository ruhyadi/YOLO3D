"""Schema module."""

import rootutils

ROOT = rootutils.autosetup()

import json
from pathlib import Path
from typing import List, Union

from pydantic import BaseModel, Field


class MultibinDataSchema(BaseModel):
    """Multibin data schema."""

    index: str = Field(..., example="042069")
    category: str = Field(..., example="car")
    box: List[float] = Field(..., example=[0.0, 0.0, 100, 100])
    alpha: float = Field(..., example=-1.0)
    orientation: List[List[float]] = Field(..., example=[[0.19, 0.98], [-0.19, -0.98]])
    confidence: List[float] = Field(..., example=[0.5, 0.5])
    dimensions: List[float] = Field(..., example=[1.2, 1.5, 1.2])


class KittiSchema(BaseModel):
    """KITTI schema."""

    type: str = Field(
        None,
        example="Car",
        description="Describes the type of object: 'Car','Pedestrian', etc",
    )
    truncated: float = Field(
        None,
        example=0.0,
        description="Float from 0 (non-truncated) to 1 (truncated), where truncated refers to the object leaving image boundaries",
    )
    occluded: int = Field(
        None,
        example=0,
        description="Integer (0,1,2,3) indicating occlusion state: 0 = fully visible, 1 = partly occluded 2 = largely occluded, 3 = unknown",
    )
    alpha: float = Field(
        None,
        example=-1.0,
        description="Observation angle of object, ranging [-pi..pi]",
    )
    bbox: List[float] = Field(
        None,
        example=[712.40, 143.00, 810.73, 307.92],
        description="2D bounding box of object in the image (0-based index): contains left, top, right, bottom pixel coordinates",
    )
    dimensions: List[float] = Field(
        None,
        example=[3.88, 1.63, 1.53],
        description="3D object dimensions: height, width, length (in meters)",
    )
    location: List[float] = Field(
        None,
        example=[2.67, 1.63, 9.0],
        description="3D object location x,y,z in camera coordinates (in meters)",
    )
    rotation_y: float = Field(
        None,
        example=-1.58,
        description="Rotation ry around Y-axis in camera coordinates [-pi..pi]",
    )
    score: float = Field(
        0.0,
        example=0.0,
        description="Only for results: Float, indicating confidence in detection, needed for p/r curves, higher is better",
    )

    def load(self, lines: List[str]) -> None:
        """Load KITTI annotations."""
        self.type = lines[0].lower()
        self.truncated = float(lines[1])
        self.occluded = int(lines[2])
        self.alpha = float(lines[3])
        self.bbox = [float(x) for x in lines[4:8]]
        self.dimensions = [float(x) for x in lines[8:11]]
        self.location = [float(x) for x in lines[11:14]]
        self.rotation_y = float(lines[14])
        if len(lines) > 15:
            self.score = float(lines[15])


class KittiDimsAvg(BaseModel):
    """kitti dimensions averager."""

    car: List[float] = Field([], example=[1.52, 1.62, 3.88])
    pedestrian: List[float] = Field([], example=[0.84, 0.50, 1.76])
    cyclist: List[float] = Field([], example=[0.60, 1.76, 1.73])

    def __getitem__(self, category: str) -> List[float]:
        """Get kitti dimension average of category."""

        return self.__dict__[str(category).lower()]

    def compute_dimensions(self, category: str, dims: List[float]) -> List[float]:
        """Compute kitti dimension average of category."""

        return dims - self.__getitem__(category)

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
                if category not in self.__dict__.keys():
                    continue
                dimension = [float(line[8]), float(line[9]), float(line[10])]
                self.__dict__[category].append(dimension)

        # average kitti dimension
        for key, value in self.__dict__.items():
            if not value:
                continue
            self.__dict__[key] = np.mean(value, axis=0).tolist()

    def generate_json(self, path: str) -> None:
        """Generate kitti dimension average json file."""
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)
