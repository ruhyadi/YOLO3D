"""Schema module."""

import rootutils
from pydantic.main import TupleGenerator

ROOT = rootutils.autosetup()

import json
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field
from tqdm import tqdm


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


class CocoSchema(BaseModel):
    """COCO schema."""

    categories: List["CocoCategorieSchema"] = Field([])
    images: List["CocoImageSchema"] = Field([])
    annotations: List["CocoAnnotationSchema"] = Field([])

    def load_kitti(self, anns_dir: str, categories: List[str]) -> "CocoSchema":
        """Load KITTI annotations."""
        anns_dir = Path(anns_dir)
        categories = [cat.lower() for cat in categories]

        # assign category
        for cat in categories:
            cat = CocoCategorieSchema(
                id=len(self.categories) + 1,  # 1-indexed
                name=cat,
                supercategory=cat,
            )
            self.categories.append(cat)

        for ann_path in tqdm(sorted(anns_dir.glob("*.txt")), desc="Loading KITTI"):
            ann_path: Path = ann_path

            # assign image
            img = CocoImageSchema(
                id=len(self.images) + 1,  # 1-indexed
                width=1242,  # default
                height=375,  # default
                file_name=ann_path.stem + ".png",
            )
            self.images.append(img)

            lines = ann_path.read_text().splitlines()
            for line in lines:
                kitti = KittiSchema()
                kitti.load(line.split())
                if kitti.type not in categories:
                    continue

                # assign annotation
                ann = CocoAnnotationSchema(
                    id=len(self.annotations) + 1,  # 1-indexed
                    image_id=img.id,
                    category_id=categories.index(kitti.type) + 1,  # 1-indexed
                    segmentation=[],
                    area=self._xyxy2area(kitti.bbox),
                    bbox=self._xyxy2xywh(kitti.bbox),
                    iscrowd=0,
                    attributes=kitti,
                )
                self.annotations.append(ann)

        return self

    def save_json(self, json_path: str) -> None:
        """Save COCO annotations to JSON."""
        data = {
            "categories": [cat.model_dump() for cat in self.categories],
            "images": [img.model_dump() for img in self.images],
            "annotations": [ann.model_dump() for ann in self.annotations],
        }

        json.dump(data, open(json_path, "w"), indent=4)

    def _xyxy2xywh(self, bbox: List[float]) -> List[float]:
        """Convert xyxy to xywh."""
        return [
            bbox[0],
            bbox[1],
            round(bbox[2] - bbox[0], 2),
            round(bbox[3] - bbox[1], 2),
        ]

    def _xyxy2area(self, bbox: List[float]) -> float:
        """Calculate area from xyxy."""
        return round((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 2)


class CocoCategorieSchema(BaseModel):
    """COCO categories schema."""

    id: int = Field(..., example=1)
    name: str = Field(..., example="car")
    supercategory: str = Field(..., example="car")


class CocoImageSchema(BaseModel):
    """COCO image schema."""

    id: int = Field(..., example=1)
    width: int = Field(..., example=640)
    height: int = Field(..., example=480)
    file_name: str = Field(..., example="42069.jpg")


class CocoAnnotationSchema(BaseModel):
    """COCO annotation schema."""

    id: int = Field(..., example=1)
    image_id: int = Field(..., example=1)
    category_id: int = Field(..., example=1)
    segmentation: List[List[float]] = Field(..., example=[[1, 2, 3, 4]])
    area: float = Field(..., example=100.0)
    bbox: List[float] = Field(..., example=[1, 2, 3, 4])
    iscrowd: int = Field(..., example=0)
    attributes: Optional[KittiSchema] = Field(None)
