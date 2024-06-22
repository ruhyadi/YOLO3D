"""Schema module."""

import rootutils

ROOT = rootutils.autosetup()

from typing import List, Optional

from pydantic import BaseModel, Field


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
        None,
        example=0.0,
        description="Only for results: Float, indicating confidence in detection, needed for p/r curves, higher is better",
    )


class CocoSchema(BaseModel):
    """COCO schema."""

    categories: List["CocoCategorieSchema"] = Field([])
    images: List["CocoImageSchema"] = Field([])
    annotations: List["CocoAnnotationSchema"] = Field([])


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
