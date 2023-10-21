"""KITTI pytorch dataloader module."""

import rootutils

ROOT = rootutils.autosetup()

from pathlib import Path
from typing import List

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from src.schema.kitti_schema import (
    KittiDimensionsAvgSchema,
    KittiLabelSchema,
    MultibinLabelSchema,
)


class KittiDataLoader(Dataset):
    """
    KITTI dataoader for MultiBin model inherited from pytorch dataset.
    """

    def __init__(
        self,
        images_dir: str = "data/kitti/images",
        labels_dir: str = "data/kitti/labels",
        calib_path: str = "assets/kitti_calib.txt",
        sets_path: str = "assets/kitti_train.txt",
        classes: List[str] = ["Car", "Pedestrian", "Cyclist"],
        n_bins: int = 2,
        overlap: float = 0.5,
    ) -> None:
        """Initialize KITTI dataset."""
        super().__init__()
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.calib_path = Path(calib_path)
        self.sets_path = Path(sets_path)
        self.classes = classes
        self.n_bins = n_bins
        self.overlap = overlap

        # get kitti dimension average
        self.dim_avg = KittiDimensionsAvgSchema()
        self.dim_avg.load(self.labels_dir, self.sets_path)
        self.dim_avg.generate_json(path="assets/kitti_dimensions.json")

    def __len__(self) -> int:
        """Return the length of the dataset."""

        return 0

    def __getitem__(self, index: int):
        """Return a sample from the dataset."""

        return None

    def setup(self) -> List[MultibinLabelSchema]:
        """Setup KITTI dataset for MultiBin model."""

        # get indexes from kitti set file (train.txt, val.txt, test.txt)
        indexes = self.sets_path.read_text().splitlines()

        # iterate over indexes
        multibins: List[MultibinLabelSchema] = []
        for i in tqdm(indexes, desc="Processing data"):
            label_path = self.labels_dir / f"{i}.txt"
            lines = label_path.read_text().splitlines()
            for line in lines:
                kitti = KittiLabelSchema()
                kitti.load(label=line.split())

                # skip if category(type) is not in classes
                if kitti.type not in self.classes:
                    continue

                # convert kitti format to MultiBin format
                multibin = self.kitti_to_multibin(i, kitti)
                multibins.append(multibin)

        return multibins

    def get_alpha(self, alpha: float) -> float:
        """
        Convert alpha value from [-pi, pi] to [0, 2pi].
        """
        new_alpha = float(alpha) + np.pi / 2.0
        if new_alpha < 0:
            new_alpha = new_alpha + 2.0 * np.pi
            # make sure angle lies in [0, 2pi]
        new_alpha = new_alpha - int(new_alpha / (2.0 * np.pi)) * (2.0 * np.pi)

        return new_alpha

    def kitti_to_multibin(
        self, index: int, kitti: KittiLabelSchema
    ) -> MultibinLabelSchema:
        """Convert kitti format to MultiBin format."""
        # get new alpha and dimensions
        alpha = self.get_alpha(alpha=kitti.alpha)
        dims = kitti.dimensions - self.dim_avg[kitti.type]

        # compute orientation and confidence bin
        ori, conf = self.compute_ori_conf(alpha=alpha)

        return MultibinLabelSchema(
            index=index,
            category=kitti.type,
            bbox=kitti.bbox,
            alpha=alpha,
            dimensions=dims,
            orientation=ori,
            confidence=conf,
        )

    def compute_ori_conf(self, alpha: float):
        """Compute orientation and confidence bin."""
        ori = np.zeros((self.n_bins, 2))
        conf = np.zeros(self.n_bins)
        anchors = self.compute_anchors(alpha, self.n_bins, self.overlap)

        for anchor in anchors:
            ori[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
            conf[anchor[0]] = 1
        conf = conf / np.sum(conf)

        return ori, conf

    def compute_anchors(self, angle: float, n_bins: int, overlap: float) -> list:
        """Compute angle offset and which bin the angle lies in."""
        anchors = []

        wedge = 2.0 * np.pi / n_bins  # 2pi / bin = pi
        l_index = int(angle / wedge)  # angle/pi
        r_index = l_index + 1

        # (angle - l_index*pi) < pi/2 * 1.05 = 1.65
        if (angle - l_index * wedge) < wedge / 2 * (1 + overlap / 2):
            anchors.append([l_index, angle - l_index * wedge])

        # (r*pi + pi - angle) < pi/2 * 1.05 = 1.65
        if (r_index * wedge - angle) < wedge / 2 * (1 + overlap / 2):
            anchors.append([r_index % n_bins, angle - r_index * wedge])

        return anchors
