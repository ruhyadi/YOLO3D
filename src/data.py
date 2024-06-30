"""Data module"""

"""Data module"""

import rootutils

ROOT = rootutils.autosetup()

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

from src.schema import KittiDimsAvg, KittiSchema, MultibinDataSchema
from src.utils import get_logger

log = get_logger()


class MultibinsData(Dataset):
    """ "Multbins dataloader."""

    def __init__(
        self,
        images_dir: str = "data/kitti/images_2",
        labels_dir: str = "data/kitti/label_2",
        calib_path: str = "assets/kitti_calib.txt",
        sets_path: str = "assets/kitti_sets.txt",
        categories: List[str] = ["car", "pedestrian", "cyclist"],
        n_bins: int = 2,
        bin_overlap: float = 0.5,
    ) -> None:
        """Initialize Multibins dataset."""
        super().__init__()
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.calib_path = Path(calib_path)
        self.sets_path = Path(sets_path)
        self.categories = categories
        self.n_bins = n_bins
        self.bin_overlap = bin_overlap

        # computer average dimensions
        self.dims_avg = KittiDimsAvg()
        self.dims_avg.load(self.labels_dir, self.sets_path)
        self.dims_avg.generate_json("data/kitti/kitti_dims_avg.json")

        # image transforms
        self.transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        # setup multibins data
        self.multibins_data = self._setup_multibins_data()

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.multibins_data)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a sample from the dataset."""
        sample = self.multibins_data[index]
        image = self._load_image(sample.index, box=sample.box)

        # to tensor
        ori = torch.tensor(sample.orientation, dtype=torch.float32)
        conf = torch.tensor(sample.confidence, dtype=torch.float32)
        dims = torch.tensor(sample.dimensions, dtype=torch.float32)

        return image, ori, conf, dims

    def _load_image(self, index: str, box: List[int]) -> torch.Tensor:
        """Get image of object (crop) from image directory."""
        img_path = self.images_dir / f"{index}.png"
        image = Image.open(img_path)
        image = image.crop(box=box)

        return self.transforms(image)

    def _setup_multibins_data(self) -> List[MultibinDataSchema]:
        """Setup data for multibins training."""
        log.info(f"Setup multibin data format...")

        # read indexes from sets file
        indexes = self.sets_path.read_text().splitlines()

        # iterate over indexes
        multibins_data: List[MultibinDataSchema] = []
        for i in tqdm(indexes, desc="KITTI to Multibin"):
            label_path = self.labels_dir / f"{i}.txt"
            lines = label_path.read_text().splitlines()
            for line in lines:
                kitti = KittiSchema()
                kitti.load(line.split())

                # skip if category not in categories
                if kitti.type not in self.categories:
                    continue

                # convert kitti to multibin schema
                multibins = self._kitti_to_multibin(i, kitti)
                multibins_data.append(multibins)

        return multibins_data

    def _kitti_to_multibin(self, index: str, kitti: KittiSchema) -> MultibinDataSchema:
        """Convert KITTI schema to Multibin schema."""
        # convert alpha to [0, 2pi]
        alpha = self._get_alpha(kitti.alpha)
        # get average dimensions
        dims = np.array(kitti.dimensions) - np.array(self.dims_avg[kitti.type])
        dims = dims.tolist()
        # get orientation and confidence of bins
        ori, conf = self._get_ori_conf(alpha)

        return MultibinDataSchema(
            index=index,
            category=kitti.type,
            box=kitti.bbox,
            alpha=alpha,
            dimensions=dims,
            orientation=ori.tolist(),
            confidence=conf.tolist(),
        )

    def _get_alpha(self, alpha: float) -> float:
        """Convert alpha value from [-pi, pi] to [0, 2pi]."""
        new_alpha = float(alpha) + np.pi / 2.0
        if new_alpha < 0:
            new_alpha = new_alpha + 2.0 * np.pi
            # make sure angle lies in [0, 2pi]
        new_alpha = new_alpha - int(new_alpha / (2.0 * np.pi)) * (2.0 * np.pi)

        return new_alpha

    def _get_ori_conf(self, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get multibin orientation and confidence."""
        ori = np.zeros((self.n_bins, 2))
        conf = np.zeros(self.n_bins)
        anchors = self._get_anchors(alpha, self.n_bins, self.bin_overlap)

        for anchor in anchors:
            ori[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
            conf[anchor[0]] = 1.0
        conf = conf / np.sum(conf)

        return ori, conf

    def _get_anchors(self, angle: float, n_bins: int, overlap: float) -> list:
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
