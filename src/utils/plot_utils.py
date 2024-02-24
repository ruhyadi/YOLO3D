"""YOLO3D plotting utils with matplotlib."""

import rootutils

ROOT = rootutils.autosetup()

from typing import List

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path

from src.utils.logger import get_logger

log = get_logger()


class Plot3DBox:
    """3D bounding box plotting module."""

    def __init__(
        self,
        proj_matrix: np.ndarray,
        classes: List[str] = ["Car", "Pedestrian", "Cyclist"],
    ) -> None:
        """
        Initialize 3D bounding box plotting module.

        Args:
            proj_matrix (np.ndarray): projection matrix P2
            classes (list, optional): list of classes
        """
        self.proj_matrix = proj_matrix
        self.classes = [c.lower() for c in classes]

        self.fig = plt.figure(figsize=(20, 5.12))
        gs = GridSpec(1, 4)
        gs.update(wspace=0)
        self.ax1 = self.fig.add_subplot(gs[0, :3])
        self.ax2 = self.fig.add_subplot(gs[0, 3:])

        self.shape = 900
        self.scale = 15

        self.color = {
            "car": "red",
            "pedestrian": "blue",
            "cyclist": "green",
        }

    def save_plot(self, save_path: str) -> None:
        """Save plot."""
        self.fig.savefig(save_path, bbox_inches="tight", pad_inches=0)

        # crop saved image to (1550 x 409) from bottom right
        width, height = 1556, 409
        img = cv2.imread(str(save_path))
        new_width = np.clip(img.shape[1] - width, 0, img.shape[1])
        new_height = np.clip(img.shape[0] - height, 0, img.shape[0])
        img = img[new_height:, new_width:]
        cv2.imwrite(str(save_path), img)

    def cleanup(self) -> None:
        """Cleanup plot."""
        self.ax1.clear()
        self.ax2.clear()

    def plot(
        self,
        img: np.ndarray,
        bbox: List[int],
        dims: List[float],
        loc: List[float],
        rot_y: float,
        category: str = "car",
        index: int = None,
    ):
        """Plot 3D bounding box."""
        # initialize bev canvas
        bev_img = np.zeros((self.shape, self.shape, 3), dtype=np.uint8)

        # draw 3D bounding box
        self.draw_3dbox(bbox, dims, loc, rot_y, category, index)
        self.draw_bev(dims, loc, rot_y, index)

        # visualize 3d bounding box
        self.ax1.imshow(img)
        self.ax1.set_title("3D Bounding Box")
        self.ax1.set_xticks([])
        self.ax1.set_yticks([])

        # plot camera view range
        x1 = np.linspace(0, self.shape / 2)
        x2 = np.linspace(self.shape / 2, self.shape)
        self.ax2.plot(
            x1, self.shape / 2 - x1, ls="--", color="grey", linewidth=1, alpha=0.5
        )
        self.ax2.plot(
            x2, x2 - self.shape / 2, ls="--", color="grey", linewidth=1, alpha=0.5
        )
        self.ax2.plot(
            self.shape / 2,
            0,
            marker="+",
            color="grey",
            markersize=16,
            markeredgecolor="red",
        )

        # visualize bev
        self.ax2.imshow(bev_img, origin="lower")
        self.ax2.set_title("Bird's Eye View")
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])

        # add legend
        # handles, labels = ax2.get_legend_handles_labels()
        # legend = ax2.legend(
        #     [handles[0], handles[1]],
        #     [labels[0], labels[1]],
        #     loc="lower right",
        #     fontsize="x-small",
        #     framealpha=0.2,
        # )
        # for text in legend.get_texts():
        #     plt.setp(text, color="w")

    def draw_3dbox(
        self,
        bbox: List[int],
        dims: List[float],
        loc: List[float],
        rot_y: float,
        category: str = "car",
        index: int = None,
    ):
        """
        Draw 3D bounding box on image plane (2D)

        Args:
            bbox (list): 2D bounding box
            dims (list): 3D bounding box dimensions
            loc (list): 3D bounding box location
            rot_y (float): 3D bounding box rotation
            category (str, optional): category of 3D bounding box
        """
        corners_2d = self.compute_3d_corners(bbox, dims, loc, rot_y)

        # draw 3D bounding box
        lines_vertices_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]
        lines_vertices = corners_2d[:, lines_vertices_idx]
        vertices = lines_vertices.T
        codes = [Path.LINETO] * vertices.shape[0]
        codes[0] = Path.MOVETO
        path = Path(vertices, codes)
        patch = patches.PathPatch(
            path, fill=False, color=self.color[category], linewidth=2
        )

        # fill front box
        width = corners_2d[:, 3][0] - corners_2d[:, 1][0]
        height = corners_2d[:, 2][1] - corners_2d[:, 1][1]
        front_fill = patches.Rectangle(
            (corners_2d[:, 1]),
            width,
            height,
            color=self.color[category],
            fill=True,
            alpha=0.4,
        )
        self.ax1.add_patch(patch)
        self.ax1.add_patch(front_fill)

        # draw text of location, dimension, and rotation
        label = f"Loc: ({loc[0]:.2f}, {loc[1]:.2f}, {loc[2]:.2f})\nDim: ({dims[0]:.2f}, {dims[1]:.2f}, {dims[2]:.2f})\nYaw: {rot_y:.2f}"
        if index:
            label = f"({index}): {label}"
        self.ax1.text(
            corners_2d[:, 1][0],
            corners_2d[:, 1][1],
            label,
            fontsize=8,
            color="white",
            bbox=dict(facecolor=self.color[category], alpha=0.4, pad=0.5),
        )

        # draw 3D bounding box corners
        # for i in range(8):
        #     scatter = patches.Circle(
        #         (corners_2d[:, i][0], corners_2d[:, i][1]),
        #         radius=5,
        #         color=self.color[category],
        #         fill=True,
        #     )
        #     self.ax1.add_patch(scatter)

    def draw_bev(
        self, dims: List[float], loc: List[float], rot_y: float, index: int = None
    ):
        """Draw bird's eye view."""
        pred_corners_2d = self.compute_bev(dims, loc, rot_y)

        codes = [Path.LINETO] * pred_corners_2d.shape[0]
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY
        path = Path(pred_corners_2d, codes)
        patch = patches.PathPatch(path, fill=False, color="green", linewidth=2)
        self.ax2.add_patch(patch)

        # draw z location of object
        label = f"z: {loc[2]:.2f}"
        if index:
            label = f"({index}): {label}"
        self.ax2.text(
            pred_corners_2d[0, 0],
            pred_corners_2d[0, 1],
            label,
            fontsize=8,
            color="white",
            bbox=dict(facecolor="green", alpha=0.4, pad=0.5),
        )

    def compute_3d_corners(
        self, bbox: List[int], dims: List[float], loc: List[float], rot_y: float
    ):
        """
        Compute 2D bounding box corners from 3D bounding box.
        Bounding box coordinate is:
            0: bottom front left corner
            1: bottom front right corner
            2: top front right corner
            3: top front left corner
            4: bottom back left corner
            5: bottom back right corner
            6: top back right corner
            7: top back left corner
        3D bounding box have shape: (8, 3)
        2D bounding box have shape: (8, 2)

        Args:
            bbox: 2D bounding box in image coordinate
            dims: 3D bounding box dimensions (height, width, length)
            loc: 3D bounding box location (x, y, z)
            rot_y: 3D bounding box rotation around y-axis

        Returns:
            corners_2d: 2D bounding box corners in image coordinate (8, 2)
        """
        h, w, l = dims
        x, y, z = loc

        # compute rotation matrix around yaw axis
        matrix_R = np.array(
            [
                [np.cos(rot_y), 0, np.sin(rot_y)],
                [0, 1, 0],
                [-np.sin(rot_y), 0, np.cos(rot_y)],
            ]
        )

        x_corners = [0, l, l, l, l, 0, 0, 0]
        y_corners = [0, 0, h, h, 0, 0, h, h]
        z_corners = [0, 0, 0, w, w, w, w, 0]

        x_corners = [x - l / 2 for x in x_corners]
        y_corners = [y - h / 2 for y in y_corners]
        z_corners = [z - w / 2 for z in z_corners]

        corners_3d = np.vstack([x_corners, y_corners, z_corners])
        corners_3d = np.dot(matrix_R, corners_3d)
        corners_3d += np.array([x, y, z]).reshape(3, 1)

        corners_3d_ = np.vstack([corners_3d, np.ones((corners_3d.shape[-1]))])
        corners_2d = np.dot(self.proj_matrix, corners_3d_)
        corners_2d = corners_2d / corners_2d[2]
        corners_2d = corners_2d[:2]

        return corners_2d

    def compute_bev(self, dims: List[float], loc: List[float], rot_y: float):
        """Compute bird's eye view."""
        h, w, l = [d * self.scale for d in dims]
        x, y, z = [d * self.scale for d in loc]
        rot_y = np.float64(rot_y)

        # compute rotation matrix around yaw axis
        matrix_R = np.array(
            [
                [-np.cos(rot_y), np.sin(rot_y)],
                [np.sin(rot_y), np.cos(rot_y)],
            ]
        )
        matrix_T = np.array([x, z]).reshape(2, 1)

        x_corners = [0, l, l, 0]
        y_corners = [0, 0, w, w]
        x_corners = [x - l / 2 for x in x_corners]
        y_corners = [y - w / 2 for y in y_corners]

        corners_2d = np.array([x_corners, y_corners])
        corners_2d = np.dot(matrix_R, corners_2d)
        corners_2d = matrix_T - corners_2d

        corners_2d[0] += int(self.shape / 2)
        corners_2d = corners_2d.astype(np.int16)
        corners_2d = corners_2d.T

        return np.vstack((corners_2d, corners_2d[0, :]))
