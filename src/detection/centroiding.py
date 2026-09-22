"""Star spot detection and subpixel centroiding module using OpenCV and image moments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.simulation.camera_simulator import CameraParameters


@dataclass
class DetectedStar:
    """A detected star spot with subpixel coordinates and unit LOS vector."""

    u: float  # Subpixel x coordinate
    v: float  # Subpixel y coordinate
    flux: float  # Estimated total flux / intensity
    area: float  # Spot area in pixels
    cam_vector: np.ndarray  # 3D normalized Line-Of-Sight vector in camera frame [Xc, Yc, Zc]


class StarCentroidDetector:
    """OpenCV-based star detection and subpixel centroid estimation pipeline."""

    def __init__(
        self,
        camera_params: Optional[CameraParameters] = None,
        sigma_threshold: float = 3.5,
        min_area: float = 2.0,
        max_area: float = 120.0,
        window_size: int = 7,
    ):
        """Initialize detector.

        Args:
            camera_params: Camera parameters with intrinsic matrix K.
            sigma_threshold: Number of standard deviations above background for thresholding.
            min_area: Minimum contour area to reject single-pixel noise.
            max_area: Maximum contour area to reject extended objects or stray light.
            window_size: Local bounding window for intensity-weighted moment calculation.
        """
        self.camera_params = camera_params or CameraParameters()
        self.K = self.camera_params.intrinsic_matrix
        self.K_inv = np.linalg.inv(self.K)
        self.sigma_threshold = sigma_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.window_size = window_size

    def pixel_to_unit_vector(self, u: float, v: float) -> np.ndarray:
        """Convert pixel coordinates (u, v) to a 3D unit LOS vector in camera frame.

        b = K^(-1) * [u, v, 1]^T / ||K^(-1) * [u, v, 1]^T||
        """
        homo = np.array([u, v, 1.0], dtype=np.float64)
        ray = self.K_inv @ homo
        norm = np.linalg.norm(ray)
        return ray / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])

    def detect(self, image: np.ndarray) -> List[DetectedStar]:
        """Detect stars and estimate subpixel centroids and LOS vectors.

        Pipeline:
            1. Estimate background mean and noise std.
            2. Compute binary mask using adaptive statistical threshold (mu + k*sigma).
            3. Find connected components / contours using OpenCV.
            4. Filter candidate regions by size.
            5. Compute subpixel centroid using image moments over background-subtracted patch.
            6. Compute calibrated 3D unit LOS vector.

        Args:
            image: Grayscale image (uint8 or float32).

        Returns:
            List[DetectedStar]: Detected stars sorted by flux descending.
        """
        img_f = image.astype(np.float32)
        h, w = img_f.shape[:2]

        # 1. Background statistical estimation
        bg_mean = float(np.mean(img_f))
        bg_std = float(np.std(img_f))
        threshold_val = bg_mean + self.sigma_threshold * bg_std

        # 2. Binary thresholding & noise cleanup
        _, binary = cv2.threshold(
            img_f,
            threshold_val,
            255.0,
            cv2.THRESH_BINARY,
        )
        binary_uint8 = binary.astype(np.uint8)

        # 3. Contour detection
        contours, _ = cv2.findContours(
            binary_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        detected_stars: List[DetectedStar] = []
        half_win = self.window_size // 2

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Some small valid spots might have contourArea == 0 if 1-2 px, check bounding rect
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            box_area = rw * rh

            if box_area < self.min_area or box_area > self.max_area:
                continue

            # Approximate center from contour moments or bounding box center
            m_cnt = cv2.moments(cnt)
            if m_cnt["m00"] > 1e-6:
                init_u = m_cnt["m10"] / m_cnt["m00"]
                init_v = m_cnt["m01"] / m_cnt["m00"]
            else:
                init_u = rx + rw / 2.0
                init_v = ry + rh / 2.0

            # 4. Extract local intensity patch for refined subpixel centroiding
            u_int = int(round(init_u))
            v_int = int(round(init_v))

            x0 = max(0, u_int - half_win)
            x1 = min(w, u_int + half_win + 1)
            y0 = max(0, v_int - half_win)
            y1 = min(h, v_int + half_win + 1)

            patch = img_f[y0:y1, x0:x1]
            # Subtract local background to avoid centroid pulling toward background
            patch_sub = np.maximum(0.0, patch - bg_mean)

            total_flux = float(np.sum(patch_sub))
            if total_flux <= 1e-6:
                continue

            # Compute intensity-weighted moments
            grid_y, grid_x = np.mgrid[y0:y1, x0:x1]
            sub_u = float(np.sum(grid_x * patch_sub) / total_flux)
            sub_v = float(np.sum(grid_y * patch_sub) / total_flux)

            # 5. Extract calibrated unit vector
            cam_vec = self.pixel_to_unit_vector(sub_u, sub_v)

            detected_stars.append(
                DetectedStar(
                    u=sub_u,
                    v=sub_v,
                    flux=total_flux,
                    area=float(box_area),
                    cam_vector=cam_vec,
                )
            )

        # Sort stars by flux in descending order (brightest first)
        detected_stars.sort(key=lambda s: s.flux, reverse=True)
        return detected_stars
