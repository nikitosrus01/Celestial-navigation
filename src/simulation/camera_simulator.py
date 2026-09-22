"""Synthetic sky image generator and camera model (Ground Truth Generator).

Simulates camera optics, pinhole projection, Point Spread Function (PSF),
and sensor noise for Star Tracker testing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from src.catalog.star_catalog import Star, StarCatalog


@dataclass
class CameraParameters:
    """Optical and sensor specifications for the star tracker camera."""

    width: int = 1920  # Sensor width in pixels
    height: int = 1080  # Sensor height in pixels
    fov_x_deg: float = 20.0  # Horizontal field of view in degrees
    psf_sigma: float = 1.2  # PSF Gaussian blur standard deviation (pixels)
    noise_sigma: float = 2.5  # Readout noise standard deviation (DN)
    background_level: float = 8.0  # Background dark offset level (DN)
    saturation_flux: float = 255.0  # Saturation peak intensity

    @property
    def focal_length(self) -> float:
        """Compute focal length in pixels based on width and horizontal FOV."""
        fov_x_rad = math.radians(self.fov_x_deg)
        return (self.width / 2.0) / math.tan(fov_x_rad / 2.0)

    @property
    def fov_y_deg(self) -> float:
        """Vertical field of view in degrees."""
        fov_y_rad = 2.0 * math.atan((self.height / 2.0) / self.focal_length)
        return math.degrees(fov_y_rad)

    @property
    def diagonal_fov_rad(self) -> float:
        """Diagonal field of view in radians."""
        diag_px = math.hypot(self.width, self.height)
        return 2.0 * math.atan((diag_px / 2.0) / self.focal_length)

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """3x3 Camera calibration intrinsic matrix K."""
        f = self.focal_length
        cx = self.width / 2.0
        cy = self.height / 2.0
        return np.array(
            [
                [f, 0.0, cx],
                [0.0, f, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


@dataclass
class ProjectedStar:
    """Information about a star projected onto the camera sensor."""

    star: Star
    u_true: float  # True pixel x coordinate
    v_true: float  # True pixel y coordinate
    cam_vector: np.ndarray  # 3D unit LOS vector in camera frame [Xc, Yc, Zc]
    flux: float  # Calculated peak intensity


class CameraSimulator:
    """Simulates realistic star field images given an attitude rotation matrix."""

    def __init__(self, camera_params: Optional[CameraParameters] = None):
        self.params = camera_params or CameraParameters()
        self.K = self.params.intrinsic_matrix
        self.K_inv = np.linalg.inv(self.K)

    @staticmethod
    def generate_random_attitude(seed: Optional[int] = None) -> np.ndarray:
        """Generate a random 3x3 rotation matrix R_true in SO(3)."""
        rng = np.random.default_rng(seed)
        # Uniform sampling via random quaternion
        rot = Rotation.random(random_state=rng)
        return rot.as_matrix()

    @staticmethod
    def attitude_from_boresight(boresight_inertial: np.ndarray, roll_angle_rad: float = 0.0) -> np.ndarray:
        """Construct a rotation matrix R_true such that camera optical axis (+Z_cam) points to boresight."""
        z_c = boresight_inertial / np.linalg.norm(boresight_inertial)
        # Pick arbitrary non-parallel reference vector to build coordinate frame
        ref = np.array([0.0, 0.0, 1.0]) if abs(z_c[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        x_c = np.cross(ref, z_c)
        x_c /= np.linalg.norm(x_c)
        y_c = np.cross(z_c, x_c)
        y_c /= np.linalg.norm(y_c)

        # Apply roll around z_c
        cos_r = math.cos(roll_angle_rad)
        sin_r = math.sin(roll_angle_rad)
        x_c_rolled = cos_r * x_c + sin_r * y_c
        y_c_rolled = -sin_r * x_c + cos_r * y_c

        # R transforms from inertial to camera: [x_c, y_c, z_c]^T
        R = np.vstack([x_c_rolled, y_c_rolled, z_c])
        return R

    def project_catalog(
        self,
        catalog: StarCatalog,
        r_inertial_to_camera: np.ndarray,
    ) -> List[ProjectedStar]:
        """Project stars from catalog into camera sensor frame.

        Args:
            catalog: Inertial StarCatalog.
            r_inertial_to_camera: 3x3 rotation matrix R (r_cam = R @ r_inertial).

        Returns:
            List[ProjectedStar]: Stars that fall within sensor bounds.
        """
        w = self.params.width
        h = self.params.height
        f = self.params.focal_length
        cx = w / 2.0
        cy = h / 2.0

        # Rotate all catalog vectors into camera frame: (N, 3) = (3, 3) @ (3, N) -> (N, 3)
        cam_vectors = (r_inertial_to_camera @ catalog.vectors.T).T

        # Visible stars must have Z_c > 0 (in front of camera)
        front_mask = cam_vectors[:, 2] > 0.001
        front_indices = np.where(front_mask)[0]

        projected: List[ProjectedStar] = []

        for idx in front_indices:
            v_c = cam_vectors[idx]
            zc = v_c[2]
            u = f * (v_c[0] / zc) + cx
            v = f * (v_c[1] / zc) + cy

            # Check if star falls within sensor boundaries (with small margin)
            if 0 <= u < w and 0 <= v < h:
                star = catalog.stars[idx]
                # Stellar flux mapping: Pogson's law
                # Star of mag 1.0 -> ~200 DN, mag 5.0 -> ~30 DN
                rel_brightness = 10.0 ** (-0.4 * (star.vmag - 1.0))
                flux = np.clip(180.0 * rel_brightness, 20.0, self.params.saturation_flux)

                # Normalized unit vector in camera frame
                v_c_unit = v_c / np.linalg.norm(v_c)

                projected.append(
                    ProjectedStar(
                        star=star,
                        u_true=float(u),
                        v_true=float(v),
                        cam_vector=v_c_unit,
                        flux=float(flux),
                    )
                )

        return projected

    def render_image(
        self,
        projected_stars: List[ProjectedStar],
        add_noise: bool = True,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Render a synthetic 8-bit grayscale image of the star field.

        Models optical Point Spread Function (PSF) as Gaussian profile for each star,
        plus background offset and sensor noise.

        Args:
            projected_stars: List of projected stars within the FOV.
            add_noise: Whether to add sensor read and shot noise.
            seed: Optional seed for reproducible noise.

        Returns:
            np.ndarray: Grayscale image with dtype uint8 of shape (height, width).
        """
        w = self.params.width
        h = self.params.height
        sigma = self.params.psf_sigma

        # Initialize image with background level in float32
        image = np.full((h, w), self.params.background_level, dtype=np.float32)

        # Radius of PSF stamp (approx 4 * sigma)
        kernel_radius = int(math.ceil(4.0 * sigma))

        for p_star in projected_stars:
            u0 = p_star.u_true
            v0 = p_star.v_true
            peak_val = p_star.flux

            # Bounding box on sensor
            x_min = max(0, int(math.floor(u0 - kernel_radius)))
            x_max = min(w, int(math.ceil(u0 + kernel_radius + 1)))
            y_min = max(0, int(math.floor(v0 - kernel_radius)))
            y_max = min(h, int(math.ceil(v0 + kernel_radius + 1)))

            if x_min >= x_max or y_min >= y_max:
                continue

            # Grid coordinates for PSF evaluation
            ys, xs = np.mgrid[y_min:y_max, x_min:x_max]
            dist_sq = (xs - u0) ** 2 + (ys - v0) ** 2
            psf_patch = peak_val * np.exp(-0.5 * dist_sq / (sigma**2))

            image[y_min:y_max, x_min:x_max] += psf_patch

        if add_noise:
            rng = np.random.default_rng(seed)
            # Readout Gaussian noise
            noise = rng.normal(0.0, self.params.noise_sigma, size=(h, w)).astype(np.float32)
            image += noise

        # Clip intensity to [0, 255] and convert to uint8
        image_uint8 = np.clip(image, 0.0, 255.0).astype(np.uint8)
        return image_uint8
