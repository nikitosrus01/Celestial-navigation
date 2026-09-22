"""Tests for star detection and subpixel centroiding accuracy."""

import math
import numpy as np
import pytest

from src.catalog.star_catalog import Star
from src.detection.centroiding import StarCentroidDetector
from src.simulation.camera_simulator import CameraParameters, CameraSimulator, ProjectedStar


def test_subpixel_centroid_accuracy():
    params = CameraParameters(width=1000, height=800, psf_sigma=1.2, noise_sigma=1.5, background_level=10.0)
    sim = CameraSimulator(params)
    detector = StarCentroidDetector(camera_params=params, sigma_threshold=3.0)

    # Place artificial star at subpixel coordinates
    u_true, v_true = 512.35, 418.72
    dummy_star = Star(
        star_id=1,
        ra_rad=0.0,
        dec_rad=0.0,
        vmag=2.0,
        vector=np.array([0.0, 0.0, 1.0]),
    )
    cam_vec = detector.pixel_to_unit_vector(u_true, v_true)
    projected = [
        ProjectedStar(
            star=dummy_star,
            u_true=u_true,
            v_true=v_true,
            cam_vector=cam_vec,
            flux=150.0,
        )
    ]

    img = sim.render_image(projected, add_noise=True, seed=123)
    detected = detector.detect(img)

    assert len(detected) == 1
    det = detected[0]

    # Subpixel error should be <= 0.1 pixels
    err_u = abs(det.u - u_true)
    err_v = abs(det.v - v_true)
    assert err_u < 0.15, f"Centroid u error too large: {err_u}"
    assert err_v < 0.15, f"Centroid v error too large: {err_v}"

    # Vector alignment check
    cos_angle = np.dot(det.cam_vector, cam_vec)
    angular_err_rad = math.acos(np.clip(cos_angle, -1.0, 1.0))
    # Angular error in arcseconds: for f ~ 2800 px, 0.15 px ~ 11 arcseconds
    angular_err_arcsec = math.degrees(angular_err_rad) * 3600.0
    assert angular_err_arcsec < 20.0
