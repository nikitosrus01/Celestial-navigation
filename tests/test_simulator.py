"""Tests for camera simulator and synthetic image generation."""

import math
import numpy as np
import pytest

from src.catalog.catalog_builder import build_default_catalog
from src.simulation.camera_simulator import CameraParameters, CameraSimulator


def test_camera_parameters():
    params = CameraParameters(width=1920, height=1080, fov_x_deg=20.0)
    assert params.focal_length > 5000.0
    assert 10.0 < params.fov_y_deg < 20.0
    K = params.intrinsic_matrix
    assert K.shape == (3, 3)
    assert K[0, 2] == 960.0
    assert K[1, 2] == 540.0


def test_projection_and_inversion():
    params = CameraParameters()
    sim = CameraSimulator(params)

    # Point boresight at Vega
    catalog = build_default_catalog(total_stars=300)
    vega = catalog.get_star_by_id(91262)
    assert vega is not None

    R = sim.attitude_from_boresight(vega.vector)
    projected = sim.project_catalog(catalog, R)

    # Vega must be near image center (960, 540)
    vega_proj = [p for p in projected if p.star.star_id == 91262]
    assert len(vega_proj) == 1
    assert math.isclose(vega_proj[0].u_true, 960.0, abs_tol=1.0)
    assert math.isclose(vega_proj[0].v_true, 540.0, abs_tol=1.0)

    # Check that cam_vector aligns with K_inv * [u, v, 1]
    pixel_homo = np.array([vega_proj[0].u_true, vega_proj[0].v_true, 1.0])
    ray = sim.K_inv @ pixel_homo
    ray_unit = ray / np.linalg.norm(ray)
    assert np.allclose(ray_unit, vega_proj[0].cam_vector, atol=1e-5)


def test_render_image():
    params = CameraParameters(width=640, height=480, fov_x_deg=20.0)
    sim = CameraSimulator(params)
    catalog = build_default_catalog(total_stars=200)

    R = sim.generate_random_attitude(seed=42)
    projected = sim.project_catalog(catalog, R)

    img = sim.render_image(projected, add_noise=True, seed=42)
    assert img.shape == (480, 640)
    assert img.dtype == np.uint8
    assert img.mean() > 5.0  # Background offset
