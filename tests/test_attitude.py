"""Tests for attitude determination (Wahba's problem) and star matching."""

import math
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.catalog.catalog_builder import build_default_catalog
from src.catalog.k_vector import StarPairDatabase
from src.detection.centroiding import DetectedStar
from src.matching.attitude_svd import compute_attitude_error, solve_attitude, solve_wahba_svd
from src.matching.triangle_matcher import TriangleMatcher
from src.simulation.camera_simulator import CameraParameters, CameraSimulator


def test_wahba_svd_noiseless():
    rng = np.random.default_rng(42)
    # Generate random true attitude
    R_true = Rotation.random(random_state=rng).as_matrix()

    # Generate 15 random inertial vectors
    inertial_v = rng.normal(size=(15, 3))
    inertial_v /= np.linalg.norm(inertial_v, axis=1, keepdims=True)

    # Transform into camera frame
    cam_b = (R_true @ inertial_v.T).T

    # Solve Wahba
    R_calc = solve_wahba_svd(cam_b, inertial_v)

    # Det must be +1
    assert np.isclose(np.linalg.det(R_calc), 1.0, atol=1e-6)

    # Total error must be zero (within float64 precision ~ 0.01 arcsec)
    tot_rad, tot_arcsec, bore_arcsec, roll_arcsec = compute_attitude_error(R_calc, R_true)
    assert tot_arcsec < 0.05, f"Noiseless error too large: {tot_arcsec} arcsec"


def test_wahba_svd_noisy():
    rng = np.random.default_rng(123)
    R_true = Rotation.random(random_state=rng).as_matrix()

    inertial_v = rng.normal(size=(10, 3))
    inertial_v /= np.linalg.norm(inertial_v, axis=1, keepdims=True)
    cam_b = (R_true @ inertial_v.T).T

    # Add 5 arcsec jitter noise to camera vectors
    noise_rad = math.radians(5.0 / 3600.0)
    jitter = rng.normal(0.0, noise_rad, size=cam_b.shape)
    cam_b_noisy = cam_b + jitter
    cam_b_noisy /= np.linalg.norm(cam_b_noisy, axis=1, keepdims=True)

    sol = solve_attitude(cam_b_noisy, inertial_v, R_true=R_true)

    assert np.isclose(np.linalg.det(sol.R_calc), 1.0, atol=1e-6)
    assert sol.error_angle_arcsec is not None
    assert sol.error_angle_arcsec < 15.0, f"Error exceeded expectation: {sol.error_angle_arcsec} arcsec"


def test_lost_in_space_matching():
    catalog = build_default_catalog(total_stars=800, max_magnitude=5.5)
    db = StarPairDatabase(catalog, max_fov_rad=math.radians(25.0))
    matcher = TriangleMatcher(catalog, db, angular_tolerance_deg=0.15)

    params = CameraParameters(fov_x_deg=22.0)
    sim = CameraSimulator(params)

    # Choose a pointing direction rich in bright navigation stars (e.g. Orion: Betelgeuse & Rigel)
    betelgeuse = catalog.get_star_by_id(27989)
    assert betelgeuse is not None
    R_true = sim.attitude_from_boresight(betelgeuse.vector, roll_angle_rad=0.3)

    projected = sim.project_catalog(catalog, R_true)
    assert len(projected) >= 5, f"Not enough stars projected: {len(projected)}"

    # Convert projected stars into DetectedStar objects
    detected = []
    for idx, p in enumerate(projected):
        detected.append(
            DetectedStar(
                u=p.u_true,
                v=p.v_true,
                flux=p.flux,
                area=10.0,
                cam_vector=p.cam_vector,
            )
        )

    # Run Lost in Space matcher
    matches = matcher.match_stars(detected)
    assert len(matches) >= 4, f"Expected at least 4 matched stars, got {len(matches)}"

    # Verify that all matches are true positive identifications
    for m in matches:
        expected_star_id = projected[m.detected_idx].star.star_id
        assert m.catalog_star_id == expected_star_id, (
            f"False match: matched {m.catalog_star_id} instead of {expected_star_id}"
        )

    # Solve attitude from matches
    cam_vecs = np.array([m.cam_vector for m in matches])
    inertial_vecs = np.array([m.inertial_vector for m in matches])
    sol = solve_attitude(cam_vecs, inertial_vecs, R_true=R_true)

    assert sol.error_angle_arcsec is not None
    assert sol.error_angle_arcsec < 1.0, f"Attitude error: {sol.error_angle_arcsec} arcsec"
