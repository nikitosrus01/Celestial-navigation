"""Attitude determination solver for Wahba's problem using Kabsch / SVD algorithm.

References:
    - Markley, F. L. (1988). "Attitude Determination Using Vector Observations: A Fast Optimal Matrix Algorithm"
    - Kabsch, W. (1976). "A solution for the best rotation to relate two sets of vectors"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class AttitudeSolution:
    """Estimated attitude orientation and error statistics."""

    R_calc: np.ndarray  # 3x3 rotation matrix (inertial to camera)
    quaternion: np.ndarray  # [w, x, y, z] scalar-first quaternion
    euler_angles_deg: np.ndarray  # [roll, pitch, yaw] in degrees
    error_angle_rad: Optional[float] = None  # Total rotation error in radians (if GT available)
    error_angle_arcsec: Optional[float] = None  # Total rotation error in arcseconds
    boresight_error_arcsec: Optional[float] = None  # Transverse / pointing axis error in arcseconds
    roll_error_arcsec: Optional[float] = None  # Error around boresight axis in arcseconds
    num_matched_stars: int = 0
    residual_rms_rad: float = 0.0


def solve_wahba_svd(
    cam_vectors: np.ndarray,
    inertial_vectors: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Solve Wahba's problem using Kabsch / Markley SVD algorithm.

    Finds R in SO(3) that minimizes:
        sum_i w_i * ||b_i - R * v_i||^2

    Args:
        cam_vectors: (N, 3) measured unit vectors in camera frame.
        inertial_vectors: (N, 3) reference unit vectors in inertial frame.
        weights: Optional (N,) array of positive weights. Defaults to uniform weights.

    Returns:
        np.ndarray: (3, 3) optimal rotation matrix R mapping inertial to camera.
    """
    b = np.asarray(cam_vectors, dtype=np.float64)
    v = np.asarray(inertial_vectors, dtype=np.float64)
    n = len(b)

    if n < 2:
        raise ValueError(f"At least 2 non-collinear vector observations required, got {n}")

    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)

    # 1. Compute attitude profile / cross-dispersion matrix B = sum(w_i * b_i * v_i^T)
    # Shape: (3, 3) = (3, N) @ (N, 3)
    B = (b.T * w) @ v

    # 2. Singular Value Decomposition
    U, S, Vt = np.linalg.svd(B)

    # 3. Ensure proper rotation (det(R) = +1, no reflection)
    d = np.linalg.det(U @ Vt)
    M = np.diag([1.0, 1.0, np.sign(d)])

    # R_opt = U @ M @ Vt
    R_opt = U @ M @ Vt
    return R_opt


def compute_attitude_error(
    R_calc: np.ndarray,
    R_true: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Compute angular error between estimated and true rotation matrices.

    Args:
        R_calc: Estimated 3x3 rotation matrix (inertial to camera).
        R_true: True 3x3 rotation matrix (inertial to camera).

    Returns:
        Tuple[float, float, float, float]:
            (total_err_rad, total_err_arcsec, boresight_err_arcsec, roll_err_arcsec)
    """
    # Relative rotation Delta_R: R_calc @ R_true^T
    delta_R = R_calc @ R_true.T

    # Total angle of rotation: Tr(delta_R) = 1 + 2*cos(phi)
    tr = float(np.trace(delta_R))
    cos_phi = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    phi_rad = float(np.arccos(cos_phi))
    phi_arcsec = math.degrees(phi_rad) * 3600.0

    # Optical axis (boresight) is Z_cam = [0, 0, 1]^T
    # True boresight in inertial frame: v_z_true = R_true.T @ [0, 0, 1]
    # Calc boresight in inertial frame: v_z_calc = R_calc.T @ [0, 0, 1]
    vz_true = R_true.T[:, 2]
    vz_calc = R_calc.T[:, 2]
    cos_boresight = np.clip(np.dot(vz_true, vz_calc), -1.0, 1.0)
    boresight_err_rad = float(np.arccos(cos_boresight))
    boresight_err_arcsec = math.degrees(boresight_err_rad) * 3600.0

    # Roll error (around boresight)
    # phi_total^2 approx boresight_err^2 + roll_err^2
    roll_err_arcsec = math.sqrt(max(0.0, phi_arcsec**2 - boresight_err_arcsec**2))

    return phi_rad, phi_arcsec, boresight_err_arcsec, roll_err_arcsec


def solve_attitude(
    cam_vectors: np.ndarray,
    inertial_vectors: np.ndarray,
    weights: Optional[np.ndarray] = None,
    R_true: Optional[np.ndarray] = None,
) -> AttitudeSolution:
    """Solve attitude and return full AttitudeSolution object with diagnostics.

    Args:
        cam_vectors: (N, 3) camera frame LOS vectors.
        inertial_vectors: (N, 3) inertial catalog vectors.
        weights: Optional weights for each star.
        R_true: Optional ground truth rotation matrix for performance evaluation.

    Returns:
        AttitudeSolution: Complete solution object.
    """
    R_calc = solve_wahba_svd(cam_vectors, inertial_vectors, weights)

    # Convert to quaternion [w, x, y, z]
    rot = Rotation.from_matrix(R_calc)
    # scipy returns [x, y, z, w], we standardize to [w, x, y, z]
    q_xyzw = rot.as_quat()
    quat = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)

    # Euler angles in degrees (roll, pitch, yaw - ZYX or XYZ)
    euler = rot.as_euler("xyz", degrees=True)

    # Compute residual RMS: ||b_i - R * v_i||
    predicted_b = (R_calc @ inertial_vectors.T).T
    residuals = np.linalg.norm(cam_vectors - predicted_b, axis=1)
    rms_rad = float(np.sqrt(np.mean(residuals**2)))

    solution = AttitudeSolution(
        R_calc=R_calc,
        quaternion=quat,
        euler_angles_deg=euler,
        num_matched_stars=len(cam_vectors),
        residual_rms_rad=rms_rad,
    )

    if R_true is not None:
        tot_rad, tot_arcsec, bore_arcsec, roll_arcsec = compute_attitude_error(R_calc, R_true)
        solution.error_angle_rad = tot_rad
        solution.error_angle_arcsec = tot_arcsec
        solution.boresight_error_arcsec = bore_arcsec
        solution.roll_error_arcsec = roll_arcsec

    return solution
