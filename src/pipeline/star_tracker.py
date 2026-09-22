"""End-to-end Star Tracker pipeline coordinating detection, matching, attitude determination, and visualization."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from src.catalog.k_vector import StarPairDatabase
from src.catalog.star_catalog import StarCatalog
from src.detection.centroiding import DetectedStar, StarCentroidDetector
from src.matching.attitude_svd import AttitudeSolution, solve_attitude
from src.matching.triangle_matcher import StarMatch, TriangleMatcher
from src.simulation.camera_simulator import CameraParameters, ProjectedStar


@dataclass
class StarTrackerResult:
    """Full telemetry output from a single star tracker solve cycle."""

    success: bool
    solution: Optional[AttitudeSolution]
    detected_stars: List[DetectedStar]
    matched_stars: List[StarMatch]
    execution_time_ms: float
    status_message: str


class StarTracker:
    """Autonomous Optical Star Tracker ('Lost in Space' attitude determination)."""

    def __init__(
        self,
        catalog: StarCatalog,
        camera_params: Optional[CameraParameters] = None,
        pair_db: Optional[StarPairDatabase] = None,
        angular_tolerance_deg: float = 0.12,
        min_inliers: int = 4,
    ):
        self.catalog = catalog
        self.camera_params = camera_params or CameraParameters()
        max_pair_fov_rad = self.camera_params.diagonal_fov_rad * 1.05
        self.pair_db = pair_db or StarPairDatabase(catalog, max_fov_rad=max_pair_fov_rad)
        self.detector = StarCentroidDetector(camera_params=self.camera_params)
        self.matcher = TriangleMatcher(
            catalog=self.catalog,
            pair_db=self.pair_db,
            angular_tolerance_deg=angular_tolerance_deg,
            min_inliers=min_inliers,
        )

    def process_frame(
        self,
        image: np.ndarray,
        R_true: Optional[np.ndarray] = None,
    ) -> StarTrackerResult:
        """Execute end-to-end attitude determination on a camera image.

        Args:
            image: Grayscale camera image (uint8 or float32).
            R_true: Optional ground truth rotation matrix for benchmark evaluation.

        Returns:
            StarTrackerResult: Status, attitude solution, matches, and telemetry.
        """
        t_start = time.perf_counter()

        # Step 1: Subpixel Star Detection & Centroiding
        detected_stars = self.detector.detect(image)
        if len(detected_stars) < 3:
            dt_ms = (time.perf_counter() - t_start) * 1000.0
            return StarTrackerResult(
                success=False,
                solution=None,
                detected_stars=detected_stars,
                matched_stars=[],
                execution_time_ms=dt_ms,
                status_message=f"Insufficient stars detected ({len(detected_stars)} < 3)",
            )

        # Step 2: Lost-in-Space Star Identification (Triangle & Pyramid Matching)
        matches = self.matcher.match_stars(detected_stars)
        if len(matches) < 3:
            dt_ms = (time.perf_counter() - t_start) * 1000.0
            return StarTrackerResult(
                success=False,
                solution=None,
                detected_stars=detected_stars,
                matched_stars=matches,
                execution_time_ms=dt_ms,
                status_message=f"Matching failed: only {len(matches)} stars identified (need >= 3)",
            )

        # Step 3: Attitude Determination via Kabsch / SVD
        cam_vecs = np.array([m.cam_vector for m in matches], dtype=np.float64)
        inertial_vecs = np.array([m.inertial_vector for m in matches], dtype=np.float64)
        # Weights proportional to detected flux
        weights = np.array([detected_stars[m.detected_idx].flux for m in matches], dtype=np.float64)
        weights /= np.sum(weights)

        solution = solve_attitude(cam_vecs, inertial_vecs, weights=weights, R_true=R_true)

        dt_ms = (time.perf_counter() - t_start) * 1000.0
        return StarTrackerResult(
            success=True,
            solution=solution,
            detected_stars=detected_stars,
            matched_stars=matches,
            execution_time_ms=dt_ms,
            status_message=f"Solved successfully with {len(matches)} matched stars",
        )

    def annotate_image(
        self,
        image: np.ndarray,
        result: StarTrackerResult,
        projected_ground_truth: Optional[List[ProjectedStar]] = None,
    ) -> np.ndarray:
        """Render visualization overlay with detected stars, matched stars, and telemetry."""
        # Convert grayscale image to BGR for color overlay
        if len(image.shape) == 2:
            annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            annotated = image.copy()

        h, w = annotated.shape[:2]

        # 1. Draw all detected star spots (thin yellow circles)
        for s in result.detected_stars:
            u_i, v_i = int(round(s.u)), int(round(s.v))
            cv2.circle(annotated, (u_i, v_i), 8, (0, 200, 255), 1, lineType=cv2.LINE_AA)

        # 2. Draw identified/matched stars (green circles with star names/IDs)
        matched_pixels = []
        for m in result.matched_stars:
            det = result.detected_stars[m.detected_idx]
            u_i, v_i = int(round(det.u)), int(round(det.v))
            matched_pixels.append((u_i, v_i))
            # Bold green circle
            cv2.circle(annotated, (u_i, v_i), 12, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            label = m.catalog_star.name if m.catalog_star.name else f"ID {m.catalog_star_id}"
            cv2.putText(
                annotated,
                f"{label} ({m.catalog_star.vmag:.1f}m)",
                (u_i + 14, v_i + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 128),
                1,
                lineType=cv2.LINE_AA,
            )

        # 3. Draw constellation pattern lines between matched stars
        if len(matched_pixels) >= 3:
            for i in range(min(5, len(matched_pixels))):
                for j in range(i + 1, min(5, len(matched_pixels))):
                    cv2.line(
                        annotated,
                        matched_pixels[i],
                        matched_pixels[j],
                        (180, 100, 30),
                        1,
                        lineType=cv2.LINE_AA,
                    )

        # 4. If solution exists, draw reprojected catalog positions (red crosses)
        if result.solution is not None:
            R_calc = result.solution.R_calc
            K = self.camera_params.intrinsic_matrix
            cam_pts = (R_calc @ self.catalog.vectors.T).T
            front_mask = cam_pts[:, 2] > 0.05
            for idx in np.where(front_mask)[0]:
                pt = cam_pts[idx]
                zc = pt[2]
                u_p = K[0, 0] * (pt[0] / zc) + K[0, 2]
                v_p = K[1, 1] * (pt[1] / zc) + K[1, 2]
                if 0 <= u_p < w and 0 <= v_p < h:
                    ui, vi = int(round(u_p)), int(round(v_p))
                    cv2.drawMarker(
                        annotated,
                        (ui, vi),
                        (0, 0, 255),
                        markerType=cv2.MARKER_CROSS,
                        markerSize=8,
                        thickness=1,
                        line_type=cv2.LINE_AA,
                    )

        # 5. Telemetry HUD Overlay (Top-Left translucent banner)
        banner_h, banner_w = 210, 520
        overlay = annotated.copy()
        cv2.rectangle(overlay, (15, 15), (15 + banner_w, 15 + banner_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
        cv2.rectangle(annotated, (15, 15), (15 + banner_w, 15 + banner_h), (80, 80, 80), 1)

        # Title
        cv2.putText(
            annotated,
            "AOCS/GNC STAR TRACKER - LOST IN SPACE",
            (28, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 230, 255),
            2,
            lineType=cv2.LINE_AA,
        )

        status_color = (0, 255, 0) if result.success else (0, 0, 255)
        cv2.putText(
            annotated,
            f"STATUS: {'TRACKING LOCKED' if result.success else 'LOST / SEARCHING'}",
            (28, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            status_color,
            1,
            lineType=cv2.LINE_AA,
        )

        cv2.putText(
            annotated,
            f"Stars: Detected={len(result.detected_stars)} | Matched={len(result.matched_stars)}",
            (28, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            lineType=cv2.LINE_AA,
        )

        cv2.putText(
            annotated,
            f"Compute Time: {result.execution_time_ms:.2f} ms ({1000.0/max(1e-3, result.execution_time_ms):.1f} Hz)",
            (28, 116),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            lineType=cv2.LINE_AA,
        )

        if result.solution is not None:
            sol = result.solution
            cv2.putText(
                annotated,
                f"Quat [w,x,y,z]: [{sol.quaternion[0]:.3f}, {sol.quaternion[1]:.3f}, {sol.quaternion[2]:.3f}, {sol.quaternion[3]:.3f}]",
                (28, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (200, 200, 200),
                1,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"Euler [RPY]: [{sol.euler_angles_deg[0]:.2f}, {sol.euler_angles_deg[1]:.2f}, {sol.euler_angles_deg[2]:.2f}] deg",
                (28, 164),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (200, 200, 200),
                1,
                lineType=cv2.LINE_AA,
            )

            if sol.error_angle_arcsec is not None:
                err_text = f"ATTITUDE ERROR: {sol.error_angle_arcsec:.2f} arcsec (Boresight: {sol.boresight_error_arcsec:.2f}\")"
                cv2.putText(
                    annotated,
                    err_text,
                    (28, 192),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (50, 255, 100),
                    2,
                    lineType=cv2.LINE_AA,
                )

        # Legend at bottom right
        leg_w, leg_h = 320, 70
        leg_x = w - leg_w - 20
        leg_y = h - leg_h - 20
        overlay2 = annotated.copy()
        cv2.rectangle(overlay2, (leg_x, leg_y), (leg_x + leg_w, leg_y + leg_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay2, 0.75, annotated, 0.25, 0, annotated)
        cv2.rectangle(annotated, (leg_x, leg_y), (leg_x + leg_w, leg_y + leg_h), (80, 80, 80), 1)

        cv2.circle(annotated, (leg_x + 18, leg_y + 22), 6, (0, 200, 255), 1)
        cv2.putText(annotated, "Detected Star Spot", (leg_x + 35, leg_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.circle(annotated, (leg_x + 18, leg_y + 48), 6, (0, 255, 0), 2)
        cv2.putText(annotated, "Identified Catalog Star", (leg_x + 35, leg_y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.drawMarker(annotated, (leg_x + 200, leg_y + 48), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=7, thickness=1)
        cv2.putText(annotated, "Reprojected GT", (leg_x + 215, leg_y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)

        return annotated
