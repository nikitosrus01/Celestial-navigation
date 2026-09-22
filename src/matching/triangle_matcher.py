"""Star pattern identification module for the 'Lost in Space' scenario.

Uses angular distance invariants, K-vector triad queries, and multi-star pyramid
verification to uniquely identify detected stars without prior attitude knowledge.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.catalog.k_vector import StarPairDatabase
from src.catalog.star_catalog import Star, StarCatalog, angular_distance
from src.detection.centroiding import DetectedStar
from src.matching.attitude_svd import solve_wahba_svd


@dataclass
class StarMatch:
    """Pairing between a detected star and a catalog star."""

    detected_idx: int  # Index in detected stars list
    catalog_star_id: int  # Unique ID of matched catalog star
    catalog_star: Star
    cam_vector: np.ndarray  # (3,) unit vector in camera frame
    inertial_vector: np.ndarray  # (3,) unit vector in inertial frame
    residual_angle_rad: float = 0.0  # Angular reprojection residual


class TriangleMatcher:
    """Lost-in-Space star identification using triangle matching and pyramid verification."""

    def __init__(
        self,
        catalog: StarCatalog,
        pair_db: StarPairDatabase,
        angular_tolerance_deg: float = 0.12,  # Angular match tolerance (~2 mrad)
        max_stars_to_match: int = 15,  # Match brightest N stars
        min_inliers: int = 4,  # Minimum confirmed stars for confident solution
    ):
        self.catalog = catalog
        self.pair_db = pair_db
        self.tolerance_rad = math.radians(angular_tolerance_deg)
        self.max_stars_to_match = max_stars_to_match
        self.min_inliers = min_inliers

    def match_stars(
        self,
        detected_stars: List[DetectedStar],
    ) -> List[StarMatch]:
        """Perform Lost-in-Space star identification on detected stars.

        Args:
            detected_stars: List of DetectedStar objects from centroiding.

        Returns:
            List[StarMatch]: Matched detected stars and corresponding catalog stars.
        """
        n_det = min(len(detected_stars), self.max_stars_to_match)
        if n_det < 3:
            return []

        # Focus on brightest detected stars
        obs_stars = detected_stars[:n_det]
        obs_vectors = np.array([s.cam_vector for s in obs_stars], dtype=np.float64)

        # Precompute pairwise angular distances between all observed stars
        obs_angles = np.zeros((n_det, n_det), dtype=np.float64)
        for i in range(n_det):
            for j in range(i + 1, n_det):
                ang = angular_distance(obs_vectors[i], obs_vectors[j])
                obs_angles[i, j] = ang
                obs_angles[j, i] = ang

        # Try triangles ordered by brightness: (0, 1, 2), (0, 1, 3), etc.
        triplets = list(itertools.combinations(range(n_det), 3))
        # Prioritize triplets with larger minimum edge (more stable geometry)
        triplets.sort(
            key=lambda t: min(obs_angles[t[0], t[1]], obs_angles[t[1], t[2]], obs_angles[t[0], t[2]]),
            reverse=True,
        )

        best_matches: List[StarMatch] = []
        best_inlier_count = 0

        for i, j, k in triplets:
            d_ij = obs_angles[i, j]
            d_jk = obs_angles[j, k]
            d_ki = obs_angles[k, i]

            # Query candidate catalog pairs for edge (i, j)
            candidate_pairs_ij = self.pair_db.query_pairs(d_ij, self.tolerance_rad)
            if not candidate_pairs_ij:
                continue

            for s1_cat_idx, s2_cat_idx, _ in candidate_pairs_ij:
                # Two possible assignments:
                # 1) i -> s1, j -> s2
                # 2) i -> s2, j -> s1
                for cat_i, cat_j in [(s1_cat_idx, s2_cat_idx), (s2_cat_idx, s1_cat_idx)]:
                    # Find candidate catalog star for k connected to both cat_i and cat_j
                    candidate_k = self._find_third_star(cat_i, cat_j, d_ki, d_jk)
                    if candidate_k is None:
                        continue

                    # We have a candidate triangle: (i->cat_i, j->cat_j, k->candidate_k)
                    triad_match = {i: cat_i, j: cat_j, k: candidate_k}

                    # Verify and expand candidate solution across all detected stars
                    confirmed = self._verify_and_expand_solution(
                        triad_match=triad_match,
                        obs_vectors=obs_vectors,
                        detected_stars=obs_stars,
                    )

                    if len(confirmed) > best_inlier_count:
                        best_inlier_count = len(confirmed)
                        best_matches = confirmed
                        # If we have confirmed >= min_inliers stars, solution is unambiguous!
                        if best_inlier_count >= max(self.min_inliers, 4):
                            return best_matches

        return best_matches

    def _find_third_star(
        self,
        cat_i: int,
        cat_j: int,
        d_ki: float,
        d_jk: float,
    ) -> Optional[int]:
        """Find a catalog star index k that satisfies distances d_ki and d_jk to cat_i and cat_j."""
        pairs_ki = self.pair_db.query_pairs(d_ki, self.tolerance_rad)
        if not pairs_ki:
            return None

        # Collect catalog stars connected to cat_i with distance ~ d_ki
        connected_to_i: Set[int] = set()
        for p1, p2, _ in pairs_ki:
            if p1 == cat_i:
                connected_to_i.add(p2)
            elif p2 == cat_i:
                connected_to_i.add(p1)

        if not connected_to_i:
            return None

        # Check which of these are also connected to cat_j with distance ~ d_jk
        pairs_jk = self.pair_db.query_pairs(d_jk, self.tolerance_rad)
        for p1, p2, _ in pairs_jk:
            cand = None
            if p1 == cat_j and p2 in connected_to_i:
                cand = p2
            elif p2 == cat_j and p1 in connected_to_i:
                cand = p1

            if cand is not None and cand != cat_i and cand != cat_j:
                return cand

        return None

    def _verify_and_expand_solution(
        self,
        triad_match: Dict[int, int],
        obs_vectors: np.ndarray,
        detected_stars: List[DetectedStar],
    ) -> List[StarMatch]:
        """Compute candidate attitude from triad and verify other detected stars against catalog."""
        det_indices = list(triad_match.keys())
        cat_indices = [triad_match[idx] for idx in det_indices]

        b_triad = obs_vectors[det_indices]
        v_triad = self.catalog.vectors[cat_indices]

        try:
            # Candidate rotation: inertial -> camera
            R_cand = solve_wahba_svd(b_triad, v_triad)
        except Exception:
            return []

        # Reproject all catalog stars into camera frame using R_cand
        # cam_est = R_cand @ v_inertial
        cat_cam_est = (R_cand @ self.catalog.vectors.T).T

        # Visible stars must have Z > 0
        front_mask = cat_cam_est[:, 2] > 0.1
        front_indices = np.where(front_mask)[0]
        front_vectors = cat_cam_est[front_indices]

        matches: List[StarMatch] = []
        used_cat_indices: Set[int] = set()

        for det_idx, det_star in enumerate(detected_stars):
            b_obs = det_star.cam_vector

            # Compute angular separation with all front catalog stars
            dots = np.dot(front_vectors, b_obs)
            best_front_idx = int(np.argmax(dots))
            max_dot = dots[best_front_idx]
            ang_err = float(np.arccos(np.clip(max_dot, -1.0, 1.0)))

            if ang_err <= self.tolerance_rad:
                matched_cat_idx = int(front_indices[best_front_idx])
                if matched_cat_idx not in used_cat_indices:
                    used_cat_indices.add(matched_cat_idx)
                    cat_star = self.catalog.stars[matched_cat_idx]
                    matches.append(
                        StarMatch(
                            detected_idx=det_idx,
                            catalog_star_id=cat_star.star_id,
                            catalog_star=cat_star,
                            cam_vector=b_obs,
                            inertial_vector=cat_star.vector,
                            residual_angle_rad=ang_err,
                        )
                    )

        return matches
