"""K-Vector range search and Star Pair / Triad database for rapid Lost-in-Space identification.

Reference:
    Mortari, D. (1996). "Search-Less Algorithm for Star Identification"
    Journal of the Astronautical Sciences.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.catalog.star_catalog import StarCatalog, angular_distance


class KVector:
    """Mortari's K-Vector data structure for O(1) range search on sorted arrays."""

    def __init__(self, values: np.ndarray, num_bins: Optional[int] = None):
        """Construct K-Vector over an array of sorted float values.

        Args:
            values: 1D array of sorted floats in non-decreasing order.
            num_bins: Number of bins. Defaults to max(len(values), 100).
        """
        self.values = np.asarray(values, dtype=np.float64)
        self.n = len(self.values)
        if self.n == 0:
            self.k = np.zeros(0, dtype=np.int64)
            self.s_min = 0.0
            self.s_max = 0.0
            self.m = 0.0
            self.q = 0.0
            return

        self.num_bins = num_bins if num_bins is not None else max(self.n, 100)
        self.s_min = float(self.values[0])
        self.s_max = float(self.values[-1])

        # Avoid zero division when all values are identical
        if abs(self.s_max - self.s_min) < 1e-12:
            self.s_max += 1e-9

        # Add small epsilon margin to cover edge cases
        self.eps = 1e-11 * (self.s_max - self.s_min)
        self.s_min_margin = self.s_min - self.eps
        self.s_max_margin = self.s_max + self.eps

        # Straight-line parameters: z(y) = m * y + q
        # Maps range [s_min_margin, s_max_margin] to index space [0, num_bins - 1]
        self.m = (self.num_bins - 1) / (self.s_max_margin - self.s_min_margin)
        self.q = -self.m * self.s_min_margin

        # Construct the integer k-vector:
        # k(i) holds the index of the last element in self.values with z(s) <= i
        bin_edges = (np.arange(self.num_bins) - self.q) / self.m
        # np.searchsorted gives first index where value > bin_edge, subtract 1
        self.k = np.searchsorted(self.values, bin_edges, side="right") - 1

    def query_range(self, val_min: float, val_max: float) -> Tuple[int, int]:
        """Query index range [start_idx, end_idx] of values within [val_min, val_max].

        Returns:
            Tuple[int, int]: (start_idx, end_idx) where values[start_idx:end_idx] are in range.
            Returns (0, 0) if no elements found.
        """
        if self.n == 0 or val_max < self.s_min or val_min > self.s_max:
            return 0, 0

        # Bound queries to the margins
        y_a = max(val_min, self.s_min_margin)
        y_b = min(val_max, self.s_max_margin)

        # Bottom index: elements strictly < ya
        idx_a = int(math.floor(self.m * y_a + self.q))
        # Top index: elements up to next bin edge > yb
        idx_b = int(math.ceil(self.m * y_b + self.q))

        idx_a = max(0, min(self.num_bins - 1, idx_a))
        idx_b = max(0, min(self.num_bins - 1, idx_b))

        start = 0 if idx_a == 0 else int(self.k[idx_a - 1]) + 1
        end = int(self.k[idx_b]) + 1

        # Clip bounds to valid array slice
        start = max(0, min(self.n, start))
        end = max(start, min(self.n, end))

        # Precision refinement at borders (due to floating point discretizations)
        while start < self.n and self.values[start] < val_min:
            start += 1
        while end > start and self.values[end - 1] > val_max:
            end -= 1

        return start, end


@dataclass
class StarPair:
    star1_idx: int  # Index in StarCatalog
    star2_idx: int  # Index in StarCatalog
    angular_dist: float  # In radians


class StarPairDatabase:
    """Precomputed database of star pairs within maximum FOV with K-Vector indexing."""

    def __init__(self, catalog: StarCatalog, max_fov_rad: float = math.radians(25.0)):
        self.catalog = catalog
        self.max_fov_rad = max_fov_rad
        self._build_pair_table()

    def _build_pair_table(self) -> None:
        """Compute all pairs of stars separated by <= max_fov_rad and index them."""
        vectors = self.catalog.vectors
        n_stars = len(vectors)

        cos_max = math.cos(self.max_fov_rad)
        pairs_list: List[Tuple[int, int, float]] = []

        # Pairwise dot product search
        for i in range(n_stars):
            v_i = vectors[i]
            # Vectorized dot products with remaining stars
            if i + 1 < n_stars:
                dots = np.dot(vectors[i + 1 :], v_i)
                valid_mask = dots >= cos_max
                valid_indices = np.where(valid_mask)[0] + (i + 1)
                for j_idx, dot_val in zip(valid_indices, dots[valid_mask]):
                    ang = float(np.arccos(np.clip(dot_val, -1.0, 1.0)))
                    pairs_list.append((i, j_idx, ang))

        # Sort pairs by angular distance
        pairs_list.sort(key=lambda p: p[2])

        self.num_pairs = len(pairs_list)
        if self.num_pairs > 0:
            self.star1 = np.array([p[0] for p in pairs_list], dtype=np.int32)
            self.star2 = np.array([p[1] for p in pairs_list], dtype=np.int32)
            self.distances = np.array([p[2] for p in pairs_list], dtype=np.float64)
            self.k_vector = KVector(self.distances)
        else:
            self.star1 = np.empty(0, dtype=np.int32)
            self.star2 = np.empty(0, dtype=np.int32)
            self.distances = np.empty(0, dtype=np.float64)
            self.k_vector = KVector(self.distances)

    def query_pairs(self, target_angle_rad: float, tolerance_rad: float) -> List[Tuple[int, int, float]]:
        """Find catalog star pairs with angular distance within target_angle_rad +- tolerance_rad.

        Returns:
            List of (star1_catalog_idx, star2_catalog_idx, angular_distance_rad).
        """
        if self.num_pairs == 0:
            return []

        min_ang = max(0.0, target_angle_rad - tolerance_rad)
        max_ang = min(self.max_fov_rad, target_angle_rad + tolerance_rad)

        start, end = self.k_vector.query_range(min_ang, max_ang)
        if start >= end:
            return []

        res = []
        for idx in range(start, end):
            res.append((int(self.star1[idx]), int(self.star2[idx]), float(self.distances[idx])))
        return res
