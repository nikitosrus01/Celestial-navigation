"""Star catalog representations, coordinate transformations, and data structures."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np


def radec_to_unit_vector(ra_rad: float, dec_rad: float) -> np.ndarray:
    """Convert Right Ascension and Declination (radians) to a 3D unit vector in J2000.

    Args:
        ra_rad: Right ascension in radians [0, 2*pi).
        dec_rad: Declination in radians [-pi/2, pi/2].

    Returns:
        np.ndarray: 3D unit vector [x, y, z] in the inertial ICRF/J2000 frame.
    """
    cos_dec = np.cos(dec_rad)
    x = cos_dec * np.cos(ra_rad)
    y = cos_dec * np.sin(ra_rad)
    z = np.sin(dec_rad)
    vec = np.array([x, y, z], dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        vec /= norm
    return vec


def unit_vector_to_radec(vec: np.ndarray) -> Tuple[float, float]:
    """Convert a 3D unit vector in J2000 to Right Ascension and Declination.

    Args:
        vec: 3D vector [x, y, z].

    Returns:
        Tuple[float, float]: (ra_rad, dec_rad) where ra_rad in [0, 2*pi) and dec_rad in [-pi/2, pi/2].
    """
    v = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return 0.0, 0.0
    v = v / norm

    dec_rad = float(np.arcsin(np.clip(v[2], -1.0, 1.0)))
    ra_rad = float(np.arctan2(v[1], v[0]) % (2.0 * np.pi))
    return ra_rad, dec_rad


def angular_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute the angular distance between two unit vectors in radians.

    Uses the numerically stable arctan2 formula:
        theta = 2 * arcsin(||v1 - v2|| / 2) or atan2(||v1 x v2||, v1 . v2)
    """
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    cross_norm = np.linalg.norm(np.cross(v1, v2))
    return float(np.arctan2(cross_norm, dot))


@dataclass
class Star:
    """Data representation for a celestial star."""

    star_id: int
    ra_rad: float  # Right Ascension in radians
    dec_rad: float  # Declination in radians
    vmag: float  # Visual magnitude (e.g. 1.5, 4.2)
    vector: np.ndarray  # 3D unit vector in J2000 (shape: (3,))
    name: str = ""  # Common or catalog name (e.g. "Sirius", "HIP 32349")

    @property
    def ra_deg(self) -> float:
        return float(np.degrees(self.ra_rad))

    @property
    def dec_deg(self) -> float:
        return float(np.degrees(self.dec_rad))

    def to_dict(self) -> dict:
        return {
            "star_id": self.star_id,
            "ra_rad": self.ra_rad,
            "dec_rad": self.dec_rad,
            "vmag": self.vmag,
            "vector": self.vector.tolist(),
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Star:
        return cls(
            star_id=int(data["star_id"]),
            ra_rad=float(data["ra_rad"]),
            dec_rad=float(data["dec_rad"]),
            vmag=float(data["vmag"]),
            vector=np.array(data["vector"], dtype=np.float64),
            name=str(data.get("name", "")),
        )


class StarCatalog:
    """Inertial Star Catalog with vectorized indexing and query capabilities."""

    def __init__(self, stars: Optional[List[Star]] = None):
        self.stars: List[Star] = stars or []
        self._sync_arrays()

    def _sync_arrays(self) -> None:
        """Synchronize internal NumPy arrays for high-performance vectorized operations."""
        if not self.stars:
            self.vectors = np.empty((0, 3), dtype=np.float64)
            self.magnitudes = np.empty((0,), dtype=np.float64)
            self.ids = np.empty((0,), dtype=np.int64)
            self.names = []
            return

        self.vectors = np.vstack([s.vector for s in self.stars]).astype(np.float64)
        self.magnitudes = np.array([s.vmag for s in self.stars], dtype=np.float64)
        self.ids = np.array([s.star_id for s in self.stars], dtype=np.int64)
        self.names = [s.name for s in self.stars]

    def add_star(self, star: Star) -> None:
        self.stars.append(star)
        self._sync_arrays()

    def __len__(self) -> int:
        return len(self.stars)

    def filter_by_magnitude(self, max_mag: float) -> StarCatalog:
        """Return a new StarCatalog with stars brighter than or equal to max_mag."""
        filtered = [s for s in self.stars if s.vmag <= max_mag]
        return StarCatalog(filtered)

    def get_star_by_id(self, star_id: int) -> Optional[Star]:
        idx = np.where(self.ids == star_id)[0]
        if len(idx) > 0:
            return self.stars[idx[0]]
        return None

    def query_cone(self, boresight: np.ndarray, max_angle_rad: float) -> List[Tuple[Star, float]]:
        """Query stars inside a cone centered around boresight vector.

        Args:
            boresight: 3D unit vector of the optical axis.
            max_angle_rad: Half-angle of the cone in radians.

        Returns:
            List of (Star, angle_rad) within the cone, sorted by angle.
        """
        if len(self.stars) == 0:
            return []

        b = boresight / np.linalg.norm(boresight)
        # Cosine of angle: v . b >= cos(max_angle_rad)
        cos_threshold = np.cos(max_angle_rad)
        dots = np.dot(self.vectors, b)
        inside_mask = dots >= cos_threshold

        results = []
        for idx in np.where(inside_mask)[0]:
            star = self.stars[idx]
            angle = float(np.arccos(np.clip(dots[idx], -1.0, 1.0)))
            results.append((star, angle))

        results.sort(key=lambda item: item[1])
        return results

    def save_json(self, filepath: Union[str, Path]) -> None:
        """Save catalog to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = [s.to_dict() for s in self.stars]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_json(cls, filepath: Union[str, Path]) -> StarCatalog:
        """Load catalog from JSON file."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        stars = [Star.from_dict(d) for d in data]
        return cls(stars)

    def save_npz(self, filepath: Union[str, Path]) -> None:
        """Save catalog to compressed NPZ file for fast binary loading."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        ra_rad = np.array([s.ra_rad for s in self.stars], dtype=np.float64)
        dec_rad = np.array([s.dec_rad for s in self.stars], dtype=np.float64)
        names = np.array([s.name for s in self.stars], dtype=str)
        np.savez_compressed(
            filepath,
            ids=self.ids,
            vectors=self.vectors,
            magnitudes=self.magnitudes,
            ra_rad=ra_rad,
            dec_rad=dec_rad,
            names=names,
        )

    @classmethod
    def load_npz(cls, filepath: Union[str, Path]) -> StarCatalog:
        """Load catalog from NPZ file."""
        filepath = Path(filepath)
        data = np.load(filepath)
        ids = data["ids"]
        vectors = data["vectors"]
        mags = data["magnitudes"]
        ra_rad = data["ra_rad"]
        dec_rad = data["dec_rad"]
        names = data["names"]

        stars = []
        for i in range(len(ids)):
            stars.append(
                Star(
                    star_id=int(ids[i]),
                    ra_rad=float(ra_rad[i]),
                    dec_rad=float(dec_rad[i]),
                    vmag=float(mags[i]),
                    vector=vectors[i],
                    name=str(names[i]),
                )
            )
        return cls(stars)
