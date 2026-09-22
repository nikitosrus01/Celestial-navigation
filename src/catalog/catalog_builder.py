"""Catalog builder: creates navigation star catalogs using standard bright stars
and realistic full-sky distribution up to magnitude +5.5.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.catalog.star_catalog import Star, StarCatalog, radec_to_unit_vector

# Standard IAU Navigational Stars with accurate J2000 coordinates (RA, Dec in degrees, Vmag, Name)
# Source: Nautical Almanac / Hipparcos bright star selection
STANDARD_BRIGHT_STARS = [
    # (HIP/ID, Name, RA_deg, Dec_deg, Vmag)
    (32349, "Sirius", 101.287, -16.716, -1.46),
    (30438, "Canopus", 95.988, -52.696, -0.74),
    (71683, "Rigil Kentaurus", 219.902, -60.834, -0.27),
    (69673, "Arcturus", 213.915, 19.182, -0.05),
    (91262, "Vega", 279.235, 38.784, 0.03),
    (24608, "Capella", 79.172, 45.998, 0.08),
    (24436, "Rigel", 78.634, -8.202, 0.13),
    (37279, "Procyon", 114.825, 5.225, 0.38),
    (7588, "Achernar", 24.429, -57.237, 0.46),
    (27989, "Betelgeuse", 88.793, 7.407, 0.50),
    (68702, "Hadar", 210.956, -60.373, 0.61),
    (97649, "Altair", 297.696, 8.868, 0.77),
    (65474, "Acrux", 186.650, -63.099, 0.76),
    (21421, "Aldebaran", 68.980, 16.509, 0.86),
    (80763, "Antares", 247.352, -26.432, 0.96),
    (65378, "Spica", 201.298, -11.161, 0.97),
    (37826, "Pollux", 116.329, 28.026, 1.14),
    (113368, "Fomalhaut", 344.413, -29.622, 1.16),
    (102098, "Deneb", 310.358, 45.280, 1.25),
    (60718, "Mimosa", 186.200, -59.689, 1.25),
    (49669, "Regulus", 152.093, 11.967, 1.35),
    (31681, "Adhara", 104.656, -28.972, 1.50),
    (33579, "Castor", 113.650, 31.888, 1.58),
    (61084, "Gacrux", 187.791, -57.113, 1.63),
    (25336, "Bellatrix", 81.283, 6.350, 1.64),
    (26727, "El Nath", 81.573, 28.608, 1.65),
    (82273, "Shaula", 253.883, -37.104, 1.62),
    (46390, "Miaplacidus", 138.300, -69.717, 1.67),
    (26311, "Alnilam", 84.053, -1.202, 1.69),
    (107315, "Enif", 326.046, 9.875, 2.38),
    (92855, "Albireo", 289.876, 27.960, 3.05),
    (11767, "Polaris", 37.954, 89.264, 1.98),
    (54061, "Dubhe", 165.932, 61.751, 1.79),
    (53910, "Merak", 165.460, 56.382, 2.37),
    (58001, "Phecda", 178.457, 53.695, 2.44),
    (59774, "Megrez", 183.857, 57.032, 3.31),
    (62956, "Alioth", 193.507, 55.960, 1.77),
    (65378, "Mizar", 200.981, 54.925, 2.23),
    (67301, "Alkaid", 206.885, 49.313, 1.86),
    (30883, "Wezen", 107.498, -26.393, 1.83),
    (90185, "Kaus Australis", 276.043, -34.384, 1.85),
    (41037, "Avior", 125.628, -59.509, 1.86),
    (26727, "Alnitak", 85.190, -1.943, 1.77),
    (86228, "Sabik", 262.664, -15.725, 2.43),
    (84012, "Rasalhague", 263.734, 12.560, 2.08),
    (72607, "Kocab", 222.676, 74.156, 2.07),
    (85927, "Shaula-B", 262.112, -43.005, 2.69),
    (100453, "Alderamin", 309.679, 62.585, 2.45),
    (34444, "Mirzam", 102.460, -17.956, 1.98),
    (45238, "Suhail", 136.999, -43.432, 2.21),
    (112029, "Scheat", 345.944, 28.083, 2.42),
    (113881, "Markab", 346.190, 15.205, 2.49),
    (113963, "Algenib", 347.457, 15.183, 2.84),
    (707, "Alpheratz", 2.097, 29.090, 2.06),
    (14135, "Algol", 47.042, 40.956, 2.12),
    (1067, "Mirach", 3.738, 35.621, 2.05),
    (9640, "Almach", 30.975, 42.332, 2.10),
    (3179, "Caph", 10.127, 59.150, 2.27),
    (4427, "Schedar", 14.177, 56.537, 2.24),
    (7415, "Gamma Cas", 24.328, 60.717, 2.47),
    (6686, "Ruchbah", 21.454, 60.235, 2.68),
]


def generate_sphere_catalog(
    total_stars: int = 1500,
    max_magnitude: float = 5.5,
    seed: int = 42,
) -> StarCatalog:
    """Generate a realistic celestial star catalog across the entire sphere.

    Combines the IAU navigation stars with a realistic stellar density distribution
    (Fibonacci celestial sphere lattice with pseudo-random perturbations and
    magnitude distribution matching astronomical log-normal / power law $N(m)$).

    Args:
        total_stars: Total number of stars to generate across the sky.
                     (For reference, total visible stars down to +5.5 mag is ~1600).
        max_magnitude: Cutoff visual magnitude.
        seed: Random seed for reproducibility.

    Returns:
        StarCatalog: Full-sky star catalog.
    """
    rng = np.random.default_rng(seed)
    stars: List[Star] = []
    used_ids = set()

    # 1. Insert standard bright navigation stars
    for hip_id, name, ra_deg, dec_deg, vmag in STANDARD_BRIGHT_STARS:
        if vmag <= max_magnitude:
            ra_rad = math.radians(ra_deg)
            dec_rad = math.radians(dec_deg)
            vec = radec_to_unit_vector(ra_rad, dec_rad)
            stars.append(
                Star(
                    star_id=hip_id,
                    ra_rad=ra_rad,
                    dec_rad=dec_rad,
                    vmag=vmag,
                    vector=vec,
                    name=name,
                )
            )
            used_ids.add(hip_id)

    # 2. Fill the remaining sphere using Fibonacci sphere sampling with random jitter
    needed_background = max(0, total_stars - len(stars))
    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0

    current_id = 100000
    for i in range(needed_background):
        while current_id in used_ids:
            current_id += 1

        # Fibonacci spiral point on sphere
        theta = 2.0 * math.pi * i / golden_ratio
        z = 1.0 - (2.0 * i + 1.0) / needed_background
        radius_xy = math.sqrt(max(0.0, 1.0 - z * z))

        # Add slight natural clustering/jitter
        jitter_angle = rng.normal(0.0, 0.015)
        jitter_z = rng.normal(0.0, 0.01)
        z = np.clip(z + jitter_z, -0.9999, 0.9999)
        radius_xy = math.sqrt(1.0 - z * z)
        theta = (theta + jitter_angle) % (2.0 * math.pi)

        x = radius_xy * math.cos(theta)
        y = radius_xy * math.sin(theta)

        vec = np.array([x, y, z], dtype=np.float64)
        vec /= np.linalg.norm(vec)

        dec_rad = float(np.arcsin(vec[2]))
        ra_rad = float(np.arctan2(vec[1], vec[0]) % (2.0 * math.pi))

        # Astronomical stellar magnitude distribution: N(m) grows exponentially with magnitude
        # Cumulative number of stars brighter than m ~ 10^(0.6 * m)
        # Power-law sampling for background stars in range [2.5, max_magnitude]
        u = rng.uniform(0.0, 1.0)
        # Transform uniform u using inverse CDF for exponential magnitude distribution
        alpha = 0.8
        vmag = 2.5 + (max_magnitude - 2.5) * (u ** (1.0 / (1.0 + alpha)))
        vmag = round(float(vmag), 2)

        stars.append(
            Star(
                star_id=current_id,
                ra_rad=ra_rad,
                dec_rad=dec_rad,
                vmag=vmag,
                vector=vec,
                name=f"CAT-{current_id}",
            )
        )
        used_ids.add(current_id)
        current_id += 1

    catalog = StarCatalog(stars)
    return catalog


def build_default_catalog(
    output_path: Optional[str | Path] = None,
    total_stars: int = 1600,
    max_magnitude: float = 5.5,
) -> StarCatalog:
    """Build and optionally cache the default navigation star catalog.

    Args:
        output_path: Path to save the NPZ file (e.g. 'data/catalogs/nav_stars_catalog.npz').
        total_stars: Total stars generated.
        max_magnitude: Limiting magnitude.

    Returns:
        StarCatalog: Initialized StarCatalog instance.
    """
    if output_path is not None:
        path = Path(output_path)
        if path.exists():
            return StarCatalog.load_npz(path)

    catalog = generate_sphere_catalog(total_stars=total_stars, max_magnitude=max_magnitude)

    if output_path is not None:
        path = Path(output_path)
        catalog.save_npz(path)

    return catalog
