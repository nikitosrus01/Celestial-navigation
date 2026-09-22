"""Tests for star catalog and k-vector indexing."""

import math
import numpy as np
import pytest

from src.catalog.catalog_builder import generate_sphere_catalog, STANDARD_BRIGHT_STARS
from src.catalog.k_vector import KVector, StarPairDatabase
from src.catalog.star_catalog import (
    StarCatalog,
    radec_to_unit_vector,
    unit_vector_to_radec,
    angular_distance,
)


def test_radec_conversions():
    # Test North Pole
    v_np = radec_to_unit_vector(0.0, np.pi / 2.0)
    assert np.allclose(v_np, [0.0, 0.0, 1.0], atol=1e-7)
    ra, dec = unit_vector_to_radec(v_np)
    assert np.isclose(dec, np.pi / 2.0, atol=1e-7)

    # Test Equator RA=0
    v_eq0 = radec_to_unit_vector(0.0, 0.0)
    assert np.allclose(v_eq0, [1.0, 0.0, 0.0], atol=1e-7)
    ra, dec = unit_vector_to_radec(v_eq0)
    assert np.isclose(ra, 0.0, atol=1e-7)
    assert np.isclose(dec, 0.0, atol=1e-7)

    # Test Equator RA=90 deg
    v_eq90 = radec_to_unit_vector(np.pi / 2.0, 0.0)
    assert np.allclose(v_eq90, [0.0, 1.0, 0.0], atol=1e-7)
    ra, dec = unit_vector_to_radec(v_eq90)
    assert np.isclose(ra, np.pi / 2.0, atol=1e-7)
    assert np.isclose(dec, 0.0, atol=1e-7)

    # Angular distance between orthogonal vectors
    assert np.isclose(angular_distance(v_eq0, v_eq90), np.pi / 2.0, atol=1e-7)


def test_catalog_builder():
    catalog = generate_sphere_catalog(total_stars=500, max_magnitude=5.0, seed=123)
    assert len(catalog) >= 500
    assert len(catalog.vectors) == len(catalog)
    assert np.all(catalog.magnitudes <= 5.0)

    # Check that Sirius is present
    sirius = catalog.get_star_by_id(32349)
    assert sirius is not None
    assert sirius.name == "Sirius"
    assert np.isclose(sirius.vmag, -1.46, atol=0.05)


def test_k_vector_range_query():
    # Sorted random array
    rng = np.random.default_rng(42)
    data = np.sort(rng.uniform(0.1, 10.0, size=1000))
    kv = KVector(data, num_bins=200)

    # Test queries
    test_ranges = [(1.5, 3.2), (0.0, 0.5), (9.0, 11.0), (4.0, 4.05)]
    for low, high in test_ranges:
        s, e = kv.query_range(low, high)
        expected_indices = np.where((data >= low) & (data <= high))[0]
        if len(expected_indices) == 0:
            assert s == e
        else:
            assert s == expected_indices[0]
            assert e == expected_indices[-1] + 1


def test_star_pair_database():
    catalog = generate_sphere_catalog(total_stars=200, max_magnitude=5.0, seed=42)
    db = StarPairDatabase(catalog, max_fov_rad=math.radians(15.0))
    assert db.num_pairs > 0

    # Pick a known pair
    star1_idx = int(db.star1[0])
    star2_idx = int(db.star2[0])
    true_angle = db.distances[0]

    # Query with small tolerance
    tol = math.radians(0.05)
    results = db.query_pairs(true_angle, tol)
    assert len(results) >= 1
    # Check that the known pair is returned
    found = any(
        (p[0] == star1_idx and p[1] == star2_idx) or (p[0] == star2_idx and p[1] == star1_idx)
        for p in results
    )
    assert found
