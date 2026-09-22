"""Catalog module."""

from src.catalog.catalog_builder import STANDARD_BRIGHT_STARS, build_default_catalog, generate_sphere_catalog
from src.catalog.k_vector import KVector, StarPairDatabase
from src.catalog.star_catalog import Star, StarCatalog, angular_distance, radec_to_unit_vector, unit_vector_to_radec

__all__ = [
    "Star",
    "StarCatalog",
    "radec_to_unit_vector",
    "unit_vector_to_radec",
    "angular_distance",
    "STANDARD_BRIGHT_STARS",
    "generate_sphere_catalog",
    "build_default_catalog",
    "KVector",
    "StarPairDatabase",
]
