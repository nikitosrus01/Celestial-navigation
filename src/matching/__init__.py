"""Matching and attitude determination module."""

from src.matching.attitude_svd import AttitudeSolution, compute_attitude_error, solve_attitude, solve_wahba_svd
from src.matching.triangle_matcher import StarMatch, TriangleMatcher

__all__ = [
    "AttitudeSolution",
    "compute_attitude_error",
    "solve_wahba_svd",
    "solve_attitude",
    "StarMatch",
    "TriangleMatcher",
]
