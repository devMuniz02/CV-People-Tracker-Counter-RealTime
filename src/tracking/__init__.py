"""Public API for the tracking package.

Exports core tracker class and helper utilities.
"""
from .centroid import CentroidTracker
from .utils import (
    calculate_centroid,
    calculate_distance,
    rect_overlap,
    calculate_iou,
    smooth_detections,
    RegionOfInterest,
)

__all__ = [
    "CentroidTracker",
    "calculate_centroid",
    "calculate_distance",
    "rect_overlap",
    "calculate_iou",
    "smooth_detections",
    "RegionOfInterest",
]
