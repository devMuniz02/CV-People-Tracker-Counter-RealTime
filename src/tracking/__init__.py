"""
__init__.py para el paquete tracking
"""

from .centroid import CentroidTracker
from .utils import (
    calculate_centroid, 
    calculate_distance, 
    calculate_iou, 
    RegionOfInterest,
    smooth_detections
)

__all__ = [
    'CentroidTracker', 
    'calculate_centroid', 
    'calculate_distance', 
    'calculate_iou', 
    'RegionOfInterest',
    'smooth_detections'
]