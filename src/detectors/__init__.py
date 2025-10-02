"""
__init__.py para el paquete detectors
"""

from .haar import HaarFaceDetector
from .dnn import DNNFaceDetector

__all__ = ['HaarFaceDetector', 'DNNFaceDetector']