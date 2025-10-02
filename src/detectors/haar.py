"""
Detector de rostros usando Haar Cascades
Implementa detección rápida y eficiente para casos con buena iluminación
"""

import cv2
import os

class HaarFaceDetector:
    def __init__(self, cascade_file=None, scale_factor=1.1, min_neighbors=5, min_size=(60, 60)):
        """
        Inicializa el detector Haar
        
        Args:
            cascade_file: Ruta al archivo XML del cascade
            scale_factor: Factor de escala para la pirámide de imágenes
            min_neighbors: Mínimo número de vecinos para considerar una detección
            min_size: Tamaño mínimo del rostro a detectar
        """
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        
        # Cargar el clasificador Haar
        if cascade_file is None:
            cascade_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        if not os.path.exists(cascade_file):
            raise FileNotFoundError(f"No se encontró el archivo cascade: {cascade_file}")
            
        self.face_cascade = cv2.CascadeClassifier(cascade_file)
        
        if self.face_cascade.empty():
            raise ValueError(f"No se pudo cargar el clasificador desde: {cascade_file}")
    
    def detect_faces(self, frame):
        """
        Detecta rostros en un frame
        
        Args:
            frame: Imagen en formato BGR
            
        Returns:
            List de tuplas (x, y, w, h, confidence) donde confidence es estimado
        """
        # Convertir a escala de grises para mejor rendimiento
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Aplicar equalización de histograma para mejorar contraste
        gray = cv2.equalizeHist(gray)
        
        # Detectar rostros
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Convertir a formato con confidence (estimada basada en tamaño)
        detections = []
        for (x, y, w, h) in faces:
            # Estimar confidence basada en el tamaño relativo del rostro
            area = w * h
            confidence = min(0.9, area / (frame.shape[0] * frame.shape[1]) * 100)
            confidence = max(0.5, confidence)  # Mínimo de confianza
            
            detections.append((x, y, w, h, confidence))
        
        return detections
    
    def get_method_name(self):
        """Retorna el nombre del método de detección"""
        return "Haar"