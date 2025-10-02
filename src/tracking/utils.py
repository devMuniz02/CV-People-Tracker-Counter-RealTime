"""
Utilidades para el sistema de tracking
"""

import numpy as np

def calculate_centroid(rect):
    """
    Calcula el centroide de un rectángulo
    
    Args:
        rect: Tupla (x, y, w, h)
        
    Returns:
        Tupla (cx, cy) con las coordenadas del centro
    """
    x, y, w, h = rect
    cx = int(x + w / 2.0)
    cy = int(y + h / 2.0)
    return (cx, cy)

def calculate_distance(point1, point2):
    """
    Calcula la distancia euclidiana entre dos puntos
    
    Args:
        point1: Tupla (x1, y1)
        point2: Tupla (x2, y2)
        
    Returns:
        Distancia euclidiana
    """
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def rect_overlap(rect1, rect2):
    """
    Calcula el área de solapamiento entre dos rectángulos
    
    Args:
        rect1: Tupla (x1, y1, w1, h1)
        rect2: Tupla (x2, y2, w2, h2)
        
    Returns:
        Área de solapamiento
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    
    # Calcular intersección
    left = max(x1, x2)
    right = min(x1 + w1, x2 + w2)
    top = max(y1, y2)
    bottom = min(y1 + h1, y2 + h2)
    
    if left < right and top < bottom:
        return (right - left) * (bottom - top)
    return 0

def calculate_iou(rect1, rect2):
    """
    Calcula el IoU (Intersection over Union) entre dos rectángulos
    
    Args:
        rect1: Tupla (x1, y1, w1, h1)
        rect2: Tupla (x2, y2, w2, h2)
        
    Returns:
        Valor IoU entre 0 y 1
    """
    overlap = rect_overlap(rect1, rect2)
    
    if overlap == 0:
        return 0.0
    
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - overlap
    
    return overlap / union if union > 0 else 0.0

def smooth_detections(detections, history, alpha=0.3):
    """
    Suaviza las detecciones usando promedio exponencial
    
    Args:
        detections: Lista de detecciones actuales
        history: Historial de detecciones previas
        alpha: Factor de suavizado (0-1)
        
    Returns:
        Detecciones suavizadas
    """
    if not history:
        return detections
    
    smoothed = []
    for detection in detections:
        x, y, w, h = detection[:4]
        
        # Buscar detección similar en el historial
        best_match = None
        best_distance = float('inf')
        
        for hist_det in history[-5:]:  # Usar últimas 5 detecciones
            hist_x, hist_y, hist_w, hist_h = hist_det[:4]
            dist = calculate_distance((x, y), (hist_x, hist_y))
            
            if dist < best_distance and dist < 50:  # Umbral de proximidad
                best_distance = dist
                best_match = hist_det
        
        if best_match:
            # Aplicar suavizado
            hist_x, hist_y, hist_w, hist_h = best_match[:4]
            smooth_x = int(alpha * x + (1 - alpha) * hist_x)
            smooth_y = int(alpha * y + (1 - alpha) * hist_y)
            smooth_w = int(alpha * w + (1 - alpha) * hist_w)
            smooth_h = int(alpha * h + (1 - alpha) * hist_h)
            
            smoothed_detection = [smooth_x, smooth_y, smooth_w, smooth_h]
            if len(detection) > 4:
                smoothed_detection.extend(detection[4:])  # Preservar confidence
            smoothed.append(tuple(smoothed_detection))
        else:
            smoothed.append(detection)
    
    return smoothed

class RegionOfInterest:
    """
    Clase para definir y verificar regiones de interés
    """
    
    def __init__(self, points=None):
        """
        Inicializa la ROI
        
        Args:
            points: Lista de puntos (x, y) que definen el polígono de la ROI
        """
        self.points = points or []
        self.enabled = len(self.points) > 2
    
    def set_points(self, points):
        """Establece los puntos de la ROI"""
        self.points = points
        self.enabled = len(self.points) > 2
    
    def is_inside(self, point):
        """
        Verifica si un punto está dentro de la ROI
        
        Args:
            point: Tupla (x, y)
            
        Returns:
            True si está dentro, False en caso contrario
        """
        if not self.enabled:
            return True
        
        x, y = point
        n = len(self.points)
        inside = False
        
        p1x, p1y = self.points[0]
        for i in range(1, n + 1):
            p2x, p2y = self.points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def filter_detections(self, detections):
        """
        Filtra detecciones que están fuera de la ROI
        
        Args:
            detections: Lista de detecciones
            
        Returns:
            Lista filtrada de detecciones
        """
        if not self.enabled:
            return detections
        
        filtered = []
        for detection in detections:
            x, y, w, h = detection[:4]
            center = (x + w // 2, y + h // 2)
            
            if self.is_inside(center):
                filtered.append(detection)
        
        return filtered