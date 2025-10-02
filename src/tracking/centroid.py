"""
Sistema de tracking centroide para seguimiento de personas
Mantiene IDs únicos y evita conteos duplicados
"""

import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict

class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=100):
        """
        Inicializa el tracker centroide
        
        Args:
            max_disappeared: Frames máximos que un objeto puede desaparecer
            max_distance: Distancia máxima para asociar detecciones
        """
        # Inicializar contador de IDs únicos
        self.next_object_id = 0
        
        # Diccionarios para almacenar objetos y frames desaparecidos
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        
        # Parámetros de configuración
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # Estadísticas
        self.total_count = 0
        self.active_count = 0
    
    def register(self, centroid):
        """
        Registra un nuevo objeto con un ID único
        
        Args:
            centroid: Tupla (x, y) del centro del objeto
            
        Returns:
            ID del nuevo objeto
        """
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        self.total_count += 1
        self.active_count += 1
        
        return self.next_object_id - 1
    
    def deregister(self, object_id):
        """
        Elimina un objeto del tracking
        
        Args:
            object_id: ID del objeto a eliminar
        """
        if object_id in self.objects:
            del self.objects[object_id]
            del self.disappeared[object_id]
            self.active_count = max(0, self.active_count - 1)
    
    def update(self, rects):
        """
        Actualiza el estado del tracker con nuevas detecciones
        
        Args:
            rects: Lista de tuplas (x, y, w, h) de las detecciones
            
        Returns:
            Diccionario {object_id: (centroid_x, centroid_y, rect)}
        """
        # Si no hay detecciones, marcar todos como desaparecidos
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                # Si ha desaparecido por mucho tiempo, eliminarlo
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            return {}
        
        # Calcular centroides de las nuevas detecciones
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        
        for (i, (x, y, w, h)) in enumerate(rects):
            cx = int(x + w / 2.0)
            cy = int(y + h / 2.0)
            input_centroids[i] = (cx, cy)
        
        # Si no hay objetos existentes, registrar todos como nuevos
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        
        else:
            # Obtener centroides existentes
            object_centroids = list(self.objects.values())
            object_ids = list(self.objects.keys())
            
            # Calcular distancias entre centroides existentes y nuevos
            D = dist.cdist(np.array(object_centroids), input_centroids)
            
            # Encontrar las asignaciones óptimas
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            # Mantener registro de índices usados
            used_row_indices = set()
            used_col_indices = set()
            
            # Iterar sobre las combinaciones (objeto, detección)
            for (row, col) in zip(rows, cols):
                # Ignorar si ya se usó
                if row in used_row_indices or col in used_col_indices:
                    continue
                
                # Verificar que la distancia sea razonable
                if D[row, col] > self.max_distance:
                    continue
                
                # Actualizar el centroide del objeto
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                # Marcar como usados
                used_row_indices.add(row)
                used_col_indices.add(col)
            
            # Calcular índices no usados
            unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)
            
            # Si hay más objetos que detecciones, marcar como desaparecidos
            if D.shape[0] >= D.shape[1]:
                for row in unused_row_indices:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            
            # Si hay más detecciones que objetos, registrar nuevos
            else:
                for col in unused_col_indices:
                    self.register(input_centroids[col])
        
        # Crear resultado con información completa
        result = {}
        for i, object_id in enumerate(self.objects.keys()):
            if i < len(rects):
                result[object_id] = {
                    'centroid': self.objects[object_id],
                    'rect': rects[i] if i < len(rects) else None,
                    'disappeared_frames': self.disappeared[object_id]
                }
        
        return result
    
    def get_total_count(self):
        """Retorna el conteo total de personas únicas detectadas"""
        return self.total_count
    
    def get_active_count(self):
        """Retorna el número de personas actualmente en escena"""
        return len(self.objects)
    
    def get_object_info(self, object_id):
        """
        Obtiene información de un objeto específico
        
        Args:
            object_id: ID del objeto
            
        Returns:
            Diccionario con información del objeto o None si no existe
        """
        if object_id in self.objects:
            return {
                'centroid': self.objects[object_id],
                'disappeared_frames': self.disappeared[object_id]
            }
        return None