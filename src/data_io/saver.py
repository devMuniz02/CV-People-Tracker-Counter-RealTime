"""
Sistema para guardar capturas de evidencia
Maneja el guardado de imágenes de rostros y frames completos
"""

import cv2
from datetime import datetime
from pathlib import Path

class ImageSaver:
    def __init__(self, output_dir, save_crops=True, save_frames=True, quality=95):
        """
        Inicializa el guardador de imágenes
        
        Args:
            output_dir: Directorio de salida
            save_crops: Si guardar recortes de rostros
            save_frames: Si guardar frames completos
            quality: Calidad JPEG (0-100)
        """
        self.output_dir = Path(output_dir)
        self.save_crops = save_crops
        self.save_frames = save_frames
        self.quality = quality
        
        # Crear directorios
        if save_crops:
            self.crops_dir = self.output_dir / "captures" / "crops"
            self.crops_dir.mkdir(parents=True, exist_ok=True)
        
        if save_frames:
            self.frames_dir = self.output_dir / "captures" / "frames"
            self.frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Control de guardado para evitar duplicados
        self.saved_persons = set()
        self.frame_counter = 0
    
    def save_person_crop(self, frame, person_id, rect, confidence=None):
        """
        Guarda el recorte de una persona detectada
        
        Args:
            frame: Frame completo en BGR
            person_id: ID único de la persona
            rect: Tupla (x, y, w, h) del rostro
            confidence: Nivel de confianza (opcional)
            
        Returns:
            Ruta del archivo guardado o None si no se guardó
        """
        if not self.save_crops:
            return None
        
        # Evitar guardar múltiples veces la misma persona
        if person_id in self.saved_persons:
            return None
        
        x, y, w, h = rect
        
        # Validar coordenadas
        if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
            return None
        
        # Extraer región del rostro con un pequeño margen
        margin = 10
        y1 = max(0, y - margin)
        y2 = min(frame.shape[0], y + h + margin)
        x1 = max(0, x - margin)
        x2 = min(frame.shape[1], x + w + margin)
        
        face_crop = frame[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            return None
        
        # Generar nombre de archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
        confidence_str = f"_conf{confidence:.2f}" if confidence else ""
        filename = f"{person_id}_{timestamp}{confidence_str}.jpg"
        filepath = self.crops_dir / filename
        
        # Guardar imagen
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        success = cv2.imwrite(str(filepath), face_crop, encode_params)
        
        if success:
            self.saved_persons.add(person_id)
            return str(filepath)
        
        return None
    
    def save_full_frame(self, frame, event_info=None):
        """
        Guarda un frame completo
        
        Args:
            frame: Frame en BGR
            event_info: Información del evento (opcional)
            
        Returns:
            Ruta del archivo guardado o None si no se guardó
        """
        if not self.save_frames:
            return None
        
        self.frame_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        # Agregar información del evento al nombre si está disponible
        event_str = ""
        if event_info:
            if 'person_id' in event_info:
                event_str += f"_p{event_info['person_id']}"
            if 'event' in event_info:
                event_str += f"_{event_info['event']}"
        
        filename = f"frame_{timestamp}{event_str}.jpg"
        filepath = self.frames_dir / filename
        
        # Guardar imagen
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        success = cv2.imwrite(str(filepath), frame, encode_params)
        
        return str(filepath) if success else None
    
    def save_annotated_frame(self, frame, detections, tracked_objects, save_interval=30):
        """
        Guarda frames anotados periódicamente
        
        Args:
            frame: Frame original
            detections: Lista de detecciones
            tracked_objects: Objetos siendo trackeados
            save_interval: Intervalo en frames para guardar
            
        Returns:
            Ruta del archivo guardado o None
        """
        if not self.save_frames:
            return None
        
        # Guardar solo cada save_interval frames
        if self.frame_counter % save_interval != 0:
            self.frame_counter += 1
            return None
        
        # Crear copia del frame para anotar
        annotated_frame = frame.copy()
        
        # Dibujar detecciones (color por fuente: person_hog azul, mtcnn verde, default amarillo)
        for detection in detections:
            x, y, w, h = detection[:4]
            source = detection[5] if len(detection) > 5 else None
            if source == 'person_hog':
                color = (255, 0, 0)
            elif source == 'mtcnn':
                color = (0, 255, 0)
            else:
                color = (0, 255, 255)

            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
        
        # Dibujar objetos trackeados
        for obj_id, obj_data in tracked_objects.items():
            if 'rect' in obj_data and obj_data['rect']:
                x, y, w, h = obj_data['rect'][:4]
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(annotated_frame, f"ID: {obj_id}", 
                          (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Agregar información de conteo
        info_text = f"Detectados: {len(detections)} | Trackeados: {len(tracked_objects)}"
        cv2.putText(annotated_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return self.save_full_frame(annotated_frame, {'event': 'annotated'})
    
    def cleanup_old_files(self, max_files=1000):
        """
        Limpia archivos antiguos si hay demasiados
        
        Args:
            max_files: Número máximo de archivos por directorio
        """
        for directory in [self.crops_dir, self.frames_dir]:
            if not directory.exists():
                continue
            
            # Obtener lista de archivos ordenados por fecha de modificación
            files = sorted(directory.glob("*.jpg"), key=lambda x: x.stat().st_mtime)
            
            # Eliminar archivos más antiguos si hay demasiados
            if len(files) > max_files:
                files_to_delete = files[:-max_files]
                for file_path in files_to_delete:
                    try:
                        file_path.unlink()
                    except OSError:
                        pass  # Ignorar errores de eliminación
    
    def get_stats(self):
        """
        Obtiene estadísticas de archivos guardados
        
        Returns:
            Diccionario con estadísticas
        """
        stats = {
            'crops_saved': 0,
            'frames_saved': 0,
            'total_size_mb': 0
        }
        
        if self.crops_dir.exists():
            crop_files = list(self.crops_dir.glob("*.jpg"))
            stats['crops_saved'] = len(crop_files)
            stats['total_size_mb'] += sum(f.stat().st_size for f in crop_files)
        
        if self.frames_dir.exists():
            frame_files = list(self.frames_dir.glob("*.jpg"))
            stats['frames_saved'] = len(frame_files)
            stats['total_size_mb'] += sum(f.stat().st_size for f in frame_files)
        
        stats['total_size_mb'] = stats['total_size_mb'] / (1024 * 1024)  # Convert to MB
        stats['unique_persons_captured'] = len(self.saved_persons)
        
        return stats