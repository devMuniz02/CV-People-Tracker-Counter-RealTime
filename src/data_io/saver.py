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
        # Do not create a frames directory; we only keep crop evidence (IDs)
        # (Frames folder creation was intentionally removed per project requirement.)
        if not save_frames:
            self.frames_dir = None
        
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

        # Compensate for possible top panel added by the annotation UI
        # Some display frames include a top info panel (black bar) that shifts
        # the visual coordinates. Detect a dark horizontal band at the top of
        # the provided frame and subtract its height from the crop Y coordinate
        # (only if it seems present and y is below the band).
        try:
            offset = self._detect_top_panel_offset(frame)
            if offset > 0:
                # only adjust if y is greater than the offset (prevents negative coords)
                y = max(0, y - offset)
        except Exception:
            # if detection fails, fall back to given rect
            pass
        
        # Clamp coordinates to image bounds (allow detections at edges)
        x = max(0, int(x))
        y = max(0, int(y))
        w = max(0, int(min(w, frame.shape[1] - x)))
        h = max(0, int(min(h, frame.shape[0] - y)))
        if w == 0 or h == 0:
            return None

        # Expand crop region asymmetrically so ID features on clothing (below face)
        # are more likely included. Use a smaller upward margin and a larger
        # downward margin (toward torso). Horizontal margin is symmetric.
        frac_h = 0.45  # horizontal margin as fraction of bbox
        frac_up = 0.25  # upward margin as fraction of bbox height
        frac_down = 1.25  # downward margin as fraction of bbox height (bigger)
        min_margin = 20
        max_margin = 600

        horiz_margin = int(max(min_margin, min(max_margin, max(w, h) * frac_h)))
        up_margin = int(max(min_margin, min(max_margin, h * frac_up)))
        down_margin = int(max(min_margin, min(max_margin, h * frac_down)))

        y1 = max(0, y - up_margin)
        y2 = min(frame.shape[0], y + h + down_margin)
        x1 = max(0, x - horiz_margin)
        x2 = min(frame.shape[1], x + w + horiz_margin)

        # We only save the expanded asymmetric crop (ID on clothing)
        id_crop = frame[y1:y2, x1:x2]
        if id_crop.size == 0:
            return None

        # Generar nombre de archivo para el ID crop
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        confidence_str = f"_conf{confidence:.2f}" if confidence else ""
        id_filename = f"{person_id}_id_{timestamp}{confidence_str}.jpg"
        id_path = self.crops_dir / id_filename

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        try:
            saved = cv2.imwrite(str(id_path), id_crop, encode_params)
        except Exception:
            saved = False

        if saved:
            self.saved_persons.add(person_id)
            return str(id_path)

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
        # Full-frame saving disabled by project requirement; return None.
        return None
    
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
        dirs = [self.crops_dir]
        if getattr(self, 'frames_dir', None):
            dirs.append(self.frames_dir)
        for directory in dirs:
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
        
        if getattr(self, 'frames_dir', None) and self.frames_dir.exists():
            frame_files = list(self.frames_dir.glob("*.jpg"))
            stats['frames_saved'] = len(frame_files)
            stats['total_size_mb'] += sum(f.stat().st_size for f in frame_files)
        
        stats['total_size_mb'] = stats['total_size_mb'] / (1024 * 1024)  # Convert to MB
        stats['unique_persons_captured'] = len(self.saved_persons)
        
        return stats

    def _detect_top_panel_offset(self, frame, max_search_frac=0.4, dark_thresh=40, var_thresh=100.0):
        """
        Detect a dark horizontal band at the top of the frame (UI info panel)

        Args:
            frame: BGR image (numpy array)
            max_search_frac: fraction of height to scan from the top (e.g. 0.4)
            dark_thresh: mean intensity threshold to consider row as dark
            var_thresh: variance threshold to consider area uniform (low var)

        Returns:
            offset (int): number of pixels in top dark band (0 if none detected)
        """
        try:
            import numpy as _np
            import cv2 as _cv2
            h = frame.shape[0]
            max_search = min(h, int(h * float(max_search_frac)))
            if max_search <= 0:
                return 0

            gray = _cv2.cvtColor(frame[:max_search, :, :], _cv2.COLOR_BGR2GRAY)

            # compute mean intensity per row
            row_means = _np.mean(gray, axis=1)
            row_vars = _np.var(gray, axis=1)

            # find initial consecutive rows that are dark and low variance
            offset = 0
            for i, (m, v) in enumerate(zip(row_means, row_vars)):
                if m < dark_thresh and v < var_thresh:
                    offset = i + 1
                else:
                    # stop when a non-dark/non-uniform row is found after we've seen some dark rows
                    if offset > 0:
                        break

            # Heuristic: if offset is very small (<6 px), ignore it
            if offset < 6:
                return 0
            return int(offset)
        except Exception:
            return 0