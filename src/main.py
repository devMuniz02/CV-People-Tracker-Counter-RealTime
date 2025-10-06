"""
Sistema Principal de Conteo de Personas en Tiempo Real
Universidad de Monterrey (UDEM)

Este programa detecta y cuenta personas en tiempo real durante 15 minutos,
evitando conteos duplicados mediante tracking y generando evidencia completa.

Autor: [Tu Nombre]
Fecha: Octubre 2025
"""

import sys
import json
import time
import cv2
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Importar módulos del proyecto
from detectors import HaarFaceDetector, DNNFaceDetector
from tracking import CentroidTracker, RegionOfInterest
from data_io import EventLogger, SessionManager, ImageSaver


# Lightweight wrappers for optional detectors (import lazily so missing packages don't break the module)
class MTCNNFaceDetector:
    """Wrapper to adapt the TensorFlow `mtcnn` package to the project's detector interface.
    """
    def __init__(self, *args, **kwargs):
        self.impl = None
        self.model = None
        self.available = False

        # Use TensorFlow 'mtcnn' package (do NOT use facenet-pytorch)
        try:
            from mtcnn import MTCNN as MT
            self.model = MT()
            self.impl = 'mtcnn'
            self.available = True
        except Exception:
            print('⚠ MTCNN (TensorFlow mtcnn) no disponible. Instala con: pip install mtcnn')
            self.model = None
            self.available = False

    def detect_faces(self, frame):
        if not self.available or self.model is None:
            return []

        try:
            import cv2
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # mtcnn package detect_faces returns list of dicts
            if self.impl == 'mtcnn':
                results = self.model.detect_faces(rgb)
                detections = []
                for r in results:
                    box = r.get('box', None)
                    conf = r.get('confidence', 0.0)
                    if box is None:
                        continue
                    x, y, w, h = box
                    x = max(0, int(x)); y = max(0, int(y)); w = int(w); h = int(h)
                    detections.append((x, y, w, h, float(conf)))
                return detections

            return []
        except Exception:
            return []

    def get_method_name(self):
        return 'MTCNN' if self.impl == 'mtcnn' else 'MTCNN (unavailable)'


class DlibFaceDetector:
    """Simple wrapper around dlib's frontal face detector (if available)."""
    def __init__(self):
        try:
            import dlib
            self.detector = dlib.get_frontal_face_detector()
            self.available = True
        except Exception:
            print('⚠ dlib no disponible (instala "pip install dlib" para usarlo).')
            self.detector = None
            self.available = False

    def detect_faces(self, frame):
        if not self.available:
            return []
        try:
            import cv2
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)
            detections = []
            h_img, w_img = gray.shape[:2]
            for r in rects:
                x = max(0, int(r.left()))
                y = max(0, int(r.top()))
                x2 = min(w_img - 1, int(r.right()))
                y2 = min(h_img - 1, int(r.bottom()))
                w = x2 - x
                h = y2 - y
                detections.append((x, y, w, h, 0.8))
            return detections
        except Exception:
            return []

    def get_method_name(self):
        return 'Dlib' if self.available else 'Dlib (unavailable)'


class PersonHOGDetector:
    """Person detector using OpenCV's HOGDescriptor + default people detector (no external models)."""
    def __init__(self):
        try:
            import cv2
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.available = True
        except Exception:
            print('⚠ HOG person detector no disponible en esta build de OpenCV.')
            self.hog = None
            self.available = False

    def detect_faces(self, frame):
        # Note: returns person detections as (x,y,w,h,confidence)
        if not self.available:
            return []
        try:
            import cv2
            # work on a resized grayscale copy for speed
            img = frame.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects, weights = self.hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
            detections = []
            for (x, y, w, h), wt in zip(rects, weights):
                detections.append((int(x), int(y), int(w), int(h), float(wt)))
            return detections
        except Exception:
            return []

    def get_method_name(self):
        return 'PersonHOG' if self.available else 'PersonHOG (unavailable)'


class RetinaFaceDetector:
    """Wrapper for RetinaFace detectors if the package is available."""
    def __init__(self):
        try:
            # The 'retinaface' package exposes RetinaFace.detect_faces
            from retinaface import RetinaFace
            self.api = RetinaFace
            self.available = True
        except Exception:
            try:
                # Some installs provide a different import path
                from retinaface import RetinaFace as RF
                self.api = RF
                self.available = True
            except Exception:
                print('⚠ RetinaFace no disponible (instala "pip install retina-face" o similar).')
                self.api = None
                self.available = False

    def detect_faces(self, frame):
        if not self.available:
            return []
        try:
            # RetinaFace.detect_faces expects RGB or BGR depending on implementation; many expect BGR
            results = self.api.detect_faces(frame)
            detections = []
            if isinstance(results, dict):
                # iterate over detected faces
                for k, v in results.items():
                    box = v.get('facial_area') if isinstance(v, dict) else None
                    conf = v.get('score', 0.0) if isinstance(v, dict) else 0.0
                    if box is None:
                        continue
                    x1, y1, x2, y2 = box
                    w = int(x2 - x1)
                    h = int(y2 - y1)
                    detections.append((int(x1), int(y1), w, h, float(conf)))
            return detections
        except Exception:
            return []

    def get_method_name(self):
        return 'RetinaFace' if self.available else 'RetinaFace (unavailable)'

class PeopleCounterSystem:
    """
    Sistema principal de conteo de personas
    """
    
    def __init__(self, config_path="config.json"):
        """
        Inicializa el sistema de conteo
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        # Cargar configuración
        self.config = self._load_config(config_path)
        
        # Inicializar componentes
        self.detector = self._initialize_detector()
        # Always initialize a person detector (HOG) to run alongside face detectors
        # PersonHOG detection removed from main runtime (handled only in tuning GUI if needed)
        self.person_hog = None
        self.tracker = self._initialize_tracker()
        self.roi = RegionOfInterest()
        
        # Variables de sesión
        self.session_dir = None
        self.logger = None
        self.image_saver = None
        self.is_running = False
        
        # Control de tiempo
        self.session_duration = timedelta(minutes=self.config['session']['duration_minutes'])
        self.start_time = None
        
        # Estadísticas
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
    def _load_config(self, config_path):
        """Carga la configuración desde archivo JSON"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = json.load(file)
            print(f"✓ Configuración cargada desde {config_path}")
            return config
        except FileNotFoundError:
            print(f"⚠ No se encontró {config_path}, usando configuración por defecto")
            return self._get_default_config()
        except json.JSONDecodeError:
            print(f"⚠ Error en formato JSON de {config_path}, usando configuración por defecto")
            return self._get_default_config()
    
    def _get_default_config(self):
        """Retorna configuración por defecto"""
        return {
            "detection": {
                "method": "mtcnn",
                "scale_factor": 1.1,
                "min_neighbors": 5,
                "min_size": [20, 20],
                "max_size": [40, 40],
                "confidence_threshold": 0.6
            },
            "tracking": {
                "max_disappeared_frames": 30,
                "max_distance_threshold": 100
            },
            "session": {
                "duration_minutes": 15,
                "location": "Lab Robotica",
                "fps_limit": 30,
                "save_full_frames": True,
                "save_face_crops": True
            },
            "preprocessing": {
                "resize_width": 640,
                "apply_blur": False,
                "apply_clahe": False
            },
            "output": {
                "base_path": "output",
                "image_quality": 95
            }
        }
    
    def _initialize_detector(self):
        """Inicializa el detector de rostros según la configuración"""
        method = self.config['detection']['method'].lower()
        # DNN and Haar are built-in
        if method == "dnn":
            print("🔍 Inicializando detector DNN...")
            return DNNFaceDetector(
                confidence_threshold=self.config['detection']['confidence_threshold']
            )
        elif method == "haar":
            # Map legacy 'haar' setting to MTCNN to avoid using Haar in this project
            print("🔍 'haar' mapeado a MTCNN (usar MTCNN en lugar de Haar)...")
            return MTCNNFaceDetector()
        elif method == 'mtcnn':
            print('🔍 Intentando inicializar MTCNN...')
            return MTCNNFaceDetector()
        elif method == 'dlib':
            print('🔍 Intentando inicializar Dlib frontal detector...')
            return DlibFaceDetector()
        elif method == 'person_hog':
            print('🔍 Inicializando detector de personas (HOG)')
            return PersonHOGDetector()
        else:
            # Default to MTCNN rather than Haar for unknown methods
            print(f"⚠ Método desconocido '{method}', usando MTCNN por defecto")
            return MTCNNFaceDetector()
    
    def _initialize_tracker(self):
        """Inicializa el sistema de tracking"""
        print("🎯 Inicializando sistema de tracking...")
        return CentroidTracker(
            max_disappeared=self.config['tracking']['max_disappeared_frames'],
            max_distance=self.config['tracking']['max_distance_threshold']
        )
    
    def setup_session(self):
        """Configura una nueva sesión de conteo"""
        print("\n" + "="*50)
        print("🚀 INICIANDO SESIÓN DE CONTEO DE PERSONAS")
        print("="*50)
        
        # Crear directorio de sesión
        location = self.config['session']['location']
        self.session_dir = SessionManager.create_session_directory(
            base_path=self.config['output']['base_path'],
            location=location
        )
        
        print(f"📁 Directorio de sesión: {self.session_dir}")
        
        # Inicializar logger; include both detectors in method information
        method_name = self.detector.get_method_name() if self.detector else 'Unknown'

        self.logger = EventLogger(
            output_dir=self.session_dir,
            location=location,
            method=method_name
        )
        
        # Inicializar guardador de imágenes
        self.image_saver = ImageSaver(
            output_dir=self.session_dir,
            save_crops=self.config['session']['save_face_crops'],
            save_frames=self.config['session']['save_full_frames'],
            quality=self.config['output']['image_quality']
        )
        
        # Guardar configuración de la sesión
        self.logger.save_session_config(self.config)
        
        print(f"📊 Duración de sesión: {self.config['session']['duration_minutes']} minutos")
        print(f"📍 Ubicación: {location}")
        print(f"🔍 Método de detección: {self.detector.get_method_name()}")
        
    def preprocess_frame(self, frame):
        """
        Preprocesa el frame según la configuración
        
        Args:
            frame: Frame original de la cámara
            
        Returns:
            Frame preprocesado
        """
        # Redimensionar si es necesario
        if 'resize_width' in self.config['preprocessing']:
            width = self.config['preprocessing']['resize_width']
            if frame.shape[1] != width:
                height = int(frame.shape[0] * width / frame.shape[1])
                frame = cv2.resize(frame, (width, height))
        
        # Aplicar blur si está habilitado
        if self.config['preprocessing'].get('apply_blur', False):
            kernel_size = self.config['preprocessing'].get('blur_kernel_size', 3)
            frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        
        # Aplicar CLAHE si está habilitado
        if self.config['preprocessing'].get('apply_clahe', False):
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Aplicar equalizeHist en escala de grises si se solicita
        if self.config['preprocessing'].get('apply_equalize', False):
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                eq = cv2.equalizeHist(gray)
                frame = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
            except Exception:
                pass
        
        return frame
    
    def draw_interface(self, frame, tracked_objects, elapsed_time, remaining_time):
        """
        Dibuja la interfaz de usuario en el frame
        
        Args:
            frame: Frame a anotar
            tracked_objects: Objetos siendo trackeados
            elapsed_time: Tiempo transcurrido
            remaining_time: Tiempo restante
            
        Returns:
            Frame anotado
        """
        # Draw the info panel outside the image so it doesn't cover the frame.
        panel_height = 120
        panel_color = (0, 0, 0)
        info_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Create a new image with space for the panel on top
        h_img, w_img = frame.shape[:2]
        out_h = h_img + panel_height
        out = np.zeros((out_h, w_img, 3), dtype=frame.dtype)

        # Fill panel area with panel_color and copy frame below
        out[0:panel_height, :, :] = panel_color
        out[panel_height:panel_height + h_img, 0:w_img, :] = frame

        # Draw bounding boxes and IDs onto the image area (offset Y by panel_height)
        y_off = panel_height
        for obj_id, obj_data in tracked_objects.items():
            if 'rect' in obj_data and obj_data['rect']:
                rect = obj_data['rect']
                x, y, w, h = rect[:4]
                source = obj_data.get('source', None)

                box_color = (0, 255, 0)  # Green
                text_color = (0, 255, 0)

                # Bounding box principal (apply vertical offset)
                cv2.rectangle(out, (x, y + y_off), (x + w, y + h + y_off), box_color, 2)

                # ID de la persona
                label = f"ID: {obj_id}"
                cv2.putText(out, label, (x, y - 10 + y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

                # Centroide (use contrasting color)
                if 'centroid' in obj_data:
                    cx, cy = obj_data['centroid']
                    cv2.circle(out, (int(cx), int(cy) + y_off), 4, (0, 128, 255), -1)
        
        # Línea 1: Conteos (draw on panel)
        total_count = self.tracker.get_total_count()
        active_count = len(tracked_objects)
        text1 = f"Personas Unicas: {total_count} | Activas: {active_count}"
        cv2.putText(out, text1, (10, 25), font, 0.7, info_color, 2)

        # Línea 2: Tiempo
        elapsed_str = str(elapsed_time).split('.')[0]
        remaining_str = str(remaining_time).split('.')[0]
        text2 = f"Transcurrido: {elapsed_str} | Restante: {remaining_str}"
        cv2.putText(out, text2, (10, 50), font, 0.7, info_color, 2)

        # Línea 3: Información técnica
        text3 = f"FPS: {self.current_fps:.1f} | Metodo: {self.detector.get_method_name()}"
        cv2.putText(out, text3, (10, 75), font, 0.7, info_color, 2)

        # Línea 4: Ubicación (usar nombre del directorio de sesión si está disponible)
        try:
            if self.session_dir:
                loc_name = Path(self.session_dir).name
            else:
                loc_name = self.config['session'].get('location', '')
        except Exception:
            loc_name = self.config['session'].get('location', '')

        text4 = f"Ubicacion: {loc_name}"
        cv2.putText(out, text4, (10, 100), font, 0.7, info_color, 2)

        # Leyenda de colores (top-right of panel): mostrar el método de detección actual
        try:
            legend_x = w_img - 300
            legend_y = 16
            box_w = 14
            box_h = 14

            method_label = self.detector.get_method_name() if self.detector else 'Detector'
            # Use green box for primary face detector visual
            cv2.rectangle(out, (legend_x, legend_y), (legend_x + box_w, legend_y + box_h), (0, 255, 0), -1)
            cv2.putText(out, method_label, (legend_x + box_w + 8, legend_y + 12), font, 0.5, info_color, 1)
        except Exception:
            pass

        # Indicador de tiempo restante (barra de progreso) on panel
        progress_width = 300
        progress_height = 10
        progress_x = w_img - progress_width - 10
        progress_y = panel_height - progress_height - 8

        # Fondo de la barra
        cv2.rectangle(out, (progress_x, progress_y),
                     (progress_x + progress_width, progress_y + progress_height),
                     (50, 50, 50), -1)

        # Progreso
        elapsed_seconds = elapsed_time.total_seconds()
        total_seconds = self.session_duration.total_seconds()
        progress = min(1.0, elapsed_seconds / total_seconds)

        progress_color = (0, 255, 0) if progress < 0.8 else (0, 255, 255) if progress < 0.95 else (0, 0, 255)
        progress_fill = int(progress_width * progress)
        cv2.rectangle(out, (progress_x, progress_y),
                     (progress_x + progress_fill, progress_y + progress_height),
                     progress_color, -1)

        return out
    
    def update_fps(self):
        """Actualiza el contador de FPS"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def process_detections(self, detections, frame):
        """
        Procesa las detecciones y actualiza el tracking
        
        Args:
            detections: Lista de detecciones del frame actual
            frame: Frame actual
            
        Returns:
            Diccionario de objetos trackeados
        """
        # Convertir detecciones (posiblemente con campo 'source') a formato para el tracker
        rects = []
        for d in detections:
            try:
                x = int(d[0]); y = int(d[1]); w = int(d[2]); h = int(d[3])
                rects.append((x, y, w, h))
            except Exception:
                continue

        # Actualizar tracker
        tracked_objects = self.tracker.update(rects)
        
        # Procesar eventos para cada objeto trackeado
        for obj_id, obj_data in tracked_objects.items():
            disappeared_frames = obj_data.get('disappeared_frames', 0)
            
            # Si es un nuevo objeto (primera vez que se ve)
            if obj_id not in self.logger.logged_persons:
                # Obtener información de confianza si está disponible
                confidence = None
                if obj_data.get('rect') and len(detections) > 0:
                    # Buscar la detección correspondiente por proximidad
                    obj_rect = obj_data['rect']
                    for det in detections:
                        det_rect = det[:4]
                        # Calcular solapamiento simple
                        if self._rects_overlap(obj_rect, det_rect):
                            # det may be (x,y,w,h,conf,source) or shorter
                            confidence = det[4] if len(det) > 4 else None
                            source = det[5] if len(det) > 5 else None
                            # Store source/confidence metadata but keep rect as (x,y,w,h)
                            try:
                                x0, y0, w0, h0 = det_rect
                                obj_data['rect'] = (int(x0), int(y0), int(w0), int(h0))
                                obj_data['source'] = source
                                obj_data['confidence'] = confidence
                            except Exception:
                                pass
                            break
                
                # Registrar entrada
                self.logger.log_person_enter(obj_id, confidence)
                
                # Guardar captura del rostro
                if obj_data.get('rect'):
                    self.image_saver.save_person_crop(
                        frame, obj_id, obj_data['rect'], confidence
                    )
                
                # Guardar frame completo del evento
                self.image_saver.save_full_frame(
                    frame, {'person_id': obj_id, 'event': 'ENTER'}
                )
            
            # Si la persona ha sido vista de forma estable
            elif disappeared_frames == 0:
                # Log periódico cada 60 frames (~2 segundos a 30 FPS)
                if self.frame_count % 60 == 0:
                    self.logger.log_person_seen(obj_id, self.frame_count)
        
        return tracked_objects

    def _filter_detections_by_size(self, detections):
        """Filtra la lista de detecciones (x,y,w,h,conf) dejando solo las que
        estén dentro del rango min_size..max_size configurado.
        """
        if not detections:
            return []

        min_size = tuple(self.config['detection'].get('min_size', (0, 0)))
        max_size = tuple(self.config['detection'].get('max_size', (99999, 99999)))

        min_w, min_h = int(min_size[0]), int(min_size[1])
        max_w, max_h = int(max_size[0]), int(max_size[1])

        filtered = []
        for det in detections:
            # det may be (x,y,w,h,conf,source) or (x,y,w,h,conf)
            try:
                x, y, w, h = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            except Exception:
                continue

            # If detection comes from person_hog, do not apply size constraints
            source = det[5] if len(det) > 5 else None
            if source == 'person_hog':
                filtered.append(det)
                continue

            # For face detectors (mtcnn, dnn, etc.) apply size limits
            if w >= min_w and h >= min_h and w <= max_w and h <= max_h:
                filtered.append(det)

        return filtered
    
    def _rects_overlap(self, rect1, rect2, threshold=0.3):
        """
        Verifica si dos rectángulos se solapan significativamente
        
        Args:
            rect1: Primer rectángulo (x, y, w, h)
            rect2: Segundo rectángulo (x, y, w, h)
            threshold: Umbral mínimo de solapamiento
            
        Returns:
            True si se solapan, False en caso contrario
        """
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        
        # Calcular intersección
        left = max(x1, x2)
        right = min(x1 + w1, x2 + w2)
        top = max(y1, y2)
        bottom = min(y1 + h1, y2 + h2)
        
        if left < right and top < bottom:
            intersection = (right - left) * (bottom - top)
            area1 = w1 * h1
            area2 = w2 * h2
            union = area1 + area2 - intersection
            
            return (intersection / union) > threshold if union > 0 else False
        
        return False
    
    def run(self):
        """
        Ejecuta el sistema de conteo principal
        """
        try:
            # Configurar sesión
            self.setup_session()
            
            # Inicializar fuente de vídeo (archivo preferido, fallback a cámara)
            video_path = Path("DeteccionDePersonas.mov")
            print("\n📹 Abriendo fuente de vídeo...")

            if video_path.exists():
                print(f"📁 Usando archivo de vídeo: {video_path}")
                cap = cv2.VideoCapture(str(video_path))
            else:
                print(f"⚠ Archivo {video_path} no encontrado. Intentando abrir la cámara...")
                cap = cv2.VideoCapture(0)  # type: ignore

            if not cap.isOpened():
                print("❌ Error: No se pudo abrir la fuente de vídeo (archivo ni cámara)")
                return

            # Intentar aplicar configuración de tamaño/FPS (puede no aplicarse a archivos)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, self.config['session']['fps_limit'])

            # Si la fuente es un archivo de vídeo, leer su FPS real y adaptarnos.
            try:
                if video_path.exists():
                    source_fps = cap.get(cv2.CAP_PROP_FPS)
                    if source_fps and source_fps > 0:
                        # Ajustar el límite de FPS de la sesión al FPS del archivo
                        print(f"ℹ️ FPS detectado en el archivo: {source_fps:.2f}")
                        # Usar un entero razonable
                        self.config['session']['fps_limit'] = int(round(source_fps))
            except Exception:
                # No crítico: si falla la lectura del FPS, continuamos con la configuración existente
                pass

            print("✓ Fuente de vídeo inicializada")
            print("\n🎬 Iniciando sesión de conteo...")
            print("Presiona 'q' para terminar antes de tiempo")
            print("Presiona 'r' para definir región de interés")
            print("Presiona 's' para guardar captura manual")
            
            # Variables de control
            self.start_time = datetime.now()
            self.is_running = True
            
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error leyendo frame de la cámara")
                    break
                
                self.frame_count += 1
                self.update_fps()
                
                # Calcular tiempo transcurrido y restante
                elapsed_time = datetime.now() - self.start_time
                remaining_time = self.session_duration - elapsed_time
                
                # Verificar si se acabó el tiempo
                if remaining_time.total_seconds() <= 0:
                    print("\n⏰ Tiempo de sesión completado!")
                    break
                
                # Preprocesar frame
                processed_frame = self.preprocess_frame(frame)

                # Detectar con detector principal y taggear con el método configurado
                raw_detections = self.detector.detect_faces(processed_frame)
                detections = []
                method_tag = self.config['detection'].get('method', 'mtcnn')
                for det in raw_detections:
                    try:
                        x, y, w, h = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                        conf = float(det[4]) if len(det) > 4 else 0.0
                        detections.append((x, y, w, h, conf, method_tag))
                    except Exception:
                        continue
                
                # Filtrar detecciones por ROI si está configurada
                detections = self.roi.filter_detections(detections)

                # Filtrar detecciones por tamaño (min/max)
                detections = self._filter_detections_by_size(detections)
                
                # Procesar tracking y eventos
                tracked_objects = self.process_detections(detections, processed_frame)
                
                # Dibujar interfaz
                display_frame = self.draw_interface(
                    processed_frame, tracked_objects, elapsed_time, remaining_time
                )
                
                # Guardar frames anotados periódicamente
                self.image_saver.save_annotated_frame(
                    display_frame, detections, tracked_objects, save_interval=60
                )
                
                # Mostrar frame (título dinámico con la ubicación - usar nombre del directorio de sesión si disponible)
                try:
                    win_loc = Path(self.session_dir).name if self.session_dir else self.config['session'].get('location', '')
                except Exception:
                    win_loc = self.config['session'].get('location', '')
                window_title = f"Contador de Personas - {win_loc}"
                cv2.imshow(window_title, display_frame)
                
                # Procesar teclas
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n🛑 Sesión terminada por el usuario")
                    break
                elif key == ord('s'):
                    # Guardar captura manual
                    self.image_saver.save_full_frame(
                        display_frame, {'event': 'manual_capture'}
                    )
                    print("📸 Captura manual guardada")
                elif key == ord('r'):
                    print("🎯 Función ROI no implementada en esta versión")
                
                # Limitar FPS si es necesario
                time.sleep(1.0 / self.config['session']['fps_limit'])
            
            # Limpiar
            cap.release()
            cv2.destroyAllWindows()
            
            # Generar reportes finales
            self.generate_final_report()
            
        except KeyboardInterrupt:
            print("\n⚠ Programa interrumpido por el usuario")
        except Exception as e:
            print(f"\n❌ Error durante la ejecución: {e}")
        finally:
            self.cleanup()
    
    def generate_final_report(self):
        """Genera reportes finales de la sesión"""
        print("\n" + "="*50)
        print("📊 GENERANDO REPORTES FINALES")
        print("="*50)
        
        # Resumen de la sesión
        summary = self.logger.get_session_summary()
        
        print(f"📈 Total personas únicas detectadas: {summary['total_unique_persons']}")
        print(f"⏱ Duración de sesión: {summary['session_duration_seconds']:.0f} segundos")
        print(f"📍 Ubicación: {summary['location']}")
        print(f"🔍 Método utilizado: {summary['method']}")
        
        # Estadísticas de imágenes
        img_stats = self.image_saver.get_stats()
        print(f"📷 Recortes guardados: {img_stats['crops_saved']}")
        print(f"🎬 Frames guardados: {img_stats['frames_saved']}")
        print(f"💾 Tamaño total: {img_stats['total_size_mb']:.2f} MB")
        
        # Exportar reporte
        report_path = self.logger.export_summary_report()
        print(f"📄 Reporte exportado: {report_path}")
        
        print(f"📁 Todos los archivos en: {self.session_dir}")
    
    def cleanup(self):
        """Limpia recursos del sistema"""
        cv2.destroyAllWindows()
        print("🧹 Limpieza completada")

def main():
    """Función principal"""
    print("🎯 Sistema de Conteo de Personas en Tiempo Real")
    print("📚 Universidad de Monterrey (UDEM)")
    print("⚡ Powered by OpenCV + Python")
    
    try:
        # Verificar que OpenCV esté instalado
        print(f"📦 OpenCV versión: {cv2.__version__}")
        
        # Crear y ejecutar sistema
        system = PeopleCounterSystem()
        system.run()
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        print("💡 Instala las dependencias con: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

if __name__ == "__main__":
    main()