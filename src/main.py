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
                "method": "haar",
                "scale_factor": 1.1,
                "min_neighbors": 5,
                "min_size": [60, 60],
                "confidence_threshold": 0.6
            },
            "tracking": {
                "max_disappeared_frames": 30,
                "max_distance_threshold": 100
            },
            "session": {
                "duration_minutes": 15,
                "location": "Cafetería UDEM",
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
        
        if method == "dnn":
            print("🔍 Inicializando detector DNN...")
            return DNNFaceDetector(
                confidence_threshold=self.config['detection']['confidence_threshold']
            )
        else:
            print("🔍 Inicializando detector Haar Cascade...")
            return HaarFaceDetector(
                scale_factor=self.config['detection']['scale_factor'],
                min_neighbors=self.config['detection']['min_neighbors'],
                min_size=tuple(self.config['detection']['min_size'])
            )
    
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
        
        # Inicializar logger
        self.logger = EventLogger(
            output_dir=self.session_dir,
            location=location,
            method=self.detector.get_method_name()
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
        overlay = frame.copy()
        
        # Dibujar bounding boxes y IDs
        for obj_id, obj_data in tracked_objects.items():
            if 'rect' in obj_data and obj_data['rect']:
                x, y, w, h = obj_data['rect'][:4]
                
                # Bounding box principal
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # ID de la persona
                label = f"ID: {obj_id}"
                cv2.putText(frame, label, (x, y - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Centroide
                if 'centroid' in obj_data:
                    cx, cy = obj_data['centroid']
                    cv2.circle(frame, (int(cx), int(cy)), 4, (255, 0, 0), -1)
        
        # Panel de información
        panel_height = 120
        panel_color = (0, 0, 0)
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_height), panel_color, -1)
        
        # Aplicar transparencia
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Información de la sesión
        info_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Línea 1: Conteos
        total_count = self.tracker.get_total_count()
        active_count = len(tracked_objects)
        text1 = f"Personas Unicas: {total_count} | Activas: {active_count}"
        cv2.putText(frame, text1, (10, 25), font, 0.7, info_color, 2)
        
        # Línea 2: Tiempo
        elapsed_str = str(elapsed_time).split('.')[0]
        remaining_str = str(remaining_time).split('.')[0]
        text2 = f"Transcurrido: {elapsed_str} | Restante: {remaining_str}"
        cv2.putText(frame, text2, (10, 50), font, 0.7, info_color, 2)
        
        # Línea 3: Información técnica
        text3 = f"FPS: {self.current_fps:.1f} | Metodo: {self.detector.get_method_name()}"
        cv2.putText(frame, text3, (10, 75), font, 0.7, info_color, 2)
        
        # Línea 4: Ubicación
        location = self.config['session']['location']
        text4 = f"Ubicacion: {location}"
        cv2.putText(frame, text4, (10, 100), font, 0.7, info_color, 2)
        
        # Indicador de tiempo restante (barra de progreso)
        progress_width = 300
        progress_height = 10
        progress_x = frame.shape[1] - progress_width - 10
        progress_y = 10
        
        # Fondo de la barra
        cv2.rectangle(frame, (progress_x, progress_y), 
                     (progress_x + progress_width, progress_y + progress_height), 
                     (50, 50, 50), -1)
        
        # Progreso
        elapsed_seconds = elapsed_time.total_seconds()
        total_seconds = self.session_duration.total_seconds()
        progress = min(1.0, elapsed_seconds / total_seconds)
        
        progress_color = (0, 255, 0) if progress < 0.8 else (0, 255, 255) if progress < 0.95 else (0, 0, 255)
        progress_fill = int(progress_width * progress)
        cv2.rectangle(frame, (progress_x, progress_y), 
                     (progress_x + progress_fill, progress_y + progress_height), 
                     progress_color, -1)
        
        return frame
    
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
        # Convertir detecciones a formato para el tracker
        rects = [(x, y, w, h) for (x, y, w, h, conf) in detections]
        
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
                            confidence = det[4]
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
            
            # Inicializar cámara
            print("\n📹 Iniciando cámara...")
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("❌ Error: No se pudo abrir la cámara")
                return
            
            # Configurar cámara
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, self.config['session']['fps_limit'])
            
            print("✓ Cámara inicializada")
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
                
                # Detectar rostros
                detections = self.detector.detect_faces(processed_frame)
                
                # Filtrar detecciones por ROI si está configurada
                detections = self.roi.filter_detections(detections)
                
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
                
                # Mostrar frame
                cv2.imshow('Contador de Personas UDEM', display_frame)
                
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