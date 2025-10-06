"""
Detector de rostros usando DNN (Deep Neural Networks)
Más robusto que Haar Cascades, especialmente con variaciones de iluminación
"""

import cv2
import numpy as np
import os

class DNNFaceDetector:
    def __init__(self, model_path=None, config_path=None, confidence_threshold=0.6):
        """
        Inicializa el detector DNN
        
        Args:
            model_path: Ruta al modelo .caffemodel
            config_path: Ruta al archivo .prototxt
            confidence_threshold: Umbral mínimo de confianza
        """
        self.confidence_threshold = confidence_threshold
        
        # Rutas por defecto si no se especifican
        if model_path is None:
            model_path = "models/opencv_face_detector_uint8.pb"
        if config_path is None:
            config_path = "models/opencv_face_detector.pbtxt"
        
        # Verificar que existan los archivos
        self.available = True
        if not os.path.exists(model_path) or not os.path.exists(config_path):
            print(f"Advertencia: No se encontró el modelo/config DNN en {model_path} / {config_path}")
            print("Coloca los archivos de modelo en la carpeta 'models/' para usar DNN.")
            self.net = None
            self.available = False
            return

        try:
            # Cargar la red neuronal
            self.net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
            print("Modelo DNN cargado exitosamente")
            self.available = True
        except Exception as e:
            print(f"Error cargando modelo DNN: {e}")
            self.net = None
            self.available = False
    
    def _download_default_model(self):
        """Descargar modelo por defecto si no existe"""
        print("Para usar DNN, descarga los archivos del modelo desde:")
        print("https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector")
        print("- opencv_face_detector_uint8.pb")
        print("- opencv_face_detector.pbtxt")
        print("Y colócalos en la carpeta 'models/'")
    
    def detect_faces(self, frame):
        """
        Detecta rostros usando DNN
        
        Args:
            frame: Imagen en formato BGR
            
        Returns:
            List de tuplas (x, y, w, h, confidence)
        """
        if not self.available or self.net is None:
            # Detector no disponible
            return []
        
        h, w = frame.shape[:2]
        
        # Crear blob para la red neuronal
        blob = cv2.dnn.blobFromImage(
            frame, 
            scalefactor=1.0, 
            size=(300, 300), 
            mean=[104, 117, 123]
        )
        
        # Pasar el blob por la red
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        
        # Procesar las detecciones
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                # Obtener coordenadas del bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x, y, x1, y1 = box.astype(int)
                
                # Convertir a formato (x, y, w, h)
                width = x1 - x
                height = y1 - y
                
                # Validar que las coordenadas estén dentro de la imagen
                if x >= 0 and y >= 0 and x + width <= w and y + height <= h:
                    faces.append((x, y, width, height, float(confidence)))
        
        return faces
    
    def get_method_name(self):
        """Retorna el nombre del método de detección"""
        return "DNN" if getattr(self, 'available', False) else "DNN (unavailable)"