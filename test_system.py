"""
Script de prueba para verificar que todos los componentes funcionen correctamente
"""

import sys
import os

# Agregar src al path para importar módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Prueba que todas las importaciones funcionen"""
    print("🔍 Probando importaciones...")
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV no está instalado")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError:
        print("❌ NumPy no está instalado")
        return False
    
    try:
        from detectors import HaarFaceDetector, DNNFaceDetector
        print("✅ Detectores importados correctamente")
    except ImportError as e:
        print(f"❌ Error importando detectores: {e}")
        return False
    
    try:
        from tracking import CentroidTracker
        print("✅ Sistema de tracking importado")
    except ImportError as e:
        print(f"❌ Error importando tracking: {e}")
        return False
    
    try:
        from data_io import EventLogger, ImageSaver
        print("✅ Sistema de I/O importado")
    except ImportError as e:
        print(f"❌ Error importando data_io: {e}")
        return False
    
    return True

def test_camera():
    """Prueba que la cámara esté disponible"""
    print("\n📹 Probando cámara...")
    
    import cv2
    
    # Probar cámaras disponibles
    available_cameras = []
    for i in range(3):  # Probar primeras 3 cámaras
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                available_cameras.append(i)
                print(f"✅ Cámara {i} disponible: {frame.shape}")
            cap.release()
        else:
            print(f"❌ Cámara {i} no disponible")
    
    if available_cameras:
        print(f"✅ Cámaras disponibles: {available_cameras}")
        return True
    else:
        print("❌ No se encontraron cámaras disponibles")
        return False

def test_face_detection():
    """Prueba el detector de rostros con imagen de prueba"""
    print("\n🔍 Probando detección de rostros...")
    
    try:
        import cv2
        import numpy as np
        from detectors import HaarFaceDetector
        
        # Crear imagen de prueba simple
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image.fill(128)  # Gris
        
        # Inicializar detector
        detector = HaarFaceDetector()
        
        # Probar detección (debería retornar lista vacía)
        detections = detector.detect_faces(test_image)
        print(f"✅ Detector funcional, detecciones: {len(detections)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en detección: {e}")
        return False

def test_tracking():
    """Prueba el sistema de tracking"""
    print("\n🎯 Probando sistema de tracking...")
    
    try:
        from tracking import CentroidTracker
        
        # Crear tracker
        tracker = CentroidTracker()
        
        # Simular algunas detecciones
        rects1 = [(100, 100, 50, 50), (200, 150, 60, 60)]
        rects2 = [(105, 105, 50, 50), (195, 145, 60, 60)]
        
        # Primera actualización
        tracked1 = tracker.update(rects1)
        print(f"✅ Frame 1: {len(tracked1)} objetos trackeados")
        
        # Segunda actualización
        tracked2 = tracker.update(rects2)
        print(f"✅ Frame 2: {len(tracked2)} objetos trackeados")
        
        print(f"✅ Total únicos: {tracker.get_total_count()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en tracking: {e}")
        return False

def test_logging():
    """Prueba el sistema de logging"""
    print("\n📊 Probando sistema de logging...")
    
    try:
        import tempfile
        from data_io import EventLogger, SessionManager
        
        # Crear directorio temporal
        with tempfile.TemporaryDirectory() as temp_dir:
            # Crear logger
            logger = EventLogger(temp_dir, "Test Location", "Test Method")
            
            # Simular algunos eventos
            logger.log_person_enter(1, 0.85)
            logger.log_person_seen(1, 30)
            logger.log_person_enter(2, 0.92)
            
            # Verificar que se creó el CSV
            csv_path = logger.csv_path
            if csv_path.exists():
                print(f"✅ CSV creado: {csv_path}")
                with open(csv_path, 'r') as f:
                    lines = f.readlines()
                    print(f"✅ Líneas en CSV: {len(lines)}")
            else:
                print("❌ CSV no creado")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error en logging: {e}")
        return False

def main():
    """Función principal de pruebas"""
    print("🧪 SISTEMA DE PRUEBAS - CONTADOR DE PERSONAS")
    print("=" * 50)
    
    tests = [
        ("Importaciones", test_imports),
        ("Cámara", test_camera),
        ("Detección de rostros", test_face_detection),
        ("Sistema de tracking", test_tracking),
        ("Sistema de logging", test_logging)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Error en {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumen final
    print("\n" + "=" * 50)
    print("📋 RESUMEN DE PRUEBAS")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Resultado: {passed}/{len(results)} pruebas exitosas")
    
    if passed == len(results):
        print("🎉 ¡Todos los componentes funcionan correctamente!")
        print("🚀 Sistema listo para usar")
    else:
        print("⚠️ Algunos componentes necesitan atención")
        print("💡 Revisa las dependencias y configuración")

if __name__ == "__main__":
    main()