# Sistema de Conteo de Personas en Tiempo Real

🎯 **Universidad de Monterrey (UDEM)**  
📚 **Proyecto de Visión Computacional**  
⚡ **Powered by OpenCV + Python**

## 📋 Descripción

Sistema inteligente de conteo de personas en tiempo real que detecta y rastrea individuos durante 15 minutos en espacios específicos del campus UDEM, evitando conteos duplicados y generando evidencia completa con reportes y capturas.

## 🚀 Características Principales

- ✅ **Detección de rostros**: Haar Cascade y DNN
- 🎯 **Tracking inteligente**: Evita conteos duplicados
- ⏰ **Sesión de 15 minutos**: Con temporizador en pantalla
- 📊 **Logging en tiempo real**: CSV con timestamps
- 📷 **Capturas automáticas**: Rostros y frames completos
- 📈 **Reportes detallados**: Estadísticas de sesión
- 🖥️ **Interfaz visual**: Información en tiempo real

## 🏗️ Estructura del Proyecto

```
CV-People-Tracker-Counter-RealTime/
├── src/
│   ├── main.py              # Programa principal
│   ├── detectors/           # Módulos de detección
│   │   ├── haar.py         # Detector Haar Cascade
│   │   └── dnn.py          # Detector DNN
│   ├── tracking/           # Sistema de tracking
│   │   ├── centroid.py     # Tracker centroide
│   │   └── utils.py        # Utilidades de tracking
│   └── data_io/            # Entrada/Salida de datos
│       ├── logger.py       # Sistema de logging
│       └── saver.py        # Guardado de imágenes
├── config.json             # Configuración del sistema
├── requirements.txt        # Dependencias Python
├── README.md              # Este archivo
├── QUICK_START.md         # Guía rápida de uso
├── PROYECTO_RESUMEN.md    # Resumen ejecutivo
├── test_system.py         # Sistema de pruebas
└── output/                # Directorio de salida (se crea automáticamente)
    └── YYYYMMDD_HHMM_location/
        ├── log.csv        # Registro de eventos
        ├── config.json    # Configuración usada
        ├── session_report.txt
        └── captures/
            ├── crops/     # Recortes de rostros
            └── frames/    # Frames completos
```

## 🔧 Instalación

### 1. Prerrequisitos
- Python 3.8 o superior
- Cámara web funcional
- Windows 10/11 (recomendado)

### 2. Clonar repositorio
```bash
git clone https://github.com/devMuniz02/CV-People-Tracker-Counter-RealTime.git
cd CV-People-Tracker-Counter-RealTime
```

### 3. Crear entorno virtual (recomendado)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 4. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 5. Verificar instalación
```bash
python -c "import cv2; print('OpenCV:', cv2.__version__)"
```

## 🎮 Uso

### Ejecución
```bash
cd src
python main.py
```

### Controles durante la ejecución
- **`q`**: Terminar sesión antes de tiempo
- **`s`**: Guardar captura manual
- **`r`**: Definir región de interés (próximamente)

### Configuración personalizada
Edita `config.json` para ajustar parámetros:

```json
{
    "detection": {
        "method": "haar",
        "scale_factor": 1.1,
        "min_neighbors": 5,
        "confidence_threshold": 0.6
    },
    "session": {
        "duration_minutes": 15,
        "location": "Tu Ubicación UDEM",
        "fps_limit": 30
    }
}
```

## 📊 Formato de Salida

### CSV de eventos (log.csv)
```csv
timestamp,person_id,event,total_count,location,method,notes
2025-10-02T15:31:05,7,ENTER,1,Cafetería UDEM,Haar,face_new,conf=0.85
2025-10-02T15:31:09,7,SEEN,1,Cafetería UDEM,Haar,stable_track,frames=120
2025-10-02T15:33:42,12,ENTER,2,Cafetería UDEM,Haar,face_new,conf=0.92
```

### Archivos generados
- **`log.csv`**: Registro completo de eventos
- **`config.json`**: Configuración utilizada
- **`session_report.txt`**: Resumen de la sesión
- **`captures/crops/`**: Recortes de rostros detectados
- **`captures/frames/`**: Frames completos con anotaciones

## 🔍 Métodos de Detección

### Haar Cascade (Predeterminado)
- ✅ Rápido y eficiente
- ✅ Bajo consumo de recursos
- ⚠️ Sensible a iluminación y ángulos
- 🎯 Ideal para condiciones controladas

### DNN (Deep Neural Network)
- ✅ Más robusto y preciso
- ✅ Mejor con variaciones de iluminación
- ⚠️ Mayor consumo computacional
- 🎯 Ideal para condiciones variables

## 📈 Resultados Esperados

### Métricas de rendimiento
- **FPS**: 15-30 (dependiendo del hardware)
- **Precisión**: 85-95% (condiciones ideales)
- **Latencia**: <100ms por frame
- **Memoria**: 200-500MB

### Archivos de evidencia
- Registro CSV con timestamps precisos
- Capturas de rostros únicos detectados
- Frames anotados con bounding boxes
- Reporte final con estadísticas

## 🛠️ Solución de Problemas

### Cámara no detectada
```bash
# Verificar cámaras disponibles
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"
```

### Baja precisión de detección
1. Ajustar `min_neighbors` (3-8)
2. Modificar `scale_factor` (1.05-1.3)
3. Cambiar a método DNN
4. Mejorar iluminación del ambiente

### Rendimiento lento
1. Reducir `resize_width` en config
2. Aumentar `fps_limit`
3. Usar método Haar en lugar de DNN
4. Cerrar aplicaciones innecesarias

## 📚 Dependencias

```txt
opencv-python==4.8.1.78
numpy==1.24.3
scipy==1.11.1
matplotlib==3.7.2
Pillow==10.0.0
pandas==2.0.3
```

## 🎯 Casos de Uso

### Ubicaciones recomendadas en UDEM
- 🍽️ **Cafetería**: Alta rotación de personas
- 📚 **Biblioteca**: Flujo constante de estudiantes
- 🏢 **CRGS**: Área de servicios estudiantiles
- 🎓 **CU**: Centro universitario
- 🚪 **Entradas principales**: Máximo tráfico

### Condiciones ideales
- 💡 Iluminación uniforme
- 📐 Ángulo frontal de la cámara
- 🎯 Distancia 2-5 metros
- 🚫 Mínimas oclusiones
- 📱 Cámara estable

## 🔄 Workflow del Sistema

1. **Inicialización**
   - Cargar configuración
   - Inicializar detector y tracker
   - Crear directorio de sesión

2. **Procesamiento en tiempo real**
   - Capturar frame de cámara
   - Detectar rostros
   - Actualizar tracking
   - Registrar eventos
   - Guardar evidencia

3. **Finalización**
   - Generar reportes
   - Exportar estadísticas
   - Limpiar recursos

## 🤝 Contribución

Este es un proyecto académico de la Universidad de Monterrey. Para mejoras:

1. Fork el repositorio
2. Crear rama feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit cambios (`git commit -am 'Agregar nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Crear Pull Request

## 📄 Licencia

Proyecto académico - Universidad de Monterrey (UDEM)  
Desarrollado para fines educativos en el curso de Visión Computacional.

## 👨‍💻 Autor

**[Tu Nombre]**  
Estudiante de Ingeniería - UDEM  
📧 [tu-email@udem.edu]

## 🔗 Enlaces Útiles

- [OpenCV Documentation](https://docs.opencv.org/)
- [NumPy User Guide](https://numpy.org/doc/stable/user/)
- [Python Official Documentation](https://docs.python.org/3/)

---

**🎓 Universidad de Monterrey - Donde la excelencia es nuestra tradición**
A real-time computer vision application for tracking and counting people using video streams. Built with OpenCV, Python, and deep learning models.
