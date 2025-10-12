
# Sistema de Conteo de Personas en Tiempo Real

🎯 **Universidad de Monterrey (UDEM)**  
📚 **Proyecto de Visión Computacional**  
⚡ **OpenCV + Python**

## 📋 Descripción

Sistema para detectar, rastrear y contar personas en tiempo real. Registra eventos en CSV, guarda capturas y genera evidencia de la sesión.

## 🚀 Características principales

- Detección configurable (métodos definidos por `config.json`)
- Tracking por centroide para evitar conteos duplicados
- Sesión configurable (por defecto 15 minutos)
- Logging en CSV con timestamps
- Guardado automático de recortes y frames

## 🏗️ Estructura del proyecto (real)

```
CV-People-Tracker-Counter-RealTime/
├── config.json               # Configuración principal
├── requirements.txt          # Dependencias Python
├── README.md                 # Este archivo
├── output/                   # Directorio de salida (creado por runtime)
│   └── YYYYMMDD_HHMM_location/
│       ├── config.json
│       ├── log.csv
│       └── captures/
│           └── crops/
└── src/
        ├── main.py               # Programa principal (entrypoint)
        ├── data_io/              # Entrada/Salida: logger y saver
        │   ├── logger.py
        │   └── saver.py
        └── tracking/             # Lógica de tracking
                ├── centroid.py
                └── utils.py
```

Nota: la estructura real no contiene una carpeta `detectors/` separada en esta versión. La detección y su configuración se manejan desde `src/main.py` y `config.json`.

## 🔧 Instalación

### Requisitos
- Python 3.8+
- Cámara web funcional

### Quick Start

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 🎮 Uso

Ejecuta el programa:

```powershell
python .\src\main.py
```

Escoge la ubicacion de tu archivo de video a processar e inicia la sesión en el GUI. 

Controles durante la ejecución:

- `q`: terminar sesión antes de tiempo
- `s`: guardar captura manual

## 🔧 Configuración

Edita `config.json` en la raíz del repositorio para ajustar parámetros. Ejemplo mínimo:

```json
{
    "detection": {
        "method": "mtcnn",
        "confidence_threshold": 0.6
    },
    "session": {
        "duration_minutes": 15,
        "location": "Ubicación Ejemplo",
        "fps_limit": 30
    }
}
```

Los parámetros exactos soportados dependen de la implementación en `src/main.py`.

## 📊 Salida / Archivos generados

- `output/YYYYMMDD_HHMM_location/log.csv`: registro de eventos (timestamp, person_id, event, total_count, ...)
- `output/.../config.json`: copia de la configuración usada
- `output/.../captures/crops/`: recortes de rostros detectados

## 🔄 Workflow resumido

1. Iniciar `main.py` (carga config y crea sesión)
2. Capturar frames y detectar rostros
3. Actualizar tracker y registrar eventos
4. Guardar evidencia y generar reporte al final
