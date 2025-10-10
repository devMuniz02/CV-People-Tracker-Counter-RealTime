# 🚀 GUÍA RÁPIDA DE USO

## ⚡ Instalación Express

### Instalación Manual
```bash
# 1. Clonar repositorio
git clone https://github.com/devMuniz02/CV-People-Tracker-Counter-RealTime.git
cd CV-People-Tracker-Counter-RealTime

# 2. Crear entorno virtual (recomendado)
python -m venv venv

# 3. Activar entorno virtual
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. Ejecutar
cd src
python main.py
```

## 🎮 Controles Durante la Ejecución

| Tecla | Acción |
|-------|--------|
| `q` | Terminar sesión antes de tiempo |
| `s` | Guardar captura manual |
| `r` | Definir región de interés (próximamente) |

## 📍 Configuraciones Predefinidas

### Cafetería (Default)
```bash
cd src
python main.py
# Usa config.json - Optimizado para espacios con mucho movimiento
```

### Biblioteca
```bash
cd src
python main.py ../config_biblioteca.json
# Usa DNN - Mayor precisión para ambiente controlado
```

### CRGS
```bash
cd src
python main.py ../config_crgs.json
# Haar rápido - Para área de servicios con flujo constante
```

## 📊 Archivos de Salida

Después de cada sesión encontrarás en `output/YYYYMMDD_HHMM_ubicacion/`:

- **`log.csv`** - Registro completo de eventos
- **`session_report.txt`** - Resumen de la sesión
- **`config.json`** - Configuración utilizada
- **`captures/crops/`** - Recortes de rostros únicos
<!-- frames folder not used -->

## 🔧 Ajustes Rápidos

### Para mejorar rendimiento
```json
{
    "preprocessing": {
        "resize_width": 480
    },
    "session": {
        "fps_limit": 20
    }
}
```

### Para mayor precisión
```json
{
    "detection": {
    "method": "mtcnn",
        "confidence_threshold": 0.7
    },
    "tracking": {
        "max_disappeared_frames": 45
    }
}
```

## 🆘 Solución Rápida de Problemas

### Cámara no detectada
- Cerrar otras aplicaciones que usen la cámara
- Verificar permisos de cámara en Windows
- Probar con `python test_system.py`

### Detección imprecisa
- Mejorar iluminación del ambiente
- Cambiar `scale_factor` a 1.05-1.1
- Usar método DNN en lugar de Haar

### Programa lento
- Reducir `resize_width` a 480 o 320
- Aumentar `fps_limit` a 15-20
- Cerrar aplicaciones innecesarias

## 📞 Soporte

Para problemas técnicos:
1. Ejecutar `python test_system.py`
2. Revisar el archivo `session_report.txt`
3. Consultar la documentación completa en `README.md`

---
**🎓 Universidad de Monterrey - Proyecto de Visión Computacional**