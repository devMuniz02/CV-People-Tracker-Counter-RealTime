"""
Sistema de logging para registrar eventos del conteo de personas
Maneja la escritura en tiempo real del CSV con timestamps y eventos
"""

import csv
import json
from datetime import datetime
from pathlib import Path

class EventLogger:
    def __init__(self, output_dir, location="UDEM", method="mtcnn"):
        """
        Inicializa el logger de eventos
        
        Args:
            output_dir: Directorio de salida
            location: Ubicación donde se realiza el conteo
            method: Método de detección utilizado
        """
        self.output_dir = Path(output_dir)
        self.location = location
        self.method = method
        
        # Crear directorio si no existe
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ruta del archivo CSV
        self.csv_path = self.output_dir / "log.csv"
        
        # Estado interno
        self.session_start = datetime.now()
        self.total_count = 0
        self.logged_persons = set()  # IDs de personas ya registradas
        
        # Inicializar archivo CSV
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Inicializa el archivo CSV con headers"""
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                'timestamp',
                'person_id', 
                'event',
                'total_count',
                'location',
                'method',
                'notes'
            ])
    
    def log_event(self, person_id, event_type, notes=""):
        """
        Registra un evento en el CSV
        
        Args:
            person_id: ID único de la persona
            event_type: Tipo de evento (ENTER, EXIT, SEEN)
            notes: Notas adicionales
        """
        timestamp = datetime.now().isoformat()
        
        # Actualizar conteo según el evento
        if event_type == "ENTER" and person_id not in self.logged_persons:
            self.total_count += 1
            self.logged_persons.add(person_id)
        
        # Escribir al CSV
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp,
                person_id,
                event_type,
                self.total_count,
                self.location,
                self.method,
                notes
            ])
    
    def log_person_enter(self, person_id, confidence=None):
        """
        Registra la entrada de una nueva persona
        
        Args:
            person_id: ID de la persona
            confidence: Nivel de confianza de la detección
        """
        notes = "face_new"
        if confidence:
            notes += f",conf={confidence:.2f}"
        
        self.log_event(person_id, "ENTER", notes)
    
    def log_person_seen(self, person_id, frames_tracked=None):
        """
        Registra que una persona sigue siendo vista
        
        Args:
            person_id: ID de la persona
            frames_tracked: Número de frames que ha sido trackeada
        """
        notes = "stable_track"
        if frames_tracked:
            notes += f",frames={frames_tracked}"
        
        self.log_event(person_id, "SEEN", notes)
    
    def log_person_exit(self, person_id, reason="timeout"):
        """
        Registra la salida de una persona
        
        Args:
            person_id: ID de la persona
            reason: Razón de la salida (timeout, out_of_frame, etc.)
        """
        notes = f"exit_reason={reason}"
        self.log_event(person_id, "EXIT", notes)
    
    def get_session_summary(self):
        """
        Obtiene un resumen de la sesión actual
        
        Returns:
            Diccionario con estadísticas de la sesión
        """
        session_duration = datetime.now() - self.session_start
        
        return {
            'session_start': self.session_start.isoformat(),
            'session_duration_seconds': session_duration.total_seconds(),
            'total_unique_persons': self.total_count,
            'location': self.location,
            'method': self.method,
            'csv_path': str(self.csv_path)
        }
    
    def save_session_config(self, config_data):
        """
        Guarda la configuración de la sesión
        
        Args:
            config_data: Diccionario con la configuración utilizada
        """
        config_path = self.output_dir / "config.json"
        
        session_config = {
            'session_info': self.get_session_summary(),
            'detection_config': config_data,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(config_path, 'w', encoding='utf-8') as file:
            json.dump(session_config, file, indent=2, ensure_ascii=False)
    
    def export_summary_report(self):
        """
        Exporta un reporte resumen en formato texto
        
        Returns:
            Ruta del archivo de reporte generado
        """
        report_path = self.output_dir / "session_report.txt"
        summary = self.get_session_summary()
        
        with open(report_path, 'w', encoding='utf-8') as file:
            file.write("=" * 50 + "\n")
            file.write("REPORTE DE SESIÓN - CONTEO DE PERSONAS\n")
            file.write("=" * 50 + "\n\n")
            
            file.write(f"Inicio de sesión: {summary['session_start']}\n")
            file.write(f"Duración: {summary['session_duration_seconds']:.0f} segundos\n")
            file.write(f"Ubicación: {summary['location']}\n")
            file.write(f"Método de detección: {summary['method']}\n")
            file.write(f"Total personas únicas detectadas: {summary['total_unique_persons']}\n\n")
            
            file.write("Archivos generados:\n")
            file.write(f"- Log CSV: {summary['csv_path']}\n")
            file.write(f"- Configuración: {self.output_dir / 'config.json'}\n")
            file.write(f"- Capturas: {self.output_dir / 'captures'}\n")
        
        return report_path

class SessionManager:
    """
    Gestor de sesiones para organizar archivos de salida
    """
    
    @staticmethod
    def create_session_directory(base_path="output", location="UDEM"):
        """
        Crea un directorio único para la sesión
        
        Args:
            base_path: Directorio base
            location: Nombre de la ubicación
            
        Returns:
            Path del directorio de sesión creado
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        # Limpiar el nombre de la ubicación para evitar caracteres problemáticos
        clean_location = "".join(c for c in location if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_location = clean_location.replace(' ', '_')
        
        session_name = f"{timestamp}_{clean_location}"
        session_dir = Path(base_path) / session_name
        
        # Crear estructura de directorios
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "captures" / "crops").mkdir(parents=True, exist_ok=True)
        # Do not create a frames directory: project stores only crop evidence (IDs)

        return session_dir