"""
Interfaz simple para ajustar parámetros de detección y ejecutar el sistema
Creado como alternativa a `main.py`. Usa `DeteccionDePersonas.mov` por defecto
y permite cambiar parámetros de detección antes de iniciar la sesión.
"""

import threading
import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path
import time
import cv2

from main import PeopleCounterSystem


class PeopleCounterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("People Counter - GUI")

        # System instance
        self.system = PeopleCounterSystem()
        self.thread = None
        # Person HOG detector (aplicado automáticamente)
        try:
            from main import PersonHOGDetector
            self.person_hog_detector = PersonHOGDetector()
        except Exception:
            self.person_hog_detector = None

        # Variables de control
        self.method_var = tk.StringVar(value='mtcnn')
        self.scale_var = tk.DoubleVar(value=1.1)
        self.min_neighbors_var = tk.IntVar(value=5)
        self.confidence_var = tk.DoubleVar(value=0.6)
        self.min_w_var = tk.IntVar(value=20)
        self.min_h_var = tk.IntVar(value=20)
        self.video_path_var = tk.StringVar(value=str(Path('DeteccionDePersonas.mov').absolute()))
        self.frame_number_var = tk.IntVar(value=0)
        self.auto_apply_var = tk.BooleanVar(value=False)
        # Preprocessing options
        self.apply_clahe_var = tk.BooleanVar(value=False)
        self.apply_equalize_var = tk.BooleanVar(value=False)
        # Max size and coordinates for min/max boxes
        self.max_w_var = tk.IntVar(value=40)
        self.max_h_var = tk.IntVar(value=40)
        self.min_x_var = tk.IntVar(value=10)
        self.min_y_var = tk.IntVar(value=10)
        self.max_x_var = tk.IntVar(value=10)
        self.max_y_var = tk.IntVar(value=80)

        # Current loaded frame and last annotated
        self.current_frame = None
        self.last_annotated = None

        # Build a simplified UI: only method selection and frame controls
        frm = ttk.Frame(self.root, padding=8)
        frm.grid(row=0, column=0, sticky='nsew')

        ttk.Label(frm, text='Detector:').grid(row=0, column=0, sticky='w')
        method_options = ['dnn', 'mtcnn', 'mediapipe', 'dlib', 'person_hog', 'retinaface']
        self.method_menu = ttk.OptionMenu(frm, self.method_var, self.method_var.get(), *method_options)
        self.method_menu.grid(row=0, column=1, sticky='ew')

        # Allow reacting to method changes while running
        try:
            # trace_add is available in newer Tk versions; fallback to trace
            if hasattr(self.method_var, 'trace_add'):
                self.method_var.trace_add('write', lambda *args: self._on_method_change())
            else:
                self.method_var.trace('w', lambda *args: self._on_method_change())
        except Exception:
            pass

        ttk.Label(frm, text='Frame #:').grid(row=1, column=0, sticky='w')
        self.frame_entry = ttk.Entry(frm, textvariable=self.frame_number_var, width=10)
        self.frame_entry.grid(row=1, column=1, sticky='w')

        load_btn = ttk.Button(frm, text='Cargar Frame', command=self.load_frame)
        load_btn.grid(row=2, column=0, pady=6)
        self.test_btn = ttk.Button(frm, text='Probar Detector', command=self.test_detector)
        self.test_btn.grid(row=2, column=1, pady=6)
        self.save_btn = ttk.Button(frm, text='Guardar Anotado', command=self.save_annotated, state='disabled')
        self.save_btn.grid(row=3, column=0, pady=6)

        # Video selection and playback controls (for running over the video)
        ttk.Label(frm, text='Video:').grid(row=6, column=0, sticky='w')
        self.video_entry = ttk.Entry(frm, textvariable=self.video_path_var, width=40)
        self.video_entry.grid(row=6, column=1, sticky='ew')
        browse_btn = ttk.Button(frm, text='Buscar...', command=self._browse_video)
        browse_btn.grid(row=7, column=0, pady=6)

        self.start_btn = ttk.Button(frm, text='Iniciar Sesión', command=self.start_session)
        self.start_btn.grid(row=7, column=1, sticky='ew', pady=6)
        self.stop_btn = ttk.Button(frm, text='Detener Sesión', command=self.stop_session, state='disabled')
        self.stop_btn.grid(row=8, column=1, sticky='ew', pady=6)

        self.status_label = ttk.Label(frm, text='Estado: listo', foreground='blue')
        self.status_label.grid(row=4, column=0, columnspan=2, sticky='w', pady=(6,0))
        self.avail_label = ttk.Label(frm, text='Detectores: comprobando...', foreground='black')
        self.avail_label.grid(row=5, column=0, columnspan=2, sticky='w')

        self.root.columnconfigure(0, weight=1)
        frm.columnconfigure(1, weight=1)

        # Check availability on startup
        self.root.after(100, self._update_detector_availability)

        # Make layout resize nicely
        self.root.columnconfigure(0, weight=1)
        frm.columnconfigure(1, weight=1)

        # Install detectors button (disabled in simplified UI)
        inst_frm = ttk.Frame(frm)
        inst_frm.grid(row=12, column=0, columnspan=3, sticky='w', pady=(6,0))
        self.install_btn = ttk.Button(inst_frm, text='Instalar detectores opcionales', command=self._open_install_dialog)
        self.install_btn.grid(row=0, column=0)
        try:
            # Disable install in simplified GUI to avoid changing environment unintentionally
            self.install_btn.config(state='disabled')
        except Exception:
            pass

    def _browse_video(self):
        path = filedialog.askopenfilename(filetypes=[('Video files', '*.mp4 *.mov *.avi'), ('All files','*.*')])
        if path:
            self.video_path_var.set(path)

    def _apply_config_to_system(self, system):
        # Apply GUI params into system.config
        det = system.config.get('detection', {})
        det['method'] = self.method_var.get()
        det['scale_factor'] = float(self.scale_var.get())
        det['min_neighbors'] = int(self.min_neighbors_var.get())
        det['confidence_threshold'] = float(self.confidence_var.get())
        det['min_size'] = [int(self.min_w_var.get()), int(self.min_h_var.get())]
        # Also store max_size
        det['max_size'] = [int(self.max_w_var.get()), int(self.max_h_var.get())]

        # Preprocessing flags
        prep = system.config.get('preprocessing', {})
        prep['apply_clahe'] = bool(self.apply_clahe_var.get())
        prep['apply_equalize'] = bool(self.apply_equalize_var.get())
        system.config['preprocessing'] = prep

        # Update session fps limit to a reasonable default if missing
        sess = system.config.get('session', {})
        sess.setdefault('fps_limit', 30)

        # Set the video path if specified
        vp = Path(self.video_path_var.get())
        if vp.exists():
            # Override by creating a Path object on the system level for convenience
            system.video_path = vp

        # Reinitialize detector and tracker with new params
        # Initialize requested detector
        new_detector = system._initialize_detector()
        # If the detector exposes 'available' and it's False, warn and keep previous detector
        if getattr(new_detector, 'available', True) is False:
            # Keep existing detector but show status to the user
            self._set_status(f"Detector '{det['method']}' no disponible en este entorno", 'orange')
        else:
            system.detector = new_detector

        # Reinitialize tracker (always safe)
        system.tracker = system._initialize_tracker()

        # If auto-apply is enabled and a frame is loaded, reprocess
        if self.auto_apply_var.get() and self.current_frame is not None:
            self.process_current_frame()

    def _update_detector_availability(self):
        """Quickly instantiate optional detectors to check availability (non-destructive)."""
        avail = []
        # Try DNN
        try:
            d = system = None
            from main import DNNFaceDetector, MTCNNFaceDetector, DlibFaceDetector, PersonHOGDetector, RetinaFaceDetector
            # Instantiate safely
            try:
                dd = DNNFaceDetector()
                avail.append('dnn' if getattr(dd, 'available', True) else 'dnn (no)')
            except Exception:
                avail.append('dnn (no)')

            try:
                m = MTCNNFaceDetector()
                avail.append('mtcnn' if getattr(m, 'available', False) else 'mtcnn (no)')
            except Exception:
                avail.append('mtcnn (no)')

            try:
                dl = DlibFaceDetector()
                avail.append('dlib' if getattr(dl, 'available', False) else 'dlib (no)')
            except Exception:
                avail.append('dlib (no)')

            try:
                ph = PersonHOGDetector()
                avail.append('person_hog' if getattr(ph, 'available', False) else 'person_hog (no)')
            except Exception:
                avail.append('person_hog (no)')

            try:
                rf = RetinaFaceDetector()
                avail.append('retinaface' if getattr(rf, 'available', False) else 'retinaface (no)')
            except Exception:
                avail.append('retinaface (no)')
        except Exception:
            # Fallback if imports fail
            avail = ['dnn (no)', 'mtcnn (no)', 'dlib (no)', 'person_hog (no)', 'retinaface (no)']

        self.avail_label.config(text='Detectores: ' + ', '.join(avail))

    def _on_method_change(self):
        """Called when the detector method selection is changed in the GUI.
        If a session is running, re-apply the configuration so the new detector is used.
        """
        try:
            # If there's a running system, attempt to reconfigure its detector
            if getattr(self, 'system', None) is not None:
                # Apply only the detection-related parts to avoid disrupting session state
                try:
                    # Apply config will create a new detector and reinitialize the tracker
                    self._apply_config_to_system(self.system)
                    self._set_status(f"Método cambiado a '{self.method_var.get()}'", 'green')
                except Exception as e:
                    # Non-fatal: just show status
                    self._set_status(f'No se pudo cambiar detector: {e}', 'orange')
        except Exception:
            pass

    def _build_detector_callable(self, method):
        """Return (callable(frame)->dets, available_bool, display_name) for the requested method.
        Detections are lists of (x,y,w,h,conf).
        """
        m = method.lower()

        # For detectors supported by PeopleCounterSystem/_initialize_detector, reuse that implementation
        if m in ('haar', 'dnn', 'mtcnn', 'dlib', 'person_hog', 'retinaface'):
            try:
                # Build a temporary system to initialize the detector according to GUI params
                tmp = PeopleCounterSystem()
                tmp.config['detection']['method'] = 'dnn' if m == 'dnn' else ('mtcnn' if m == 'mtcnn' else m)
                tmp.config['detection']['scale_factor'] = float(self.scale_var.get())
                tmp.config['detection']['min_neighbors'] = int(self.min_neighbors_var.get())
                tmp.config['detection']['confidence_threshold'] = float(self.confidence_var.get())
                detector = tmp._initialize_detector()
                available = getattr(detector, 'available', True)
                name = detector.get_method_name() if hasattr(detector, 'get_method_name') else method

                def _call(f):
                    try:
                        return detector.detect_faces(f)
                    except Exception:
                        return []

                return _call, bool(available), name
            except Exception:
                return (lambda f: [], False, method)

        # MediaPipe
        if m == 'mediapipe':
            try:
                import mediapipe as mp

                def _mp(f):
                    img_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    h, w = img_rgb.shape[:2]
                    boxes = []
                    scores = []
                    try:
                        with mp.solutions.face_detection.FaceDetection(model_selection=0,
                                                                      min_detection_confidence=float(self.confidence_var.get())) as fd:
                            res = fd.process(img_rgb)
                            if res.detections:
                                for det in res.detections:
                                    rel = det.location_data.relative_bounding_box
                                    x1 = int(rel.xmin * w)
                                    y1 = int(rel.ymin * h)
                                    x2 = int((rel.xmin + rel.width) * w)
                                    y2 = int((rel.ymin + rel.height) * h)
                                    boxes.append((x1, y1, x2 - x1, y2 - y1, float(det.score[0]) if det.score else 1.0))
                    except Exception:
                        return []
                    return boxes

                return _mp, True, 'MediaPipe'
            except Exception:
                return (lambda f: [], False, 'MediaPipe')

        # For MTCNN use the TensorFlow 'mtcnn' package (handled via PeopleCounterSystem._initialize_detector)
        if m == 'mtcnn':
            try:
                # Reuse the system initialization which prefers the mtcnn package
                tmp = PeopleCounterSystem()
                tmp.config['detection']['method'] = 'mtcnn'
                detector = tmp._initialize_detector()
                available = getattr(detector, 'available', True)

                def _call(f):
                    try:
                        return detector.detect_faces(f)
                    except Exception:
                        return []

                name = detector.get_method_name() if hasattr(detector, 'get_method_name') else 'MTCNN'
                return _call, bool(available), name
            except Exception:
                return (lambda f: [], False, 'MTCNN')

        # Unknown
        return (lambda f: [], False, method)

    def test_detector(self):
        """Run the selected detector on the loaded frame and show a small summary."""
        if self.current_frame is None:
            self._set_status('Carga un frame antes de probar el detector', 'red')
            return

        # Temporarily apply config and instantiate detector
        system = PeopleCounterSystem()
        self._apply_config_to_system(system)
        det = system.detector
        if getattr(det, 'available', True) is False:
            self._set_status(f"Detector seleccionado '{self.method_var.get()}' no disponible", 'orange')
            return

        try:
            proc = system.preprocess_frame(self.current_frame.copy())

            # Primary detector (could be face or person)
            dets = det.detect_faces(proc)

            # Also run person-hog detector if available to compare (person hog should not be filtered)
            ph_dets = []
            try:
                if getattr(self, 'person_hog_detector', None):
                    ph_dets = self.person_hog_detector.detect_faces(proc) or []
            except Exception:
                ph_dets = []

            total = len(dets) + len(ph_dets)
            self._set_status(f'Detecciones encontradas: {total} (detector: {self.method_var.get()})', 'green')

            # Annotate and show a small preview: faces -> green, person_hog -> blue
            preview = proc.copy()
            # faces (primary detector) as green
            for (x, y, w, h, conf) in dets:
                cv2.rectangle(preview, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            # person hog as blue
            for (x, y, w, h, conf) in ph_dets:
                cv2.rectangle(preview, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)

            cv2.imshow('Prueba Detector', preview)
            cv2.waitKey(1)
        except Exception as e:
            self._set_status(f'Error en prueba de detector: {e}', 'red')

    def _open_install_dialog(self):
        """Open a small dialog to select optional packages to install via pip."""
        dlg = tk.Toplevel(self.root)
        dlg.title('Instalar detectores')

        opts = {
            'mtcnn': tk.BooleanVar(value=False),
            'retinaface': tk.BooleanVar(value=False),
            'dlib': tk.BooleanVar(value=False)
        }

        ttk.Label(dlg, text='Selecciona detectores a instalar:').grid(row=0, column=0, columnspan=2, pady=(6,6))
        ttk.Checkbutton(dlg, text='MTCNN (mtcnn)', variable=opts['mtcnn']).grid(row=1, column=0, sticky='w')
        ttk.Checkbutton(dlg, text='RetinaFace (retina-face / retinaface)', variable=opts['retinaface']).grid(row=2, column=0, sticky='w')
        ttk.Checkbutton(dlg, text='Dlib (dlib) — puede requerir build tools', variable=opts['dlib']).grid(row=3, column=0, sticky='w')

        def do_install():
            chosen = [k for k, v in opts.items() if v.get()]
            dlg.destroy()
            if not chosen:
                return
            self._set_status('Instalando paquetes seleccionados... (ver consola)', 'orange')
            threading.Thread(target=self._install_packages_thread, args=(chosen,), daemon=True).start()

        ttk.Button(dlg, text='Instalar', command=do_install).grid(row=4, column=0, pady=(8,8))

    def _install_packages_thread(self, packages):
        """Install packages using the same Python interpreter's pip and refresh availability."""
        import sys
        import subprocess
        failures = []
        for pkg in packages:
            try:
                # Map friendly names to pip package names
                pkg_name = pkg
                if pkg == 'retinaface':
                    # try common package names
                    candidates = ['retina-face', 'retinaface', 'retinaface-pytorch']
                else:
                    candidates = [pkg]

                success = False
                for candidate in candidates:
                    try:
                        subprocess.check_call([sys.executable, '-m', 'pip', 'install', candidate])
                        success = True
                        break
                    except Exception:
                        continue
                if not success:
                    failures.append(pkg)
            except Exception:
                failures.append(pkg)

        # Refresh availability indicator
        try:
            self._update_detector_availability()
        except Exception:
            pass

        if failures:
            self._set_status(f'Error instalando: {",".join(failures)} (ver consola)', 'red')
        else:
            self._set_status('Instalación completada, actualiza disponibilidad', 'green')

    def load_frame(self):
        """Carga un frame específico del vídeo y lo muestra listo para tuning"""
        video_path = Path(self.video_path_var.get())
        if not video_path.exists():
            self._set_status('Archivo de vídeo no encontrado', 'red')
            return

        frame_no = int(self.frame_number_var.get() or 0)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self._set_status('No se pudo abrir el vídeo', 'red')
            return

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_no < 0 or (total and frame_no >= total):
            self._set_status(f'Frame fuera de rango (0 - {max(0,total-1)})', 'red')
            cap.release()
            return

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            self._set_status('No se pudo leer el frame seleccionado', 'red')
            return

        # Ensure session artifacts exist for saving if needed
        try:
            self.system.setup_session()
        except Exception:
            pass

        self.current_frame = frame
        self._set_status(f'Frame {frame_no} cargado', 'green')
        self.save_btn.config(state='normal')

        # Apply current params immediately (if auto apply) or draw once
        if self.auto_apply_var.get():
            self.apply_params()
        else:
            self.process_current_frame()

    def process_current_frame(self):
        """Procesa la imagen cargada con los parámetros actuales y la muestra anotada"""
        if self.current_frame is None:
            self._set_status('No hay frame cargado', 'red')
            return

        # Apply preprocessing
        try:
            proc = self.system.preprocess_frame(self.current_frame.copy())
        except Exception:
            proc = self.current_frame.copy()

        # Detect with primary detector
        try:
            face_dets = self.system.detector.detect_faces(proc) or []
        except Exception as e:
            face_dets = []
            print('Error en detect_faces (primary):', e)

        # Always run person-hog detector in parallel (if available) and do NOT apply min/max to it
        ph_dets = []
        try:
            if getattr(self, 'person_hog_detector', None):
                ph_dets = self.person_hog_detector.detect_faces(proc) or []
        except Exception as e:
            ph_dets = []
            print('Error en detect_faces (person_hog):', e)

        # Apply size filter only to face detections (do not filter person_hog)
        try:
            try:
                filtered_faces = self.system._filter_detections_by_size(face_dets)
            except Exception:
                filtered_faces = face_dets
        except Exception:
            filtered_faces = face_dets

        # Combine detections for display/tracking: faces first, then person_hog
        dets = list(filtered_faces) + list(ph_dets)

        # Build tracked_objects-like dict for draw_interface
        tracked = {}
        for i, d in enumerate(dets):
            x, y, w, h, conf = d
            cx = int(x + w/2)
            cy = int(y + h/2)
            tracked[i] = {
                'rect': (int(x), int(y), int(w), int(h), conf),
                'centroid': (cx, cy),
                'disappeared_frames': 0
            }

        # Use zero elapsed/remaining for display
        from datetime import timedelta, datetime
        elapsed = timedelta(0)
        remaining = timedelta(0)

        try:
            annotated = self.system.draw_interface(proc.copy(), tracked, elapsed, remaining)
        except Exception:
            annotated = proc

        # Draw face and person_hog boxes with explicit colors on top of annotated image
        try:
            # faces: green (0,255,0). person_hog: blue (255,0,0)
            # Draw faces (they were added first)
            for (x, y, w, h, conf) in filtered_faces:
                cv2.rectangle(annotated, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            for (x, y, w, h, conf) in ph_dets:
                cv2.rectangle(annotated, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
        except Exception:
            pass

        # Draw a red box showing the chosen minimum size (top-right corner with margin)
        try:
            # Read values
            min_w = int(self.min_w_var.get())
            min_h = int(self.min_h_var.get())
            max_w = int(self.max_w_var.get())
            max_h = int(self.max_h_var.get())

            min_x = int(self.min_x_var.get())
            min_y = int(self.min_y_var.get())
            max_x = int(self.max_x_var.get())
            max_y = int(self.max_y_var.get())

            h_img, w_img = annotated.shape[:2]
            # Clamp positions to image bounds
            min_x = max(0, min(min_x, w_img - 1))
            min_y = max(0, min(min_y, h_img - 1))
            max_x = max(0, min(max_x, w_img - 1))
            max_y = max(0, min(max_y, h_img - 1))

            # Clamp sizes so boxes stay inside image
            min_w_clamped = max(1, min(min_w, w_img - min_x))
            min_h_clamped = max(1, min(min_h, h_img - min_y))
            max_w_clamped = max(1, min(max_w, w_img - max_x))
            max_h_clamped = max(1, min(max_h, h_img - max_y))

            # Draw min box in red
            cv2.rectangle(annotated, (min_x, min_y), (min_x + min_w_clamped, min_y + min_h_clamped), (0, 0, 255), 2)
            label_min = f"min: {min_w}x{min_h} @ ({min_x},{min_y})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(annotated, label_min, (min_x, max(min_y - 6, 0)), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Draw max box in green
            cv2.rectangle(annotated, (max_x, max_y), (max_x + max_w_clamped, max_y + max_h_clamped), (0, 255, 0), 2)
            label_max = f"max: {max_w}x{max_h} @ ({max_x},{max_y})"
            cv2.putText(annotated, label_max, (max_x, max(max_y - 6, 0)), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        except Exception:
            pass

        # Show annotated frame
        try:
            cv2.imshow('Tuning - Frame', annotated)
            cv2.waitKey(1)
        except Exception:
            pass

        # Keep last annotated for saving
        self.last_annotated = annotated

    def apply_params(self):
        """Aplica los parámetros del GUI al sistema y re-procesa el frame"""
        try:
            self._apply_config_to_system(self.system)
            self._set_status('Parámetros aplicados', 'green')
        except Exception as e:
            self._set_status(f'Error aplicando parámetros: {e}', 'red')
            return

        # Reprocess current frame to reflect changes
        self.process_current_frame()

    def save_annotated(self):
        """Guarda la última imagen anotada en el directorio de sesión"""
        if not hasattr(self, 'last_annotated') or self.last_annotated is None:
            self._set_status('No hay imagen anotada para guardar', 'red')
            return

        try:
            # Use image_saver if available
            if self.system and getattr(self.system, 'image_saver', None):
                self.system.image_saver.save_full_frame(self.last_annotated, {'event': 'tuning_save'})
                self._set_status('Imagen anotada guardada', 'green')
            else:
                # Fallback: save using OpenCV to session dir
                import os
                out_dir = Path(self.system.session_dir) if getattr(self.system, 'session_dir', None) else Path('output')
                out_dir = Path(out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                filename = out_dir / f'tuning_annotated_{int(time.time())}.jpg'
                cv2.imwrite(str(filename), self.last_annotated)
                self._set_status(f'Guardado: {filename}', 'green')
        except Exception as e:
            self._set_status(f'Error guardando: {e}', 'red')

    def start_session(self):
        if self.thread and self.thread.is_alive():
            self._set_status('Ya hay una sesión en ejecución', 'orange')
            return

        # Create a fresh system instance
        self.system = PeopleCounterSystem()

        # If user provided an explicit video path, ensure the system will use it by
        # setting the Path object used in main.run when choosing the file.
        # We attach the attribute and the run method in main.py already checks for Detec...mov, so we will
        # monkeypatch the Path check by setting an attribute 'video_path' used by temp_main only.

        # Apply GUI parameters
        try:
            self._apply_config_to_system(self.system)
        except Exception as e:
            self._set_status(f'Error aplicando configuración: {e}', 'red')
            return

        # Start run in a background thread
        def target_run():
            try:
                # If we attached a video_path, let the system use it by temporarily changing cwd
                # The PeopleCounterSystem.run uses Path('DeteccionDePersonas.mov') so if user selected another
                # file we will replace that file path by copying or by monkeypatching Path.exists -- simpler approach:
                # set attribute system._temp_video_path and monkeypatch run behavior via checking for it before opening.
                # But to avoid modifying the original class further, we set the current working directory to the
                # folder containing the chosen video and create a symlink-like copy named 'DeteccionDePersonas.mov' if needed.
                # To keep things simple and safe, we will set an attribute and rely on the existing open code which checks Path('DeteccionDePersonas.mov').
                # If the selected file is not the default name, we set system._override_video to the path and patch cv2.VideoCapture below by
                # temporarily replacing Path.exists via monkeypatching; however, to avoid fragile monkeypatching across modules,
                # we will create a small wrapper: if system has attribute 'video_path', we will call cv2.VideoCapture with that path inside a
                # minimal run wrapper. For this, we call system.run() but first monkeypatch its run method to prefer system.video_path.
                original_run = self.system.run

                def patched_run():
                    # Producer-consumer: display at 30 FPS regardless of processing speed.
                    import cv2
                    import queue
                    from datetime import datetime
                    from pathlib import Path as _P

                    # Setup session (creates logger, image_saver, etc.)
                    self.system.setup_session()

                    video_path = getattr(self.system, 'video_path', None)
                    print('\n📹 Abriendo fuente de vídeo (GUI)...')
                    if not video_path or not _P(video_path).exists():
                        print('❌ No se seleccionó un archivo de vídeo válido')
                        return

                    cap = cv2.VideoCapture(str(video_path))
                    if not cap.isOpened():
                        print('❌ No se pudo abrir la fuente de vídeo seleccionada')
                        return

                    print('✓ Fuente de vídeo inicializada (GUI)')
                    print('\n🎬 Iniciando sesión de conteo (GUI)...')

                    # Playback target: 30 FPS
                    target_fps = 30.0
                    target_interval = 1.0 / target_fps

                    # Create processing queue and synchronization primitives
                    frame_queue = queue.Queue(maxsize=8)
                    latest_tracked = {}
                    latest_lock = threading.Lock()
                    stop_event = threading.Event()

                    # Worker: consumes frames, runs detection/tracking, updates latest_tracked
                    def worker():
                        while not stop_event.is_set():
                            try:
                                frm = frame_queue.get(timeout=0.5)
                            except queue.Empty:
                                continue

                            try:
                                proc = self.system.preprocess_frame(frm)
                                dets = self.system.detector.detect_faces(proc)
                                dets = self.system.roi.filter_detections(dets)
                                tracked = self.system.process_detections(dets, proc)

                                # update shared tracked objects
                                with latest_lock:
                                    latest_tracked.clear()
                                    latest_tracked.update(tracked)

                                # image saving is handled inside process_detections/image_saver
                            except Exception as e:
                                print('Error en worker procesando frame:', e)

                    worker_thread = threading.Thread(target=worker, daemon=True)
                    worker_thread.start()

                    self.system.start_time = datetime.now()
                    self.system.is_running = True

                    try:
                        while self.system.is_running:
                            loop_start = time.time()

                            ret, frame = cap.read()
                            if not ret:
                                print('\n🔚 Fin del archivo de vídeo')
                                break

                            # Enqueue frame for processing (drop oldest if queue full)
                            try:
                                frame_copy = frame.copy()
                                try:
                                    frame_queue.put_nowait(frame_copy)
                                except queue.Full:
                                    # drop oldest then enqueue
                                    try:
                                        _ = frame_queue.get_nowait()
                                    except Exception:
                                        pass
                                    try:
                                        frame_queue.put_nowait(frame_copy)
                                    except Exception:
                                        pass
                            except Exception:
                                pass

                            # Prepare display frame (resize/preprocess lightly to match pipeline)
                            try:
                                display_frame = self.system.preprocess_frame(frame.copy())
                            except Exception:
                                display_frame = frame

                            # Overlay latest tracked objects if available
                            with latest_lock:
                                tracked_copy = dict(latest_tracked)

                            elapsed_time = datetime.now() - self.system.start_time
                            remaining_time = self.system.session_duration - elapsed_time

                            try:
                                annotated = self.system.draw_interface(display_frame, tracked_copy, elapsed_time, remaining_time)
                            except Exception:
                                annotated = display_frame

                            # Show frame at 30 FPS
                            cv2.imshow('Contador de Personas UDEM', annotated)

                            # Handle keypresses (non-blocking)
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q'):
                                print('\n🛑 Sesión terminada por el usuario')
                                break
                            elif key == ord('s'):
                                try:
                                    self.system.image_saver.save_full_frame(annotated, {'event': 'manual_capture'})
                                    print('📸 Captura manual guardada')
                                except Exception:
                                    pass
                            elif key == ord('r'):
                                print('🎯 Función ROI no implementada en esta versión')

                            # Sleep to maintain 30 FPS playback regardless of processing speed
                            elapsed = time.time() - loop_start
                            sleep_time = max(0.0, target_interval - elapsed)
                            if sleep_time > 0:
                                time.sleep(sleep_time)

                    finally:
                        # Stop worker and cleanup
                        stop_event.set()
                        worker_thread.join(timeout=1.0)
                        cap.release()
                        cv2.destroyAllWindows()
                        try:
                            self.system.generate_final_report()
                        except Exception:
                            pass
                        self.system.cleanup()

                # Replace and call
                self.system.run = patched_run
                self.system.run()
            except Exception as e:
                print('Error en sesión:', e)
            finally:
                # Ensure buttons reflect stopped state
                self.root.after(0, lambda: self._on_thread_finish())

        self.thread = threading.Thread(target=target_run, daemon=True)
        self.thread.start()

        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self._set_status('Sesión iniciada', 'green')

    def _on_thread_finish(self):
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self._set_status('Sesión detenida', 'blue')

    def stop_session(self):
        if self.system:
            try:
                # Signal the system to stop
                self.system.is_running = False
                # Some cleanup functions expect OpenCV windows
                self.system.cleanup()
                self._set_status('Deteniendo sesión...', 'orange')
            except Exception as e:
                self._set_status(f'Error deteniendo sesión: {e}', 'red')

    def _set_status(self, text, color='black'):
        self.status_label.config(text=f'Estado: {text}', foreground=color)


def run_gui():
    root = tk.Tk()
    app = PeopleCounterGUI(root)
    root.protocol('WM_DELETE_WINDOW', lambda: (app.stop_session(), root.destroy()))
    root.mainloop()


if __name__ == '__main__':
    run_gui()
