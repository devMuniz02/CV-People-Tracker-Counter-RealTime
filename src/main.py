"""
Interfaz simple para ajustar parámetros de detección y ejecutar el sistema
Creado como alternativa a `main.py`. Usa `DeteccionDePersonas.mov` por defecto
y permite cambiar parámetros de detección antes de iniciar la sesión.
"""

import threading
import tkinter as tk
from tkinter import ttk, filedialog
import time
import cv2

import sys
import json
import numpy as np
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

# --- Inlined minimal detectors and PeopleCounterSystem (extracted from src/main.py) ---
try:
    # Lightweight MTCNN wrapper (TensorFlow mtcnn package)
    class MTCNNFaceDetector:
        def __init__(self, *args, **kwargs):
            self.impl = None
            self.model = None
            self.available = False
            try:
                from mtcnn import MTCNN as MT
                self.model = MT()
                self.impl = 'mtcnn'
                self.available = True
            except Exception:
                self.model = None
                self.available = False

        def detect_faces(self, frame):
            if not self.available or self.model is None:
                return []
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
            except Exception:
                return []

        def get_method_name(self):
            return 'MTCNN'
except Exception:
    # Fallback if mtcnn import fails at class creation time
    class MTCNNFaceDetector:
        def __init__(self, *args, **kwargs):
            self.available = False
        def detect_faces(self, frame):
            return []
        def get_method_name(self):
            return 'MTCNN (unavailable)'


class DlibFaceDetector:
    def __init__(self):
        try:
            import dlib
            self.detector = dlib.get_frontal_face_detector()
            self.available = True
        except Exception:
            self.detector = None
            self.available = False

    def detect_faces(self, frame):
        if not getattr(self, 'available', False):
            return []
        try:
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


class PersonHOGDetector:
    def __init__(self):
        try:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.available = True
        except Exception:
            self.hog = None
            self.available = False

    def detect_faces(self, frame):
        if not getattr(self, 'available', False):
            return []
        try:
            img = frame.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects, weights = self.hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
            detections = []
            for (x, y, w, h), wt in zip(rects, weights):
                detections.append((int(x), int(y), int(w), int(h), float(wt)))
            return detections
        except Exception:
            return []


class RetinaFaceDetector:
    def __init__(self):
        try:
            from retinaface import RetinaFace
            self.api = RetinaFace
            self.available = True
        except Exception:
            self.api = None
            self.available = False

    def detect_faces(self, frame):
        if not getattr(self, 'available', False):
            return []
        try:
            results = self.api.detect_faces(frame)
            detections = []
            if isinstance(results, dict):
                for _, v in results.items():
                    box = v.get('facial_area') if isinstance(v, dict) else None
                    conf = v.get('score', 0.0) if isinstance(v, dict) else 0.0
                    if box is None:
                        continue
                    x1, y1, x2, y2 = box
                    w = int(x2 - x1); h = int(y2 - y1)
                    detections.append((int(x1), int(y1), w, h, float(conf)))
            return detections
        except Exception:
            return []


class PeopleCounterSystem:
    def __init__(self, config_path="config.json"):
        self.config = self._load_config(config_path)
        self.detector = self._initialize_detector()
        self.person_hog = None
        # Initialize lightweight tracker and ROI (import from tracking module in original)
        try:
            from tracking import CentroidTracker, RegionOfInterest
            # compute sensible max_disappeared based on fps_limit and allowed seconds
            fps_limit = int(self.config.get('session', {}).get('fps_limit', 30) or 30)
            max_seconds = float(self.config.get('tracking', {}).get('max_disappeared_seconds', 2.0) or 2.0)
            computed_max = max(1, int(round(max_seconds * fps_limit)))
            self.tracker = CentroidTracker(max_disappeared=computed_max, max_distance=self.config['tracking']['max_distance_threshold'])
            self.roi = RegionOfInterest()
        except Exception:
            # Fallback minimal tracker stub
            class _StubTracker:
                def __init__(self, max_disappeared=30, max_distance=100, max_seconds=2.0):
                    self._total = 0
                    self.max_disappeared = max_disappeared
                    self.max_distance = max_distance
                    self.max_seconds = float(max_seconds)
                    # store last seen timestamp per id-like key
                    self._next_id = 0
                    self._objects = {}  # id -> {'rect':(..), 'centroid':(..), 'last_seen': timestamp}

                def update(self, rects, frame_size=None):
                    # naive assignment: each rect becomes a unique id unless close to existing
                    out = {}
                    now = time.time()
                    for r in rects:
                        x, y, w, h = r
                        cx = int(x + w/2)
                        cy = int(y + h/2)
                        # find nearest existing object
                        assigned = None
                        for oid, data in list(self._objects.items()):
                            # only consider candidates seen recently
                            if now - data.get('last_seen', 0.0) > self.max_seconds:
                                continue
                            ox, oy = data['centroid']
                            dist = ((ox - cx) ** 2 + (oy - cy) ** 2) ** 0.5
                            if dist <= self.max_distance:
                                assigned = oid
                                break
                        if assigned is None:
                            oid = self._next_id
                            self._next_id += 1
                            self._objects[oid] = {'rect': (x, y, w, h), 'centroid': (cx, cy), 'last_seen': now}
                        else:
                            oid = assigned
                            self._objects[oid]['rect'] = (x, y, w, h)
                            self._objects[oid]['centroid'] = (cx, cy)
                            self._objects[oid]['last_seen'] = now

                    # remove objects not seen within max_seconds OR whose last known centroid is outside frame (left camera)
                    for oid, data in list(self._objects.items()):
                        # If frame_size is provided, consider objects outside the frame as gone
                        if frame_size is not None:
                            h_img, w_img = frame_size
                            cx, cy = data.get('centroid', (None, None))
                            if cx is None or cy is None or cx < 0 or cy < 0 or cx >= w_img or cy >= h_img:
                                # object left camera area -> remove permanently
                                del self._objects[oid]
                                continue
                        # Otherwise, remove if timed out
                        if now - data.get('last_seen', 0.0) > self.max_seconds:
                            del self._objects[oid]

                    # build output mapping
                    for oid, data in self._objects.items():
                        x, y, w, h = data['rect']
                        cx, cy = data['centroid']
                        out[oid] = {'rect': (int(x), int(y), int(w), int(h), 1.0), 'centroid': (int(cx), int(cy)), 'disappeared_frames': 0}
                    return out

                def get_total_count(self):
                    return len(self._objects)
            class _StubROI:
                def filter_detections(self, dets):
                    return dets
            # compute fallback parameters consistent with config
            fps_limit = int(self.config.get('session', {}).get('fps_limit', 30) or 30)
            max_seconds = float(self.config.get('tracking', {}).get('max_disappeared_seconds', 2.0) or 2.0)
            computed_max = max(1, int(round(max_seconds * fps_limit)))
            max_distance = int(self.config.get('tracking', {}).get('max_distance_threshold', 10) or 10)
            self.tracker = _StubTracker(max_disappeared=computed_max, max_distance=max_distance)
            self.roi = _StubROI()

        self.session_dir = None
        self.logger = None
        self.image_saver = None
        self.is_running = False
        self.session_duration = timedelta(minutes=self.config['session']['duration_minutes'])
        self.start_time = None
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0

    def _load_config(self, config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return self._get_default_config()

    def _get_default_config(self):
        return {
            "detection": {"method": "mtcnn", "scale_factor": 1.1, "min_neighbors": 5, "min_size": [20, 20], "max_size": [40, 40], "confidence_threshold": 0.6},
            "tracking": {"max_disappeared_frames": 30, "max_disappeared_seconds": 2.0, "max_distance_threshold": 100},
            "session": {"duration_minutes": 15, "location": "Lab de robótica", "fps_limit": 30, "save_full_frames": True, "save_face_crops": True},
            "preprocessing": {"resize_width": 640, "apply_blur": False, "apply_clahe": False},
            "output": {"base_path": "output", "image_quality": 95}
        }

    def _initialize_detector(self):
        method = self.config['detection'].get('method', 'mtcnn').lower()
        if method == 'mtcnn':
            return MTCNNFaceDetector()
        elif method == 'dlib':
            return DlibFaceDetector()
        elif method == 'person_hog':
            return PersonHOGDetector()
        elif method == 'retinaface':
            return RetinaFaceDetector()
        else:
            return MTCNNFaceDetector()

    def _initialize_tracker(self):
        try:
            from tracking import CentroidTracker
            return CentroidTracker(max_disappeared=self.config['tracking']['max_disappeared_frames'], max_distance=self.config['tracking']['max_distance_threshold'])
        except Exception:
            return None

    def setup_session(self):
        # Minimal setup: create session directory if needed
        base = Path(self.config.get('output', {}).get('base_path', 'output'))
        base.mkdir(parents=True, exist_ok=True)
        # Use SessionManager to create a timestamped session dir
        try:
            from data_io import SessionManager, EventLogger, ImageSaver
            loc = self.config.get('session', {}).get('location', 'UDEM')
            session_dir = SessionManager.create_session_directory(base_path=str(base), location=loc)
            self.session_dir = str(session_dir)

            # Initialize logger and image saver
            method_name = self.detector.get_method_name() if hasattr(self.detector, 'get_method_name') else self.config.get('detection', {}).get('method', 'mtcnn')
            self.logger = EventLogger(self.session_dir, location=loc, method=method_name)
            save_crops = bool(self.config.get('session', {}).get('save_face_crops', True))
            # Force disabling full-frame saving: only save crops as requested
            save_frames = False
            quality = int(self.config.get('output', {}).get('image_quality', 95) or 95)
            self.image_saver = ImageSaver(self.session_dir, save_crops=save_crops, save_frames=save_frames, quality=quality)

            # Save config used for the session
            try:
                self.logger.save_session_config(self.config)
            except Exception:
                pass
        except Exception:
            # Fallback: keep simple directory
            self.session_dir = str(base)

    def preprocess_frame(self, frame):
        # Reuse logic from main: resize and optional preprocessing
        if 'resize_width' in self.config.get('preprocessing', {}):
            width = self.config['preprocessing']['resize_width']
            if frame.shape[1] != width:
                height = int(frame.shape[0] * width / frame.shape[1])
                frame = cv2.resize(frame, (width, height))
        if self.config.get('preprocessing', {}).get('apply_clahe', False):
            try:
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            except Exception:
                pass
        return frame

    def draw_interface(self, frame, tracked_objects, elapsed_time, remaining_time, fps=None, source_label=None):
        # Reuse the same drawing panel as main.py
        panel_height = 120
        panel_color = (0, 0, 0)
        info_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        h_img, w_img = frame.shape[:2]
        # Place the frame at the top and the info panel below it so
        # detection coordinates match frame pixel coordinates directly.
        out_h = h_img + panel_height
        out = np.zeros((out_h, w_img, 3), dtype=frame.dtype)
        out[0:h_img, 0:w_img, :] = frame
        out[h_img:out_h, :, :] = panel_color

        # Draw bounding boxes directly on the top area (frame) without offset
        for obj_id, obj_data in tracked_objects.items():
            if 'rect' in obj_data and obj_data['rect']:
                rect = obj_data['rect']
                x, y, w, h = rect[:4]
                box_color = (0, 255, 0)
                text_color = (0, 255, 0)
                cv2.rectangle(out, (x, y), (x + w, y + h), box_color, 2)
                label = f"ID: {obj_id}"
                cv2.putText(out, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                if 'centroid' in obj_data:
                    cx, cy = obj_data['centroid']
                    cv2.circle(out, (int(cx), int(cy)), 4, (0, 128, 255), -1)

        # Prefer counting unique IDs seen by our internal matching logic if available
        if hasattr(self, '_pc_seen_ids') and isinstance(self._pc_seen_ids, (set, list)):
            try:
                total_count = len(self._pc_seen_ids)
            except Exception:
                total_count = getattr(self.tracker, 'get_total_count', lambda: 0)()
        else:
            total_count = getattr(self.tracker, 'get_total_count', lambda: 0)()
        active_count = len(tracked_objects)
        # Build status lines (draw a single info box inside the top panel)
        try:
            elapsed_str = str(elapsed_time).split('.')[0]
            remaining_str = str(remaining_time).split('.')[0]
        except Exception:
            elapsed_str = '0:00:00'
            remaining_str = '0:00:00'

        # Use provided fps if given (smoothed), otherwise fall back to stored current_fps
        try:
            disp_fps = float(fps) if fps is not None else float(self.current_fps or 0.0)
        except Exception:
            disp_fps = float(self.current_fps or 0.0)

        try:
            if self.session_dir:
                loc_name = Path(self.session_dir).name
            else:
                loc_name = self.config['session'].get('location', '')
        except Exception:
            loc_name = self.config['session'].get('location', '')

        lines = [
            f"Personas Unicas: {total_count} | Activas: {active_count}",
            f"Transcurrido: {elapsed_str} | Restante: {remaining_str}",
            f"FPS: {disp_fps:.1f} | Metodo: {self.detector.get_method_name()}",
            "Ubicacion: Lab de robotica"
        ]

        # Optionally include a short source label if provided
        if source_label:
            s = str(source_label)
            if len(s) > 48:
                s = '...' + s[-45:]
            lines.insert(0, f'Fuente: {s}')

        # Draw semi-opaque box inside the top panel
        try:
            box_x = 10
            box_y = h_img + 8
            pad = 8
            scale = 0.6
            thickness = 1
            # measure
            text_sizes = [cv2.getTextSize(t, font, scale, thickness)[0] for t in lines]
            max_w = max((w for (w, h) in text_sizes), default=0)
            text_h = text_sizes[0][1] if text_sizes else 12
            line_h = text_h + 8
            box_w = max_w + pad * 2
            box_h = line_h * len(lines) + pad * 2

            # Ensure box fits within panel area (panel starts at h_img)
            box_w = min(box_w, out.shape[1] - box_x - 10)
            # limit box_h to panel_height
            box_h = min(box_h, panel_height - 8)

            overlay = out.copy()
            cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

            # draw text lines
            start_y = box_y + pad + text_h
            for i, line in enumerate(lines):
                y = start_y + i * line_h
                color = (0, 255, 0) if line.startswith('Personas Unicas') or line.startswith('Activas') else info_color
                cv2.putText(out, line, (box_x + pad, y), font, scale, color, thickness, cv2.LINE_AA)
        except Exception:
            # fallback: draw previous separate texts if something fails
            try:
                # fallback: draw texts inside the bottom panel area
                base_y = h_img + 10
                text1 = f"Personas Unicas: {total_count} | Activas: {active_count}"
                cv2.putText(out, text1, (10, base_y + 15), font, 0.7, info_color, 2)
                text2 = f"Transcurrido: {elapsed_str} | Restante: {remaining_str}"
                cv2.putText(out, text2, (10, base_y + 40), font, 0.7, info_color, 2)
                text3 = f"FPS: {self.current_fps:.1f} | Metodo: {self.detector.get_method_name()}"
                cv2.putText(out, text3, (10, base_y + 65), font, 0.7, info_color, 2)
                text4 = f"Ubicacion: {loc_name}"
                cv2.putText(out, text4, (10, base_y + 90), font, 0.7, info_color, 2)
            except Exception:
                pass

        # progress bar
        progress_width = 300
        progress_height = 10
        progress_x = w_img - progress_width - 10
        progress_y = h_img + (panel_height - progress_height - 8)
        cv2.rectangle(out, (progress_x, progress_y), (progress_x + progress_width, progress_y + progress_height), (50, 50, 50), -1)
        elapsed_seconds = elapsed_time.total_seconds()
        total_seconds = self.session_duration.total_seconds()
        progress = min(1.0, elapsed_seconds / total_seconds) if total_seconds > 0 else 0
        progress_color = (0, 255, 0) if progress < 0.8 else (0, 255, 255) if progress < 0.95 else (0, 0, 255)
        progress_fill = int(progress_width * progress)
        cv2.rectangle(out, (progress_x, progress_y), (progress_x + progress_fill, progress_y + progress_height), progress_color, -1)

        # Draw 'no-detection' border zone overlay based on tracking.remove_on_border_margin
        try:
            try:
                border_margin = int(self.config.get('tracking', {}).get('remove_on_border_margin', 40) or 40)
            except Exception:
                border_margin = 20
            # Clamp border to reasonable size (not more than half of smallest dimension)
            if h_img is not None and w_img is not None and border_margin > 0:
                max_margin = min(h_img, w_img) // 2
                border_margin = max(0, min(border_margin, max_margin))

                overlay_zone = out.copy()
                zone_color = (0, 0, 255)  # red
                alpha_zone = 0.25

                # top strip
                if border_margin > 0:
                    cv2.rectangle(overlay_zone, (0, 0), (w_img, border_margin), zone_color, -1)
                    # bottom strip
                    cv2.rectangle(overlay_zone, (0, max(0, h_img - border_margin)), (w_img, h_img), zone_color, -1)
                    # left strip
                    cv2.rectangle(overlay_zone, (0, 0), (border_margin, h_img), zone_color, -1)
                    # right strip
                    cv2.rectangle(overlay_zone, (max(0, w_img - border_margin), 0), (w_img, h_img), zone_color, -1)

                    # Blend only the overlay area into the output
                    cv2.addWeighted(overlay_zone, alpha_zone, out, 1 - alpha_zone, 0, out)
        except Exception:
            # If anything fails drawing the zone, ignore and return the frame as-is
            pass

        return out

    def update_fps(self):
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time

    def process_detections(self, detections, frame):
        # Convert incoming detections to rects and centroids
        rects = []
        centroids = []
        for d in detections:
            try:
                x = int(d[0]); y = int(d[1]); w = int(d[2]); h = int(d[3])
                rects.append((x, y, w, h))
                centroids.append((int(x + w/2), int(y + h/2)))
            except Exception:
                continue

        # If tracker supports update(rects, frame_size) use it; otherwise perform distance-based matching
        try:
            h_img, w_img = frame.shape[:2]
        except Exception:
            h_img, w_img = (None, None)

        # Use internal distance-based matching to produce stable IDs across frames.
        # We intentionally avoid delegating to self.tracker.update here so that
        # ID assignment is consistent and based on centroid-distance to previous
        # detections stored in this PeopleCounterSystem instance.

        # Fallback: implement distance-based matching here to produce stable IDs
        # We'll keep an internal mapping on the PeopleCounterSystem instance: _pc_objects
        now = time.time()
        max_seconds = float(self.config.get('tracking', {}).get('max_disappeared_seconds', 10.0) or 10.0)
        # Use a tighter default matching distance to avoid merging nearby people.
        max_distance = float(self.config.get('tracking', {}).get('max_distance_threshold', 10) or 10)
        # Determine border margin early so we can ignore new detections inside it
        try:
            border_margin = int(self.config.get('tracking', {}).get('remove_on_border_margin', 20) or 20)
        except Exception:
            border_margin = 20
        if not hasattr(self, '_pc_objects'):
            # id -> {'rect':(x,y,w,h), 'centroid':(cx,cy), 'last_seen':ts}
            self._pc_objects = {}
            self._pc_next_id = 0
            # set of unique ids seen during the session (used to compute Personas Unicas)
            self._pc_seen_ids = set()

        assigned_ids = set()
        new_tracked = {}

        # For each detection, find nearest existing object within max_distance
        for rect, centroid in zip(rects, centroids):
            cx, cy = centroid
            best_id = None
            best_dist = None
            for oid, data in list(self._pc_objects.items()):
                # ignore objects that timed out
                if now - data.get('last_seen', 0.0) > max_seconds:
                    continue
                ox, oy = data['centroid']
                dist = ((ox - cx) ** 2 + (oy - cy) ** 2) ** 0.5
                if dist <= max_distance and (best_dist is None or dist < best_dist):
                    best_dist = dist
                    best_id = oid

            if best_id is None:
                # If detection centroid is inside the configured border margin (red zone),
                # treat it as not a valid new detection (ignore it). This prevents
                # creating a new persistent ID for objects appearing inside the
                # no-detection border.
                try:
                    if h_img is not None and w_img is not None and border_margin and border_margin > 0:
                        if (cx <= border_margin or cy <= border_margin or
                                (w_img - 1 - cx) <= border_margin or (h_img - 1 - cy) <= border_margin):
                            # skip creating a new id for this detection
                            continue
                except Exception:
                    pass

                # create new id
                oid = self._pc_next_id
                self._pc_next_id += 1
                self._pc_objects[oid] = {'rect': rect, 'centroid': centroid, 'last_seen': now}
                assigned_ids.add(oid)
                # record unique id seen
                try:
                    self._pc_seen_ids.add(oid)
                except Exception:
                    pass
                new_tracked[oid] = {'rect': (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]), 1.0), 'centroid': (int(cx), int(cy)), 'disappeared_frames': 0}
            else:
                # update existing object
                self._pc_objects[best_id]['rect'] = rect
                self._pc_objects[best_id]['centroid'] = centroid
                self._pc_objects[best_id]['last_seen'] = now
                assigned_ids.add(best_id)
                r = rect
                new_tracked[best_id] = {'rect': (int(r[0]), int(r[1]), int(r[2]), int(r[3]), 1.0), 'centroid': (int(cx), int(cy)), 'disappeared_frames': 0}

        # Remove objects that either moved outside the frame or were last seen
        # at/near the border (interpreted as having left the scene). We no
        # longer remove objects purely by timeout here — objects remain
        # available for matching as long as their last known centroid stays
        # inside the frame away from the border. This makes re-entering the
        # scene after briefly disappearing at the border create a new unique
        # id (as requested).
        #
        # Configurable margin: number of pixels from the edge to consider the
        # object as 'at the border'. It can be set in config under
        # tracking.remove_on_border_margin. Default is 20 pixels.
        try:
            border_margin = int(self.config.get('tracking', {}).get('remove_on_border_margin', 20) or 20)
        except Exception:
            border_margin = 20

        for oid, data in list(self._pc_objects.items()):
            if oid in assigned_ids:
                continue
            # if we have frame size, check if centroid is outside frame -> consider left camera
            if h_img is not None and w_img is not None:
                ox, oy = data.get('centroid', (None, None))
                if ox is None or oy is None:
                    # can't reason about position, remove conservatively
                    del self._pc_objects[oid]
                    continue
                # If centroid is completely outside image bounds -> remove
                if ox < 0 or oy < 0 or ox >= w_img or oy >= h_img:
                    del self._pc_objects[oid]
                    continue
                # If centroid is within border_margin of any edge, treat object as
                # potentially leaving the scene. We remove it so that if it
                # re-enters later it will be considered a new unique id.
                if (ox <= border_margin or oy <= border_margin or
                        (w_img - 1 - ox) <= border_margin or (h_img - 1 - oy) <= border_margin):
                    del self._pc_objects[oid]
                    continue
            else:
                # No frame size known: do not remove by timeout anymore; keep object
                # so that matching will prefer closest id when available.
                pass

        return new_tracked

    def _filter_detections_by_size(self, detections):
        if not detections:
            return []
        min_size = tuple(self.config['detection'].get('min_size', (0, 0)))
        max_size = tuple(self.config['detection'].get('max_size', (30, 30)))
        min_w, min_h = int(min_size[0]), int(min_size[1])
        max_w, max_h = int(max_size[0]), int(max_size[1])
        filtered = []
        for det in detections:
            try:
                x, y, w, h = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            except Exception:
                continue
            source = det[5] if len(det) > 5 else None
            if source == 'person_hog':
                filtered.append(det)
                continue
            if w >= min_w and h >= min_h and w <= max_w and h <= max_h:
                filtered.append(det)
        return filtered

    def _rects_overlap(self, rect1, rect2, threshold=0.3):
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
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
        # Minimal run to satisfy calls from GUI; delegate heavy lifting to patched_run in GUI
        self.setup_session()

    def generate_final_report(self):
        try:
            print('Generating final report...')
        except Exception:
            pass

    def cleanup(self):
        try:
            pass
        except Exception:
            pass

# --- End of inlined classes ---


class PeopleCounterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("People Counter - GUI")

        # System instance
        self.system = PeopleCounterSystem()
        self.thread = None
        # Person HOG detector removed — GUI uses only MTCNN as primary detector

        # Variables de control
        # Fixed: only mtcnn is selectable in this simplified GUI
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
        self.max_w_var = tk.IntVar(value=50)
        self.max_h_var = tk.IntVar(value=50)
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
        # Show a static label explaining the available detector
        ttk.Label(frm, text='MTCNN (TensorFlow)').grid(row=0, column=1, sticky='w')

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
        # Force MTCNN as primary detector
        det['method'] = 'mtcnn'
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
        # Only check MTCNN (via PeopleCounterSystem initialization) and a locally-instantiated
        # PersonHOGDetector (if available). Keep this check non-destructive and tolerant to
        # import failures so the GUI still starts in minimal environments.
        avail = []
        # Check MTCNN availability via the system initializer
        try:
            tmp = PeopleCounterSystem()
            tmp.config['detection']['method'] = 'mtcnn'
            det = tmp._initialize_detector()
            avail.append('mtcnn' if getattr(det, 'available', True) else 'mtcnn (no)')
        except Exception:
            avail.append('mtcnn (no)')

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
                    self._set_status("Método cambiado a 'mtcnn'", 'green')
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
        # Only support MTCNN in this GUI helper
        if m == 'mtcnn':
            try:
                tmp = PeopleCounterSystem()
                tmp.config['detection']['method'] = 'mtcnn'
                tmp.config['detection']['confidence_threshold'] = float(self.confidence_var.get())
                detector = tmp._initialize_detector()
                available = getattr(detector, 'available', True)

                def _call(f):
                    try:
                        return detector.detect_faces(f) or []
                    except Exception:
                        return []

                name = detector.get_method_name() if hasattr(detector, 'get_method_name') else 'MTCNN'
                return _call, bool(available), name
            except Exception:
                return (lambda f: [], False, 'MTCNN')

        # Unknown method -> no-op
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
            self._set_status("Detector seleccionado 'mtcnn' no disponible", 'orange')
            return

        try:
            proc = system.preprocess_frame(self.current_frame.copy())

            # Primary detector (MTCNN)
            dets = det.detect_faces(proc)

            total = len(dets)
            self._set_status(f'Detecciones encontradas: {total} (detector: mtcnn)', 'green')

            # Annotate and show a small preview: faces -> green
            preview = proc.copy()
            for (x, y, w, h, conf) in dets:
                cv2.rectangle(preview, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

            cv2.imshow('Prueba Detector', preview)
            cv2.waitKey(1)
        except Exception as e:
            self._set_status(f'Error en prueba de detector: {e}', 'red')

    def _open_install_dialog(self):
        """Open a small dialog to select optional packages to install via pip."""
        dlg = tk.Toplevel(self.root)
        dlg.title('Instalar detectores')

        # Only provide installation option for MTCNN (TensorFlow mtcnn package).
        opts = {'mtcnn': tk.BooleanVar(value=False)}

        ttk.Label(dlg, text='Selecciona detectores a instalar:').grid(row=0, column=0, columnspan=2, pady=(6,6))
        ttk.Checkbutton(dlg, text='MTCNN (mtcnn)', variable=opts['mtcnn']).grid(row=1, column=0, sticky='w')

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
        failures = []
        for pkg in packages:
            try:
                # Map friendly names to pip package names (only mtcnn supported here)
                if pkg == 'mtcnn':
                    candidates = ['mtcnn']
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

        # Apply size filter to face detections
        try:
            try:
                filtered_faces = self.system._filter_detections_by_size(face_dets)
            except Exception:
                filtered_faces = face_dets
        except Exception:
            filtered_faces = face_dets

        # Combine detections for display/tracking: faces only
        dets = list(filtered_faces)

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
        elapsed = timedelta(0)
        remaining = timedelta(0)

        try:
            annotated = self.system.draw_interface(proc.copy(), tracked, elapsed, remaining)
        except Exception:
            annotated = proc

    # Draw face boxes with explicit colors on top of annotated image
        try:
            # faces: green (0,255,0)
            for (x, y, w, h, conf) in filtered_faces:
                cv2.rectangle(annotated, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
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
            self._set_status('Error aplicando parámetros: ' + str(e), 'red')
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
                out_dir = Path(self.system.session_dir) if getattr(self.system, 'session_dir', None) else Path('output')
                out_dir = Path(out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                filename = out_dir / f'tuning_annotated_{int(time.time())}.jpg'
                cv2.imwrite(str(filename), self.last_annotated)
                self._set_status('Guardado: ' + str(filename), 'green')
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
                    import queue
                    from datetime import datetime

                    # Setup session (creates logger, image_saver, etc.)
                    self.system.setup_session()

                    video_path = getattr(self.system, 'video_path', None)
                    print('\n📹 Abriendo fuente de vídeo (GUI)...')
                    if not video_path or not Path(video_path).exists():
                        print('❌ No se seleccionó un archivo de vídeo válido')
                        return

                    cap = cv2.VideoCapture(str(video_path))
                    if not cap.isOpened():
                        print('❌ No se pudo abrir la fuente de vídeo seleccionada')
                        return

                    print('✓ Fuente de vídeo inicializada (GUI)')
                    print('\n🎬 Iniciando sesión de conteo (GUI)...')

                    # Determine playback FPS from video (fall back to 30 FPS)
                    try:
                        video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
                    except Exception:
                        video_fps = 30.0
                    target_fps = max(1.0, video_fps)
                    target_interval = 1.0 / target_fps

                    # Seek to a small startup offset to skip camera adjustment at the start
                    # Default: 12 seconds, override by setting `self.system.start_seconds` before run
                    try:
                        start_seconds = float(getattr(self.system, 'start_seconds', 12.0) or 12.0)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                        start_frame = int(round(start_seconds * video_fps))
                        if total_frames and start_frame >= total_frames:
                            start_frame = max(0, total_frames - 1)
                        # Set capture position
                        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                        print(f'✅ Saltando a segundo {start_seconds} (frame {start_frame})')
                    except Exception:
                        pass

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

                                # Apply ROI filter to primary detections (faces/persons depending on detector)
                                try:
                                    dets = self.system.roi.filter_detections(dets)
                                except Exception:
                                    pass

                                tracked = self.system.process_detections(dets, proc)

                                # Determine newly entered and exited IDs by comparing previous snapshot
                                try:
                                    prev_ids = set(latest_tracked.keys())
                                except Exception:
                                    prev_ids = set()
                                curr_ids = set(tracked.keys())

                                new_ids = curr_ids - prev_ids
                                exited_ids = prev_ids - curr_ids

                                # Log and save captures for new IDs
                                for nid in new_ids:
                                    try:
                                        # Log enter
                                        if getattr(self.system, 'logger', None):
                                            try:
                                                # We don't have a per-detection confidence here; pass None
                                                self.system.logger.log_person_enter(nid, confidence=None)
                                            except Exception:
                                                pass
                                        # Save crop and frame for evidence
                                        if getattr(self.system, 'image_saver', None):
                                            try:
                                                rect = tracked[nid].get('rect')
                                                if rect:
                                                    x, y, w, h = rect[:4]
                                                    # Use the preprocessed frame (proc) so coordinates and image content match
                                                    self.system.image_saver.save_person_crop(proc, nid, (x, y, w, h), confidence=None)
                                                # save full frame once per new id (still disabled in ImageSaver by config)
                                                self.system.image_saver.save_full_frame(proc, {'person_id': nid, 'event': 'enter'})
                                            except Exception:
                                                pass
                                    except Exception:
                                        pass

                                # Log exited ids
                                for eid in exited_ids:
                                    try:
                                        if getattr(self.system, 'logger', None):
                                            try:
                                                self.system.logger.log_person_exit(eid, reason='timeout_or_out_of_frame')
                                            except Exception:
                                                pass
                                    except Exception:
                                        pass

                                # Post a non-blocking UI update with the latest counts (mtcnn only)
                                try:
                                    total = len(dets)
                                    try:
                                        self.root.after(0, lambda t=total: self._set_status(f'Detecciones encontradas: {t} (mtcnn)', 'green'))
                                    except Exception:
                                        pass
                                except Exception:
                                    pass

                                # update shared tracked objects
                                with latest_lock:
                                    latest_tracked.clear()
                                    latest_tracked.update(tracked)

                                # image saving is handled inside process_detections/image_saver
                            except Exception as e:
                                print('Error en worker procesando frame:', e)

                    worker_thread = threading.Thread(target=worker, daemon=True)
                    worker_thread.start()

                    # Mark session start: keep datetime for logs but use a monotonic epoch for elapsed time
                    self.system.start_time = datetime.now()
                    # epoch_start is used to compute elapsed time using real seconds (time.time())
                    self.system._epoch_start = time.time()
                    self.system.is_running = True
                    # Video writer for recording the displayed annotated window (lazy init)
                    video_writer = None
                    video_out_path = None
                    # Count displayed frames (used to compute elapsed time based on a fixed FPS)
                    frames_counted = 0

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

                            # Compute elapsed/remaining using real seconds (monotonic epoch) so UI shows wall-clock time
                            try:
                                if hasattr(self.system, '_epoch_start') and self.system._epoch_start:
                                    elapsed_secs = max(0.0, time.time() - float(self.system._epoch_start))
                                    elapsed_time = timedelta(seconds=elapsed_secs)
                                else:
                                    elapsed_time = datetime.now() - self.system.start_time
                            except Exception:
                                elapsed_time = datetime.now() - self.system.start_time
                            try:
                                remaining_time = self.system.session_duration - elapsed_time
                            except Exception:
                                remaining_time = self.system.session_duration

                            # Compute a smoothed FPS for display (measure based on processing/display loop)
                            try:
                                if not hasattr(self, '_fps_history'):
                                    self._fps_history = []
                                now = time.time()
                                if not hasattr(self, '_last_frame_time'):
                                    self._last_frame_time = now
                                frame_dt = now - getattr(self, '_last_frame_time', now)
                                self._last_frame_time = now
                                if frame_dt > 0:
                                    inst_fps = 1.0 / frame_dt
                                else:
                                    inst_fps = target_fps
                                self._fps_history.append(inst_fps)
                                # keep last N values
                                if len(self._fps_history) > 15:
                                    self._fps_history.pop(0)
                                smooth_fps = sum(self._fps_history) / len(self._fps_history)
                            except Exception:
                                smooth_fps = target_fps

                            try:
                                # Centralize all text drawing inside draw_interface so offsets are consistent
                                # Use frames_counted (fixed 30 FPS) to compute elapsed/remaining for the UI
                                try:
                                    frames_counted += 1
                                except Exception:
                                    pass
                                try:
                                    elapsed_secs = float(frames_counted) / 30.0
                                    elapsed_time = timedelta(seconds=elapsed_secs)
                                except Exception:
                                    # fallback to wall-clock if something goes wrong
                                    try:
                                        if hasattr(self.system, '_epoch_start') and self.system._epoch_start:
                                            elapsed_secs = max(0.0, time.time() - float(self.system._epoch_start))
                                            elapsed_time = timedelta(seconds=elapsed_secs)
                                        else:
                                            elapsed_time = datetime.now() - self.system.start_time
                                    except Exception:
                                        elapsed_time = datetime.now() - self.system.start_time
                                try:
                                    remaining_time = self.system.session_duration - elapsed_time
                                except Exception:
                                    remaining_time = self.system.session_duration

                                annotated = self.system.draw_interface(display_frame, tracked_copy, elapsed_time, remaining_time, fps=smooth_fps, source_label=None)
                            except Exception:
                                annotated = display_frame

                            # Show frame at video FPS (simulate live camera)
                            cv2.imshow('Contador de Personas UDEM', annotated)

                            # Lazy-init VideoWriter once we have the annotated frame size
                            try:
                                if video_writer is None and getattr(self.system, 'session_dir', None):
                                    video_out_path = Path(self.system.session_dir) / 'ContadorDePersonas.mp4'
                                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                    h_a, w_a = annotated.shape[:2]
                                    # Force recording at 30 FPS as requested
                                    rec_fps = 30.0
                                    # Note: VideoWriter expects (width, height)
                                    video_writer = cv2.VideoWriter(str(video_out_path), fourcc, rec_fps, (w_a, h_a))
                            except Exception:
                                video_writer = None

                            # Write annotated frame to video if writer initialized
                            wrote_frame = False
                            if video_writer is not None:
                                try:
                                    video_writer.write(annotated)
                                    wrote_frame = True
                                except Exception:
                                    wrote_frame = False

                            # (frames_counted increment and elapsed calculation are handled before draw_interface)

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

                            # Stop the session when the configured session duration elapses (based on frames)
                            try:
                                if elapsed_time >= self.system.session_duration:
                                    print('\n⏱️  Sesión completa: tiempo de sesión alcanzado (por frames)')
                                    break
                            except Exception:
                                pass

                            # Sleep to maintain video FPS (simulate live camera). We account for time spent
                            elapsed = time.time() - loop_start
                            sleep_time = max(0.0, target_interval - elapsed)
                            # If the processing is faster than video framerate, wait a bit to match camera rate
                            if sleep_time > 0:
                                time.sleep(sleep_time)

                    finally:
                        # Stop worker and cleanup
                        stop_event.set()
                        worker_thread.join(timeout=1.0)
                        cap.release()
                        cv2.destroyAllWindows()
                        # Release video writer if opened
                        try:
                            if video_writer is not None:
                                video_writer.release()
                        except Exception:
                            pass
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
