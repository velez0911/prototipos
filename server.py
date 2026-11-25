import argparse
import csv
import ctypes
import hashlib
import json
import mimetypes
import os
import shutil
import tempfile
import threading
import time
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

import cv2

import main


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "produccion"
PHOTOS = OUTPUT / "imagenes"
CAPTURES = OUTPUT / "capturas"
CSV_PATH = OUTPUT / "inspecciones.csv"
TRAINING_DATA = ROOT / "datos"
TRAINING_FOLDERS = {"BUENA": "buena", "DANADA": "danada"}
PRODUCTION_STOP_SECONDS = 30
PRODUCT_COLOR_CONFIRM_FRAMES = 2
PRODUCT_COLOR_CLEAR_FRAMES = 3
DEFECT_REASONS = (
    "Grieta visible",
    "Fragmentos separados",
    "Color defectuoso",
    "Otros",
)


def defect_reason_index(reason):
    normalized = (reason or "").casefold()
    if "grieta" in normalized:
        return 0
    if "fragmento" in normalized:
        return 1
    if "color defectuoso" in normalized:
        return 2
    return 3


def defect_reason_counts(pieces):
    counts = [0] * len(DEFECT_REASONS)
    for piece in pieces:
        if piece["resultado"] == "Defectuosa":
            counts[defect_reason_index(piece.get("motivo"))] += 1
    return counts


class WindowsInstanceLock:
    def __init__(self, kernel32, handle):
        self.kernel32 = kernel32
        self.handle = handle

    def close(self):
        self.kernel32.CloseHandle(self.handle)


def acquire_instance_lock(project_root=None):
    lock_root = Path(project_root).resolve() if project_root is not None else ROOT
    lock_name = hashlib.sha256(str(lock_root).encode("utf-8")).hexdigest()[:12]
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p)
        kernel32.CreateMutexW.restype = ctypes.c_void_p
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        kernel32.CloseHandle.restype = ctypes.c_int
        kernel32.SetLastError.argtypes = (ctypes.c_ulong,)
        kernel32.GetLastError.restype = ctypes.c_ulong
        kernel32.SetLastError(0)
        handle = kernel32.CreateMutexW(None, False, "Local\\VisionLine_{}".format(lock_name))
        if not handle:
            raise ctypes.WinError(kernel32.GetLastError())
        if kernel32.GetLastError() == 183:
            kernel32.CloseHandle(handle)
            raise SystemExit("VisionLine ya tiene una instancia activa en este proyecto.")
        return WindowsInstanceLock(kernel32, handle)

    lock_path = Path(tempfile.gettempdir()) / "visionline_{}.lock".format(lock_name)
    lock_file = lock_path.open("a+b")
    try:
        import fcntl

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        lock_file.close()
        raise SystemExit("VisionLine ya tiene una instancia activa en este proyecto.") from error
    return lock_file


class ProductionState:
    def __init__(self, model_path):
        self.lock = threading.RLock()
        self.counts = {"BUENA": 0, "DANADA": 0, "REVISAR": 0}
        self.total = 0
        self.last = None
        self.lots = {}
        self.last_piece_at = time.monotonic()
        OUTPUT.mkdir(parents=True, exist_ok=True)
        PHOTOS.mkdir(parents=True, exist_ok=True)
        CAPTURES.mkdir(parents=True, exist_ok=True)
        if not CSV_PATH.exists():
            with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as file:
                csv.writer(file).writerow(
                    ("timestamp", "lote", "pieza", "resultado", "probabilidad_danada", "motivo", "foto")
                )
        self.normalize_duplicate_piece_ids()
        self.load_history()

    @staticmethod
    def write_history_rows(fieldnames, rows):
        if not fieldnames:
            raise ValueError("El CSV de producción no tiene encabezados.")
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                newline="",
                encoding="utf-8-sig",
                prefix=CSV_PATH.stem + "_",
                suffix=".tmp",
                dir=str(CSV_PATH.parent),
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                writer = csv.DictWriter(temporary_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            os.replace(str(temporary_path), str(CSV_PATH))
        except Exception:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
            raise

    def normalize_duplicate_piece_ids(self):
        with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames
            rows = list(reader)
        if not fieldnames or "pieza" not in fieldnames:
            raise ValueError("El CSV de producción no tiene la columna pieza.")

        seen = set()
        changed = False
        for row in rows:
            piece_id = row.get("pieza", "")
            if not piece_id:
                continue
            if piece_id in seen:
                while True:
                    replacement = "{}-{}".format(piece_id, uuid.uuid4().hex[:8].upper())
                    if replacement not in seen:
                        break
                row["pieza"] = replacement
                piece_id = replacement
                changed = True
            seen.add(piece_id)
        if changed:
            self.write_history_rows(fieldnames, rows)

    def load_history(self):
        if not CSV_PATH.exists():
            return
        with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames
            rows = list(reader)
        csv_changed = False
        for row in rows:
            if not row:
                continue
            timestamp = row.get("timestamp", "")
            lot_id = row.get("lote", "")
            piece_id = row.get("pieza", "")
            label = row.get("resultado", "REVISAR")
            try:
                probability = float(row.get("probabilidad_danada", "0"))
            except ValueError:
                probability = 0.0
            reason = row.get("motivo", "")
            photo = row.get("foto", "")
            if not timestamp or not lot_id or not piece_id:
                continue

            manual_label = self.saved_training_label(piece_id)
            if manual_label:
                manual_result = "Buena" if manual_label == "BUENA" else "Defectuosa"
                if label != manual_result or ("motivo" in row and reason != "Clasificacion manual"):
                    row["resultado"] = manual_result
                    if "motivo" in row:
                        row["motivo"] = "Clasificacion manual"
                    csv_changed = True
                label = manual_result
                reason = "Clasificacion manual"

            date = timestamp[:10]
            lot = self.lots.setdefault(
                lot_id,
                self.empty_lot(lot_id, date, timestamp),
            )
            piece = self.piece(piece_id, lot_id, date, timestamp, label, probability, reason, photo)
            piece["etiqueta_manual"] = manual_label
            lot["piezas"].append(piece)
            lot["cantidad"] = len(lot["piezas"])
            lot["buenas"] = sum(p["resultado"] == "Buena" for p in lot["piezas"])
            lot["defectuosas"] = sum(p["resultado"] == "Defectuosa" for p in lot["piezas"])
            if label == "Defectuosa":
                lot["byReason"][defect_reason_index(reason)] += 1
            self.counts["BUENA"] += label == "Buena"
            self.counts["DANADA"] += label == "Defectuosa"
            self.counts["REVISAR"] += label not in ("Buena", "Defectuosa")
            self.total += 1
        if csv_changed:
            self.write_history_rows(fieldnames, rows)

    @staticmethod
    def shift(turn_hour):
        return "Matutino" if 6 <= turn_hour < 14 else "Vespertino"

    def empty_lot(self, lot_id, date, timestamp):
        hour = int(timestamp[11:13]) if len(timestamp) >= 13 else 6
        turn = self.shift(hour)
        return {
            "id": lot_id,
            "fecha": date,
            "inicio": "{:02d}:00".format(hour if turn == "Matutino" else max(hour - 8, 14)),
            "fin": "--:--",
            "turno": turn,
            "operador": "Línea automática",
            "cantidad": 0,
            "buenas": 0,
            "defectuosas": 0,
            "estado": "En curso",
            "horas": [0, 0],
            "startH": hour,
            "byReason": [0] * len(DEFECT_REASONS),
            "piezas": [],
        }

    @staticmethod
    def piece(piece_id, lot_id, date, timestamp, label, probability, reason, photo):
        return {
            "pieza": piece_id,
            "lote": lot_id,
            "fecha": date,
            "hora": timestamp[11:16],
            "estacion": "Camara IA",
            "operador": "Línea automática",
            "resultado": label,
            "motivo": reason or ("—" if label == "Buena" else "Defecto detectado por IA"),
            "_probabilidad": probability,
            "_foto": photo,
            "etiqueta_manual": "",
        }

    def get_lot(self, now):
        date = now.strftime("%Y-%m-%d")
        hour = now.hour
        turn_hour = hour
        shift_start = 6 if 6 <= hour < 14 else 14 if 14 <= hour < 22 else 22
        lot_prefix = date.replace("-", "")[-6:]
        with self.lock:
            candidates = [
                lot for lot in self.lots.values()
                if lot["fecha"] == date and lot["turno"] == self.shift(turn_hour)
                and lot["estado"] == "En curso"
            ]
            if candidates:
                lot = candidates[-1]
            else:
                number = sum(
                    lot["fecha"] == date and lot["turno"] == self.shift(turn_hour)
                    for lot in self.lots.values()
                ) + 1
                lot_id = "L-{}-{:02d}".format(lot_prefix, number)
                lot = self.empty_lot(lot_id, date, now.isoformat(timespec="seconds"))
                lot["inicio"] = "{:02d}:00".format(shift_start)
                lot["turno"] = self.shift(turn_hour)
                lot["piezas"] = []
                self.lots[lot_id] = lot
            return lot

    def record(self, label, probability, frame, threshold, reason=None):
        now = datetime.now().astimezone()
        with self.lock:
            lot = self.get_lot(now)
            sequence = len(lot["piezas"]) + 1
            day_code = now.strftime("%y%m%d")
            lot_code = lot["id"].rsplit("-", 1)[-1]
            piece_id = "P-{}{}-{:03d}-{}".format(
                day_code, lot_code, sequence, uuid.uuid4().hex[:8].upper()
            )
            reason = reason or (
                "Detectado como dañado"
                if label == "DANADA"
                else ("Sin defecto" if label == "BUENA" else "Requiere revision")
            )
            photo_name = ""
            photo_url = ""
            if frame is not None:
                photo_name = "{}_{}.jpg".format(now.strftime("%Y%m%d_%H%M%S_%f"), piece_id)
                photo_path = PHOTOS / photo_name
                cv2.imwrite(str(photo_path), frame)
                photo_url = "/produccion/imagenes/" + photo_name
            lot["cantidad"] = len(lot["piezas"]) + 1
            lot["piezas"].append(
                self.piece(
                    piece_id,
                    lot["id"],
                    now.strftime("%Y-%m-%d"),
                    now.isoformat(timespec="seconds"),
                    "Buena" if label == "BUENA" else ("Defectuosa" if label == "DANADA" else "REVISAR"),
                    probability,
                    reason,
                    photo_url,
                )
            )
            lot["buenas"] = sum(p["resultado"] == "Buena" for p in lot["piezas"])
            lot["defectuosa"] = sum(p["resultado"] == "Defectuosa" for p in lot["piezas"])
            lot["defectuosas"] = lot["defectuosa"]
            if label == "BUENA":
                self.counts["BUENA"] += 1
            elif label == "DANADA":
                self.counts["DANADA"] += 1
                lot["byReason"][defect_reason_index(reason)] += 1
            else:
                self.counts["REVISAR"] += 1
            self.total += 1
            self.last = {
                "pieza": piece_id,
                "lote": lot["id"],
                "resultado": "Buena" if label == "BUENA" else ("Defectuosa" if label == "DANADA" else "REVISAR"),
                "probabilidad": probability,
                "timestamp": now.isoformat(timespec="seconds"),
                "motivo": reason,
            }
            with CSV_PATH.open("a", newline="", encoding="utf-8-sig") as file:
                csv.writer(file).writerow(
                    (now.isoformat(timespec="seconds"), lot["id"], piece_id,
                     self.last["resultado"], "{:.6f}".format(probability), reason, photo_url)
                )
            self.last_piece_at = time.monotonic()
            return self.last

    @staticmethod
    def saved_training_label(piece_id):
        for label, folder in TRAINING_FOLDERS.items():
            if (TRAINING_DATA / folder / (piece_id + ".jpg")).is_file():
                return label
        return ""

    def label_for_training(self, piece_id, label):
        folder = TRAINING_FOLDERS[label]
        result = "Buena" if label == "BUENA" else "Defectuosa"
        with self.lock:
            piece = next(
                (piece for lot in self.lots.values() for piece in lot["piezas"]
                 if piece["pieza"] == piece_id),
                None,
            )
            if piece is None:
                raise KeyError("No existe la pieza solicitada.")
            photo_url = piece.get("_foto", "")
            if not photo_url:
                raise FileNotFoundError("La pieza no tiene una foto guardada.")
            source = PHOTOS / Path(urlsplit(photo_url).path).name
            if not source.is_file():
                raise FileNotFoundError("No se encontró la foto guardada de la pieza.")

            target_dir = TRAINING_DATA / folder
            other_label = "DANADA" if label == "BUENA" else "BUENA"
            target = target_dir / (piece_id + ".jpg")
            previous = TRAINING_DATA / TRAINING_FOLDERS[other_label] / (piece_id + ".jpg")
            target_dir.mkdir(parents=True, exist_ok=True)
            if target.exists() and previous.exists():
                raise OSError("La pieza aparece etiquetada en ambas carpetas de entrenamiento.")

            moved = False
            copied = False
            try:
                if not target.exists():
                    if previous.exists():
                        previous.replace(target)
                        moved = True
                    else:
                        copied = True
                        shutil.copy2(source, target)
                self.update_manual_result_in_csv(piece_id, result)
            except Exception:
                if moved and target.exists():
                    target.replace(previous)
                elif copied and target.exists():
                    target.unlink()
                raise

            piece["etiqueta_manual"] = label
            piece["resultado"] = result
            piece["motivo"] = "Clasificacion manual"
            for current_lot in self.lots.values():
                current_lot["cantidad"] = len(current_lot["piezas"])
                current_lot["buenas"] = sum(item["resultado"] == "Buena" for item in current_lot["piezas"])
                current_lot["defectuosas"] = sum(item["resultado"] == "Defectuosa" for item in current_lot["piezas"])
                current_lot["defectuosa"] = current_lot["defectuosas"]
                current_lot["byReason"] = defect_reason_counts(current_lot["piezas"])
            all_pieces = [item for current_lot in self.lots.values() for item in current_lot["piezas"]]
            self.counts = {
                "BUENA": sum(item["resultado"] == "Buena" for item in all_pieces),
                "DANADA": sum(item["resultado"] == "Defectuosa" for item in all_pieces),
                "REVISAR": sum(item["resultado"] not in ("Buena", "Defectuosa") for item in all_pieces),
            }
            if self.last and self.last["pieza"] == piece_id:
                self.last["resultado"] = result
                self.last["motivo"] = "Clasificacion manual"
            return {
                "pieza": piece_id,
                "etiqueta": label,
                "resultado": result,
                "motivo": piece["motivo"],
            }

    def update_manual_result_in_csv(self, piece_id, result):
        with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames
            rows = list(reader)
        if not fieldnames or "pieza" not in fieldnames or "resultado" not in fieldnames:
            raise ValueError("El CSV de producción no tiene las columnas esperadas.")
        matches = [row for row in rows if row.get("pieza") == piece_id]
        if len(matches) != 1:
            raise ValueError("El registro de la pieza no aparece exactamente una vez en el CSV.")
        matches[0]["resultado"] = result
        if "motivo" in fieldnames:
            matches[0]["motivo"] = "Clasificacion manual"
        self.write_history_rows(fieldnames, rows)

    def delete_piece(self, piece_id):
        with self.lock:
            lot = next(
                (lot for lot in self.lots.values()
                 if any(piece["pieza"] == piece_id for piece in lot["piezas"])),
                None,
            )
            if lot is None:
                raise KeyError("No existe la pieza solicitada.")
            piece = next(piece for piece in lot["piezas"] if piece["pieza"] == piece_id)
            photo_url = piece.get("_foto", "")
            photo_path = PHOTOS / Path(urlsplit(photo_url).path).name if photo_url else None

            with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as file:
                rows = list(csv.reader(file))
            if not rows or "pieza" not in rows[0] or "timestamp" not in rows[0] or "resultado" not in rows[0]:
                raise ValueError("El CSV de producción no tiene las columnas esperadas.")
            piece_index = rows[0].index("pieza")
            remaining_rows = [rows[0]]
            removed_count = 0
            for row in rows[1:]:
                if len(row) > piece_index and row[piece_index] == piece_id:
                    removed_count += 1
                else:
                    remaining_rows.append(row)
            if removed_count != 1:
                raise ValueError("El registro de la pieza no aparece exactamente una vez en el CSV.")

            temporary_csv = None
            staged_photo = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    newline="",
                    encoding="utf-8-sig",
                    prefix=CSV_PATH.stem + "_",
                    suffix=".tmp",
                    dir=str(CSV_PATH.parent),
                    delete=False,
                ) as temporary_file:
                    temporary_csv = Path(temporary_file.name)
                    csv.writer(temporary_file).writerows(remaining_rows)
                if photo_path is not None and photo_path.is_file():
                    with tempfile.NamedTemporaryFile(
                        prefix=photo_path.name + "_",
                        suffix=".deleted",
                        dir=str(PHOTOS),
                        delete=False,
                    ) as temporary_photo:
                        staged_photo = Path(temporary_photo.name)
                    staged_photo.unlink()
                    photo_path.replace(staged_photo)
                os.replace(str(temporary_csv), str(CSV_PATH))
                temporary_csv = None
            except Exception:
                if staged_photo is not None and staged_photo.exists() and photo_path is not None and not photo_path.exists():
                    staged_photo.replace(photo_path)
                if temporary_csv is not None and temporary_csv.exists():
                    temporary_csv.unlink()
                raise

            lot["piezas"] = [item for item in lot["piezas"] if item["pieza"] != piece_id]
            if not lot["piezas"]:
                self.lots.pop(lot["id"])
            else:
                lot["cantidad"] = len(lot["piezas"])
                lot["buenas"] = sum(item["resultado"] == "Buena" for item in lot["piezas"])
                lot["defectuosas"] = sum(item["resultado"] == "Defectuosa" for item in lot["piezas"])
                lot["defectuosa"] = lot["defectuosas"]
                lot["byReason"] = defect_reason_counts(lot["piezas"])

            remaining_pieces = [item for current_lot in self.lots.values() for item in current_lot["piezas"]]
            self.total = len(remaining_pieces)
            self.counts = {
                "BUENA": sum(item["resultado"] == "Buena" for item in remaining_pieces),
                "DANADA": sum(item["resultado"] == "Defectuosa" for item in remaining_pieces),
                "REVISAR": sum(item["resultado"] not in ("Buena", "Defectuosa") for item in remaining_pieces),
            }
            if remaining_rows[1:]:
                latest_row = max(
                    remaining_rows[1:],
                    key=lambda row: row[rows[0].index("timestamp")] if len(row) > rows[0].index("timestamp") else "",
                )
                header = rows[0]
                latest = dict(zip(header, latest_row))
                try:
                    probability = float(latest.get("probabilidad_danada", "0"))
                except ValueError:
                    probability = 0.0
                self.last = {
                    "pieza": latest.get("pieza", ""),
                    "lote": latest.get("lote", ""),
                    "resultado": latest.get("resultado", "REVISAR"),
                    "probabilidad": probability,
                    "timestamp": latest.get("timestamp", ""),
                    "motivo": latest.get("motivo", ""),
                }
            else:
                self.last = None

            warning = ""
            if staged_photo is not None:
                try:
                    staged_photo.unlink()
                except OSError as error:
                    warning = "El registro se eliminó, pero no se pudo borrar el archivo temporal de la foto: {}".format(error)
            return {"pieza": piece_id, "eliminada": True, "aviso": warning}

    def production_status(self, camera_enabled=True):
        if not camera_enabled:
            return "camara_apagada"
        with self.lock:
            if time.monotonic() - self.last_piece_at >= PRODUCTION_STOP_SECONDS:
                return "parada"
            return "produciendo"

    def reset_production_timeout(self):
        with self.lock:
            self.last_piece_at = time.monotonic()

    def snapshot(self, camera_enabled=True, camera_error=None):
        with self.lock:
            lots = []
            for lot in sorted(self.lots.values(), key=lambda item: (item["fecha"], item["inicio"])):
                copied = dict(lot)
                copied["piezas"] = [
                    dict(piece, foto=piece["_foto"]) if "_foto" in piece else dict(piece)
                    for piece in lot["piezas"]
                ]
                lots.append(copied)
            return {
                "status": "operando",
                "counts": dict(self.counts),
                "total": self.total,
                "last": self.last,
                "camera_enabled": camera_enabled,
                "camera_error": camera_error,
                "production_status": self.production_status(camera_enabled),
                "lots": lots,
            }


class CameraCapture:
    def __init__(self, source):
        self.source_label = source
        self.source = int(source) if source.isdigit() else source
        self.lock = threading.Lock()
        self.frame = None
        self.error = None
        self.stop_event = threading.Event()
        self.enabled = True
        self.thread = None
        self.capture = self._open_capture()
        if self.capture is None:
            self.enabled = False

    def _open_capture(self):
        if isinstance(self.source, int) and os.name == "nt":
            capture = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            capture.release()
            self.error = "No se pudo abrir la camara: {}".format(self.source_label)
            return None
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.error = None
        return capture

    def start(self):
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        while not self.stop_event.is_set():
            with self.lock:
                if not self.enabled or self.capture is None:
                    capture = None
                else:
                    capture = self.capture
                    ok, frame = capture.read()
                    if ok and frame is not None:
                        self.frame = frame
                        self.error = None
                    else:
                        self.frame = None
                        self.error = "Se perdio la lectura de la camara."
            if capture is None or not ok:
                time.sleep(0.1)
                continue

    def set_enabled(self, enabled):
        with self.lock:
            if enabled:
                if self.enabled and self.capture is not None:
                    return
                self.capture = self._open_capture()
                self.enabled = self.capture is not None
                self.frame = None
            else:
                self.enabled = False
                capture = self.capture
                self.capture = None
                self.frame = None
                self.error = None
        if not enabled and capture is not None:
            capture.release()

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.stop_event.set()
        with self.lock:
            self.enabled = False
            capture = self.capture
            self.capture = None
            self.frame = None
        if capture is not None:
            capture.release()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2)


def create_handler(state, camera):
    class Handler(BaseHTTPRequestHandler):
        server_version = "VisionLineServer/1.0"

        def send_json(self, payload, code=200):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            if path == "/api/state":
                self.send_json(state.snapshot(camera.enabled, camera.error))
            elif path == "/api/status":
                self.send_json({
                    "operating": camera.enabled and camera.error is None,
                    "camera_enabled": camera.enabled,
                    "camera_error": camera.error,
                    "production_status": state.production_status(camera.enabled),
                    "error": camera.error,
                    "last": state.last,
                    "counts": state.counts,
                })
            elif path == "/api/frame":
                frame = camera.get_frame()
                if frame is None:
                    self.send_json({"error": camera.error or "Sin fotogramas"}, 503)
                    return
                ok, encoded = cv2.imencode(".jpg", frame)
                if not ok:
                    self.send_json({"error": "No se pudo codificar el fotograma"}, 500)
                    return
                body = encoded.tobytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif path == "/" or path == "/index.html":
                self.serve_file(ROOT / "index.html", "text/html")
            elif path.startswith("/produccion/imagenes/"):
                name = Path(path).name
                self.serve_file(PHOTOS / name, "image/jpeg")
            elif path.startswith("/produccion/capturas/"):
                name = Path(path).name
                self.serve_file(CAPTURES / name, "image/jpeg")
            elif path.startswith("/static/"):
                self.serve_file(ROOT / path.lstrip("/"))
            else:
                self.send_json({"error": "Ruta no encontrada"}, 404)

        def do_POST(self):
            path = urlsplit(self.path).path
            if path not in ("/api/record", "/api/label", "/api/delete-piece", "/api/camera"):
                self.send_json({"error": "Ruta no encontrada"}, 404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.send_json({"error": "JSON invalido"}, 400)
                return
            if not isinstance(payload, dict):
                self.send_json({"error": "El cuerpo JSON debe ser un objeto"}, 400)
                return
            if path == "/api/camera":
                enabled = payload.get("enabled")
                if not isinstance(enabled, bool):
                    self.send_json({"error": "enabled debe ser true o false"}, 400)
                    return
                camera.set_enabled(enabled)
                if enabled and camera.enabled:
                    state.reset_production_timeout()
                result = state.snapshot(camera.enabled, camera.error)
                result["error"] = camera.error
                self.send_json(result, 200 if not enabled or camera.enabled else 503)
                return
            if path == "/api/delete-piece":
                piece_id = payload.get("piece")
                if not isinstance(piece_id, str) or not piece_id:
                    self.send_json({"error": "Falta el identificador de la pieza"}, 400)
                    return
                try:
                    self.send_json(state.delete_piece(piece_id))
                except KeyError as error:
                    self.send_json({"error": str(error)}, 404)
                except (OSError, ValueError) as error:
                    self.send_json({"error": str(error)}, 500)
                return
            if path == "/api/label":
                piece_id = payload.get("piece")
                label = payload.get("label")
                if not isinstance(piece_id, str) or not piece_id:
                    self.send_json({"error": "Falta el identificador de la pieza"}, 400)
                    return
                if label not in TRAINING_FOLDERS:
                    self.send_json({"error": "La etiqueta debe ser BUENA o DANADA"}, 400)
                    return
                try:
                    self.send_json(state.label_for_training(piece_id, label))
                except KeyError as error:
                    self.send_json({"error": str(error)}, 404)
                except FileNotFoundError as error:
                    self.send_json({"error": str(error)}, 404)
                except OSError as error:
                    self.send_json({"error": str(error)}, 500)
                return
            label = str(payload.get("label", "REVISAR")).upper()
            try:
                probability = float(payload.get("probability", 0.5))
            except (TypeError, ValueError):
                probability = 0.5
            if label not in ("BUENA", "DANADA", "REVISAR"):
                label = "REVISAR"
            probability = max(0.0, min(1.0, probability))
            result = state.record(label, probability, camera.get_frame(), float(payload.get("threshold", 0.70)))
            self.send_json(result)

        def serve_file(self, path, content_type=None):
            if not path.is_file():
                self.send_json({"error": "Archivo no encontrado"}, 404)
                return
            body = path.read_bytes()
            if content_type is None:
                content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format_string, *args):
            return

    return Handler


def save_periodic_snapshot(frame):
    CAPTURES.mkdir(parents=True, exist_ok=True)
    name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
    path = CAPTURES / name
    if not cv2.imwrite(str(path), frame):
        raise OSError("No se pudo guardar la captura periódica: {}".format(path))
    return path


def run_detection(state, camera, classifier, threshold, interval):
    next_snapshot = time.monotonic()
    color_frames = 0
    clear_frames = 0
    product_active = False
    while not camera.stop_event.is_set():
        if not camera.enabled:
            color_frames = 0
            clear_frames = 0
            product_active = False
            time.sleep(0.1)
            continue
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        now = time.monotonic()
        if now >= next_snapshot:
            try:
                save_periodic_snapshot(frame)
            except OSError as error:
                print(error)
            next_snapshot = now + interval

        if main.has_product_color(frame):
            color_frames += 1
            clear_frames = 0
            if not product_active and color_frames >= PRODUCT_COLOR_CONFIRM_FRAMES:
                label, probability, reason = main.classify_image_with_reason(
                    classifier, frame, threshold
                )
                state.record(label, probability, frame, threshold, reason)
                product_active = True
        else:
            color_frames = 0
            if product_active:
                clear_frames += 1
                if clear_frames >= PRODUCT_COLOR_CLEAR_FRAMES:
                    product_active = False
                    clear_frames = 0

        time.sleep(min(0.08, max(0.0, next_snapshot - now)))


def build_parser():
    parser = argparse.ArgumentParser(description="Servidor local de VisionLine")
    parser.add_argument("--source", default="0", help="indice de webcam o URL RTSP")
    parser.add_argument("--model", default="modelo.joblib")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument(
        "--interval",
        "--cooldown",
        dest="interval",
        type=main.positive_interval,
        default=10.0,
        help="segundos entre fotos (por defecto: 10)",
    )
    return parser


def main_server():
    args = build_parser().parse_args()
    if not 0.5 <= args.threshold < 1:
        raise SystemExit("El umbral debe estar entre 0.5 y 1.")
    instance_lock = acquire_instance_lock()
    server = None
    camera = None
    serving = False
    try:
        server = ThreadingHTTPServer(
            (args.host, args.port),
            BaseHTTPRequestHandler,
            bind_and_activate=False,
        )
        server.server_bind()
        server.server_activate()
    except OSError as error:
        if server is not None:
            server.server_close()
        instance_lock.close()
        raise SystemExit(
            "No se pudo reservar {}:{}; puede que VisionLine ya este activo: {}".format(
                args.host, args.port, error
            )
        ) from error

    try:
        state = ProductionState(args.model)
        try:
            classifier = main.load_model(args.model)
        except (FileNotFoundError, ValueError):
            classifier = None
            print("Aviso: no hay modelo valido '{}'; las detecciones quedaran como REVISAR.".format(args.model))
        camera = CameraCapture(args.source)
        camera.start()
        threading.Thread(
            target=run_detection,
            args=(state, camera, classifier, args.threshold, args.interval),
            daemon=True,
        ).start()
        server.RequestHandlerClass = create_handler(state, camera)
        print("VisionLine: http://{}:{}/".format(args.host, args.port))
        print("Camara: {}".format(args.source))
        serving = True
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None and serving:
            server.shutdown()
        if server is not None:
            server.server_close()
        if camera is not None:
            camera.stop()
        instance_lock.close()


if __name__ == "__main__":
    main_server()
