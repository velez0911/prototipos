import argparse
import csv
import os
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
IMAGE_SIZE = (128, 128)
FEATURE_VERSION = 1
MIN_COLOR_FRACTION = 0.01
CLASSES = ("BUENA", "DANADA")
HOG = cv2.HOGDescriptor(
    _winSize=IMAGE_SIZE,
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9,
)


def extract_features(image):
    """Convierte una imagen BGR o en escala de grises a un vector HOG fijo."""
    if image is None or image.size == 0:
        raise ValueError("La imagen esta vacia o no se pudo decodificar.")
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError("Formato de imagen no compatible.")

    gray = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return HOG.compute(gray).reshape(-1).astype(np.float32)


def load_model(path):
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(
            "No existe el modelo '{}'. Entrena uno con el comando 'train'.".format(model_path)
        )
    bundle = joblib.load(model_path)
    if bundle.get("feature_version") != FEATURE_VERSION:
        raise ValueError("El modelo usa una version de caracteristicas no compatible.")
    classifier = bundle.get("classifier")
    if classifier is None or set(classifier.classes_) != set(CLASSES):
        raise ValueError("El archivo no contiene un modelo BUENA/DANADA valido.")
    return classifier


def physical_break_reason(image):
    if image is None or image.size == 0:
        raise ValueError("La imagen esta vacia o no se pudo decodificar.")
    if image.ndim != 3 or image.shape[2] != 3:
        return None

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    colored_pixels = ((hsv[:, :, 1] >= 75) & (hsv[:, :, 2] >= 40)).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        colored_pixels, connectivity=8
    )
    min_fragment_area = max(100, int(colored_pixels.size * 0.005))
    fragments = [
        index for index in range(1, component_count)
        if stats[index, cv2.CC_STAT_AREA] >= min_fragment_area
    ]
    if len(fragments) < 2:
        return None
    if len(fragments) > 2:
        return "Fragmentos separados"

    first, second = fragments
    distance_map = cv2.distanceTransform(
        (labels != second).astype(np.uint8), cv2.DIST_L2, 3
    )
    gap = float(distance_map[labels == first].min())
    crack_gap = max(4.0, min(image.shape[:2]) * 0.03)
    if gap <= crack_gap:
        return "Grieta visible"
    return "Fragmentos separados"


def product_color_counts(image):
    if image is None or image.size == 0:
        raise ValueError("La imagen esta vacia o no se pudo decodificar.")
    if image.ndim != 3 or image.shape[2] != 3:
        return None

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturated = (hsv[:, :, 1] >= 75) & (hsv[:, :, 2] >= 40)
    damaged = saturated & (((hue >= 36) & (hue <= 85)) | ((hue >= 90) & (hue <= 121)))
    good = saturated & (((hue >= 18) & (hue <= 34)) | ((hue >= 123) & (hue <= 170)))
    damaged_count = int(np.count_nonzero(damaged))
    good_count = int(np.count_nonzero(good))
    return damaged_count, good_count


def has_product_color(image):
    counts = product_color_counts(image)
    if counts is None:
        return False
    color_count = sum(counts)
    min_color_pixels = max(
        100, int(image.shape[0] * image.shape[1] * MIN_COLOR_FRACTION)
    )
    return color_count >= min_color_pixels


def classify_by_color(image, threshold):
    counts = product_color_counts(image)
    if counts is None:
        return None
    damaged_count, good_count = counts
    color_count = damaged_count + good_count
    if color_count < max(100, int(image.shape[0] * image.shape[1] * MIN_COLOR_FRACTION)):
        return None

    damaged_probability = damaged_count / color_count
    if damaged_probability >= threshold:
        return "DANADA", damaged_probability
    if damaged_probability <= 1.0 - threshold:
        return "BUENA", damaged_probability
    return "REVISAR", damaged_probability


def classify_image_with_reason(classifier, image, threshold):
    break_reason = physical_break_reason(image)
    if break_reason:
        return "DANADA", 1.0, break_reason

    color_result = classify_by_color(image, threshold)
    if color_result is not None:
        label, probability = color_result
        reason = {
            "DANADA": "Color defectuoso",
            "BUENA": "Sin defecto",
            "REVISAR": "Requiere revision",
        }[label]
        return label, probability, reason

    if classifier is None:
        return "REVISAR", 0.5, "Requiere revision"

    probabilities = classifier.predict_proba([extract_features(image)])[0]
    danada_probability = float(probabilities[list(classifier.classes_).index("DANADA")])
    if danada_probability >= threshold:
        label = "DANADA"
        reason = "Defecto detectado por IA"
    elif danada_probability <= 1.0 - threshold:
        label = "BUENA"
        reason = "Sin defecto"
    else:
        label = "REVISAR"
        reason = "Requiere revision"
    return label, danada_probability, reason


def classify_image(classifier, image, threshold):
    label, probability, _ = classify_image_with_reason(classifier, image, threshold)
    return label, probability


def read_training_data(data_dir):
    root = Path(data_dir)
    features = []
    labels = []
    counts = {name: 0 for name in CLASSES}
    for folder, label in (("buena", "BUENA"), ("danada", "DANADA")):
        class_dir = root / folder
        if not class_dir.is_dir():
            raise FileNotFoundError("Falta la carpeta de entrenamiento: {}".format(class_dir))
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image = cv2.imread(str(path))
            if image is None:
                print("Aviso: se omite imagen ilegible: {}".format(path), file=sys.stderr)
                continue
            features.append(extract_features(image))
            labels.append(label)
            counts[label] += 1

    for label, count in counts.items():
        if count < 2:
            raise ValueError(
                "Se necesitan al menos 2 imagenes legibles en cada clase; "
                "{} tiene {}.".format(label, count)
            )
    return np.asarray(features), np.asarray(labels), counts


def make_classifier():
    return RandomForestClassifier(
        n_estimators=120,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def train(args):
    features, labels, counts = read_training_data(args.data)
    print("Imagenes cargadas: BUENA={}, DANADA={}".format(counts["BUENA"], counts["DANADA"]))

    class_count = len(CLASSES)
    test_count = max(class_count, int(round(len(labels) * 0.2)))
    can_validate = test_count <= len(labels) - class_count
    if can_validate:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=test_count,
            random_state=42,
            stratify=labels,
        )
        validation_model = make_classifier()
        validation_model.fit(x_train, y_train)
        predictions = validation_model.predict(x_test)
        print("Exactitud en particion reservada: {:.1%}".format(accuracy_score(y_test, predictions)))
        print("Matriz de confusion (filas=real, columnas=predicho; BUENA, DANADA):")
        print(confusion_matrix(y_test, predictions, labels=list(CLASSES)))
        print(classification_report(y_test, predictions, labels=list(CLASSES), zero_division=0))
    else:
        print(
            "Aviso: hay muy pocas imagenes para reservar una evaluacion independiente; "
            "el rendimiento aun no esta validado."
        )

    classifier = make_classifier()
    classifier.fit(features, labels)
    output = Path(args.model)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=output.stem + "_",
            suffix=".joblib",
            dir=str(output.parent),
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
        joblib.dump({"feature_version": FEATURE_VERSION, "classifier": classifier}, temp_path)
        os.replace(str(temp_path), str(output))
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    print("Modelo guardado en: {}".format(output.resolve()))


def inspect_still(args):
    classifier = load_model(args.model)
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError("No se pudo abrir la imagen: {}".format(args.image))
    label, probability = classify_image(classifier, image, args.threshold)
    print("{} | probabilidad de DANADA: {:.1%}".format(label, probability))


class LatestFrame:
    def __init__(self, source):
        self.source = source
        self.capture = None
        self.frame = None
        self.error = None
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._read, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _read(self):
        source = self.source
        if source.isdigit():
            source = int(source)
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            self.error = "No se pudo abrir la camara o stream: {}".format(self.source)
            self.stopped.set()
            return
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while not self.stopped.is_set():
            ok, frame = self.capture.read()
            if ok and frame is not None:
                self.frame = frame
            else:
                self.error = "No se recibio un fotograma de la camara."
                time.sleep(0.05)

    def stop(self):
        self.stopped.set()
        if self.capture is not None:
            self.capture.release()
        if self.thread.is_alive():
            self.thread.join(timeout=2)


def append_inspection(csv_path, source, label, probability, image=None, save_images=False):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    image_path = ""
    if save_images and image is not None:
        image_dir = csv_path.parent / "imagenes"
        image_dir.mkdir(parents=True, exist_ok=True)
        safe_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_file = image_dir / "{}_{}.jpg".format(safe_time, label.lower())
        if not cv2.imwrite(str(image_file), image):
            raise OSError("No se pudo guardar la imagen de inspeccion: {}".format(image_file))
        image_path = str(image_file)
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        if is_new:
            writer.writerow(("timestamp", "source", "resultado", "probabilidad_danada", "imagen"))
        writer.writerow((timestamp, source, label, "{:.6f}".format(probability), image_path))


def safe_source(source):
    parts = urlsplit(source)
    if not parts.scheme or not parts.netloc:
        return source
    host = parts.hostname or ""
    if ":" in host and not host.startswith("["):
        host = "[{}]".format(host)
    if parts.port:
        host = "{}:{}".format(host, parts.port)
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def draw_overlay(frame, label, probability, fps, counts, threshold):
    colors = {
        "BUENA": (45, 170, 45),
        "DANADA": (40, 40, 220),
        "REVISAR": (0, 170, 230),
        "ESPERANDO": (150, 150, 150),
    }
    color = colors.get(label, colors["ESPERANDO"])
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 122), (28, 30, 34), -1)
    cv2.putText(
        frame,
        "VisionLine | {}".format(label),
        (18, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "DANADA: {:.1%}  |  umbral: {:.0%}  |  {:.1f} FPS".format(
            probability, threshold, fps
        ),
        (18, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Registradas  BUENA: {}  DANADA: {}  REVISAR: {}".format(
            counts["BUENA"], counts["DANADA"], counts["REVISAR"]
        ),
        (18, 101),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return frame


def inspect_camera(args):
    classifier = load_model(args.model)
    stream = LatestFrame(args.source).start()
    log_source = safe_source(args.source)
    print("Camara: {} | ESPACIO registra | Q/ESC sale".format(log_source))
    print("CSV: {}".format(Path(args.log).resolve()))
    counts = {name: 0 for name in ("BUENA", "DANADA", "REVISAR")}
    label = "ESPERANDO"
    probability = 0.0
    latest = None
    last_inference = 0.0
    last_log = 0.0
    last_frame_time = time.perf_counter()
    fps = 0.0

    try:
        while True:
            if stream.error:
                raise RuntimeError(stream.error)
            if stream.stopped.is_set():
                break
            frame = stream.frame
            if frame is None:
                time.sleep(0.01)
                continue
            now = time.perf_counter()
            elapsed = now - last_frame_time
            if elapsed > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / elapsed) if fps else 1.0 / elapsed
            last_frame_time = now

            if now - last_inference >= 0.03:
                label, probability = classify_image(classifier, frame, args.threshold)
                latest = frame.copy()
                last_inference = now

            display = draw_overlay(frame.copy(), label, probability, fps, counts, args.threshold)
            cv2.imshow("VisionLine - Inspeccion", display)
            key = cv2.waitKey(1) & 0xFF
            should_log = key == ord(" ") or (
                args.auto_log and now - last_log >= args.interval
            )
            if should_log and latest is not None:
                append_inspection(
                    args.log, log_source, label, probability, latest, args.save_images
                )
                counts[label] += 1
                last_log = now
                print("{} | DANADA {:.1%}".format(label, probability))
            if key in (ord("q"), 27):
                break
    finally:
        stream.stop()
        cv2.destroyAllWindows()


def positive_threshold(value):
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("El umbral debe ser un numero entre 0.5 y 1.") from exc
    if not 0.5 < number < 1.0:
        raise argparse.ArgumentTypeError("El umbral debe estar entre 0.5 y 1 (sin incluirlos).")
    return number


def positive_interval(value):
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("El intervalo debe ser un numero positivo.") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("El intervalo debe ser mayor que cero.")
    return number


def build_parser():
    parser = argparse.ArgumentParser(
        description="VisionLine: entrenamiento e inspeccion visual local de piezas."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="entrena con carpetas buena/ y danada/")
    train_parser.add_argument("--data", required=True, help="carpeta con buena/ y danada/")
    train_parser.add_argument("--model", default="modelo.joblib", help="ruta de salida del modelo")
    train_parser.set_defaults(func=train)

    image_parser = commands.add_parser("image", help="inspecciona una imagen")
    image_parser.add_argument("--model", default="modelo.joblib")
    image_parser.add_argument("--image", required=True, help="ruta de la imagen")
    image_parser.add_argument("--threshold", type=positive_threshold, default=0.70)
    image_parser.set_defaults(func=inspect_still)

    camera_parser = commands.add_parser("camera", help="inspecciona una webcam o stream de video")
    camera_parser.add_argument("--model", default="modelo.joblib")
    camera_parser.add_argument("--source", default="0", help="indice de webcam o URL compatible con OpenCV")
    camera_parser.add_argument("--threshold", type=positive_threshold, default=0.70)
    camera_parser.add_argument("--log", default="resultados/inspecciones.csv")
    camera_parser.add_argument("--auto-log", action="store_true", help="registra muestras por intervalo")
    camera_parser.add_argument("--interval", type=positive_interval, default=0.5)
    camera_parser.add_argument("--save-images", action="store_true", help="guarda los fotogramas registrados")
    camera_parser.set_defaults(func=inspect_camera)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.exit(1, "Error: {}\n".format(error))


if __name__ == "__main__":
    main()
