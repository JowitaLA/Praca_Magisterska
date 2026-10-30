from __future__ import annotations

# Plik extraction.py odpowiada za ekstrakcję cech twarzy.
# - DETEKCJA TWARZY: YOLO, jeżeli nie MediaPipe, jeżeli nie fallback (fixed 70% obrazu)
# - PREPROCESSING: Gaussian blur + Median + CLAHE (histogram equalization)
# - WYRÓWNANIE: Alignment po oczach (rotation, scale i translation)
# - MASKA TWARZY: FaceMesh landmarks => convex hull + erosion
# - EMBEDDING: ArcFace (ONNX 512-dim) lub fallback (classic_plus_landmarks)
#
# WYJŚCIE:
# - embeddings.csv: Znormalizowane embeddingu (L2, cosine-ready)
# - dataset_summary.csv: Statystyki źródeł (profile/ff_train/ff_test)
# - masks/, preprocessed/, figures/: Artefakty (dla debugowania i walidacji)
import sys

import argparse
import csv
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
    
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from skimage.feature import local_binary_pattern
from ultralytics import settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from resources.config import *

# Docelowy rozmiar twarzy dla ArcFace / fallbacku
TARGET_W, TARGET_H = 112, 112

# Preprocessing obrazu
THRESHOLD_DUPLICATES = 0.99
GAUSS_SIGMA = 1.2
MEDIAN_K = 3
CLAHE_CLIP = 2.0
CLAHE_GRID = 8

# Fallback klasyczny
P = 8
R = 2
CANNY_LO, CANNY_HI = 50, 150
MIN_CC_AREA_DEFAULT = 9

# Parametry detekcji / wyrównania
FACE_SCALE_FACTOR = 1.2
FACE_MIN_NEIGH = 5
FACE_MIN_SIZE_FRAC = 0.15
FACE_MARGIN_FRAC = 0.08
EYE_DIST_FRAC_DEFAULT = 0.33

MIN_DATA_SIZE = 30

# Nazwy katalogów danych
PROFILE_DIR_NAME = "profile"
VGG_DIR_NAME = "FairFace"
ff_train_NAME = "train"
ff_test_NAME = "test"
VERIFICATION_DIR_NAME = "verification"

# Flagi sterujące
store_outputs = False # Czy mają być generowane outputy (tj. masks, figure, preprocessed, validation)
delete_photos = False # Czy zdjęcia z bazy danych mają być usuwane (RODO)

@dataclass(frozen=True)
class Settings:
    target_w: int = TARGET_W
    target_h: int = TARGET_H
    gauss_sigma: float = GAUSS_SIGMA
    median_k: int = MEDIAN_K
    clahe_clip: float = CLAHE_CLIP
    clahe_grid: int = CLAHE_GRID
    lbp_p: int = P
    lbp_r: int = R
    canny_lo: int = CANNY_LO
    canny_hi: int = CANNY_HI
    min_cc_area: int = MIN_CC_AREA_DEFAULT
    random_state: int = RANDOM_STATE
    eye_dist_frac: float = EYE_DIST_FRAC_DEFAULT
    threshold_duplicates: float = THRESHOLD_DUPLICATES
    use_gaussian_blur: bool = False
    use_median_filter: bool = False
    use_clahe: bool = False
    use_face_mask: bool = False


@dataclass(frozen=True)
class FaceParams:
    scale_factor: float = FACE_SCALE_FACTOR
    min_neighbors: int = FACE_MIN_NEIGH
    min_size_frac: float = FACE_MIN_SIZE_FRAC
    margin_frac: float = FACE_MARGIN_FRAC


@dataclass(frozen=True)
class SampleInfo:
    file_path: Path
    identity: str
    source: str  # profile / ff_train / ff_test
    min_data_size: int = MIN_DATA_SIZE


_YOLO_MODEL = None
_MP_FACE_DET = None
_FACEMESH = None
_ARCFACE_SESSION = None
_ARCFACE_INPUT_NAME = None
_ARCFACE_AVAILABLE = None
_YOLO_AVAILABLE = None


# FUNKCJA Z ŚCIEŻKAMI I TWORZENIU KATALOGÓW WYJŚCIOWYCH
def build_paths(base_dir: Path, db: str, store_outputs: bool) -> Dict[str, Path]:
    ek_dir = Path(__file__).resolve().parent  
    out = ek_dir / "outputs"  
    weights = base_dir / "weights"  
    
    paths: Dict[str, Path] = {
        "base": base_dir,  
        "weights": weights,  
        "data": base_dir / db,  
        "profiles": base_dir / db / PROFILE_DIR_NAME,  
        "vgg_root": base_dir / db / VGG_DIR_NAME,  
        "ff_train": base_dir / db / VGG_DIR_NAME / ff_train_NAME,  
        "ff_test": base_dir / db / VGG_DIR_NAME / ff_test_NAME,  
        "verification": base_dir / db / VERIFICATION_DIR_NAME,  
        "ek": ek_dir,  
        "out": out,
        # Logi i główny plik wyjściowy zawsze zapisujemy:
        "embeddings_csv": out / "embeddings.csv", 
        "ff_train_csv": base_dir / db / "ff_train.csv",
        "ff_test_csv": base_dir / db / "ff_test.csv",
    }
    
    for key in ["weights", "out"]:
        paths[key].mkdir(parents=True, exist_ok=True)
        
    if store_outputs:
        figs = out / "figures"
        paths.update({
            "prep": out / "preprocessed",
            "mask": out / "masks",
            "figs": figs,
            "fig_pipeline": figs / "pipeline",
            "fig_roi": figs / "roi_overlays",
            "valid": out / "validation",
            "valid_figs": out / "validation" / "figures",
            "dataset_summary_csv": out / "dataset_summary.csv",
            
            "sample_index": out / "sample_index.csv",
            "preprocess_log": out / "preprocess_log.csv",
            "meta": out / "meta.json",
        })
        
        for key in ["prep", "mask", "figs", "fig_pipeline", "fig_roi", "valid", "valid_figs"]:
            paths[key].mkdir(parents=True, exist_ok=True)
            
    return paths


# FUNKCJA WYSZUKUJĄCA WSZYSTKIE OBRAZY W FOLDERZE I PODFOLDERACH
def list_images(folder: Path) -> List[Path]:
    exts = (
        "*.bmp",
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.tif",
        "*.tiff",
        "*.webp",
    )  # Obsługiwane rozszerzenia obrazów
    files: List[Path] = []  # Lista plików obrazów
    if not folder.exists():  # Jeśli folder nie istnieje, zwróć pustą listę
        return files
    for (
        ext
    ) in exts:  # Dla każdego rozszerzenia, wyszukaj pliki w folderze i podfolderach
        files.extend(folder.rglob(ext))  # Dodaj znalezione pliki do listy
    return sorted(files)  # Zwróć posortowaną listę plików (alfabetycznie)


def _collect_dataset_samples(folder: Path, source: str) -> List[SampleInfo]:
    """Treat flat files as distinct identities and subfolders as labeled identities."""
    samples: List[SampleInfo] = []
    if not folder.exists():
        return samples

    image_extensions = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
    flat_images = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    )
    samples.extend(
        SampleInfo(image, f"{source}__{image.name}", source)
        for image in flat_images
    )

    for person_dir in sorted(
        (path for path in folder.iterdir() if path.is_dir()),
        key=lambda path: path.name.lower(),
    ):
        samples.extend(
            SampleInfo(image, person_dir.name, source)
            for image in list_images(person_dir)
        )

    return samples


# FUNKCJA TWORZĄCA BEZPIECZNĄ NAZWĘ PLIKU NA PODSTAWIE ŹRÓDŁA, TOŻSAMOŚCI I ŚCIEŻKI PLIKU
def make_safe_name(source: str, identity: str, file_path: Path) -> str:
    return f"{source}__{identity.replace(' ', '_')}__{file_path.stem.replace(' ', '_')}"


# FUNKCJA ZBIERAJĄCA WSZYSTKIE PRÓBKI OBRAZÓW (DO NAUKI MODELU)
def collect_samples(paths: Dict[str, Path], user_name: str, profile_only: bool = False) -> List[SampleInfo]:
    samples: List[SampleInfo] = []
    for f in list_images(
        paths["profiles"] / user_name
    ):  # Przegląda wszystkie obrazy w katalogu użytkownika
        samples.append(
            SampleInfo(f, user_name, "profile")
        )  # Dodaje próbkę do listy z informacjami o źródle i tożsamości
    
    if profile_only:
        return samples  # W trybie profilu zwracamy tylko zdjęcia użytkownika, pomijając FairFace

    if paths["ff_train"].exists() and not paths["ff_train_csv"].exists():  # Jeśli katalog ff_train istnieje i nie ma jeszcze pliku csv, zbiera próbki impostorów (do dobierania progu)
        samples.extend(_collect_dataset_samples(paths["ff_train"], "ff_train"))

    if paths["ff_test"].exists() and not paths["ff_test_csv"].exists():  # Jeśli katalog ff_test istnieje i nie ma jeszcze pliku csv, zbiera próbki impostorów (do testowania)
        samples.extend(_collect_dataset_samples(paths["ff_test"], "ff_test"))

    return samples  # Zwraca listę wszystkich próbek z metadanymi

# FUNKCJE ZAPISUJĄCE DANE DO PLIKÓW CSV
def _write_csv(path: Path, header: List[str], rows: List[List[object]]) -> None:
    path.parent.mkdir(
        parents=True, exist_ok=True
    )  # Utwórz katalog nadrzędny, jeśli nie istnieje
    last_exc = None
    for _ in range(
        5
    ):  # Próbuje zapisać plik CSV do 5 razy, jeśli wystąpi błąd PermissionError
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                wr = csv.writer(f)
                wr.writerow(header)
                wr.writerows(rows)
            return
        except PermissionError as e:
            last_exc = e
            time.sleep(0.5)
    raise last_exc if last_exc else PermissionError(f"Nie można zapisać {path}")


# FUNKCJE ZAPISUJĄCE DANE DO PLIKÓW CSV
# ZAWIERA INDES PRÓBEK, LOG PREPROCESSINGU, EMBEDDINGI, PODSUMOWANIE DATASETU
def write_sample_index(sample_rows: List[dict], out_csv: Path) -> None:
    rows = [
        [r["sample_name"], r["identity"], r["source"], r["file_path"]]
        for r in sample_rows
    ]
    _write_csv(out_csv, ["sample_name", "identity", "source", "file_path"], rows)


# FUNKCJA ZAPISUJĄCA LOG PREPROCESSINGU DO PLIKU CSV
# ZAWIERA INFORMACJE O PRÓBCE, TOŻSAMOŚCI, ŹRÓDLE, ŚCIEŻCE PLIKU, STATUSIE, DETEKTORZE, WYNIKU DETEKCJI, MASKACH, FIGURACH I BOKSACH TWARZY
def write_preprocess_log(log_rows: List[dict], out_csv: Path) -> None:
    header = [
        "sample_name",
        "identity",
        "source",
        "file_path",
        "status",
        "detector_backend",
        "detector_score",
        "landmarks_found",
        "mask_coverage",
        "prep_file",
        "mask_file",
        "roi_figure",
        "pipeline_figure",
        "face_bbox_x",
        "face_bbox_y",
        "face_bbox_w",
        "face_bbox_h",
    ]
    rows = [[r.get(col, "") for col in header] for r in log_rows]
    _write_csv(out_csv, header, rows)


# FUNKCJA ZAPISUJĄCA EMBEDDINGI DO PLIKU CSV
# ZAWIERA INFORMACJE O PRÓBCE, TOŻSAMOŚCI, ŹRÓDLE, DETEKTORZE, WYNIKU DETEKCJI, MASKACH, FIGURACH I BOKSACH TWARZY ORAZ 512-WYMIAROWYM WEKTORZE EMBEDDINGU
def write_embeddings_csv(records: List[dict], out_csv: Path) -> None:
    if not records:
        return
    dim = len(records[0]["embedding"])
    header = [
        "sample_name",
        "id",
        "source",
        "backend",
        "detector_backend",
        "detector_score",
        "landmarks_found",
        "prep_file",
        "mask_file",
        "roi_figure",
        "pipeline_figure",
    ] + [f"e_{i:03d}" for i in range(dim)]
    rows = []
    for r in records:
        rows.append(
            [
                r["sample_name"],
                r["id"],
                r["source"],
                r["backend"],
                r["detector_backend"],
                r["detector_score"],
                int(r["landmarks_found"]),
                r["prep_file"],
                r["mask_file"],
                r["roi_figure"],
                r["pipeline_figure"],
            ]
            + r["embedding"].tolist()
        )
    _write_csv(out_csv, header, rows)


# FUNKCJA AKTUALIZUJĄCA EMBEDDING CSV
def update_embeddings_csv_for_profile(
    records: List[dict], out_csv: Path, user_name: str
) -> List[List[object]]:
    if not records:
        raise ValueError(f"Brak nowych rekordów profilu użytkownika {user_name}")

    dim = len(records[0]["embedding"])
    header = [
        "sample_name",
        "id",
        "source",
        "backend",
        "detector_backend",
        "detector_score",
        "landmarks_found",
        "prep_file",
        "mask_file",
        "roi_figure",
        "pipeline_figure",
    ] + [f"e_{i:03d}" for i in range(dim)]

    old_rows: List[List[object]] = []
    user_rows: List[dict] = []

    if out_csv.exists():
        with open(out_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                is_same_user_profile = (
                    row.get("source") == "profile" and row.get("id") == user_name
                )
                if is_same_user_profile:
                    # Dodajemy stary rekord (przy autoryzacji dopisujemy nowy wiersz, nie nadpisujemy)
                    user_rows.append(row)
                else:
                    old_rows.append([row.get(col, "") for col in header])

    # Dopisywanie nowych rekordów z autoryzacji
    for r in records:
        new_row = {
            "sample_name": r["sample_name"],
            "id": r["id"],
            "source": r["source"],
            "backend": r["backend"],
            "detector_backend": r["detector_backend"],
            "detector_score": str(r["detector_score"]),
            "landmarks_found": str(int(r["landmarks_found"])),
            "prep_file": r["prep_file"],
            "mask_file": r["mask_file"],
            "roi_figure": r["roi_figure"],
            "pipeline_figure": r["pipeline_figure"],
        }
        emb = r["embedding"].tolist()
        for i in range(dim):
            new_row[f"e_{i:03d}"] = str(emb[i])
        user_rows.append(new_row)

    # Usuwanie duplikatów odpalane przy autoryzacji, gdy liczba danych wychodzi ponad 25
    if len(user_rows) > MIN_DATA_SIZE:
        import numpy as np
        while len(user_rows) > MIN_DATA_SIZE:
            embs = []
            for row in user_rows:
                e = np.array([float(row[f"e_{i:03d}"]) for i in range(dim)], dtype=np.float32)
                embs.append(e)
            embs = np.array(embs)

            # Szukamy najbardziej podobnych do siebie embeddingów
            sims = np.dot(embs, embs.T)
            np.fill_diagonal(sims, -1.0)
            
            i, j = np.unravel_index(np.argmax(sims), sims.shape)
            idx_to_remove = min(i, j) # Z pary usuwamy ten starszy
            user_rows.pop(idx_to_remove)

    final_user_rows = []
    for row_dict in user_rows:
        final_user_rows.append([row_dict.get(col, "") for col in header])

    _write_csv(out_csv, header, old_rows + final_user_rows)
    
    # Zwracamy listę do wykorzystania w extract_profile, by uaktualnić user.csv
    return final_user_rows

# FUNKCJA ZAPISUJĄCA PODSUMOWANIE DATASETU DO PLIKU CSV
# ZAWIERA INFORMACJE O ŹRÓDLE, BACKENDZIE EMBEDDINGU, BACKENDZIE DETEKTORA ORAZ LICZBIE PRÓBEK
def write_dataset_summary(records: List[dict], out_csv: Path) -> None:
    counts = {}
    for r in records:
        key = (r["source"], r["backend"], r["detector_backend"])
        counts[key] = counts.get(key, 0) + 1
    rows = [[src, back, det, cnt] for (src, back, det), cnt in sorted(counts.items())]
    _write_csv(
        out_csv, ["source", "embedding_backend", "detector_backend", "count"], rows
    )


# FUNKCJE ZAPISUJĄCE METADANE DO PLIKU JSON
# ZAWIERA INFORMACJE O PARAMETRACH, ŚCIEŻKACH, LICZBIE PRÓBEK, LICZBIE PRÓBEK Z WYKRYTYMI TWARZAMI, LICZBIE PRÓBEK Z WYKRYTYMI LANDMARKAMI ORAZ LICZBIE PRÓBEK Z WYKRYTYMI MASKAMI
def _lazy_import_yolo():
    global _YOLO_AVAILABLE
    if _YOLO_AVAILABLE is None:
        try:
            from ultralytics import YOLO  # type: ignore

            _YOLO_AVAILABLE = YOLO
        except Exception:
            _YOLO_AVAILABLE = False
    return _YOLO_AVAILABLE


# FUNKCJA ZWRACAJĄCA MODEL YOLO DO DETEKCJI TWARZY
def get_yolo_model(weights_path: Optional[Path]):
    global _YOLO_MODEL  # Używa globalnej zmiennej _YOLO_MODEL do przechowywania instancji modelu YOLO
    if _YOLO_MODEL is not None:  # Jeśli model YOLO został już załadowany, zwróć go
        return _YOLO_MODEL
    YOLO = _lazy_import_yolo()  # Jeśli YOLO nie jest dostępny, zwróć None
    if (
        not YOLO or weights_path is None or not Path(weights_path).exists()
    ):  # Jeśli wagi modelu nie istnieją, zwróć None
        return None
    try:  # Jeśli wagi modelu istnieją, załaduj model YOLO z podanej ścieżki wag
        _YOLO_MODEL = YOLO(str(weights_path))
        return _YOLO_MODEL
    except Exception:
        return None


# FUNKCJA ZWRACAJĄCA DETEKTOR TWARZY MEDIA PIPE
def get_mp_face_detector():
    global _MP_FACE_DET  # Używa globalnej zmiennej _MP_FACE_DET do przechowywania instancji detektora twarzy MediaPipe
    if (
        _MP_FACE_DET is None
    ):  # Jeśli detektor twarzy MediaPipe nie został jeszcze załadowany, utwórz nową instancję detektora twarzy MediaPipe z model_selection=1 (dla wykrywania twarzy w różnych orientacjach) i min_detection_confidence=0.5 (minimalna pewność wykrycia twarzy)
        _MP_FACE_DET = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
    return _MP_FACE_DET


# FUNKCJA ZWRACAJĄCA FACEMESH MEDIA PIPE
def get_facemesh(max_faces: int = 1):
    global _FACEMESH  # Używa globalnej zmiennej _FACEMESH do przechowywania instancji FaceMesh MediaPipe
    if (
        _FACEMESH is None
    ):  # Jeśli FaceMesh MediaPipe nie został jeszcze załadowany, utwórz nową instancję FaceMesh MediaPipe z static_image_mode=True (dla przetwarzania pojedynczych obrazów), max_num_faces=max_faces (maksymalna liczba twarzy do wykrycia), refine_landmarks=True (dla dokładniejszego wykrywania punktów charakterystycznych twarzy) i min_detection_confidence=0.5 (minimalna pewność wykrycia twarzy)
        _FACEMESH = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,  # przetwarzanie pojedynczych obrazów
            max_num_faces=max_faces,  # maksymalna liczba twarzy do wykrycia
            refine_landmarks=True,  # dokładniejsze wykrywanie punktów charakterystycznych twarzy (np. kontur ust, oczu)
            min_detection_confidence=0.5,  # minimalna pewność wykrycia twarzy
        )
    return _FACEMESH


# FUNKCJA WYKRYWAJĄCA BOKS TWARZY ZA POMOCĄ YOLO
def detect_face_bbox_yolo(
    img_bgr: np.ndarray, weights_path: Optional[Path]
) -> Optional[Tuple[int, int, int, int, float]]:
    model = get_yolo_model(
        weights_path
    )  # Pobiera model YOLO do detekcji twarzy z podanej ścieżki wag
    if model is None:  # Jeśli model YOLO nie został załadowany, zwróć None
        return None
    try:
        results = model.predict(
            img_bgr, verbose=False
        )  # Przeprowadza predykcję na obrazie BGR za pomocą modelu YOLO, zwracając listę wyników detekcji
        if not results:  # Jeśli lista wyników detekcji jest pusta, zwróć None
            return None
        boxes = results[
            0
        ].boxes  # Pobiera obiekty Box z pierwszego wyniku detekcji (zakładając, że jest tylko jeden wynik)
        if (
            boxes is None or len(boxes) == 0
        ):  # Jeśli nie ma wykrytych boksów, zwróć None
            return None
        xyxy = (
            boxes.xyxy.cpu().numpy()
        )  # Pobiera współrzędne boksów w formacie (x1, y1, x2, y2) i konwertuje je na tablicę NumPy
        conf = (
            boxes.conf.cpu().numpy()
            if boxes.conf is not None
            else np.ones((xyxy.shape[0],), dtype=float)
        )  # Pobiera pewności wykrycia boksów i konwertuje je na tablicę NumPy, jeśli pewności nie są dostępne, tworzy tablicę jedynek
        idx = int(
            np.argmax(conf)
        )  # Wybiera indeks boksu z najwyższą pewnością wykrycia
        x1, y1, x2, y2 = (
            xyxy[idx].astype(int).tolist()
        )  # Pobiera współrzędne boksu z najwyższą pewnością wykrycia i konwertuje je na listę liczb całkowitych
        h, w = img_bgr.shape[:2]  # Pobiera wysokość i szerokość obrazu BGR
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))  # Ogranicza współrzędne boksu do granic obrazu
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(
            y1 + 1, min(y2, h)
        )  # Ogranicza współrzędne boksu do granic obrazu, zapewniając, że x2 > x1 i y2 > y1
        return (
            x1,
            y1,
            x2 - x1,
            y2 - y1,
            float(conf[idx]),
        )  # Zwraca współrzędne boksu w formacie (x, y, width, height) oraz pewność wykrycia
    except Exception:
        return None


# FUNKCJA WYKRYWAJĄCA BOKS TWARZY ZA POMOCĄ MEDIA PIPE
def detect_face_bbox_mediapipe(
    img_bgr: np.ndarray, fpar: FaceParams
) -> Optional[Tuple[int, int, int, int, float]]:
    det = get_mp_face_detector()  # Pobiera detektor twarzy MediaPipe
    h, w = img_bgr.shape[:2]  # Pobiera wysokość i szerokość obrazu BGR
    res = det.process(
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    )  # Przetwarza obraz BGR na RGB i przeprowadza detekcję twarzy za pomocą detektora MediaPipe
    if not res.detections:  # Jeśli nie wykryto żadnych twarzy, zwróć None
        return None
    best = max(
        res.detections, key=lambda d: float(d.score[0])
    )  # Wybiera detekcję z najwyższą pewnością wykrycia
    bbox = (
        best.location_data.relative_bounding_box
    )  # Pobiera współrzędne boksu twarzy w formacie (xmin, ymin, width, height) z detekcji MediaPipe
    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    bw = int(bbox.width * w)
    bh = int(
        bbox.height * h
    )  # Konwertuje współrzędne boksu twarzy z formatu względnego (0-1) na piksele, uwzględniając szerokość i wysokość obrazu
    mx = int(bw * fpar.margin_frac)
    my = int(
        bh * fpar.margin_frac
    )  # Oblicza margines wokół boksu twarzy na podstawie szerokości i wysokości boksu oraz parametru margin_frac
    x0 = max(0, x - mx)
    y0 = max(
        0, y - my
    )  # Oblicza współrzędne lewego górnego rogu boksu twarzy z marginesem, ograniczając je do granic obrazu
    x1 = min(w, x + bw + mx)
    y1 = min(
        h, y + bh + my
    )  # Oblicza współrzędne prawego dolnego rogu boksu twarzy z marginesem, ograniczając je do granic obrazu
    if (
        x1 <= x0 or y1 <= y0
    ):  # Jeśli współrzędne boksu twarzy są niepoprawne (x1 <= x0 lub y1 <= y0), zwróć None
        return None
    return (
        x0,
        y0,
        x1 - x0,
        y1 - y0,
        float(best.score[0]),
    )  # Zwraca współrzędne boksu twarzy w formacie (x, y, width, height) oraz pewność wykrycia


# FUNKCJA WYKRYWAJĄCA BOKS TWARZY ZA POMOCĄ DETEKTORÓW
def detect_face_bbox(
    img_bgr: np.ndarray, paths: Dict[str, Path], fpar: FaceParams
) -> Tuple[Tuple[int, int, int, int], str, float]:
    # Próba 1: YOLO
    yolo_det = detect_face_bbox_yolo(img_bgr, paths["weights"] / "yolo_face.pt")
    if yolo_det is not None:
        x, y, w, h, score = yolo_det
        return (x, y, w, h), "yolo", score
    # Próba 2: MediaPipe (jeśli YOLO nie zadziała)
    mp_det = detect_face_bbox_mediapipe(img_bgr, fpar)
    if mp_det is not None:
        x, y, w, h, score = mp_det
        return (x, y, w, h), "mediapipe", score
    # Próba 3: Fallback fixed (jeśli żaden detektor nie zadziała)
    H, W = img_bgr.shape[:2]
    side = int(0.7 * min(H, W))
    x0 = (W - side) // 2
    y0 = (H - side) // 2
    return (x0, y0, side, side), "fixed", 0.0


# FUNKCJA PRZETWARZAJĄCA OBRAZ SZAROŚCI: Gaussian blur, Median filter i CLAHE
def preprocess_gray(gray: np.ndarray, settings: Settings) -> np.ndarray:
    processed = gray

    if settings.use_gaussian_blur:
        k = max(3, int(2 * round(settings.gauss_sigma * 2) + 1))
        processed = cv2.GaussianBlur(processed, (k, k), settings.gauss_sigma)

    if settings.median_k >= 3 and settings.median_k % 2 == 1:
        if settings.use_median_filter:
            processed = cv2.medianBlur(processed, settings.median_k)

    if settings.use_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=settings.clahe_clip,
            tileGridSize=(settings.clahe_grid, settings.clahe_grid),
        )
        processed = clahe.apply(processed)

    return processed


# FUNKCJA WYKRYWAJĄCA LANDMARKI TWARZY ZA POMOCĄ MEDIA PIPE
def detect_landmarks_mediapipe(gray: np.ndarray):
    bgr = cv2.cvtColor(
        gray, cv2.COLOR_GRAY2BGR
    )  # Konwertuje obraz szarości na BGR, ponieważ MediaPipe FaceMesh wymaga obrazu w formacie BGR lub RGB
    fm = get_facemesh(
        max_faces=1
    )  # Pobiera instancję FaceMesh MediaPipe z maksymalną liczbą twarzy do wykrycia ustawioną na 1
    res = fm.process(
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    )  # Przetwarza obraz BGR na RGB i przeprowadza detekcję punktów charakterystycznych twarzy za pomocą FaceMesh MediaPipe
    if not res.multi_face_landmarks:  # Jeśli nie wykryto żadnych twarzy, zwróć None
        return None
    h, w = gray.shape[:2]  # Pobiera wysokość i szerokość obrazu szarości
    pts = (
        []
    )  # Tworzy pustą listę do przechowywania współrzędnych punktów charakterystycznych twarzy w pikselach
    for p in res.multi_face_landmarks[
        0
    ].landmark:  # Iteruje po wszystkich 468 punktach charakterystycznych twarzy wykrytych przez FaceMesh MediaPipe dla pierwszej twarzy
        pts.append((int(p.x * w), int(p.y * h)))
    return pts  # Zwraca listę 468 tupli (x, y) w pikselach reprezentujących współrzędne punktów charakterystycznych twarzy


# FUNKCJA WYODRĘBNIAJĄCA ŚRODKI OCZU Z LANDMARKÓW MEDIA PIPE
def pick_eye_centers_from_lm(pts):
    # https://pishi.ai/help/facemesh-detect-faces-and-expressions-in-scratch/
    left_idx = [
        33,
        133,
        159,
        145,
    ]  # Indeksy punktów charakterystycznych twarzy dla lewego oka (MediaPipe FaceMesh definiuje 468 punktów charakterystycznych twarzy, a te indeksy odpowiadają punktom wokół lewego oka)
    right_idx = [
        263,
        362,
        386,
        374,
    ]  # Indeksy punktów charakterystycznych twarzy dla prawego oka
    lx = np.mean([pts[i][0] for i in left_idx])
    ly = np.mean(
        [pts[i][1] for i in left_idx]
    )  # Oblicza średnią współrzędną x i y dla lewego oka, biorąc pod uwagę punkty charakterystyczne twarzy zdefiniowane w left_idx
    rx = np.mean([pts[i][0] for i in right_idx])
    ry = np.mean([pts[i][1] for i in right_idx])  # To samo dla prawego oka
    return (int(lx), int(ly)), (
        int(rx),
        int(ry),
    )  # Zwraca tuplę z dwoma tuplami reprezentującymi współrzędne (x, y) środków lewego i prawego oka w pikselach


# FUNKCJA WYRÓWNANIA TWARZY NA PODSTAWIE POZYCJI OCZU
def align_face_by_eyes(
    gray: np.ndarray,
    left_eye: tuple,
    right_eye: tuple,
    out_size=(112, 112),
    eye_y=0.35,
    eye_dist_frac=0.33,
):
    # Oblicz kąt obrotu między oczami i skalę
    (lx, ly), (rx, ry) = (
        left_eye,
        right_eye,
    )  # Pobiera współrzędne (x, y) lewego i prawego oka
    dx, dy = rx - lx, ry - ly  # Oblicza różnicę współrzędnych x i y między oczami
    angle = np.degrees(
        np.arctan2(dy, dx)
    )  # Oblicza kąt obrotu w stopniach między oczami za pomocą funkcji arctan2, która zwraca kąt w radianach, a następnie konwertuje go na stopnie
    dist = np.hypot(
        dx, dy
    )  # Oblicza odległość między oczami za pomocą funkcji hypot, która zwraca pierwiastek kwadratowy z sumy kwadratów różnic współrzędnych x i y
    target_dist = (
        out_size[0] * eye_dist_frac
    )  # Oblicza docelową odległość między oczami w pikselach na podstawie szerokości wyjściowego obrazu i parametru eye_dist_frac
    scale = target_dist / (
        dist + 1e-6
    )  # Oblicza skalę, aby dopasować odległość między oczami do docelowej odległości, dodając małą wartość (1e-6) do odległości, aby uniknąć dzielenia przez zero
    # Macierz rotacji + skalowania
    cx, cy = (lx + rx) / 2.0, (
        ly + ry
    ) / 2.0  # Oblicza środek między oczami jako punkt obrotu
    Mrot = cv2.getRotationMatrix2D(
        (cx, cy), angle, scale
    )  # Tworzy macierz transformacji afine (rotacja + skalowanie) za pomocą funkcji getRotationMatrix2D, która przyjmuje punkt obrotu, kąt obrotu i skalę
    warped = cv2.warpAffine(
        gray, Mrot, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_CUBIC
    )  # Przeprowadza transformację afine (rotacja + skalowanie) na obrazie szarości za pomocą funkcji warpAffine, która przyjmuje obraz wejściowy, macierz transformacji, rozmiar wyjściowego obrazu i flagę interpolacji (INTER_CUBIC dla lepszej jakości)

    # Wylicz pozycję oczu po rotacji
    def warp_pt(x, y):
        X = (
            Mrot[0, 0] * x + Mrot[0, 1] * y + Mrot[0, 2]
        )  # Oblicza nową współrzędną x po transformacji afine (rotacja + skalowanie) za pomocą macierzy Mrot
        Y = (
            Mrot[1, 0] * x + Mrot[1, 1] * y + Mrot[1, 2]
        )  # Oblicza nową współrzędną y po transformacji afine (rotacja + skalowanie) za pomocą macierzy Mrot
        return X, Y

    # Wylicz pozycję oczu po rotacji
    lx2, ly2 = warp_pt(lx, ly)
    rx2, ry2 = warp_pt(
        rx, ry
    )  # Oblicza nowe współrzędne (x, y) lewego i prawego oka po transformacji afine (rotacja + skalowanie) za pomocą funkcji warp_pt
    out_w, out_h = out_size  # Pobiera szerokość i wysokość wyjściowego obrazu
    eye_y_px = (
        eye_y * out_h
    )  # Oblicza docelową współrzędną y oczu w pikselach na podstawie wysokości wyjściowego obrazu i parametru eye_y
    center_x = (lx2 + rx2) / 2.0
    center_y = (
        ly2 + ry2
    ) / 2.0  # Oblicza środek między oczami po transformacji afine (rotacja + skalowanie) jako punkt odniesienia do translacji
    # Macierz translacji - przesuń oczy do standardowej pozycji
    tx = out_w / 2.0 - center_x
    ty = (
        eye_y_px - center_y
    )  # Oblicza przesunięcie w pikselach w kierunku x i y, aby umieścić środek między oczami w docelowej pozycji (środek obrazu i docelowa współrzędna y oczu)
    Mtrans = np.array(
        [[1, 0, tx], [0, 1, ty]], dtype=np.float32
    )  # Tworzy macierz translacji (przesunięcie) w formacie 2x3, która przesuwa obraz o tx w kierunku x i ty w kierunku y
    return cv2.warpAffine(
        warped, Mtrans, (out_w, out_h), flags=cv2.INTER_CUBIC
    )  # Przeprowadza transformację afine (translacja) na obrazie po rotacji i skalowaniu za pomocą funkcji warpAffine, która przyjmuje obraz wejściowy, macierz translacji, rozmiar wyjściowego obrazu i flagę interpolacji (INTER_CUBIC dla lepszej jakości)


# FUNKCJA CZYSZCZĄCA KRAWĘDZIE OBRAZU
def _clean_edges(edges: np.ndarray, min_area: int) -> np.ndarray:
    if (
        edges.dtype != np.uint8
    ):  # Konwertuje obraz krawędzi na typ uint8, jeśli nie jest już w tym formacie
        edges = edges.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        edges, connectivity=8
    )  # Wykrywa połączone komponenty w obrazie krawędzi za pomocą funkcji connectedComponentsWithStats, która zwraca liczbę etykiet, mapę etykiet, statystyki (w tym obszar) i centroidy
    out = (
        edges.copy()
    )  # Tworzy kopię obrazu krawędzi, aby zachować oryginalne krawędzie
    for lbl in range(
        1, num_labels
    ):  # Iteruje po wszystkich etykietach (pomijając etykietę 0, która reprezentuje tło)
        if (
            stats[lbl, cv2.CC_STAT_AREA] < min_area
        ):  # Jeśli obszar komponentu jest mniejszy niż minimalny obszar, usuwa go z obrazu krawędzi
            out[labels == lbl] = (
                0  # Ustawia piksele należące do komponentu o etykiecie lbl na 0 (czarne), aby usunąć małe komponenty z obrazu krawędzi
            )
    return out


# FUNKCJA WYLICZAJĄCA CECHY GEOMETRYCZNE Z LANDMARKÓW
def landmark_geometry_features(pts, image_shape):
    # Normalizacja landmarków do zakresu [0, 1]
    pts_np = np.array(pts, dtype=np.float32)
    # Normalizacja współrzędnych landmarków do zakresu [0, 1] względem rozmiarów obrazu
    h, w = image_shape[:2]  # Pobiera wysokość i szerokość obrazu
    pts_norm = np.stack([pts_np[:, 0] / max(1, w), pts_np[:, 1] / max(1, h)], axis=1)
    # Definicja punktów (MediaPipe FaceMesh)
    ids = {
        "l_eye": 33,
        "r_eye": 263,
        "nose_tip": 1,
        "mouth_l": 61,
        "mouth_r": 291,
        "chin": 199,
        "brow_l": 70,
        "brow_r": 300,
    }

    # Pobieranie znormalizowanych współrzędnych punktu charakterystycznego twarzy na podstawie indeksu
    def p(idx):
        return pts_norm[idx]

    # Wyliczanie cech geometrycznych
    feats = []
    # Dystanse i kąty między istotnymi punktami
    pairs = [
        ("l_eye", "r_eye"),
        ("l_eye", "nose_tip"),
        ("r_eye", "nose_tip"),
        ("mouth_l", "mouth_r"),
        ("nose_tip", "chin"),
    ]
    # Obliczanie dystansów i kątów między parami punktów charakterystycznych twarzy
    for a, b in pairs:
        va, vb = p(ids[a]), p(ids[b])
        feats += [
            np.linalg.norm(vb - va),
            (np.degrees(np.arctan2(vb[1] - va[1], vb[0] - va[0])) % 180.0),
        ]
    # Wyliczanie pozycji punktów charakterystycznych twarzy względem środka oczu
    le, re = p(ids["l_eye"]), p(ids["r_eye"])
    # Obliczanie środka między oczami
    mid = (le + re) / 2.0
    # Obliczanie wektorów od środka oczu do istotnych punktów charakterystycznych twarzy i dodawanie ich współrzędnych x, y oraz normy do cech
    for k in ["mouth_l", "mouth_r", "nose_tip", "chin", "brow_l", "brow_r"]:
        v = p(ids[k]) - mid
        feats += [v[0], v[1], np.linalg.norm(v)]

    # Konwersja do float32 i dopasowanie do 64 cech (padding lub przycinanie)
    feats = np.array(feats, dtype=np.float32)
    # Dopasowanie do 64 cech (padding lub przycinanie)
    if feats.size < 64:
        feats = np.pad(feats, (0, 64 - feats.size))
    else:  # Przycinanie do 64 cech, jeśli jest ich więcej
        feats = feats[:64]
    return feats


# FUNKCJA ŁADUJĄCA SESJĘ ARCFACE ONNX (tryb lazy loading, ponieważ model ArcFace jest duży a nie zawsze potrzebny np. przy detekcji twarzy bez ekstrakcji embeddingów)
def _lazy_arcface_session(weights_path: Path):
    # Używa globalnych zmiennych do przechowywania sesji ArcFace ONNX, nazwy wejścia i dostępności modelu
    global _ARCFACE_SESSION, _ARCFACE_INPUT_NAME, _ARCFACE_AVAILABLE
    # Jeśli ArcFace nie jest dostępny, zwróć None
    if _ARCFACE_AVAILABLE is False:
        return None
    # Jeśli sesja ArcFace już istnieje, zwróć ją
    if _ARCFACE_SESSION is not None:
        return _ARCFACE_SESSION
    try:  # Jeśli sesja ArcFace nie istnieje, spróbuj ją załadować
        import onnxruntime as ort  # type: ignore

        if not weights_path.exists():
            _ARCFACE_AVAILABLE = False
            return None
        # Tworzy sesję ONNX Runtime dla modelu ArcFace z podanej ścieżki wag, używając CPU jako dostawcy wykonania
        sess = ort.InferenceSession(
            str(weights_path), providers=["CPUExecutionProvider"]
        )
        _ARCFACE_SESSION = (
            sess  # Przechowuje sesję ArcFace ONNX w globalnej zmiennej _ARCFACE_SESSION
        )
        _ARCFACE_INPUT_NAME = sess.get_inputs()[
            0
        ].name  # Pobiera nazwę wejścia modelu ArcFace ONNX i przechowuje ją w globalnej zmiennej _ARCFACE_INPUT_NAME
        _ARCFACE_AVAILABLE = True  # Ustawia flagę dostępności modelu ArcFace na True
        return sess  # Zwraca sesję ArcFace ONNX
    except Exception:
        _ARCFACE_AVAILABLE = False  # Ustawia flagę dostępności modelu ArcFace na False, jeśli wystąpił błąd podczas ładowania sesji ONNX
        return None  # Zwraca None, jeśli wystąpił błąd podczas ładowania sesji ONNX


# FUNKCJA WYLICZAJĄCA EMBEDDING ARCFACE Z WYRÓWNANEGO OBRAZU SZAROŚCI
def arcface_embedding(
    aligned_image: np.ndarray, weights_path: Path
) -> Optional[np.ndarray]:
    # Lazy load ArcFace ONNX session
    sess = _lazy_arcface_session(
        weights_path
    )  # weights_path to ścieżka do pliku arcface.onnx, który zawiera wytrenowany model ArcFace w formacie ONNX.
    # Jeśli sesja ArcFace ONNX nie jest dostępna, zwróć None
    if sess is None:
        return None
    # Pobierz kształt wejścia modelu ArcFace ONNX
    input_shape = sess.get_inputs()[0].shape
    # Przygotuj wejście: RGB, resample do 112x112, normalizuj
    if aligned_image.ndim == 2:
        img = cv2.cvtColor(aligned_image, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA).astype(
        np.float32
    )  # Resample obrazu do rozmiaru 112x112 pikseli za pomocą interpolacji AREA, a następnie konwertuje go na typ float32
    img = (img / 127.5) - 1.0  # Normalizacja do [-1, 1]
    if (
        len(input_shape) == 4 and input_shape[1] == 3
    ):  # Jeśli model ArcFace ONNX oczekuje wejścia w formacie NCHW (batch_size, channels, height, width), przekształca obraz do tego formatu, kiedy liczba kanałów wynosi 3
        img = np.transpose(img, (2, 0, 1))[None, ...]  # NCHW format
    else:  # Jeśli model ArcFace ONNX oczekuje wejścia w formacie NHWC (batch_size, height, width, channels), przekształca obraz do tego formatu
        img = img[None, ...]  # NHWC format
    # Uruchom inference
    out = sess.run(None, {_ARCFACE_INPUT_NAME: img.astype(np.float32)})[0]
    emb = out[0].astype(np.float32)
    # Normalizuj (L2) - gotowe do cosine similarity, None jeśli embedding jest zerowy (np. model ArcFace nie zadziałał)
    norm = np.linalg.norm(emb) + 1e-8
    if norm == 0:
        return None
    return emb / norm


# FUNKCJA WYLICZAJĄCA EMBEDDING Z CECH HANDCRAFTED (LBP + GEOMETRIA LANDMARKÓW)
def classic_plus_landmarks_embedding(
    img_gray: np.ndarray, settings: Settings
) -> np.ndarray:
    # Detekcja krawędzi (Canny) i czyszczenie małych komponentów
    edges = cv2.Canny(img_gray, settings.canny_lo, settings.canny_hi)
    # Czyszczenie małych komponentów (usuwanie szumów)
    edges = _clean_edges(edges, settings.min_cc_area)
    # Obliczenie LBP histogramu
    lbp = local_binary_pattern(
        img_gray, P=settings.lbp_p, R=settings.lbp_r, method="uniform"
    )
    # Histogram LBP (z normalizacją)
    lbp_hist = np.histogram(
        lbp, bins=settings.lbp_p + 2, range=(0, settings.lbp_p + 2)
    )[0].astype(np.float32)
    # Normalizacja histogramu LBP
    lbp_hist /= lbp_hist.sum() + 1e-8
    # Detekcja landmarków MediaPipe i wyliczenie cech geometrycznych
    pts = detect_landmarks_mediapipe(img_gray)

    # Wyliczenie cech geometrycznych z landmarków (dystanse, kąty, pozycje względem oczu)
    if pts is not None:
        geo = landmark_geometry_features(pts, img_gray.shape)
    else:  # Jeśli nie wykryto landmarków, użyj wektora zerowego
        geo = np.zeros(64, dtype=np.float32)

    # Statystyki skalarnych: średnia (mean), odchylenie standardowe (std) i stosunek krawędzi do całego obrazu (edge ratio)
    scalar = np.array(
        [
            float((edges > 0).mean()),
            float(np.mean(img_gray) / 255.0),
            float(np.std(img_gray) / 255.0),
        ],
        dtype=np.float32,
    )
    vec = np.concatenate([lbp_hist, scalar, geo]).astype(np.float32)
    return vec / (
        np.linalg.norm(vec) + 1e-8
    )  # Znormalizowany wektor cech (L2) gotowy do porównań (np. cosine similarity)


# FUNKCJA WYLICZAJĄCA EMBEDDING Z MODELI (ArcFace, classic_plus_landmarks)
def embed_aligned_face(
    aligned_image: np.ndarray, settings: Settings, paths: Dict[str, Path]
) -> Tuple[np.ndarray, str]:
    # Próba 1: ArcFace ONNX
    emb = arcface_embedding(aligned_image, paths["weights"] / "arcface.onnx")
    try:
        return emb.astype(np.float32), "arcface"
    except Exception as e:
        print(f"Błąd ekstrakcji cech: {e}")
        return None
        
    # ZAKOMENTOWANE - Zgodnie z normami biometrycznymi: błąd ekstrakcji to Failure to Enroll (FTE)
    # if emb is not None:
    #     return emb.astype(np.float32), "arcface"
    # # Próba 2: Fallback - handcrafted features (LBP + edge + geometria landmarków)
    # return (
    #     classic_plus_landmarks_embedding(aligned_gray, settings).astype(np.float32),
    #     "classic_plus_landmarks",
    # )

# FUNKCJA TWORZĄCA NAKŁADKĘ ROI NA OBRAZIE ORYGINALNYM
def _make_roi_overlay(
    original_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    detector_backend: str,
    detector_score: float,
) -> np.ndarray:
    img = (
        original_bgr.copy()
    )  # Tworzy kopię oryginalnego obrazu BGR, aby nie modyfikować go bezpośrednio
    x, y, w, h = (
        bbox  # Pobiera współrzędne boksu twarzy (x, y, width, height) z wykrytego boksu
    )
    cv2.rectangle(
        img, (x, y), (x + w, y + h), (0, 255, 0), 2
    )  # Rysuje prostokąt wokół wykrytego boksu twarzy na obrazie BGR w kolorze zielonym (0, 255, 0) o grubości 2 pikseli
    txt = f"{detector_backend}: {detector_score:.3f}"  # Tworzy napis z nazwą detektora i jego pewnością wykrycia w formacie "detector_backend: score"
    cv2.putText(
        img,
        txt,
        (x, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )  # Dodaje napis z nazwą detektora i jego pewnością wykrycia nad prostokątem boksu twarzy na obrazie BGR w kolorze zielonym (0, 255, 0) o grubości 2 pikseli i antyaliasingiem
    return img


# FUNKCJA TWORZĄCA FIGURĘ Z KROKAMI PRZETWARZANIA OBRAZU
def _build_pipeline_figure(
    original_bgr: np.ndarray,
    crop_gray: np.ndarray,
    aligned_gray: np.ndarray,
    mask_roi: np.ndarray,
) -> np.ndarray:
    # Tworzy figurę z krokami przetwarzania obrazu: oryginalny obraz, wycięty ROI, wyrównany obraz i maska ROI
    a = cv2.resize(original_bgr, (240, 240), interpolation=cv2.INTER_AREA)
    b = cv2.cvtColor(cv2.resize(crop_gray, (240, 240)), cv2.COLOR_GRAY2BGR)
    c = cv2.cvtColor(cv2.resize(aligned_gray, (240, 240)), cv2.COLOR_GRAY2BGR)
    d = cv2.cvtColor(cv2.resize(mask_roi, (240, 240)), cv2.COLOR_GRAY2BGR)

    # Dodaje podpisy do każdego obrazu w figurze
    cv2.putText(
        a,
        "oryginal",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        b,
        "crop ROI",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        c,
        "aligned",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        d,
        "maska",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Tworzy siatkę 2x2 z obrazów a, b, c, d i zwraca wynikowy obraz
    top = np.hstack([a, b])
    bottom = np.hstack([c, d])
    return np.vstack([top, bottom])


# FUNKCJA PRZETWARZAJĄCA JEDEN OBRAZ: Pełna kaskada od wczytania do embeddingu
def preprocess_one(
    file: Path,
    settings: Settings,
    paths: Dict[str, Path],
    fpar: FaceParams,
    store_outputs: bool = True,
    sample_name: Optional[str] = None,
    data_root: Optional[Path | str] = None,
) -> dict:
    result = {
        "sample_name": sample_name,
        "status": "error",
        "detector_backend": "none",
        "detector_score": 0.0,
        "landmarks_found": False,
        "mask_coverage": 0.0,
        "prep_file": "",
        "mask_file": "",
        "roi_figure": "",
        "pipeline_figure": "",
        "face_bbox_x": 0,
        "face_bbox_y": 0,
        "face_bbox_w": 0,
        "face_bbox_h": 0,
        "embedding": None,
        "backend": "none",
        "edges": None,
        "aligned_gray": None,
    }
    # Wczytaj obraz i przetwórz go krok po kroku
    try:
        base_dir = Path(__file__).resolve().parents[1]
        if paths is None:
            resolved_data_root = data_root
            if resolved_data_root is None:
                resolved_data_root = getattr(settings, "db_name", None)
            if resolved_data_root is None:
                step = getattr(settings, "step", None)
                if step == "test":
                    resolved_data_root = TEST_DATA_DIR
                else:
                    resolved_data_root = DATA_DIR
                    
            paths = build_paths(base_dir, resolved_data_root, store_outputs)
        img_bgr = cv2.imread(str(file), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError("Nie można wczytać obrazu.")

        bbox, detector_backend, detector_score = detect_face_bbox(
            img_bgr, paths, fpar
        )  # Wykrywa boks twarzy w obrazie za pomocą detektorów (YOLO, MediaPipe lub fallback fixed)
        x, y, w, h = (
            bbox  # Wyciąga współrzędne boksu twarzy (x, y, width, height) z wykrytego boksu
        )
        result.update(
            {
                "detector_backend": detector_backend,
                "detector_score": detector_score,
                "face_bbox_x": x,
                "face_bbox_y": y,
                "face_bbox_w": w,
                "face_bbox_h": h,
            }
        )
        # Wytnij ROI twarzy i przetwórz obraz szarości
        roi_bgr = img_bgr[y : y + h, x : x + w]
        # Przetwarzanie obrazu szarości: Gaussian blur, Median filter i CLAHE
        roi_gray = preprocess_gray(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY), settings)
        # Wykryj landmarki MediaPipe i wyciągnij środki oczu
        pts = detect_landmarks_mediapipe(roi_gray)
        # Jeśli wykryto landmarki, wyciągnij środki oczu i wyrównaj twarz; w przeciwnym razie przeskaluj ROI do docelowego rozmiaru
        eyes = pick_eye_centers_from_lm(pts) if pts is not None else None
        # Wyrównanie twarzy na podstawie pozycji oczu lub przeskalowanie ROI do docelowego rozmiaru
        if eyes is not None:
            aligned = align_face_by_eyes(
                roi_gray,
                eyes[0],
                eyes[1],
                out_size=(settings.target_w, settings.target_h),
                eye_y=0.35,
                eye_dist_frac=settings.eye_dist_frac,
            )
            aligned_color = align_face_by_eyes(
                roi_bgr,
                eyes[0],
                eyes[1],
                out_size=(settings.target_w, settings.target_h),
                eye_y=0.35,
                eye_dist_frac=settings.eye_dist_frac,
            )
            pts_aligned = detect_landmarks_mediapipe(aligned)
        else:
            aligned = cv2.resize(
                roi_gray,
                (settings.target_w, settings.target_h),
                interpolation=cv2.INTER_AREA,
            )
            aligned_color = cv2.resize(
                roi_bgr,
                (settings.target_w, settings.target_h),
                interpolation=cv2.INTER_AREA,
            )
            pts_aligned = None

        # Tworzenie maski ROI na podstawie wykrytych landmarków MediaPipe
        result["landmarks_found"] = pts_aligned is not None
        if (
            pts_aligned is not None and len(pts_aligned) >= 400
        ):  # Jeśli wykryto wystarczającą liczbę landmarków, utwórz maskę ROI na podstawie konturu twarzy
            oval = (
                mp.solutions.face_mesh.FACEMESH_FACE_OVAL
            )  # Pobiera indeksy punktów charakterystycznych twarzy, które tworzą kontur twarzy (MediaPipe FaceMesh)
            idxs = (
                set()
            )  # Tworzy pusty zbiór do przechowywania unikalnych indeksów punktów charakterystycznych twarzy, które tworzą kontur twarzy
            # Iteruje po wszystkich parach punktów charakterystycznych twarzy w konturze twarzy i dodaje ich indeksy do zbioru idxs
            for (
                a,
                b,
            ) in (
                oval
            ):  # Pobiera pary punktów charakterystycznych twarzy, które tworzą kontur twarzy (MediaPipe FaceMesh)
                idxs.update((int(a), int(b)))
            poly = np.array([pts_aligned[i] for i in sorted(idxs)], dtype=np.int32)
            if (
                poly.shape[0] >= 3
            ):  # Jeśli kontur twarzy ma co najmniej 3 punkty, utwórz maskę ROI na podstawie konturu twarzy
                hull = cv2.convexHull(poly)
                mask_roi = np.zeros(
                    (settings.target_h, settings.target_w), dtype=np.uint8
                )
                cv2.fillConvexPoly(mask_roi, hull, 255)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                mask_roi = cv2.erode(mask_roi, kernel, iterations=1)
            else:  # Jeśli kontur twarzy ma mniej niż 3 punkty, utwórz pełną maskę ROI (wszystkie piksele ustawione na 255)
                mask_roi = np.full(
                    (settings.target_h, settings.target_w), 255, np.uint8
                )
        else:  # Jeśli nie wykryto wystarczającej liczby landmarków, utwórz pełną maskę ROI (wszystkie piksele ustawione na 255)
            mask_roi = np.full((settings.target_h, settings.target_w), 255, np.uint8)
        # Oblicz pokrycie maski ROI (procent pikseli niezerowych w masce)
        result["mask_coverage"] = float((mask_roi > 0).mean())        
        
        # Warunkowy zapis plików
        if store_outputs:
            prep_path = paths["prep"] / f"{sample_name}_prep.png"
            mask_path = paths["mask"] / f"{sample_name}_mask.png"
            cv2.imwrite(str(prep_path), aligned)
            cv2.imwrite(str(mask_path), mask_roi)
            result["prep_file"] = str(prep_path)
            result["mask_file"] = str(mask_path)
            
            roi_fig = paths["fig_roi"] / f"{sample_name}_roi.png"
            pipeline_fig = paths["fig_pipeline"] / f"{sample_name}_pipeline.png"
            cv2.imwrite(str(roi_fig), _make_roi_overlay(img_bgr, bbox, detector_backend, detector_score))
            cv2.imwrite(str(pipeline_fig), _build_pipeline_figure(img_bgr, roi_gray, aligned, mask_roi))
            result["roi_figure"] = str(roi_fig)
            result["pipeline_figure"] = str(pipeline_fig)

        # Wylicz embedding z wyrównanego obrazu i maski ROI
        masked = aligned.copy()
        if settings.use_face_mask:
            masked[mask_roi == 0] = 0
        emb, backend = embed_aligned_face(aligned_color, settings, paths)
        edges = cv2.Canny(masked, settings.canny_lo, settings.canny_hi)
        edges = _clean_edges(edges, settings.min_cc_area)

        # Zaktualizuj wynik z informacjami o embeddingu, backendzie i krawędziach
        result["backend"] = backend
        result["embedding"] = emb.astype(np.float32)
        result["edges"] = (edges > 0).astype(np.uint8)
        result["aligned_gray"] = masked
        result["status"] = "ok"
        return result
    except (
        Exception
    ) as e:  # W przypadku wystąpienia błędu podczas przetwarzania obrazu, wyświetl komunikat o błędzie i zaktualizuj wynik z informacjami o błędzie
        print(f"Pomiń {sample_name}: {e}")
        result["status"] = f"error: {e}"
        return result


# FUNKCJA ZAPISUJĄCA WALIDACJĘ EKSTRAKCJI: Rozkład backendów embeddingów i detektorów
def save_extraction_validation(records: List[dict], paths: Dict[str, Path]) -> None:
    if (
        not records
    ):  # Jeśli lista rekordów jest pusta, wyświetl komunikat i zakończ funkcję
        return
    valid_rows = (
        []
    )  # Lista przechowująca informacje o każdej próbce, w tym nazwę próbki, źródło, identyfikator, backend embeddingu, backend detektora i gęstość krawędzi. Służy do tworzenia pliku CSV z wynikami walidacji ekstrakcji.
    backend_counts: Dict[str, int] = (
        {}
    )  # Lista przechowująca liczbę próbek dla każdego backendu embeddingu. Kluczem jest nazwa backendu, a wartością liczba próbek. Służy do tworzenia wykresu słupkowego z rozkładem backendów embeddingów.
    det_counts: Dict[str, int] = (
        {}
    )  # Lista przechowująca liczbę próbek dla każdego backendu detektora. Kluczem jest nazwa backendu, a wartością liczba próbek. Służy do tworzenia wykresu słupkowego z rozkładem backendów detektorów.
    edge_densities = (
        []
    )  # Lista przechowująca gęstość krawędzi dla każdej próbki. Gęstość krawędzi jest obliczana jako stosunek liczby pikseli krawędzi do całkowitej liczby pikseli w obrazie. Służy do tworzenia histogramu gęstości krawędzi po preprocessingu.
    for r in records:  # Dla każdego rekordu przetwórz dane i oblicz gęstość krawędzi
        edge_density = (
            float((r["edges"] > 0).mean()) if r.get("edges") is not None else 0.0
        )
        valid_rows.append(
            [
                r["sample_name"],
                r["source"],
                r["id"],
                r["backend"],
                r["detector_backend"],
                edge_density,
            ]
        )
        backend_counts[r["backend"]] = backend_counts.get(r["backend"], 0) + 1
        det_counts[r["detector_backend"]] = det_counts.get(r["detector_backend"], 0) + 1
        edge_densities.append(edge_density)
    # Zapisz wyniki
    _write_csv(
        paths["valid"] / "validation_results.csv",
        [
            "sample_name",
            "source",
            "id",
            "embedding_backend",
            "detector_backend",
            "edge_density",
        ],
        valid_rows,
    )

    # Wykres backendów
    fig1 = paths["valid_figs"] / "backend_counts.png"
    plt.figure(figsize=(7, 4))
    plt.bar(list(backend_counts.keys()), list(backend_counts.values()))
    plt.title("Liczba próbek według backendu embeddingu")
    plt.xlabel("Backend")
    plt.ylabel("Liczba próbek")
    plt.tight_layout()
    plt.savefig(fig1, dpi=150)
    plt.close()

    # Wykres detektorów
    fig2 = paths["valid_figs"] / "detector_counts.png"
    plt.figure(figsize=(7, 4))
    plt.bar(list(det_counts.keys()), list(det_counts.values()))
    plt.title("Liczba próbek według backendu detekcji")
    plt.xlabel("Detektor")
    plt.ylabel("Liczba próbek")
    plt.tight_layout()
    plt.savefig(fig2, dpi=150)
    plt.close()

    # Histogram gęstości krawędzi
    fig3 = paths["valid_figs"] / "edge_density_hist.png"
    plt.figure(figsize=(7, 4))
    plt.hist(edge_densities, bins=30)
    plt.title("Histogram gęstości krawędzi po preprocessingu")
    plt.xlabel("Gęstość krawędzi")
    plt.ylabel("Liczba próbek")
    plt.tight_layout()
    plt.savefig(fig3, dpi=150)
    plt.close()

# def filter_duplicate_samples(samples: List[SampleInfo], threshold: float = 0.96) -> List[SampleInfo]:
#     # Usuwa duplikaty z surowych plików przed ich preprocessingiem na podstawie korelacji histogramów.
#     unique_samples = []
#     seen_histograms = []
    
#     for s in samples:
#         if len(unique_samples) >= MIN_DATA_SIZE:
#             break
        
#         img = cv2.imread(str(s.file_path), cv2.IMREAD_GRAYSCALE)
#         if img is None:
#             continue
        
#         # Skalujemy do stałego rozmiaru, aby histogramy były porównywalne niezależnie od rozdzielczości zdjęć
#         img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
#         hist = cv2.calcHist([img_resized], [0], None, [64], [0, 256]).astype(np.float32)
#         hist /= hist.sum() + 1e-8
#         hist = hist.flatten()
        
#         is_dup = False
#         for h_existing in seen_histograms:
#             corr = np.corrcoef(hist, h_existing)[0, 1]
#             if corr >= threshold:
#                 is_dup = True
#                 break
                
#         if not is_dup:
#             seen_histograms.append(hist)
#             unique_samples.append(s)
            
#     print(f"Deduplikacja surowych obrazów: zredukowano z {len(samples)} do {len(unique_samples)} unikalnych próbek.")
#     dup_removed = len(samples) - len(unique_samples)
#     return unique_samples

# GŁÓWNA FUNKCJA EKSTRAKCJI: Przetwórz wszystkie obrazy użytkownika (Profile + VGG), embeddings
def extract_all(
    paths: Dict[str, Path], settings: Settings, fpar: FaceParams, user: str
) -> None:
    # Zbierz wszystkie obrazy z bazy
    samples = collect_samples(paths, user)
    if not samples:
        print("Brak obrazów w nowej strukturze bazy danych.")
        return

    print(f"{SEPARATOR}\n\nKrok 1: Preprocessing\n\n{SEPARATOR}\n")
    print(f"Liczba próbek: {len(samples)}")
    print("Wybrane ROI: detekcja twarzy YOLO -> MediaPipe -> fixed fallback")
    print("Wybrana maska: facemesh")
    print(SECTION)
    
    # Przy rejestracji nie można usuwać duplikatów
    dup_removed = 0

    sample_rows: List[dict] = (
        []
    )  # Lista przechowująca informacje o każdej próbce, w tym nazwę próbki, tożsamość, źródło i ścieżkę do pliku obrazu. Służy do tworzenia indeksu próbek.
    preprocess_rows: List[dict] = (
        []
    )  # Lista przechowująca wyniki preprocessingu dla każdej próbki, w tym status, bounding box, landmarki, maskę ROI, embedding i inne informacje. Służy do tworzenia logu preprocessingu.
    good_records: List[dict] = (
        []
    )  # Lista przechowująca tylko te rekordy, które przeszły preprocessingu pomyślnie (status "ok"). Służy do dalszej analizy i ekstrakcji embeddingów.
    histograms: Dict[str, np.ndarray] = (
        {}
    )  # Lista przechowująca histogramy dla każdej próbki, gdzie kluczem jest nazwa próbki, a wartością jest histogram obrazu w skali szarości. Służy do wykrywania duplikatów na podstawie podobieństwa histogramów.

    # Preprocessing każdego obrazu
    for i, s in enumerate(samples, start=1):
        # Tworzenie bezpiecznej nazwy próbki na podstawie źródła, tożsamości i ścieżki pliku. Funkcja make_safe_name zapewnia, że nazwa próbki jest unikalna i nie zawiera niedozwolonych znaków.
        sample_name = make_safe_name(s.source, s.identity, s.file_path)
        # Dodanie informacji o próbce do listy sample_rows, która będzie używana do tworzenia indeksu próbek. Zawiera nazwę próbki, tożsamość, źródło i ścieżkę do pliku obrazu.
        sample_rows.append(
            {
                "sample_name": sample_name,
                "identity": s.identity,
                "source": s.source,
                "file_path": str(s.file_path),
            }
        )
        # Wywołanie funkcji preprocess_one, która przetwarza pojedynczy obraz, wykonując detekcję twarzy, wyrównanie, ekstrakcję cech i zapis wyników. Funkcja zwraca słownik z wynikami preprocessingu, który jest następnie aktualizowany o dodatkowe informacje o próbce.
        rec = preprocess_one(
            file=s.file_path,
            settings=settings,
            paths=paths,
            fpar=fpar,
            store_outputs=("prep" in paths),
            sample_name=sample_name,
        )

        # Aktualizacja rekordu preprocessingu o dodatkowe informacje o próbce, takie jak tożsamość, identyfikator i źródło. Te informacje są przydatne do dalszej analizy i tworzenia logu preprocessingu.
        rec.update(
            {
                "identity": s.identity,
                "id": s.identity,
                "source": s.source,
                "file_path": str(s.file_path),
            }
        )
        # Dodanie rekordu preprocessingu do listy preprocess_rows, która będzie używana do tworzenia logu preprocessingu. Każdy rekord zawiera szczegółowe informacje o wyniku preprocessingu dla danej próbki.
        preprocess_rows.append(rec)
        if rec["status"] == "ok":
            # Jeśli preprocessingu zakończył się pomyślnie (status "ok"), dodaj rekord do listy good_records, która będzie używana do dalszej analizy i ekstrakcji embeddingów.
            good_records.append(rec)
            hist = cv2.calcHist([rec["aligned_gray"]], [0], None, [64], [0, 256]).astype(np.float32)
            hist /= hist.sum() + 1e-8
            histograms[sample_name] = hist.flatten()
        print(f"({i}/{len(samples)}) Zapisano plik: {sample_name}.")

    if "sample_index" in paths:
        # Zapisanie indeksu próbek
        write_sample_index(sample_rows, paths["sample_index"])
        print(f"Zapisano indeks próbek: {paths['sample_index']}")

    if "preprocess_log" in paths:    
        # Zapisanie logu preprocessingu
        write_preprocess_log(preprocess_rows, paths["preprocess_log"])
        print(f"Zapisano log preprocessingu: {paths['preprocess_log']}")

    print(SECTION)

    # Tylko rekordy, które przetrwały
    records_kept = [r for r in good_records if r["status"] == "ok"]

    records_profile = [r for r in records_kept if r["source"] == "profile"]
    records_ff_train = [r for r in records_kept if r["source"] == "ff_train"]
    records_ff_test = [r for r in records_kept if r["source"] == "ff_test"]
    
    print(f"\n{SEPARATOR}\n\nKrok 2: Ekstrakcja embeddingów\n\n{SEPARATOR}\n")
    print(f"Liczba plików po preprocessingu: {len(records_kept)}")
    print("Backend embeddingu: ArcFace (ONNX) z fallbackiem classic_plus_landmarks")
    print(SECTION)

    # Zapisz embeddings i statystyki
    if records_profile:
        user_csv = paths["profiles"] / user / f"{user}.csv"
        write_embeddings_csv(records_profile, user_csv)
    
    # Zapis ff_train (tylko jeśli jeszcze nie istnieje lub wymuszamy nadpisanie)
    if records_ff_train and not paths["ff_train_csv"].exists():
        write_embeddings_csv(records_ff_train, paths["ff_train_csv"])
        
    # Zapis ff_test
    if records_ff_test and not paths["ff_test_csv"].exists():
        write_embeddings_csv(records_ff_test, paths["ff_test_csv"])
        
    write_embeddings_csv(records_kept, paths["embeddings_csv"])
    print(f"Zapisano embeddings: {paths['embeddings_csv']}")
    
    if getattr(settings, "delete_photos", False):
        print(f"\n{SEPARATOR}\nRODO: Trwa usuwanie oryginalnych zdjęć ze wszystkich źródeł...\n{SEPARATOR}")
        deleted_count = 0
        for s in samples:
            if s.file_path and s.file_path.exists():
                try:
                    s.file_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"Nie udało się usunąć pliku {s.file_path.name}: {e}")
        print(f"Usunięto pomyślnie {deleted_count} zdjęć z dysku.")
    
    if "dataset_summary_csv" in paths:
        write_dataset_summary(records_kept, paths["dataset_summary_csv"])
        print(f"Zapisano podsumowanie zbioru: {paths['dataset_summary_csv']}")
        
    if "valid" in paths:
        save_extraction_validation(records_kept, paths)
        print(f"Zapisano walidację ekstrakcji: {paths['valid'] / 'validation_results.csv'}")
    
    # Zapisz metadane aby mieć pełną dokumentację procesu ekstrakcji, w tym: ustawienia użyte do preprocessingu i ekstrakcji embeddingów,
    # parametry detekcji twarzy, stan losowy użyty w procesie, liczbę próbek przetworzonych, liczbę próbek pomyślnie przetworzonych,
    # liczbę usuniętych duplikatów oraz układ katalogów i plików w projekcie. Dodatkowo zapisuje wersje używanych bibliotek (Python, OpenCV, NumPy)
    # i platformę systemową, co jest istotne dla reprodukowalności wyników i diagnostyki ewentualnych problemów.
    if "meta" in paths:
        meta = {
            "settings": asdict(settings),
            "face_params": asdict(fpar),
            "random_state": RANDOM_STATE,
            "counts": {
                "samples_total": len(samples),
                "samples_ok": len([r for r in records_kept if r["status"] == "ok"]),
                "duplicates_removed": dup_removed,
            },
            "dataset_layout": {
                "profiles": str(paths["profiles"]),
                "ff_train": str(paths["ff_train"]),
                "ff_test": str(paths["ff_test"]),
                "verification": str(paths["verification"]),
            },
            "weights": {
                "yolo_face": str(paths["weights"] / "yolo_face.pt"),
                "arcface_onnx": str(paths["weights"] / "arcface.onnx"),
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "opencv": cv2.__version__,
                "numpy": np.__version__,
            },
            "store_outputs": store_outputs,
            "delete_photos": delete_photos,
            "min_data_size": MIN_DATA_SIZE
        }
        with open(paths["meta"], "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"Zapisano meta: {paths['meta']}")

# FUNKCJA EKSTRAKCJI PROFILU: Przetwórz tylko zdjęcia profilu wskazanego użytkownika
def extract_profile(
    paths: Dict[str, Path], settings: Settings, fpar: FaceParams, user: str, step: str
) -> None:
    samples = collect_samples(paths, user, profile_only=True)

    if not samples:
        raise ValueError(f"Brak zdjęć profilu użytkownika: {user}")
    print(
        f"{SEPARATOR}\n\nTryb PROFILE: aktualizacja profilu użytkownika\n\n{SEPARATOR}\n"
    )
    print(f"Użytkownik: {user}")
    print(f"Liczba próbek profilu: {len(samples)}")
    print(SECTION)

    sample_rows: List[dict] = []
    preprocess_rows: List[dict] = []
    good_records: List[dict] = []
    histograms: Dict[str, np.ndarray] = {}

    for i, s in enumerate(samples, start=1):
        sample_name = make_safe_name(s.source, s.identity, s.file_path)

        sample_rows.append(
            {
                "sample_name": sample_name,
                "identity": s.identity,
                "source": s.source,
                "file_path": str(s.file_path),
            }
        )

        rec = preprocess_one(
            file=s.file_path,
            settings=settings,
            paths=paths,
            fpar=fpar,
            store_outputs=("prep" in paths),
            sample_name=sample_name,
        )

        rec.update(
            {
                "identity": s.identity,
                "id": s.identity,
                "source": s.source,
                "file_path": str(s.file_path),
            }
        )

        preprocess_rows.append(rec)

        if rec["status"] == "ok":
            good_records.append(rec)
            hist = cv2.calcHist([rec["aligned_gray"]], [0], None, [64], [0, 256]).astype(np.float32)
            hist /= hist.sum() + 1e-8
            histograms[sample_name] = hist.flatten()

        print(f"({i}/{len(samples)}) Zapisano plik: {sample_name}")

    if "sample_index" in paths:
        write_sample_index(sample_rows, paths["sample_index"])
        print(f"Zapisano indeks próbek profilu: {paths['sample_index']}")

    if "preprocess_log" in paths:
        write_preprocess_log(preprocess_rows, paths["preprocess_log"])
        print(f"Zapisano log preprocessingu profilu: {paths['preprocess_log']}")

    print(SECTION)

    records_kept = [r for r in good_records if r["status"] == "ok"]
    records_ff_train = [r for r in records_kept if r["source"] == "ff_train"]
    records_ff_test = [r for r in records_kept if r["source"] == "ff_test"]
        
    if not records_kept:
        raise ValueError(f"Brak poprawnych rekordów profilu użytkownika {user}")

    print(
        f"\n{SEPARATOR}\n\nAktualizacja embeddings.csv dla użytkownika\n\n{SEPARATOR}\n"
    )
    print(f"Liczba nowych embeddingów użytkownika: {len(records_kept)}")
    
    if records_ff_train and not paths["ff_train_csv"].exists():
        write_embeddings_csv(records_ff_train, paths["ff_train_csv"])
            
    if records_ff_test and not paths["ff_test_csv"].exists():
        write_embeddings_csv(records_ff_test, paths["ff_test_csv"])
    print(SECTION)

    # Pobieramy połączone i odfiltrowane z duplikatów wiersze (limit: 25)
    final_user_rows = update_embeddings_csv_for_profile(records_kept, paths["embeddings_csv"], user)
    user_csv_path = paths["profiles"] / user / f"{user}.csv"
    
    # Tworzymy nagłówek dla pliku CSV
    dim = len(records_kept[0]["embedding"])
    header = [
        "sample_name", "id", "source", "backend", "detector_backend",
        "detector_score", "landmarks_found", "prep_file", "mask_file",
        "roi_figure", "pipeline_figure"
    ] + [f"e_{i:03d}" for i in range(dim)]
    
    # Nadpisujemy lokalny plik użytkownika POŁĄCZONĄ bazą danych (a nie tylko nowymi zdjęciami)
    _write_csv(user_csv_path, header, final_user_rows)

    print(f"Zapisano zaktualizowany profil użytkownika {user} w: {user_csv_path}")
    print(f"Zaktualizowano główny plik embeddings: {paths['embeddings_csv']}")

    if "dataset_summary_csv" in paths:
        write_dataset_summary(records_kept, paths["dataset_summary_csv"])
        print(f"Zapisano podsumowanie profilu: {paths['dataset_summary_csv']}")

    if "valid" in paths:
        save_extraction_validation(records_kept, paths)
        print(f"Zapisano walidację ekstrakcji: {paths['valid'] / 'validation_results.csv'}")

    if getattr(settings, "delete_photos", False):
        print(f"\n{SEPARATOR}\nRODO: Trwa usuwanie oryginalnych zdjęć profilowych...\n{SEPARATOR}")
        deleted_count = 0
        for s in samples:
            if s.file_path and s.file_path.exists():
                try:
                    s.file_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"Nie udało się usunąć pliku {s.file_path.name}: {e}")
        print(f"Usunięto pomyślnie {deleted_count} zdjęć z dysku.")
    
    if "meta" in paths:
        meta = {
            "mode": "profile",
            "user": user,
            "settings": asdict(settings),
            "face_params": asdict(fpar),
            "random_state": RANDOM_STATE,
            "step": getattr(settings, "step", None),
            "counts": {
                "profile_samples_total": len(samples),
                "profile_samples_ok": len(records_kept),
            },
            "profile_dir": str(paths["profiles"] / user),
            "embeddings_csv": str(paths["embeddings_csv"]),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "opencv": cv2.__version__,
                "numpy": np.__version__,
            },
            "store_outputs": store_outputs,
            "delete_photos": delete_photos,
            "min_data_size": MIN_DATA_SIZE
        }

        with open(paths["meta"], "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"Zapisano meta: {paths['meta']}")
# FUNKCJA PRZETWARZAJĄCA JEDEN OBRAZ WERYFIKACYJNY (np. z katalogu verification) I ZWRACAJĄCA EMBEDDING + METADANE
# UŻYWANE W varify.py DO PRZETWORZENIA POJEDYNCZEGO ZDJĘCIA WERYFIKACYJNEGO
def embed_image(
    image_path: Path,
    base_dir: Path,
    store_outputs: str = "yes",
    settings: Optional[Settings] = None,
    fpar: Optional[FaceParams] = None,
    prefix: str = "check",
    delete_photo: str = "no",
    step: str = "test",
) -> Tuple[np.ndarray, dict]:
    settings = settings or Settings()
    fpar = fpar or FaceParams()
    
    paths = build_paths(
        base_dir,
        TEST_DATA_DIR if step == "test" else DATA_DIR,
        store_outputs=(store_outputs == "yes")
    )
    
    if store_outputs == "yes":
        paths["prep"] = paths["out"] / "_tmp_check_prep"
        paths["mask"] = paths["out"] / "_tmp_check_mask"
        paths["fig_pipeline"] = paths["out"] / "_tmp_check_fig"
        paths["fig_roi"] = paths["out"] / "_tmp_check_roi"
        for p in ["prep", "mask", "fig_pipeline", "fig_roi"]:
            paths[p].mkdir(parents=True, exist_ok=True)
            
    rec = preprocess_one(
        file=image_path,
        settings=settings,
        paths=paths,
        fpar=fpar,
        store_outputs=(store_outputs == "yes"),
        sample_name=f"{prefix}__{image_path.stem}",
    )
    
    if rec["status"] != "ok":
        raise RuntimeError(f"Nie udało się przetworzyć {image_path}: {rec['status']}")
    
    # Usuwanie zdjęcia (RODO)
    if delete_photo == "yes":
        try:
            if image_path and image_path.exists():
                image_path.unlink()
                print(f"RODO: Usunięto plik weryfikacyjny -> {image_path.name}")
        except Exception as e:
            print(f"Nie udało się usunąć pliku {image_path.name}: {e}")
    return rec["embedding"], rec

# FUNKCJA PARSUJĄCA ARGUMENTY WIERSZA POLECEŃ
def parse_args():
    # Tworzenie parsera argumentów wiersza poleceń z opisem funkcji ekstrakcji cech twarzy
    ap = argparse.ArgumentParser(
        description="Ekstrakcja cech twarzy: detekcja, preprocessing, embeddingi i logi"
    )  

    # Argument określający, co uruchomić:
    # "profile"- utworzenie samego pliku embeddings.csv dla wskazanego użytkownika (Profile/<user>) bez przeliczania VGG
    # "all" - przetworzenie wszystkich obrazów użytkownika (Profile + VGG) i zapisanie embeddings.csv, dataset_summary.csv oraz walidacji ekstrakcji
    ap.add_argument(
        "cmd", choices=["profile", "all", "impostors"], help="Co uruchomić"
    )
    
    # Argument określający bazowy katalog projektu, domyślnie ustawiony na katalog nadrzędny skryptu
    ap.add_argument(
        "--base-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent),
        help="Katalog bazowy projektu",
    ) 
    
    # Argument określający użytkownika referencyjnego
    ap.add_argument(
        "--user",
        type=str,
        default="Jowi",
        help="Użytkownik referencyjny z bazy danych",
    )
    
    # Argument określający do czego ma służyć przetwarzanie: 
    # "test" - przetwarzanie testowe (korzysta z testowej bazy danych, nie usuwa/dodaje/edytuje obrazów wejściowych),
    # "register" - rejestracja użytkownika (nowa baza danych, przetwarzanie wszystkich obrazów wejściowych, wyciągnięcie cech twarzy oraz usunięcie obrazów wejściowych po przetworzeniu)
    
    ap.add_argument(
        "--step",
        choices=["test", "normal"],
        default="test",
        help="Rodzaj przetwarzania",
    )
    
    ap.add_argument(
        "--store-outputs",
        choices=["yes", "no"],
        default="yes",
        help="Zapisanie dodatkowych danych wyjściowych",
    )
    
    ap.add_argument(
        "--delete-photos",
        choices=["yes","no"],
        default="no",
        help="Usuwanie zdjęć z bazy danych (RODO)",
    )

    for name, dest in (
        ("gaussian-blur", "use_gaussian_blur"),
        ("median-filter", "use_median_filter"),
        ("clahe", "use_clahe"),
        ("face-mask", "use_face_mask"),
    ):
        ap.add_argument(
            f"--{name}", dest=dest, nargs="?", const="yes", choices=["yes", "no"],
            default="no", help=f"Włącz/wyłącz operację: {name}"
        )
        ap.add_argument(
            f"--no-{name}", dest=dest, action="store_const", const="no",
            help=f"Wyłącz operację: {name}"
        )
    
    # Argument określający współczynnik skalowania dla detektora twarzy
    ap.add_argument(
        "--face-scale-factor", type=float, default=FACE_SCALE_FACTOR
    ) 
    
    # Argument określający minimalną liczbę sąsiadów dla detektora twarzy
    ap.add_argument(
        "--face-min-neighbors", type=int, default=FACE_MIN_NEIGH
    )
    
    # Argument określający minimalny ułamek rozmiaru twarzy dla detektora twarzy
    ap.add_argument(
        "--face-min-size-frac", type=float, default=FACE_MIN_SIZE_FRAC
    )
    
    # Argument określający ułamek marginesu dla detektora twarzy
    ap.add_argument(
        "--face-margin-frac", type=float, default=FACE_MARGIN_FRAC
    )
    
    # Argument określający minimalny obszar komponentu spójnego (CC) dla czyszczenia krawędzi
    ap.add_argument(
        "--min-cc-area", type=int, default=MIN_CC_AREA_DEFAULT
    )
    
    # Argument określający ułamek odległości między oczami dla wyrównania twarzy
    ap.add_argument(
        "--eye-dist-frac", type=float, default=EYE_DIST_FRAC_DEFAULT
    )
    return ap.parse_args()
 
# GŁÓWNA FUNKCJA
def main():
    args = parse_args()  # Parsuje argumenty wiersza poleceń
    
    match args.step:
        case "test":     
            db = TEST_DATA_DIR
        case _:
            db = DATA_DIR
    store_outputs = True if args.store_outputs == "yes" else False  # Czy obrazy wejściowe mają być usuwane
    delete_photos = True if args.delete_photos == "yes" else False  # Czy obrazy wejściowe mają być usuwane

    base_dir = Path(
        args.base_dir
    ).resolve()  # Ustala katalog bazowy projektu na podstawie argumentu --base-dir lub domyślnie na katalog nadrzędny skryptu
    paths = build_paths(
        base_dir, db, store_outputs
    )  # Buduje słownik ścieżek do różnych katalogów i plików w projekcie na podstawie katalogu bazowego
    settings = (
        Settings()
    )  # Tworzy obiekt Settings z domyślnymi parametrami przetwarzania obrazu
    object.__setattr__(
        settings, "min_cc_area", args.min_cc_area
    )  # Ustawia minimalny obszar komponentu spójnego (CC) w obiekcie Settings na podstawie argumentu --min-cc-area
    object.__setattr__(
        settings, "eye_dist_frac", args.eye_dist_frac
    )  # Ustawia ułamek odległości między oczami w obiekcie Settings na podstawie argumentu --eye-dist-frac
    object.__setattr__(
        settings, "delete_photos", delete_photos
    ) # Ustawia flagę usuwania zdjęć (pod RODO)
    for dest in ("use_gaussian_blur", "use_median_filter", "use_clahe", "use_face_mask"):
        object.__setattr__(settings, dest, getattr(args, dest) == "yes")
    fpar = FaceParams(  # Tworzy obiekt FaceParams z parametrami detekcji twarzy na podstawie argumentów wiersza poleceń
        scale_factor=args.face_scale_factor,  # Ustawia współczynnik skalowania dla detektora twarzy na podstawie argumentu --face-scale-factor
        min_neighbors=args.face_min_neighbors,  # Ustawia minimalną liczbę sąsiadów dla detektora twarzy na podstawie argumentu --face-min-neighbors
        min_size_frac=args.face_min_size_frac,  # Ustawia minimalny ułamek rozmiaru twarzy dla detektora twarzy na podstawie argumentu --face-min-size-frac
        margin_frac=args.face_margin_frac,  # Ustawia ułamek marginesu dla detektora twarzy na podstawie argumentu --face-margin-frac
    )

    if args.cmd == "all":
        extract_all(
            paths, settings, fpar, args.user
        )  # Uruchamia funkcję extract_all, która przetwarza wszystkie obrazy użytkownika w katalogu bazowym, wykonując detekcję twarzy, preprocessing, ekstrakcję embeddingów i zapis wyników.
    elif args.cmd == "profile":
        extract_profile(paths, settings, fpar, args.user, args.step)


if __name__ == "__main__":
    main()
