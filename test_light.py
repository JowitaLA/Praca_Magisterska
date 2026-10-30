import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from resources.config import *   # BASE, TEST_DATA_DIR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
CENTER_FRACTION = 0.55          # jaka część ramki twarzy (środek) idzie do pomiaru
YOLO_CONF = 0.25                # minimalna pewność detekcji YOLO
WEIGHTS_PATH = BASE / "weights" / "yolo_face.pt"
FALLBACK_W, FALLBACK_H = 0.40, 0.50   # zapasowa "ramka twarzy": środek kadru (ułamek szer./wys.)

# detektory
try:
    from ultralytics import YOLO
    _yolo = YOLO(str(WEIGHTS_PATH))
except Exception as e:           # brak biblioteki albo wag -> tylko Haar
    print(f"Uwaga: YOLO niedostępne ({type(e).__name__}: {e}). Używam wyłącznie Haara.")
    _yolo = None

_haar = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_face(img: Image.Image, gray: np.ndarray):
    # Zwraca (x, y, w, h, nazwa_detektora, pewność) albo None, gdy brak twarzy.
    H, W = gray.shape
    if _yolo is not None:
        res = _yolo.predict(img, verbose=False, conf=YOLO_CONF)[0]
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()
            area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            i = int(np.argmax(area))                       # największa twarz
            x0, y0, x1, y1 = xyxy[i]
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(W, int(x1)), min(H, int(y1))
            if x1 - x0 > 20 and y1 - y0 > 20:
                return x0, y0, x1 - x0, y1 - y0, "yolo", float(conf[i])
    faces = _haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return int(x), int(y), int(w), int(h), "haar", float("nan")
    return None


def measure_face_light(image_path: Path) -> dict:
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float64)
    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    H, W = lum.shape

    det = detect_face(img, lum.astype(np.uint8))
    if det is None:
        # twarz jest widoczna dla człowieka, ale detektory zawiodły -> klatka zostaje w analizie
        w, h = int(W * FALLBACK_W), int(H * FALLBACK_H)
        x, y = (W - w) // 2, (H - h) // 2
        detector, conf, fallback = "none", float("nan"), 1
    else:
        x, y, w, h, detector, conf = det
        fallback = 0

    margin = (1 - CENTER_FRACTION) / 2
    mx, my = int(w * margin), int(h * margin)
    roi = lum[y + my:y + h - my, x + mx:x + w - mx]
    if roi.size == 0:
        raise ValueError("pusty obszar pomiaru")

    # asymetria: lewy vs prawy POLICZEK (bez oczu, okularów i słuchawek)
    def patch(x0, x1):
        return lum[y + int(h * 0.55):y + int(h * 0.80), x + int(w * x0):x + int(w * x1)]
    left, right = patch(0.15, 0.40), patch(0.60, 0.85)

    p05, p95 = np.percentile(roi, [5, 95])
    return {
        "brightness": float(roi.mean()),
        "contrast": float(roi.std()),
        "over": float((roi >= 250).mean()),
        "under": float((roi <= 5).mean()),
        "asym": float(abs(left.mean() - right.mean())),
        "dyn_range": float(p95 - p05),
        "sharp": float(cv2.Laplacian(roi, cv2.CV_64F).var()),
        "img_w": int(W), "img_h": int(H),
        "face_rel": float(w / W) if not fallback else float("nan"),
        "box_x": int(x), "box_y": int(y), "box_w": int(w), "box_h": int(h),
        "detector": detector,
        "detector_conf": conf,
        "roi_fallback": fallback,
    }


def find_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--probes", type=Path, default=TEST_DATA_DIR / "verification" / "2",
                    help="folder ze zdjęciami do zmierzenia (próby lub wzorce)")
    ap.add_argument("--out", type=Path, default=BASE / "resources" / "light.csv")
    a = ap.parse_args()

    folder, output_csv = a.probes, a.out
    if not folder.is_dir():
        print(f"Błąd: {folder} nie jest folderem.")
        sys.exit(1)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    images = find_images(folder)
    if not images:
        print(f"Nie znaleziono żadnych zdjęć w {folder} (obsługiwane: {IMAGE_EXTENSIONS})")
        sys.exit(1)

    print(f"Znaleziono {len(images)} zdjęć w {folder}. Pomiar...")

    rows, failed = [], []
    for i, img_path in enumerate(images, 1):
        try:
            feats = measure_face_light(img_path)
            rows.append({
                "filename": img_path.name,
                "path": str(img_path),
                **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in feats.items()},
            })
        except Exception as e:
            failed.append(img_path.name)
            print(f"  Błąd pomiaru {img_path.name}: {e}")
        if i % 20 == 0 or i == len(images):
            print(f"  przetworzono {i}/{len(images)}")

    if not rows:
        print("Nie udało się zmierzyć żadnego zdjęcia.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    if df["filename"].duplicated().any():
        print("UWAGA: powtarzają się nazwy plików w podfolderach - łączenie z wynikami po nazwie będzie niejednoznaczne!")
    df.to_csv(output_csv, index=False)

    # klatki z zapasowym ROI - do sprawdzenia wzrokiem (czy twarz faktycznie widoczna)
    fb = df.loc[df["roi_fallback"] == 1, "filename"].tolist()
    if fb:
        (output_csv.parent / (output_csv.stem + "_fallback.txt")).write_text("\n".join(fb), encoding="utf-8")
    if failed:
        (output_csv.parent / (output_csv.stem + "_errors.txt")).write_text("\n".join(failed), encoding="utf-8")

    print(f"\nZapisano: {output_csv}")
    print(f"Zmierzono: {len(df)}, z zapasowym ROI (detektory zawiodły): {len(fb)}, błędy odczytu: {len(failed)}")
    print("Które detektory znalazły twarz:")
    print(df["detector"].value_counts().to_string())
    print("\nRozkład cech (opis w pracy: zakres i liczność, nie grupy):")
    print(df[["brightness", "contrast", "over", "under", "asym", "dyn_range", "sharp", "face_rel"]].describe().round(2))


if __name__ == "__main__":
    main()