from __future__ import annotations

import sys

import argparse
import csv
import secrets
from datetime import datetime
from pathlib import Path
from typing import List

import joblib
import numpy as np

# Plik verify.py weryfikuje tożsamość na podstawie zdjęcia.
# - WCZYTANIE VERIFIER'U: verifier.pkl z training.py (galerię + próg + meta)
# - EKSTRAKCJA EMBEDINGU: embed_image() - przetworzenie nowego zdjęcia
# - OBLICZANIE SCORE'U: cosine_similarity(query_embedding, galeria)
# - DECYZJA: score >= threshold, AKCEPTACJA : ODRZUCENIE
# - FALLBACK OTP: Jeśli odrzucenie, generowanie 6-cyfrowego OTP (tymczasowo)
#
# WYJŚCIE:
# - verification_results.csv: Szczegółowe wyniki dla każdego zdjęcia
# - verification_summary.csv: Agregat (ile akceptacji/odrzuceń)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from resources.config import *  # Importowanie zmiennych globalnych i ścieżek z config.py
from extraction.extraction import Settings, FaceParams, build_paths as build_extraction_paths, embed_image  # Import funkcji i klas z modułu ekstrakcji cech

# FUNKCJA WCZYTUJĄCA VERIFIER
def load_verifier(base_dir: Path, user: str):
    pred_out = TRAIN_DIR / "outputs" # Katalog wyjściowy etapu modelowania
    vf_path = pred_out / f"verifier_{user}.pkl" # Ścieżka do pliku verifier.pkl
    # Sprawdzenie czy model został wcześniej utworzony
    if not vf_path.exists():
        raise FileNotFoundError(f"Brak {vf_path}. Uruchom najpierw: python modeling/training.py")
    return joblib.load(vf_path)

# FUNKCJA PRZYGOTOWUJĄCA LISTĘ OBRAZÓW DO WERYFIKACJI
def resolve_targets(base_dir: Path, check_arg: str, step: str, user: str = "Jowi") -> List[Path]:
    image_extensions = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}

    def images_in(folder: Path) -> List[Path]:
        return sorted(
            path for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in image_extensions
        )

    # Próba 1: Ścieżka absolutna
    p = Path(check_arg)
    if p.is_absolute() and p.exists(): # Jeśli ścieżka absolutna istnieje, zwracany jest pojedynczy plik albo pliki z folderu
        return [p] if p.is_file() else images_in(p)
    
    # "all" oznacza zdjęcia wyłącznie z folderu deklarowanego użytkownika.
    if step.lower() in {"normal"}:
        verification_dir = DATA_DIR / "verification"
    else:
        verification_dir = TEST_DATA_DIR / "verification"
    if check_arg.lower() in {"all", "*", "folder"}:
        return images_in(verification_dir / user)

    # Pozwól podać nazwę pliku bez rozszerzenia, np. --check 1 dla 1.jpg.
    relative_path = verification_dir / check_arg
    if relative_path.is_file():
        return [relative_path]
    if not relative_path.suffix:
        for extension in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"):
            image_path = relative_path.with_suffix(extension)
            if image_path.is_file():
                return [image_path]
    if relative_path.is_dir():
        return images_in(relative_path)
    return [relative_path]

# FUNKCJA OBLICZAJĄCA SCORE PODOBIEŃSTWA DO GALERII
def cosine_scores(query: np.ndarray, gallery: np.ndarray, topk: int = 3) -> tuple[float, float]:
    # Obliczenie cosine similarity między wszystkimi embeddingami galerii a embeddingiem query
    sims = gallery @ query.reshape(-1, 1) 
    sims = sims.ravel() # Spłaszczenie wyniku do tablicy 1D
    score_max = float(np.max(sims)) # Najwyższy score podobieństwa do pojedynczej próbki z galerii
    k = max(1, min(topk, len(sims))) # Ustalenie liczby wyników używanych do średniej top-k
    score_topk = float(np.mean(np.sort(sims)[-k:])) # Obliczenie średniej z k najwyższych score'ów
    return score_max, score_topk # Zwrócenie obu wariantów score'u

# FUNKCJA DOPISUJĄCA WIERSZE DO PLIKU CSV
def append_rows(path: Path, header: List[str], rows: List[List[object]]) -> None:
    # Upewnienie się, że katalog docelowy istnieje
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() # Sprawdzenie czy trzeba zapisać nagłówek
    with open(path, "a", newline="", encoding="utf-8") as f: # Otworzenie pliku w trybie append, żeby nie kasować wcześniejszych wyników
        wr = csv.writer(f)
        if write_header:
            wr.writerow(header)
        wr.writerows(rows)

# GŁÓWNA FUNKCJA
def main():
    ap = argparse.ArgumentParser(description="Weryfikacja tożsamości na pojedynczym zdjęciu lub całym folderze") # Utworzenie parsera argumentów
    ap.add_argument(
        "--check",
        default="all",
        type=str,
        help="Zdjęcie, folder lub 'all' dla wszystkich zdjęć z verification/<user>",
    )
    ap.add_argument("--user", default="Jowi", type=str, help="Deklarowany użytkownik") # Argument określający deklarowanego użytkownik
    ap.add_argument("--store-outputs", choices=["yes", "no"], default="yes", help="Zapis pomocniczych artefaktów")
    ap.add_argument("--delete-photo", choices=["yes", "no"], default="no", help="Usuwanie zdjęcia")
    ap.add_argument("--step", default="test", choices=["test", "normal"], help="Krok do wykonania") # Argument określający krok do wykonania
    args = ap.parse_args() # Odczyt argumentów z linii komend

    # Wczytanie verifier.pkl zapisanego przez model.py
    vf = load_verifier(BASE, args.user)
    # Przygotowanie listy obrazów do weryfikacji
    targets = resolve_targets(BASE, args.check, args.step, args.user)
    # Sprawdzenie czy znaleziono pliki wejściowe
    if not targets or not all(p.exists() for p in targets):
        raise FileNotFoundError(f"Nie znaleziono wejścia do weryfikacji dla argumentu: {args.check}")

    # Pobranie galerii embeddingów użytkownika z verifier.pkl
    gallery = np.asarray(vf["gallery_embeddings"], dtype=np.float32)
    # Pobranie progu decyzyjnego z verifier.pkl
    threshold = float(vf.get("threshold", 0.45)) # Jeśli próg nie istnieje, używany jest domyślny próg 0.45
    # Pobranie wartości topk z verifier.pkl
    topk = int(vf.get("topk", 3))
    settings = Settings() # Ustawienia preprocessingu i detekcji używane przez embed_image
    fpar = FaceParams()
    # Określenie bazy danych w zależności od parametru --step
    db = DATA_DIR if args.step in {"normal"} else TEST_DATA_DIR

    # Zbudowanie ścieżek z przekazaniem bazy danych
    paths = build_extraction_paths(BASE, db, args.store_outputs)    # Katalog wyjściowy etapu weryfikacji
    
    if args.store_outputs == "yes":
        outputs_dir = VERIFICATION_DIR / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # Ścieżki do plików wynikowyc
        results_csv = outputs_dir / "verification_results.csv"
        summary_csv = outputs_dir / "verification_summary.csv"

    # Wyświetlenie wyników etapu 3 w konsoli
    print(f"{SEPARATOR}\n\nEtap 3: Weryfikacja tożsamości\n\n{SEPARATOR}\n")
    print(f"Liczba plików do sprawdzenia: {len(targets)}")
    print(f"Deklarowany użytkownik:       {args.user}")
    print(f"Użytkownik modelu:            {vf['user']}")
    print(f"Próg decyzji:                 {threshold:.6f}")
    print(SECTION)

    # Lista wierszy do zapisania w verification_results.csv
    rows = []
    # Liczniki decyzji
    n_accept = 0
    n_reject = 0
    for img_path in targets: # Przetwarzanie każdego obrazu wejściowego
        # Przetworzenie obrazu do embeddingu
        # Funkcja embed_image wykonuje preprocessing, detekcję, alignment, maskę i embedding
        emb, info = embed_image(
            img_path, BASE, args.store_outputs, settings=settings, fpar=fpar,
            prefix="verify", delete_photo=args.delete_photo, step=args.step,
        )
        emb = emb.astype(np.float32) # Konwersja embeddingu do float32
        emb = emb / (np.linalg.norm(emb) + 1e-8) # Normalizacja L2 embeddingu
        score_max, score_topk = cosine_scores(emb, gallery, topk=topk) # Obliczenie score'u względem galerii użytkownika
        # Decyzja końcowa
        decision = (score_max >= threshold) and (args.user == vf["user"]) # Przekroczenie progu + zgodność deklarowanego użytkownika z modelem
        otp = "" # Domyślnie OTP jest puste
        if decision: # Decyzja pozytywna
            n_accept += 1
        else: # Decyzja negatywna
            n_reject += 1
            otp = ''.join(str(secrets.randbelow(10)) for _ in range(6)) # tymczasowe OTP jako 6-cyfrowy znak
        print(f"Plik: {img_path.name}")
        print(f"  score_max:   {score_max:.6f}")
        print(f"  score_topk:  {score_topk:.6f}")
        print(f"  decyzja:     {'AKCEPTACJA' if decision else 'ODRZUCENIE'}")
        if otp:
            print(f"  fallback OTP (demo): {otp}") 

        # Dodanie wyniku bieżącego pliku do listy rows
        rows.append([
            datetime.now().isoformat(timespec="seconds"),
            str(img_path),
            args.user,
            vf["user"],
            float(threshold),
            float(score_max),
            float(score_topk),
            int(decision),
            info["backend"],
            info["detector_backend"],
            float(info["detector_score"]),
            int(info["landmarks_found"]),
            info["prep_file"],
            info["mask_file"],
            info["roi_figure"],
            info["pipeline_figure"],
            otp,
        ])

    if args.store_outputs == "yes":
        # Zapis szczegółowych wyników weryfikacji do CSV
        append_rows(
            results_csv,
            [
                "timestamp", "check_file", "claimed_user", "model_user", "threshold", "score_max", "score_topk",
                "decision", "embedding_backend", "detector_backend", "detector_score", "landmarks_found",
                "prep_file", "mask_file", "roi_figure", "pipeline_figure", "otp_demo",
            ],
            rows,
        )
        
        # Zapis podsumowania całego uruchomienia do CSV
        append_rows(
            summary_csv,
            ["timestamp", "claimed_user", "model_user", "n_files", "n_accept", "n_reject", "accept_rate"],
            [[datetime.now().isoformat(timespec="seconds"), args.user, vf["user"], len(targets), n_accept, n_reject, (n_accept / max(1, len(targets)))]]
        )
        print(SECTION)
        print(f"Zapisano wyniki jednostkowe:    {results_csv}")
        print(f"Zapisano podsumowanie:          {summary_csv}")

if __name__ == "__main__":
    main()
