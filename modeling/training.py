from __future__ import annotations

import sys

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from scipy.stats import gumbel_r
from datetime import datetime

# Plik training.py odpowiada za trenowanie i ewaluację systemu weryfikacji twarzy.
# - Wczytuje osadzenia (embeddings) z outputs/embeddings.csv
# - Dzieli próbki użytkownika profile na galerię i holdout do walidacji genuine
# - Oblicza score'y cosinusowe między wzorcami galerii a próbkami genuine i impostor
# - Dobiera próg EER bazując na zbiorze impostorów z ff_train lub ff_test
# - Generuje raporty, wykresy i zapisuje obiekt weryfikatora do verifier.pkl
#
# WYJŚCIE:
# - verifier.pkl: Znormalizowane embeddingi (L2, cosine-ready)
# - model_report.json: Statystyki źródeł (profile/ff_train/ff_test)
# - wykresy ROC/DET/histogram/PCA: Artefakty (dla debugowania i walidacji)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from resources.config import * 

db = DATA_DIR
MIN_OPERATIONAL_THRESHOLD = 0.50
# Poziomy zabezpieczeń wybierane przez użytkownika w ustawieniach: klucz -> docelowy FAR.
# Dla porównania: Android wymaga 1/50 000 (2e-5), Face ID deklaruje ok. 1e-6.
SECURITY_LEVELS = {"high": 2e-5, "standard": 1e-4, "comfort": 1e-3}
DEFAULT_SECURITY_LEVEL = "standard"   # poziom zapisywany jako "threshold" (zgodność ze starym kodem)
GATE_LEVEL = "comfort"                # rejestracja jest odrzucana dopiero, gdy nawet ten poziom nie przechodzi
TARGET_FAR = SECURITY_LEVELS[DEFAULT_SECURITY_LEVEL]
MAX_ENROLLMENT_FRR = 0.10             # holdout to ok. 12 próbek: 1 pomyłka = 8.3%, 2 = 16.7%

@dataclass
class Paths:
    base: Path
    ek_out: Path
    pred_out: Path
    figs: Path


def build_paths(base_dir: Path, store_outputs: str) -> Paths:
    ek_out =EXTRACTION_DIR / "outputs"
    pred_out = TRAIN_DIR / "outputs"
    figs = pred_out / "figures"
    
    pred_out.mkdir(parents=True, exist_ok=True)
    if store_outputs == "yes":
        figs.mkdir(parents=True, exist_ok=True)
        
    return Paths(base=base_dir, ek_out=ek_out, pred_out=pred_out, figs=figs)

# FUNKCJA ZAŁADOWANIA EMBEDDINGÓW
def load_embeddings(paths: Paths, user: str, db_name: str) -> pd.DataFrame:
    db_path = paths.base / db_name
    
    # 1. Ścieżki do poszczególnych plików CSV
    user_csv = db_path / "profile" / user / f"{user}.csv"
    ff_train_csv = db_path / "ff_train.csv"
    ff_test_csv = db_path / "ff_test.csv"
    
    dfs = []
    
    # 2. Wczytanie profilu użytkownika
    if user_csv.exists():
        dfs.append(pd.read_csv(user_csv))
    else:
        raise FileNotFoundError(f"Brak pliku embeddingów użytkownika: {user_csv}")
        
    # 3. Wczytanie bazy treningowej VGG
    if ff_train_csv.exists():
        dfs.append(pd.read_csv(ff_train_csv))
    else:
        print(f"Ostrzeżenie: Brak pliku {ff_train_csv}")
        
    # 4. Wczytanie bazy testowej VGG
    if ff_test_csv.exists():
        dfs.append(pd.read_csv(ff_test_csv))
    else:
        print(f"Ostrzeżenie: Brak pliku {ff_test_csv}")
        
    if not dfs:
        raise ValueError("Nie wczytano żadnych danych embeddingów.")
        
    # Połączenie w jeden DataFrame
    df = pd.concat(dfs, ignore_index=True)
    
    req = {"sample_name", "id", "source", "backend"}
    if not req.issubset(df.columns):
        raise ValueError(f"Pliki CSV muszą zawierać kolumny: {sorted(req)}")
    df["id"] = df["id"].astype(str)
    
    return df

# FUNKCJA WYODRĘBNIANIA NAZWY KOLUMN EMBEDDINGÓW
# tj. e_000, e_001, ..., e_511.
def embedding_columns(df: pd.DataFrame) -> List[str]:
    # Wybierz kolumny numeryczne, które nie są w known meta
    meta_cols = {"sample_name", "id", "source", "backend",
                 "detector_backend", "detector_score",
                 "landmarks_found", "prep_file", "mask_file",
                 "roi_figure", "pipeline_figure"}
    emb_cols = [c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]
    return emb_cols

# FUNKCJA ZAWIERAJĄCA PRÓBKOWANIE ZRÓŻNICOWANE PO TOŻSAMOŚCIACH
def balance_embeddings(df: pd.DataFrame, user: str, ratio: float = 1.0) -> pd.DataFrame:
    df_user = df[df["source"] == "profile"]
    df_vgg = df[df["source"] != "profile"]
    
    n_user_samples = len(df_user)
    target_vgg_count = int(n_user_samples * ratio)
    
    if target_vgg_count == 0 or len(df_vgg) == 0:
        return df

    # Wybieramy po jednym zdjęciu dla każdej unikalnej osoby z VGG
    df_vgg_unique_ids = df_vgg.groupby("id").first().reset_index()
    
    # Losujemy spośród nich wymaganą liczbę tożsamości
    n_to_take = min(target_vgg_count, len(df_vgg_unique_ids))
    df_vgg_sampled = df_vgg_unique_ids.sample(n=n_to_take, random_state=RANDOM_STATE)
    
    df_balanced = pd.concat([df_user, df_vgg_sampled], ignore_index=True)
    print(f"[INFO] Wybrano {len(df_vgg_sampled)} unikalnych intruzów dla {len(df_user)} zdjęć użytkownika (ratio={ratio}).")
    return df_balanced

# FUNKCJA NORMALIZUJĄCA WEKTORY DO JEDNOSTKOWEJ DŁUGOŚCI (L2)
def l2_normalize(X: np.ndarray) -> np.ndarray:
    # Oblicza normę L2 dla każdego wiersza zabezpieczając przed dzieleniem przez zero (+ 1e-8)..
    norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return X / norm # Dzieli każdy wektor przez jego długość.


# FUNKCJA OBLICZAJĄCA SCORE'Y
# max cosine similarity + średnia z top-K podobnych
def cosine_scores_to_gallery(X_queries: np.ndarray, X_gallery: np.ndarray, topk: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    # Cosine similarity = X @ X.T (obie już L2-znormalizowane)
    sims = X_queries @ X_gallery.T
    # Score maksymalny (jeden najbardziej podobny z galerii)
    max_scores = sims.max(axis=1)
    # Score jako średnia z K najbardziej podobnych (bardziej odporna na szumy)
    k = max(1, min(topk, X_gallery.shape[0]))
    topk_scores = np.partition(sims, kth=sims.shape[1] - k, axis=1)[:, -k:]
    mean_topk = topk_scores.mean(axis=1)
    return max_scores, mean_topk

# FUNKCJA OBLICZAJĄCA WSKAŚCIKI BŁĘDÓW 
# - FAR (False Accept Rate): ile impostorów przyjmujemy błędnie (wyżej = gorzej)
# - FRR (False Reject Rate): ile użytkowników odrzucamy błędnie (wyżej = gorzej)
def compute_far_frr(scores_genuine: np.ndarray, scores_impostor: np.ndarray, thr: float) -> Tuple[float, float]:
    far = float(np.mean(scores_impostor >= thr)) if scores_impostor.size else 0.0
    frr = float(np.mean(scores_genuine < thr)) if scores_genuine.size else 0.0
    return far, frr

# FUNKCJA DOBIERANIA PROGU, GDZIE gdzie FAR = FRR (Equal Error Rate)
def choose_threshold_eer(scores_genuine: np.ndarray, scores_impostor: np.ndarray) -> Tuple[float, float, List[Tuple[float, float, float]]]:
    candidates = np.unique(np.concatenate([scores_genuine, scores_impostor]))
    if candidates.size == 0:
        return 0.0, 0.0, []
    metrics = []
    best_thr = float(candidates[0])
    best_gap = 1e9
    best_eer = 1.0
    # Testuj każdą możliwą wartość progu
    for thr in candidates:
        far, frr = compute_far_frr(scores_genuine, scores_impostor, float(thr))
        gap = abs(far - frr)  # Szukaj miejsca gdzie FAR ≈ FRR
        metrics.append((float(thr), far, frr))
        if gap < best_gap:
            best_gap = gap
            best_thr = float(thr)
            best_eer = (far + frr) / 2.0
    return best_thr, float(best_eer), metrics

# FUNKCJA DOBIERANIA PROGU, gdzie FAR <= 10^(-4)
def choose_threshold_far(
    scores_genuine: np.ndarray,
    scores_impostor: np.ndarray,
    target_far: float = 1e-4,
    allow_extrapolation: bool = True,
) -> Tuple[float, float, float]:

    genuine = np.asarray(scores_genuine, dtype=np.float64)
    impostor = np.asarray(scores_impostor, dtype=np.float64)

    if impostor.size == 0:
        raise ValueError("Brak prób impostorów do kalibracji progu.")

    required_trials = int(np.ceil(1.0 / target_far))

    # 1. Scieżka empiryczna (gdy mamy wystarczająco dużo danych)
    if impostor.size >= required_trials:
        max_false_accepts = int(np.floor(target_far * impostor.size))
        sorted_scores = np.sort(impostor)[::-1]
        if max_false_accepts == 0:
            threshold = float(np.nextafter(sorted_scores[0], np.inf))
        else:
            threshold = float(np.nextafter(sorted_scores[max_false_accepts], np.inf))
        far, frr = compute_far_frr(genuine, impostor, threshold)
        print(f"[INFO] Empiryczny próg dla FAR={target_far:.1e} wyznaczony na {impostor.size} próbkach.")
        
    # 2. Scieżka ekstrapolacji (dla norm i publikacji naukowych)
    elif allow_extrapolation:
        print(f"[WARN] Za mało próbek intruzów ({impostor.size}) dla empirycznego FAR={target_far:.1e}.")
        print(f"[INFO] Dopasowuję rozkład Gumbela (EVD) i ekstrapoluję próg...")
        
        # Dopasowanie rozkładu Gumbela do score'ów impostorów
        # Gumbel dobrze modeluje wartości maksymalne (ekstremalne) podobieństw
        loc, scale = gumbel_r.fit(impostor)
        
        # Wyznaczamy kwantyl 1 - target_far.
        # ppf (Percent Point Function) zwraca próg, dla którego CDF = 1 - target_far
        threshold = float(gumbel_r.ppf(1.0 - target_far, loc, scale))
        
        # Zabezpieczenie: jeśli ekstrapolacja daje absurdalny wynik (np. > 1.0), 
        # ustaw próg na maksymalny zaobserwowany wynik impostora + epsilon
        if threshold > 1.0 or threshold <= np.max(impostor):
            print("Ekstrapolacja dała niepewny wynik. Używam bezpiecznego maksimum empirycznego.")
            threshold = float(np.nextafter(np.max(impostor), np.inf))
            
        far, frr = compute_far_frr(genuine, impostor, threshold)
        
        # Skoryguj teoretyczne FAR (bo ekstrapolacja to model, nie pomiar)
        far = target_far 
        
    else:
        raise ValueError(
            f"FAR={target_far:.1e} wymaga co najmniej {required_trials} prób impostorów. "
            f"Dostępne próby: {impostor.size}."
        )
        
    # Zabezpieczenie minimalnego progu operacyjnego na 0.5
    threshold = max(threshold, MIN_OPERATIONAL_THRESHOLD)

    return threshold, far, frr

# def compute_adaptive_threshold(
#     scores_genuine: np.ndarray,
#     scores_impostor: np.ndarray,
#     n_profile: int,
# ) -> Tuple[float, bool]:
#     threshold_raw, _, _ = choose_threshold_eer(scores_genuine, scores_impostor)

#     if scores_genuine.size == 0:
#         return float(MIN_OPERATIONAL_THRESHOLD), True

#     genuine_low = float(np.quantile(scores_genuine, 0.10))
#     impostor_high = float(np.quantile(scores_impostor, 0.90)) if scores_impostor.size else genuine_low
#     count_bonus = min(0.03, 0.003 * max(0, n_profile - 2))

#     candidate = max(
#         float(MIN_OPERATIONAL_THRESHOLD),
#         max(float(threshold_raw), genuine_low + count_bonus, (genuine_low + impostor_high) / 2.0),
#     )
#     candidate = min(candidate, 0.95)
#     return float(candidate), bool(candidate <= MIN_OPERATIONAL_THRESHOLD)

# FUNKCJA DZIELĄCA PROFIL UŻYTKOWNIKA NA 2 CZĘŚCI
def split_profile(df_user: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(df_user) < 2:
        return df_user.copy(), pd.DataFrame()

    ordered = df_user.sort_values("sample_name").reset_index(drop=True)
    registration_pose_count = 6
    n_groups = min(registration_pose_count, len(ordered))
    group_size = len(ordered) // n_groups
    gallery_parts = []
    holdout_parts = []

    for group_idx in range(n_groups):
        start = group_idx * group_size
        end = (group_idx + 1) * group_size if group_idx < n_groups - 1 else len(ordered)
        group = ordered.iloc[start:end]
        if len(group) == 1:
            gallery_parts.append(group)
            continue

        n_holdout = max(1, int(round(len(group) * 0.2)))
        holdout_parts.append(group.iloc[-n_holdout:])
        gallery_parts.append(group.iloc[:-n_holdout])

    gallery = pd.concat(gallery_parts, ignore_index=True)
    holdout = pd.concat(holdout_parts, ignore_index=True) if holdout_parts else pd.DataFrame()
    return gallery, holdout

# FUNKCJA DO OBLICZANIA SCORE'ÓW GENUINE METODĄ LEAVE-ONE-OUT
def gallery_scores_with_exclusion(X_profile: np.ndarray) -> np.ndarray:
    # Normalizacja L2 embeddingów
    X_profile = l2_normalize(X_profile)
    
    # Obliczenie macierzy podobieństw między wszystkimi próbkami użytkownika
    # Wynik ma wymiar (N, N), gdzie sims[i, j] oznacza podobieństwo próbki i do próbki j
    sims = X_profile @ X_profile.T
    np.fill_diagonal(sims, -np.inf) # Wykluczenie porównania próbki samej ze sobą
    return np.max(sims, axis=1) # Dla każdej próbki wybieramy najwyższy score względem pozostałych próbek użytkownika otrzymując jeden wynik podobieństwa dla każdej

# FUNKCJA  ZAPISUJĄCA WSZYSTKIE METRYKI PROGU (FAR, FRR) DO CSV
def save_threshold_metrics(metrics: List[Tuple[float, float, float]], path: Path) -> None:
    # Sprawdzenie czy katalog istnieje
    path.parent.mkdir(parents=True, exist_ok=True)

    # Otwórz plik CSV w trybie zapisu
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        # Zapisz nagłówek pliku CSV
        wr.writerow(["threshold", "far", "frr"])

        # Zapisz kolejne wartości progu oraz odpowiadające im metryki FAR i FRR
        for threshold, far, frr in metrics:
            wr.writerow([
                round(float(threshold), 6), # wartość progu decyzyjnego
                round(float(far), 6),       # False Accept Rate
                round(float(frr), 6),       # False Reject Rate
            ])


# FUNKCJA DO ZAPISU HISTOGRAMU WYNIKÓW GENUINE I IMPOSTOR
def save_score_histogram(scores_genuine: np.ndarray, scores_impostor: np.ndarray, thr: float, path: Path) -> None:
    if scores_genuine.size == 0 or scores_impostor.size == 0:
        print("Pomijam histogram - brak danych genuine lub impostor.")
        return
    plt.figure(figsize=(8, 4.5)) # Utworzenie nowej figury wykresu o określonym rozmiarze
    plt.hist(scores_genuine, bins=30, alpha=0.6, label="genuine") # Histogram wyników dla prawdziwego użytkownika
    plt.hist(scores_impostor, bins=30, alpha=0.6, label="impostor") # Histogram wyników dla impostorów (powinny być mniejsze niż genuie)
    plt.axvline(thr, color="red", linestyle="-", label=f"threshold={thr:.3f}")  # Linia oznaczająca próg decyzyjny
    plt.xlabel("Cosine similarity") # Opis osi X jako wartości podobieństwa cosinusowego
    plt.ylabel("Liczba próbek")  # Opis osi Y jako liczby próbek w danym przedziale score'u
    plt.title("Rozkład wyników dla genuine i impostor")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

# FUNKCJA GENERUJĄCA KRZYWE ROC I DET
def save_roc_det(scores_genuine: np.ndarray, scores_impostor: np.ndarray, figs_dir: Path, output_dir: Path, store_files: str = "yes") -> Dict[str, float]:
    # Gdy gen_max lub impostor_for_threshold są puste (np. galeria z jedną próbką i brak holdoutu), funkcja roc_curve wyświetli błąd.
    if scores_genuine.size == 0 or scores_impostor.size == 0:
        return {"roc_auc": 0.0, "eer": 0.0, "eer_threshold": 0.0}
    
    # Połączenie wyników genuine i impostor w jedną tablicę wyników
    y_score = np.concatenate([scores_genuine, scores_impostor])

    
    # Utworzenie etykiet klas dla krzywej ROC
    y_true = np.concatenate([
        np.ones_like(scores_genuine),   # 1 oznacza próbki prawdziwego użytkownika
        np.zeros_like(scores_impostor)  # 0 oznacza próbki impostorów
    ])
    
    # Obliczenie punktów krzywej ROC
    # fpr - False Positive Rate,
    # tpr - True Positive Rate, 
    # thr - wartości progów decyzyjnych
    fpr, tpr, thr = roc_curve(y_true, y_score) 
    roc_auc = float(auc(fpr, tpr)) # Obliczenie pola pod krzywą ROC
    fnr = 1.0 - tpr # Obliczenie False Negative Rate

    # Zapis punktów krzywej ROC do pliku CSV
    if store_files == "yes":
        roc_csv = output_dir / "roc_curve.csv" # Ścieżka
        with open(roc_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["threshold", "fpr", "tpr", "fnr"])
            for i in range(len(thr)):
                wr.writerow([float(thr[i]), float(fpr[i]), float(tpr[i]), float(fnr[i])])

        # Zapis punktów krzywej DET do pliku CSV
        det_csv = output_dir / "det_curve.csv" # Ścieżka
        with open(det_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["fpr", "fnr"])
            for i in range(len(fpr)):
                wr.writerow([float(fpr[i]), float(fnr[i])])

        # Utworzenie wykresu ROC
        roc_fig = figs_dir / "roc_curve.png"
        plt.figure(figsize=(5.5, 5.5))
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Accept Rate (FAR)")
        plt.ylabel("True Accept Rate (TAR)")
        plt.title("Krzywa ROC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(roc_fig, dpi=150)
        plt.close()

        # Utworzenie wykresu DET
        det_fig = figs_dir / "det_curve.png"
        plt.figure(figsize=(5.5, 5.5))
        plt.plot(fpr, fnr)
        plt.xlabel("False Accept Rate (FAR)")
        plt.ylabel("False Reject Rate (FRR)")
        plt.title("Krzywa DET")
        plt.tight_layout()
        plt.savefig(det_fig, dpi=150)
        plt.close()

    # EER z ROC
    idx = int(np.argmin(np.abs(fpr - fnr))) # Wyznaczenie punktu EER
    eer = float((fpr[idx] + fnr[idx]) / 2.0) # Obliczenie wartości EER jako średniej z FAR i FRR w znalezionym punkcie
    eer_threshold = float(thr[idx]) # Pobranie progu odpowiadającego punktowi EER
    return {"roc_auc": roc_auc, "eer": eer, "eer_threshold": eer_threshold} # Zwrócenie najważniejszych metryk do raportu i verifier.pkl

# FUNKCJA TWORZĄCA WIZUALIZACJĘ EMBEDDINGÓW PCA
def save_pca_scatter(X: np.ndarray, labels: List[str], path: Path) -> Tuple[StandardScaler, PCA] | Tuple[None, None]:
    scaler = StandardScaler() # Standaryzacja
    Xs = scaler.fit_transform(X) # Dopasowanie skalera do danych i przekształcenie embeddingów
    n_comp = min(10, Xs.shape[0], Xs.shape[1]) # Wyznaczenie liczby komponentów PCA
    # Jeśli liczba komponentów jest mniejsza niż 2, nie da się narysować wykresu 2D
    if n_comp < 2: 
        return None, None
    
    # Utworzenie obiektu PCA
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    # Dopasowanie PCA do danych i transformacja embeddingów
    Xp = pca.fit_transform(Xs)
    # Utworzenie wykresu
    plt.figure(figsize=(7, 5))
    # Pobranie etykiet
    unique = sorted(set(labels))
    # Wygenerowanie kolorów dla poszczególnych etykiet
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(unique)))
    # Rysowanie punktów dla każdej etykiety osobno
    for color, lab in zip(colors, unique):
        idx = [i for i, v in enumerate(labels) if v == lab]
        plt.scatter(Xp[idx, 0], Xp[idx, 1], s=25, alpha=0.7, label=lab, color=color)
    plt.xlabel("PC1") # Opis osi X jako pierwszej składowej głównej
    plt.ylabel("PC2") # Opis osi Y jako drugiej składowej głównej
    plt.title("PCA embeddingów")
    if len(unique) <= 12: # Dodanie legendy tylko wtedy, gdy liczba etykiet nie jest zbyt duża
        plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return scaler, pca # Zwrócenie skalera i obiektu PCA

# GŁÓWNA FUNKCJA
def main():
    ap = argparse.ArgumentParser(description="Modelowanie i ewaluacja systemu weryfikacji twarzy")      # Utworzenie parsera argumentów uruchomieniowych
    ap.add_argument("--user", type=str, default="Jowi", help="Użytkownik referencyjny")                 # Argument określający użytkownika, dla którego budowany jest model
    ap.add_argument("--step", choices=["test", "normal"], default="normal", help="Jakiej użyć bazy danych")
    ap.add_argument("--store-outputs", choices=["yes", "no"], default="yes", help="Zapis pomocniczych artefaktów")
    ap.add_argument("--max-frr", type=float, default=MAX_ENROLLMENT_FRR, help="Maks. FRR na holdout przy rejestracji")
    ap.add_argument("--topk", type=int, default=3, help="Ile najlepszych score'ów galerii uśredniać")   # Argument określający ile najlepszych podobieństw do galerii uśredniać
    ap.add_argument("--pca", choices=["yes", "no"], default="no", help="Wykres PCA embeddingów (wolny)")
    # Odczytanie argumentów z linii komend
    args = ap.parse_args()
    db = TEST_DATA_DIR if args.step == "test" else DATA_DIR

    base_dir = Path(__file__).resolve().parents[1] # Wyznaczenie katalogu bazowego projektu
    paths = build_paths(base_dir, args.store_outputs) # Zbudowanie ścieżek do katalogów wejściowych i wyjściowych
    user_outputs_dir = paths.pred_out / "users" / args.user
    user_figs_dir = user_outputs_dir / "figures"
    if args.store_outputs == "yes":
        user_figs_dir.mkdir(parents=True, exist_ok=True)

    print(f"{SEPARATOR}\n\nEtap 2: Modelowanie i ewaluacja\n\n{SEPARATOR}\n")
    
    # Wczytanie embeddingów
    df = load_embeddings(paths, user=args.user, db_name=db) # Wczytanie embeddings.csv
    # Przycięcie bazy intruzów (5 razy więcej intruzów niż zdjęć profilowych)
    # df = balance_embeddings(df, user=args.user, ratio=5.0)
    # Wyodrębnienie kolumn zawierających embeddingi twarzy
    emb_cols = embedding_columns(df)
    if not emb_cols: # Sprawdzenie, czy znaleziono kolumny embeddingów
        raise ValueError("Brak kolumn embeddingów w embeddings.csv")

    # Przygotowanie zbiorów treningowych 
    df_user = df[(df["source"] == "profile") & (df["id"] == args.user)].copy().sort_values("sample_name") 
    df_imp_dev = df[df["source"] == "ff_train"].copy().sort_values("sample_name")  # Impostorzy do dobierania progu
    df_imp_test = df[df["source"] == "ff_test"].copy().sort_values("sample_name")   # Impostorzy do testowania

    # Sprawdzenie, czy istnieją próbki użytkownika referencyjnego
    if df_user.empty: 
        raise ValueError(f"Brak próbek użytkownika {args.user} w embeddings.csv")
    
    # Sprawdzenie, czy istnieją jakiekolwiek próbki impostorów
    if df_imp_dev.empty and df_imp_test.empty:
        raise ValueError("Brak próbek impostorów (ff_train/ff_test)")

    # Podział: galeria (60%) + holdout test genuine (40%)
    gallery_df, genuine_holdout_df = split_profile(df_user)
    X_gallery = l2_normalize(gallery_df[emb_cols].to_numpy(dtype=np.float32))

    # Obliczenie score'ów
    # Score genuine (jeśli jest holdout) - nowe próby użytkownika
    if len(genuine_holdout_df) > 0:
        X_genuine_dev = l2_normalize(genuine_holdout_df[emb_cols].to_numpy(dtype=np.float32))
        gen_max, gen_topk = cosine_scores_to_gallery(X_genuine_dev, X_gallery, topk=args.topk)
    elif len(gallery_df) >= 2:
        loo = gallery_scores_with_exclusion(gallery_df[emb_cols].to_numpy(dtype=np.float32))
        gen_max, gen_topk = loo, loo
    else:
        # Przy jednym zdjęciu profilu nie da się policzyć leave-one-out, więc
        # podajemy neutralny, ale poprawny score 1.0, aby nie wywołać błędu
        # przy zapisie raportów i kalibracji progu.
        gen_max = np.array([1.0], dtype=np.float32)
        gen_topk = np.array([1.0], dtype=np.float32)

    # Score impostorów DEV i TEST - inne osoby z VGG
    # Jeśli zbiór jest pusty, tworzona jest pusta macierz o odpowiednim wymiarze
    X_imp_dev = l2_normalize(df_imp_dev[emb_cols].to_numpy(dtype=np.float32)) if len(df_imp_dev) else np.empty((0, len(emb_cols)), dtype=np.float32)
    X_imp_test = l2_normalize(df_imp_test[emb_cols].to_numpy(dtype=np.float32)) if len(df_imp_test) else np.empty((0, len(emb_cols)), dtype=np.float32)
    
    imp_dev_max, imp_dev_topk = cosine_scores_to_gallery(X_imp_dev, X_gallery, topk=args.topk) if len(X_imp_dev) else (np.array([]), np.array([]))
    imp_test_max, imp_test_topk = cosine_scores_to_gallery(X_imp_test, X_gallery, topk=args.topk) if len(X_imp_test) else (np.array([]), np.array([]))

    # Dobieranie progu EER
    # Wybór zbioru impostorów do doboru progu
    impostor_for_threshold = imp_dev_max if len(imp_dev_max) else imp_test_max

    # threshold_raw to próg wyliczony metodą EER
    verifier_path = paths.pred_out / f"verifier_{args.user}.pkl"
    
    # --- Progi dla WSZYSTKICH poziomów zabezpieczeń naraz (suwak w ustawieniach wybiera aktywny) ---
    n_gen = len(gen_max)

    # Diagnostyka (trafia do logu): pozwala zobaczyć DLACZEGO rejestracja przeszła lub nie.
    if n_gen:
        print(f"[DIAG] genuine holdout: n={n_gen}, min={float(np.min(gen_max)):.3f}, "
              f"mediana={float(np.median(gen_max)):.3f}, max={float(np.max(gen_max)):.3f}")
        print(f"[DIAG] genuine score'y: {[round(float(x), 3) for x in gen_max]}")
    print(f"[DIAG] galeria={len(gallery_df)}, impostorzy DEV={len(impostor_for_threshold)}, "
          f"backendy próbek użytkownika={gallery_df['backend'].value_counts().to_dict()}")

    levels = {}
    for level_key, level_far in SECURITY_LEVELS.items():
        thr_c, far_c, frr_c = choose_threshold_far(
            gen_max,
            impostor_for_threshold,
            target_far=level_far,
            allow_extrapolation=True,
        )
        levels[level_key] = {
            "target_far": float(level_far),
            "threshold": float(thr_c),
            "far_dev": float(far_c),
            "frr_holdout": float(frr_c),
            "n_holdout": int(n_gen),
            "empirical": bool(len(impostor_for_threshold) >= int(np.ceil(1.0 / level_far))),
        }
        print(f"[DIAG] poziom={level_key:<8} FAR={level_far:.0e}: próg={thr_c:.3f}, FRR(holdout)={frr_c:.1%}")

    gate = levels[GATE_LEVEL]
    if n_gen and gate["frr_holdout"] > args.max_frr:
        raise RuntimeError(
            f"Rejestracja odrzucona: przy FAR={gate['target_far']:.1e} "
            f"FRR wynosi {gate['frr_holdout']:.1%}, limit to {args.max_frr:.1%}. "
            "Zbierz bardziej spójne zdjęcia twarzy."
        )

    # Wartości "domyślne" (poziom standardowy) - używane przez resztę skryptu i stary kod
    std = levels[DEFAULT_SECURITY_LEVEL]
    threshold = std["threshold"]
    far_dev = std["far_dev"]
    frr_dev = std["frr_holdout"]
    used_far = std["target_far"]
    threshold_source = "calibrated_at_target_far"

    threshold = float(threshold)
    # threshold_raw = threshold
    #threshold_was_clamped = False

    # threshold to próg finalny, który zostanie zapisany do verifier.pkl.
    # W przeciwieństwie do samego EER, próg jest teraz też zależny od aktualnego
    # rozkładu score'ów genuinnych i liczby zaakceptowanych próbek użytkownika.
    # threshold, threshold_was_clamped = compute_adaptive_threshold(
    #     gen_max,
    #     impostor_for_threshold,
    #     n_profile=int(len(df_user)),
    # )
    
    # threshold = float(threshold)
    
    #_, _, threshold_metrics = choose_threshold_eer(gen_max, impostor_for_threshold)

    roc_info = save_roc_det(
        gen_max,
        impostor_for_threshold,
        user_figs_dir,
        user_outputs_dir,
        store_files=args.store_outputs
    )
    
    # Ewaluacja na TEST set
    genuine_eval = gen_max # Genuine eval to próbki użytkownika użyte do oceny
    #  Do ewaluacji impostorów preferowany jest zbiór TEST
    impostor_eval = imp_test_max if len(imp_test_max) else imp_dev_max # Jeśli TEST nie istnieje, używany jest DEV
    # Obliczenie FAR i FRR dla wybranego progu
    far_eval, frr_eval = compute_far_frr(genuine_eval, impostor_eval, threshold)
    tar_eval = 1.0 - frr_eval # TAR to True Accept Rate, czyli 1 minus FRR

    # Zapisz wyników
    if args.store_outputs == "yes":
        # save_threshold_metrics(
        #     threshold_metrics,
        #     paths.pred_out / "threshold_metrics.csv"
        # )

        save_score_histogram(
            gen_max,
            impostor_for_threshold,
            threshold,
            user_figs_dir / "score_histogram.png"
        )
        
        scores_csv = user_outputs_dir / "scores.csv" # Ścieżka
        
        def _block(split, label, frame, smax, stopk):
            return pd.DataFrame({
                "split": split, "label": label,
                "sample_name": frame["sample_name"].to_numpy(),
                "id": frame["id"].to_numpy(),
                "source": frame["source"].to_numpy(),
                "score_max": np.asarray(smax, dtype=float),
                "score_mean_topk": np.asarray(stopk, dtype=float),
            })

        gen_frame = genuine_holdout_df if len(genuine_holdout_df) > 0 else gallery_df
        parts = [_block("dev", "genuine", gen_frame, gen_max, gen_topk)]
        if len(df_imp_dev):
            parts.append(_block("dev", "impostor", df_imp_dev, imp_dev_max, imp_dev_topk))
        if len(df_imp_test):
            parts.append(_block("test", "impostor", df_imp_test, imp_test_max, imp_test_topk))
        pd.concat(parts, ignore_index=True).to_csv(scores_csv, index=False)

        # PCA
        # Pobranie wszystkich embeddingów do analizy wizualnej PCA [PY, str(Path(TRAIN_DIR) /
        if args.pca == "yes":
            X_all = df[emb_cols].to_numpy(dtype=np.float32)
            labels = [f"{src}:{pid}" for src, pid in zip(df["source"].tolist(), df["id"].tolist())] # Utworzenie etykiet dla punktów na wykresie PCA
            scaler_pca, pca_obj = save_pca_scatter(X_all, labels, user_figs_dir / "pca_scatter.png") # Zapis wykresu PCA i pobranie obiektów skalera oraz PCA
            if scaler_pca is not None and pca_obj is not None: # Zapis obiektu StandardScaler i PCA do późniejszej analizy
                joblib.dump(scaler_pca, user_outputs_dir / "analysis_scaler.pkl")
                joblib.dump(pca_obj, user_outputs_dir / "analysis_pca.pkl")

    # Zapis verifier.pkl
    # Verifier zawiera: galerię embeddingów, próg i wszystko co potrzeba do weryfikacji
    verifier = {
        "user": args.user,
        "threshold": float(threshold),
        "threshold_target_far": float(threshold),
        "target_far": float(used_far),
        "default_level": DEFAULT_SECURITY_LEVEL,
        "levels": levels,
        "far_dev": float(far_dev),
        "frr_dev": float(frr_dev),
        "threshold_source": threshold_source,
        "min_operational_threshold": float(MIN_OPERATIONAL_THRESHOLD),
        "threshold_eer": float(roc_info["eer_threshold"]),
        "eer": float(roc_info["eer"]),
        "roc_auc": float(roc_info["roc_auc"]),
        "score_mode": "cosine_max",
        "topk": int(args.topk),
        "embedding_backend_gallery_majority": gallery_df["backend"].mode().iloc[0] if len(gallery_df) else "unknown",
        "gallery_sample_names": gallery_df["sample_name"].tolist(),
        "gallery_embeddings": X_gallery.astype(np.float32),
        "embedding_dim": int(X_gallery.shape[1]),
        "n_gallery": int(X_gallery.shape[0]),
        
    }
    verifier["created_at"] = datetime.now().isoformat()
    verifier_filename = f"verifier_{args.user}.pkl"
    joblib.dump(verifier, paths.pred_out / verifier_filename)

    # Raport końcowy
    if args.store_outputs == "yes":
        report = {
            "user": args.user,
            "step": args.step,
            "store_outputs": args.store_outputs,
            "n_profile": int(len(df_user)),
            "n_gallery": int(len(gallery_df)),
            "n_holdout_genuine": int(len(genuine_holdout_df)),
            "n_impostor_dev": int(len(df_imp_dev)),
            "n_impostor_test": int(len(df_imp_test)),
            "threshold": float(threshold),
            "threshold_target_far": float(threshold),
            "target_far": float(used_far),
        "default_level": DEFAULT_SECURITY_LEVEL,
        "levels": levels,
            "far_dev": float(far_dev),
            "frr_dev": float(frr_dev),
            "threshold_source": threshold_source,
            "min_operational_threshold": float(MIN_OPERATIONAL_THRESHOLD),
            "eer": float(roc_info["eer"]),
            "roc_auc": float(roc_info["roc_auc"]),
            "far_eval": float(far_eval),
            "frr_eval": float(frr_eval),
            "tar_eval": float(tar_eval),
        }
        with open(user_outputs_dir / "model_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        with open(user_outputs_dir / "model_report.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["metric", "value"])
            for k, v in report.items():
                if k == "levels":
                    for lvl, d in v.items():
                        for kk, vv in d.items():
                            wr.writerow([f"level_{lvl}_{kk}", vv])
                    continue
                wr.writerow([k, v])

    # Wyświetlenie wyników w konsoli
    print(f"Użytkownik referencyjny:    {args.user}")
    print(f"Liczba próbek użytkownika:  {len(df_user)}")
    print(f"Liczba próbek w galerii:    {len(gallery_df)}")
    print(f"Liczba impostorów DEV:      {len(df_imp_dev)}")
    print(f"Liczba impostorów TEST:     {len(df_imp_test)}")
    print(f"Próg FAR-target:            {threshold:.6f}")
    print(f"Poziom domyślny:           {DEFAULT_SECURITY_LEVEL} (FAR {used_far:.1e})")
    for _lvl, _d in levels.items():
        print(f"  {_lvl:<9} FAR={_d['target_far']:.0e}  próg={_d['threshold']:.4f}  FRR(holdout)={_d['frr_holdout']:.1%}")
    print(f"Źródło progu:               {threshold_source}")
    
    # if threshold_was_clamped:
    #     print("INFO: Próg EER był zbyt niski, dlatego zastosowano minimalny próg operacyjny")
    # else:
    #     print("INFO: Próg EER był wystarczający, nie zastosowano podbicia do minimum")

    print(f"EER:                        {roc_info['eer']:.6f}")
    print(f"ROC AUC:                    {roc_info['roc_auc']:.6f}")
    print(f"FAR(test):                  {far_eval:.6f}")
    print(f"FRR(test):                  {frr_eval:.6f}")
    print(SECTION)
    print(f"Zapisano verifier:          {paths.pred_out / verifier_filename}")
    if args.store_outputs == "yes":
        print(f"Zapisano raport modelu:     {user_outputs_dir / 'model_report.json'}")
        print(f"Zapisano wykresy:           {user_figs_dir}")

if __name__ == "__main__":
    main()