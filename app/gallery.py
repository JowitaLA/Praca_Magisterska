from __future__ import annotations

import csv
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, TYPE_CHECKING

from .config import safe_user_name, MAX_GALLERY_EMBEDDINGS, UPDATE_GALLERY_MARGIN

if TYPE_CHECKING:
    from .i18n import LanguageManager


def cosine_scores(query: np.ndarray, gallery: np.ndarray, topk: int = 3) -> tuple[float, float]:
    sims = gallery @ query.reshape(-1, 1)
    sims = sims.ravel()
    score_max = float(np.max(sims))
    k = max(1, min(topk, len(sims)))
    score_topk = float(np.mean(np.sort(sims)[-k:]))
    return score_max, score_topk


def append_embedding_row_to_csv(
    embeddings_csv: Path,
    user: str,
    embedding: np.ndarray,
    metadata: Dict[str, object],
    sample_name: str | None = None,
) -> None:
    embeddings_csv.parent.mkdir(parents=True, exist_ok=True)
    emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
    dim = int(emb.shape[0])

    default_header = [
        "sample_name", "id", "source", "backend", "detector_backend",
        "detector_score", "landmarks_found", "prep_file", "mask_file",
        "roi_figure", "pipeline_figure",
    ] + [f"e_{i:03d}" for i in range(dim)]

    existing_rows: List[List[object]] = []
    header = list(default_header)

    if embeddings_csv.exists():
        with open(embeddings_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                header = list(default_header)
            for row in reader:
                existing_rows.append(row)

    if not header or len(header) != 11 + dim:
        header = list(default_header)

    sample_name = sample_name or f"profile__{safe_user_name(user)}__auth__{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    row = [
        sample_name, user, "profile",
        metadata.get("backend", "arcface"), metadata.get("detector_backend", ""),
        metadata.get("detector_score", ""), int(metadata.get("landmarks_found", 0)),
        metadata.get("prep_file", ""), metadata.get("mask_file", ""),
        metadata.get("roi_figure", ""), metadata.get("pipeline_figure", ""),
    ] + [float(x) for x in emb.tolist()]

    with open(embeddings_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for existing_row in existing_rows:
            writer.writerow(existing_row)
        writer.writerow(row)

def update_user_gallery(
    vf: dict,
    emb: np.ndarray,
    user: str,
    score_max: float,
    threshold: float,
    log_fn=print,
    i18n: LanguageManager | None = None,
) -> bool:
    tr = i18n.tr if i18n else (lambda k, d: d)

    if score_max < threshold + UPDATE_GALLERY_MARGIN:
        log_fn(tr("log_gallery_not_added_margin", "Nie dodano embeddingu do galerii, score był zbyt blisko progu"))
        return False

    gallery = np.asarray(vf["gallery_embeddings"], dtype=np.float32)
    emb = emb.astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-8)

    gallery = np.vstack([gallery, emb.reshape(1, -1)])
    sample_names = list(vf.get("gallery_sample_names", []))
    new_name = f"live_auth__{user}__{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    sample_names.append(new_name)

    if gallery.shape[0] > MAX_GALLERY_EMBEDDINGS:
        gallery = gallery[-MAX_GALLERY_EMBEDDINGS:]
        sample_names = sample_names[-MAX_GALLERY_EMBEDDINGS:]

    vf["gallery_embeddings"] = gallery.astype(np.float32)
    vf["gallery_sample_names"] = sample_names
    vf["n_gallery"] = int(gallery.shape[0])
    vf["last_gallery_update"] = datetime.now().isoformat(timespec="seconds")

    log_fn(tr("log_gallery_added_new", "Dodano nowy embedding do galerii użytkownika"))
    log_fn(tr("log_gallery_count_fmt", "Aktualna liczba embeddingów w galerii: {count}").format(count=vf['n_gallery']))
    return True