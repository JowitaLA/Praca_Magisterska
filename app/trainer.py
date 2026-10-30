from __future__ import annotations

import os
import sys
import queue
import math
import joblib
import threading
import subprocess
import tempfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
import logging

from resources.config import EXTRACTION_DIR, TRAIN_DIR
from .config import BASE, LOG_ROOT, python_exe
from .security_levels import LEVEL_KEYS
import shutil
from resources.config import DATA_DIR

if TYPE_CHECKING:
    from .i18n import LanguageManager

logger = logging.getLogger("FaceRegisterApp")

MIN_OPERATIONAL_THRESHOLD = 0.50
MAX_OPERATIONAL_THRESHOLD = 0.95


def _validate_verifier_thresholds(vf: Mapping) -> str | None:
    try:
        floor = float(vf.get("min_operational_threshold", MIN_OPERATIONAL_THRESHOLD))
    except (OverflowError, TypeError, ValueError):
        return "Nieprawidłowy minimalny próg operacyjny w verifierze."

    if not math.isfinite(floor) or not MIN_OPERATIONAL_THRESHOLD <= floor < MAX_OPERATIONAL_THRESHOLD:
        return "Minimalny próg operacyjny jest poza dopuszczalnym zakresem [0.5; 0.95)."

    thresholds = [("standard (pole zgodności)", vf.get("threshold"))]
    levels = vf.get("levels")
    if not isinstance(levels, Mapping):
        return "Verifier nie zawiera progów dla wszystkich poziomów bezpieczeństwa."

    for level in LEVEL_KEYS:
        entry = levels.get(level)
        if not isinstance(entry, Mapping) or "threshold" not in entry:
            return f"Verifier nie zawiera progu dla poziomu {level}."
        thresholds.append((level, entry["threshold"]))

    for level, raw_threshold in thresholds:
        try:
            threshold = float(raw_threshold)
        except (OverflowError, TypeError, ValueError):
            return f"Nieprawidłowy próg dla poziomu {level}."

        operational_threshold = max(threshold, floor)
        if not math.isfinite(threshold) or not MIN_OPERATIONAL_THRESHOLD <= operational_threshold < MAX_OPERATIONAL_THRESHOLD:
            return (
                f"Próg dla poziomu {level} ({threshold}) jest poza "
                "dopuszczalnym zakresem operacyjnym [0.5; 0.95)."
            )

    return None


def _script_cmd(script, *args) -> list[str]:
    # Polecenie uruchamiające skrypt pomocniczy (extraction.py / training.py).
    # W wersji .exe nie ma interpretera Pythona ani .venv, więc skrypt jest
    # wykonywany przez ten sam plik .exe w trybie roboczym (patrz main.py: --worker).
    if getattr(sys, "frozen", False):
        return [sys.executable, "--worker", str(script), *args]
    return [python_exe(), str(script), *args]

class ModelTrainer:
    def __init__(
        self,
        log_queue: queue.Queue[str],
        log_fn,
        i18n: LanguageManager | None = None
    ):
        self.log_queue = log_queue
        self.log_fn = log_fn
        self.i18n = i18n
        
    # Funckcja do tłumaczenia tekstu
    def _tr(self, key: str, default: str) -> str:
        if self.i18n:
            return self.i18n.tr(key, default)
        return default

    # Funkcja do rozpoczęcia procesu uczenia w osobnym wątku
    def start_training(self, user: str, reason: str = "manual") -> threading.Thread:
        if reason == "after_auth":
            msg = self._tr("log_retrain_after_auth", "Rozpoczęto ponowne uczenie modelu po zaakceptowaniu twarzy")
        else:
            msg = self._tr("log_start_training_user", "Start uczenia użytkownika")
        
        self.log_fn(msg)

        t = threading.Thread(target=self._train_user_worker, args=(user, reason), daemon=True)
        t.start()
        return t

    # Funkcja do ponownego wczytania progu weryfikatora po zakończeniu uczenia
    def reload_verifier_threshold(self, user: str, reason: str = "manual") -> str:
        verifier_path = TRAIN_DIR / "outputs" / f"verifier_{user}.pkl"
        if not verifier_path.exists():
            msg = f"Nie znaleziono pliku verifier_{user}.pkl po uczeniu."
            self.log_fn(msg)
            return msg

        try:
            vf = joblib.load(verifier_path)
            threshold = float(vf.get("threshold", 0.0))
            threshold_raw = float(vf.get("threshold_eer", 0.0))
            model_user = vf.get("user", "")

            if model_user and model_user != user:
                return f"Błąd: Verifier jest dla innego użytkownika: {model_user}"

            validation_error = _validate_verifier_thresholds(vf)
            if validation_error:
                self.log_fn(validation_error)

                if reason == "manual":
                    user_dir = DATA_DIR / "profile" / user
                    if user_dir.exists():
                        shutil.rmtree(user_dir, ignore_errors=True)
                        self.log_fn(f"Usunięto folder użytkownika: {user_dir}")

                    if verifier_path.exists():
                        verifier_path.unlink()
                        self.log_fn(f"Usunięto wadliwy model: {verifier_path}")

                return f"Nieprawidłowa konfiguracja progów bezpieczeństwa: {validation_error}"

            msg_thresh = f"Próg po retrainie dla {user}: {threshold:.6f}"
            self.log_fn(msg_thresh)

            self.log_fn(f"Próg EER po retrainie dla {user}: {threshold_raw:.6f}")
            return "ok"

        except Exception as exc:
            return f"Nie udało się odczytać pliku verifier: {exc}"
        
    def _resolve_extraction_mode(self, user: str, reason: str) -> str:
        if reason != "after_auth":
            return "all"

        profile_dir = DATA_DIR / "profile" / user
        if not profile_dir.exists():
            self.log_fn(f"Brak katalogu profilu użytkownika {user}; fallback do pełnego retrainu po autoryzacji.")
            return "all"

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        has_profile_images = any(
            p.is_file() and p.suffix.lower() in image_extensions
            for p in profile_dir.iterdir()
        )

        if not has_profile_images:
            self.log_fn(f"Brak zdjęć profilu użytkownika {user}; fallback do pełnego retrainu po autoryzacji.")
            return "all"

        return "profile"

    # Funkcja do wykonywania procesu uczenia użytkownika w osobnym wątku
    def _train_user_worker(self, user: str, reason: str):
        extraction_mode = self._resolve_extraction_mode(user, reason)
        verifier_path = TRAIN_DIR / "outputs" / f"verifier_{user}.pkl"
        verifier_backup = None

        if reason == "after_auth":
            if not verifier_path.is_file():
                msg = f"Nie znaleziono poprzedniego modelu weryfikatora: {verifier_path}"
                self.log_fn(msg)
                self.log_queue.put(("error", msg))
                return
            try:
                with tempfile.NamedTemporaryFile(
                    dir=verifier_path.parent,
                    prefix=f"{verifier_path.name}.",
                    suffix=".bak",
                    delete=False,
                ) as handle:
                    verifier_backup = Path(handle.name)
                shutil.copy2(verifier_path, verifier_backup)
            except OSError as exc:
                msg = f"Nie udało się zabezpieczyć poprzedniego modelu {verifier_path}: {exc}"
                self.log_fn(msg)
                self.log_queue.put(("error", msg))
                if verifier_backup is not None:
                    verifier_backup.unlink(missing_ok=True)
                return

        def restore_previous_verifier() -> str | None:
            nonlocal verifier_backup
            if verifier_backup is None:
                return None
            try:
                os.replace(verifier_backup, verifier_path)
                verifier_backup = None
                self.log_fn(f"Przywrócono poprzedni model weryfikatora: {verifier_path}")
                return None
            except OSError as exc:
                return (
                    f"Nie udało się przywrócić poprzedniego modelu z "
                    f"{verifier_backup}: {exc}"
                )

        def remove_verifier_backup() -> None:
            nonlocal verifier_backup
            if verifier_backup is not None:
                try:
                    verifier_backup.unlink()
                except OSError as exc:
                    self.log_fn(f"Nie udało się usunąć kopii modelu {verifier_backup}: {exc}")
                verifier_backup = None

        def cleanup_failed_registration():
            if reason != "manual":
                return
            user_dir = DATA_DIR / "profile" / user
            user_csv = user_dir / f"{user}.csv"
            verifier_path = TRAIN_DIR / "outputs" / f"verifier_{user}.pkl"
            for path in (user_csv, verifier_path):
                try:
                    if path.exists():
                        path.unlink()
                except OSError as exc:
                    self.log_fn(f"Nie udało się usunąć artefaktu rejestracji {path}: {exc}")
            if user_dir.exists():
                shutil.rmtree(user_dir, ignore_errors=True)

        commands = [
            _script_cmd(EXTRACTION_DIR / "extraction.py", extraction_mode, "--user", user, "--step", "normal", "--store-outputs", "no", "--delete-photos", "yes", "--gaussian-blur", "no", "--median-filter", "no", "--clahe", "no", "--face-mask", "no"),
            _script_cmd(TRAIN_DIR / "training.py", "--user", user, "--step", "normal", "--store-outputs", "no"),
        ]

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONPATH"] = str(BASE)

        error_detail = None  # powód przerwania zgłoszony przez skrypt (np. odrzucona rejestracja)

        for cmd in commands:
            header = f"\n{'=' * 100}\nCMD: {' '.join(cmd)}\n{'=' * 100}\n"
            logger.info(header.strip())
            self.log_queue.put(header)

            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                    bufsize=1,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            except OSError as exc:
                err_msg = f"Nie udało się uruchomić procesu uczenia: {exc}"
                cleanup_failed_registration()
                restore_error = restore_previous_verifier()
                if restore_error:
                    err_msg = f"{err_msg}\n{restore_error}"
                self.log_queue.put(("error", err_msg))
                return

            assert proc.stdout is not None
            for line in proc.stdout:
                line_clean = line.rstrip("\n")
                logger.info(line_clean)
                self.log_queue.put(line_clean)
                if line_clean.startswith("NURAFACE_ERROR::"):
                    # Linia z komunikatem zapisana przez tryb roboczy w main.py (wersja .exe)
                    parts = line_clean.split("::", 2)
                    if len(parts) == 3:
                        _, etype, emsg = parts
                        error_detail = emsg if etype == "RuntimeError" else f"{etype}: {emsg}"
                elif line_clean.startswith("RuntimeError: "):
                    # Uruchomienie z kodu źródłowego (bez trybu roboczego): ostatnia linia tracebacku
                    error_detail = line_clean[len("RuntimeError: "):].strip()

            code = proc.wait()
            footer = f"[EXIT CODE] {code}"
            logger.info(footer)
            self.log_queue.put(footer)

            if code != 0:
                # Gdy skrypt przerwał z czytelnym powodem (np. "Rejestracja odrzucona: ..."),
                # pokazujemy go w oknie zamiast ogólnego komunikatu; pełny traceback zostaje w logu.
                err_msg = error_detail or self._tr("log_training_failed", "BŁĄD: uczenie przerwane")
                logger.error(err_msg)
                cleanup_failed_registration()
                restore_error = restore_previous_verifier()
                if restore_error:
                    err_msg = f"{err_msg}\n{restore_error}"
                self.log_queue.put(("error", err_msg))
                return

        # Po zakończeniu wszystkich komend poprawnie sprawdzamy próg
        result = self.reload_verifier_threshold(user, reason=reason)
        
        if result == "ok":
            remove_verifier_backup()
            # WYŚWIETLAMY KOMUNIKAT SUKCESU TYLKO JEŚLI TO NIE JEST "after_auth"
            if reason != "after_auth":
                msg_success = self._tr(
                    "log_training_success_registered",
                    "Użytkownik został zarejestrowany i model został utworzony"
                )
                logger.info(msg_success)
                self.log_queue.put(("success", msg_success))
            else:
                # W tle zapisujemy do loggera (bez wyrzucania powiadomienia do UI)
                logger.info("Model i profil zaktualizowany po autoryzacji.")
        else:
            cleanup_failed_registration()
            restore_error = restore_previous_verifier()
            if restore_error:
                result = f"{result}\n{restore_error}"
            self.log_queue.put(("error", result))