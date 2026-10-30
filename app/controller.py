from __future__ import annotations

import os
import time
import queue
import shutil
import joblib
import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
import threading
import secrets
from tkinter import messagebox
from PIL import Image, ImageTk
from pathlib import Path
from datetime import datetime
from tkinter import simpledialog, Toplevel, Label
from .authenticator import AuthenticatorManager
from .credential_hash import hash_credential, verify_credential

from resources.config import EXTRACTION_DIR, TRAIN_DIR, VERIFICATION_DIR
from extraction.extraction import Settings, FaceParams, embed_image

from .config import (
    BASE, PROFILE_ROOT, POSES, SHOTS_PER_POSE,
    AUTO_STABLE_FRAMES, AUTO_CAPTURE_DELAY, IMG_EXT, safe_user_name, LOG_ROOT, APP_NAME
)
from collections import deque
from .camera import list_connected_cameras
from .face_geometry import FaceGeometryAnalyzer
from .liveness import LivenessChecker
from .gallery import cosine_scores, update_user_gallery
from .attendance import append_attendance_record
from .security_levels import DEFAULT_LEVEL, normalize_level, resolve_threshold, estimate_frr
from .trainer import ModelTrainer
from .views import MainView
from .i18n import LanguageManager
from .config_manager import ConfigManager
import logging
logger = logging.getLogger("FaceRegisterApp")

class FaceRegisterApp:
    def apply_theme(self, theme: str):
        theme_name = "litera" if theme == "light" else "superhero"
        style = getattr(self.root, "style", None)
        if style is not None:
            style.theme_use(theme_name)

    def __init__(self, root: tk.Tk):
        self.root = root
        self.config = ConfigManager()
        # Zsynchronizowanie wpisu autostartu w rejestrze z ustawieniem w settings.ini
        # (także po przeniesieniu aplikacji w inne miejsce).
        try:
            self.config.apply_autostart()
        except Exception:
            logger.exception("Nie udało się zsynchronizować autostartu")

        saved_theme = self.config.get("App", "theme", fallback="dark")
        self.apply_theme(saved_theme)
        
        self.auth_attempts = 0
        self.last_auth_time = time.time()
        self.is_locked = False
        self.is_authorized = False
        
        self.frame_buffer = deque(maxlen=10)
        self.frame_counter = 0
        
        self.is_authenticated = False
        self.failed_silent_attempts = 0
        self.silent_timer_id = None
        # Konfiguracja czasów (w milisekundach)
        self.CHECK_INTERVAL_MS = 5 * 60 * 1000  # 5 minut (lub 1h)
        self.RETRY_DELAY_MS = 2000  # 2 sekundy zwłoki na drugą próbę
        
        lang_dir = BASE / "resources" / "lang"
        
        self.auth_in_progress = False
        self.liveness_in_progress = False
        self.auth_generation = 0
        self.fallback_auth_open = False
        self.password_prompt_open = False
        self.password_verified = False
        self.auth_after_id = None
        self.authentication_started_at = None
        self.current_login_attempts = 0
        self.ui_interaction_until = 0.0
        self._face_continuous_start = 0

        start_ui_lang = self.config.get("App", "ui_lang", fallback="pl_PL")
        start_log_lang = self.config.get("App", "log_lang", fallback="en_US")
        
        self.ui_i18n = LanguageManager(lang_dir=lang_dir, default_lang=start_ui_lang)
        self.log_i18n = LanguageManager(lang_dir=lang_dir, default_lang=start_log_lang)
        
        self.root.title(self.ui_i18n.tr("app_title", APP_NAME))

        self.root.geometry("1080x760")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._closing = False
        self._close_pending = False  # zamknięcie odroczone do końca uczenia modelu
        self.cap: cv2.VideoCapture | None = None
        self.current_camera_idx = 0
        self.current_frame = None

        # Kamera jest włączana tylko na czas autoryzacji / rejestracji / cichej weryfikacji
        self._camera_opening = False
        self._camera_retry_at = 0.0
        self._camera_warmup_until = 0.0
        self._silent_capture_lock = threading.Lock()

        self.current_pose_idx = 0
        self.current_shot_idx = 0
        self.stable_count = 0
        self.last_capture_time = 0.0

        self.user_dir: Path | None = None
        self.saved_files: list[Path] = []

        self.training_queue: queue.Queue[str] = queue.Queue()
        self.training_running = False
        self.training_reason = "manual"  # "manual" (rejestracja) albo "after_auth" (retrain po autoryzacji)
        self.training_anim_id = None
        self.training_anim_step = 0

        self.cameras = list_connected_cameras(i18n=self.ui_i18n)
        self.camera_name_to_index = {name: idx for idx, name in self.cameras}

        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.45
        )
        self.preview_face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.45
        )
        
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.45, min_tracking_confidence=0.45,
        )

        # Dedykowana instancja do analizy liveness w tle (zapobiega wyciekom pamięci!)
        # self.liveness_mesh = mp.solutions.face_mesh.FaceMesh(
        #     static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        #     min_detection_confidence=0.45, min_tracking_confidence=0.45,
        # )
        self.liveness_lock = threading.Lock()

        self.geometry_analyzer = FaceGeometryAnalyzer(
            self.face_detector, self.face_mesh, i18n=self.ui_i18n
        )
        self.liveness_checker = LivenessChecker(
            self.face_mesh, i18n=self.ui_i18n
        )
        
        self.auth_manager = AuthenticatorManager(app_name=APP_NAME)

        callbacks = {
            "on_camera_change": self.change_camera,
            "on_register": self.start_registration,
            "on_authorize": self.start_authorization,
            "on_capture": self.capture_pose,
            "on_train": self.train_user,
            "on_close": self.on_close,
            "on_language_changed": self.handle_ui_language_change,
            "on_log_language_changed": self.handle_log_language_change,
            "on_theme_changed": self.handle_theme_change,
            "on_generate_qr": self.generate_and_show_qr,
            "on_generate_offline_code": self.generate_offline_code,
            "on_delete_user": self.delete_user,
            "on_fallback_auth_clicked": self.open_fallback_auth_window,
            "on_restore_from_tray": self.handle_restore_from_tray,
            "on_view_changed": self.handle_view_changed,
            "on_security_level_change": self.handle_security_level_change,
            "get_security_level_hint": self.get_security_level_hint
        }

        self.view = MainView( 
            self.root, self.cameras, self.ui_i18n, callbacks, config_manager=self.config
        )

        # Odliczanie cichej weryfikacji ma startować dopiero po minimalizacji do tła
        minimize_btn = getattr(self.view, "btn_minimize", None)
        if minimize_btn is not None:
            minimize_btn.configure(command=self._on_minimize_clicked)

        self.root.bind_all(
            "<ButtonPress>", self._on_global_click, add="+"
        )

        self.view.show_frame("SettingsView")
        
        self.trainer = ModelTrainer(self.training_queue, self.log, i18n=self.log_i18n)

        # Inicjalizacja buforów podręcznych dla płynnego renderowania ramek
        self._cached_auth_det = (False, None)
        self._cached_reg_data = (False, self.ui_i18n.tr("registration_instruction", "Wpisz nową nazwę konta i wciśnij rejestruj"), None, 0.0, 0.0, 0.0, False, False)
        self._face_detection_lock = threading.Lock()
        self._face_detection_event = threading.Event()
        self._face_detection_stop = threading.Event()
        self._face_detection_frame = None
        self._face_detection_thread = threading.Thread(
            target=self._face_detection_worker,
            name="face-detection-worker",
            daemon=True,
        )
        self._face_detection_thread.start()

        saved_camera_name = self.config.get("App", "camera_name", fallback="")
        if self.cameras:
            pref_idx = None
            pref_name = None
            
            for idx, name in self.cameras:
                if name == saved_camera_name:
                    pref_idx = idx
                    pref_name = name
                    break
                    
            if pref_idx is None:
                pref_idx = 0 if any(idx == 0 for idx, _ in self.cameras) else self.cameras[0][0]
                pref_name = next(name for idx, name in self.cameras if idx == pref_idx)

            self.view.camera_var.set(pref_name)
            # Kamera nie jest otwierana na starcie - włączy się dopiero w widoku autoryzacji/rejestracji
            self.current_camera_idx = pref_idx
        else:
            self.log_tr("no_camera_detected", "Nie wykryto żadnej kamery")
                
        self.update_camera_frame()
        self.drain_training_log()
        self._overlay_watchdog()
        self.current_pose_idx = 0

        self.root.bind_all("<ButtonPress>", self._on_global_click, add="+")
    
    def log_tr(self, key: str, default: str, **kwargs):
        text = self.log_i18n.tr(key, default)
        if kwargs:
            text = text.format(**kwargs)
        self.log(text)

    @staticmethod
    def _safe_preview_to_image(frame):
        if frame is None or not isinstance(frame, np.ndarray):
            return None
        if frame.size == 0 or frame.ndim != 3:
            return None
        try:
            preview = cv2.resize(frame, (640, 480))
            preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            return Image.fromarray(preview)
        except (TypeError, ValueError, RuntimeError):
            return None

    def _start_authentication_timer(self):
        if self.authentication_started_at is None:
            self.authentication_started_at = time.perf_counter()

    def _log_authentication_duration(self):
        started_at = self.authentication_started_at
        self.authentication_started_at = None
        if started_at is None:
            return 0.0
        duration = time.perf_counter() - started_at
        self.log_tr(
            "log_auth_duration_fmt",
            "Czas zalogowania: {duration:.2f} sec",
            duration=duration,
        )
        return duration
    
    def on_login_success(self):
        self.is_authenticated = True
        self.failed_silent_attempts = 0
        self.start_silent_timer(self.CHECK_INTERVAL_MS)

    def start_silent_timer(self, delay_ms: int):
        # Funkcja odpowiedzialna za zaplanowanie cichego sprawdzenia obecności.
        if getattr(self, "silent_timer_id", None):
            self.root.after_cancel(self.silent_timer_id)
        self.silent_timer_id = self.root.after(int(delay_ms), self._perform_silent_check)

    def cancel_silent_timer(self):
        timer_id = getattr(self, "silent_timer_id", None)
        if timer_id is not None:
            try:
                self.root.after_cancel(timer_id)
            except (ValueError, tk.TclError):
                pass
            self.silent_timer_id = None
        
    # Funkcja tłumacząca ustawienia sprawdzania obecności na milisekundy
    def get_security_level(self) -> str:
        # Aktywny poziom zabezpieczeń (docelowy FAR) wybrany w ustawieniach.
        return normalize_level(self.config.get("App", "security_level", fallback=DEFAULT_LEVEL))

    def handle_security_level_change(self, level_key: str):
        # Zapis wartości robi widok (_save_setting); tutaj tylko wpis do logu.
        self.log_tr(
            "log_security_level_changed_fmt",
            "Zmieniono poziom zabezpieczeń na: {level}",
            level=self.log_i18n.tr(f"security_{level_key}", level_key),
        )

    def get_security_level_hint(self, level_key: str) -> str:
        # Szacunek odrzuceń właściciela dla zalogowanego profilu (z danych testowych rejestracji).
        user = getattr(self, "current_auth_user", None)
        if not user:
            return ""
        try:
            vf = joblib.load(TRAIN_DIR / "outputs" / f"verifier_{user}.pkl")
        except Exception:
            return ""
        est = estimate_frr(vf, level_key)
        if est is None:
            return self.ui_i18n.tr(
                "security_hint_legacy",
                "This profile was created with an older version. Re-train it to calculate the estimated rejection rate.",
            )
        frr, n = est
        return self.ui_i18n.tr(
            "security_hint_frr_fmt",
            "Estimated for your profile: about {frr:.0f}% owner rejection rate ({samples} test photos).",
        ).format(frr=frr * 100, samples=n)

    def _log_attendance(self, user, method, duration, attempts, level):
        # Wpis do attendance_log.csv wraz z poziomem zabezpieczeń użytym przy logowaniu.
        try:
            append_attendance_record(
                user, method=method, time_value=duration, attempts=attempts, security_level=level
            )
        except TypeError:
            # attendance.py w starej wersji (bez kolumny poziomu) - zapis jak dotychczas
            append_attendance_record(user, method=method, time_value=duration, attempts=attempts)

    def get_presence_interval_ms(self) -> int:
        val = self.config.get("App", "presence_check", fallback="Brak")
        mapping = {
            "Brak": 0,
            "Never": 0,
            "Co minutę": 60 * 1000,
            "Every minute": 60 * 1000,
            "Co 5 minut": 5 * 60 * 1000,
            "Every 5 minutes": 5 * 60 * 1000,
            "Co 10 minut": 10 * 60 * 1000,
            "Every 10 minutes": 10 * 60 * 1000,
            "Co 30 minut": 30 * 60 * 1000,
            "Every 30 minutes": 30 * 60 * 1000,
            "Co 1 godzinę": 60 * 60 * 1000,
            "Every hour": 60 * 60 * 1000,
            "Co 2 godziny": 120 * 60 * 1000,
            "Every 2 hours": 120 * 60 * 1000,
            "Co 5 godzin": 300 * 60 * 1000,
            "Every 5 hours": 300 * 60 * 1000,
        }
        return mapping.get(val, 0)

    def _is_minimized(self) -> bool:
        # Funkcja sprawdzająca, czy aplikacja jest zminimalizowana do tła (ukryta w zasobniku).
        try:
            return self.root.state() == "withdrawn"
        except tk.TclError:
            return False

    def _on_minimize_clicked(self):
        # Funkcja odpowiedzialna za minimalizację do tła i dopiero wtedy start odliczania cichej weryfikacji.
        self.view.lock_manager.minimize_to_tray()
        self.failed_silent_attempts = 0
        interval = self.get_presence_interval_ms()
        if self.is_authorized and interval > 0:
            self.start_silent_timer(interval)

    def _perform_silent_check(self):
        self.silent_timer_id = None
        if (
            not getattr(self, "is_authorized", False)
            or getattr(self.view, "current_view", "") != "SettingsView"
            or not self._is_minimized()
            or self._close_pending
        ):
            return

        # Model jeszcze się uczy - pierwsza próba dopiero po jego nauczeniu.
        # Czas jest przedłużany (sprawdzamy co 2 s), aż uczenie się skończy.
        if self.training_running:
            self.start_silent_timer(2000)
            return

        user = getattr(self, "current_auth_user", None)

        # Bez zapisanego użytkownika traktujemy jako niepowodzenie
        if not user:
            self._handle_silent_check_result(False)
            return

        self.log_tr(
            "log_silent_check_started",
            "Rozpoczynam cichą weryfikację...",
        )
        # Kamera jest wyłączona w Ustawieniach - na czas sprawdzenia włączamy ją na chwilę w tle
        threading.Thread(
            target=self._silent_capture_worker,
            args=(user,),
            daemon=True
        ).start()

    def _silent_capture_worker(self, user: str):
        # Funkcja odpowiedzialna za krótkie włączenie kamery, zrobienie klatki i wyłączenie kamery.
        if not self._silent_capture_lock.acquire(blocking=False):
            return
        try:
            frame = self._grab_single_frame(self.current_camera_idx)
        except Exception:
            frame = None
        finally:
            self._silent_capture_lock.release()

        if frame is None:
            self.root.after(0, self._handle_silent_check_result, False)
            return

        self._verify_frame_silently_worker(frame, user)

    def _verify_frame_silently_worker(self, frame: np.ndarray, user: str):
        try:
            vf = joblib.load(TRAIN_DIR / "outputs" / f"verifier_{user}.pkl")
            
            # Zapis tymczasowej klatki dla embed_image
            tmp_img = VERIFICATION_DIR / "outputs" / "_silent_auth" / f"tmp_{time.time()}.jpg"
            tmp_img.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(tmp_img), frame)

            settings = Settings()
            fpar = FaceParams()
            # Ekstrakcja na niewidce - bez info na UI
            emb, _ = embed_image(
                tmp_img, BASE, settings=settings, fpar=fpar, prefix="silent",
                step="normal",
            )
            
            if tmp_img.exists():
                try:
                    tmp_img.unlink()
                except OSError:
                    pass

            emb = emb.astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            
            gallery = np.asarray(vf["gallery_embeddings"], dtype=np.float32)
            threshold, _used_level, _legacy = resolve_threshold(vf, self.get_security_level())
            
            score_max, _ = cosine_scores(emb, gallery, topk=int(vf.get("topk", 3)))
            is_valid = score_max >= threshold

            # Przekazanie wyniku do głównego wątku (bezpieczne dla tkintera)
            self.root.after(0, self._handle_silent_check_result, is_valid)
        except Exception as e:
            self.log_tr(
                "log_silent_check_error_fmt",
                "Błąd cichej weryfikacji obecności: {error}",
                error=e,
            )
            self.root.after(0, self._handle_silent_check_result, False)

    def _handle_silent_check_result(self, is_valid: bool):
        if not getattr(self, "is_authorized", False) or self._close_pending:
            return

        # Użytkownik przywrócił okno albo zmienił widok w trakcie sprawdzania (kamera mogła być zajęta) -
        # wynik nie jest wiarygodny. Odliczanie startuje od nowa dopiero po kolejnej minimalizacji.
        if getattr(self.view, "current_view", "") != "SettingsView" or not self._is_minimized():
            return

        if is_valid:
            # Wszystko gra (twarz zautoryzowana) - resetujemy błędy i liczymy czas od nowa
            self.log_tr(
                "log_silent_check_success",
                "Cicha weryfikacja powiodła się.",
            )
            self.failed_silent_attempts = 0
            
            self.failed_silent_attempts = 0
            interval = self.get_presence_interval_ms()
            if interval > 0:
                self.start_silent_timer(interval)
        else:
            self.failed_silent_attempts = getattr(self, "failed_silent_attempts", 0) + 1
            if self.failed_silent_attempts < 3:
                # Nie udało się (1. lub 2. błąd), za 5 sekund sprawdzamy znowu
                self.log_tr(
                    "log_silent_check_retry_fmt",
                    "Nieudana cicha weryfikacja obecności (próba {attempt}). Ponowna próba za {delay_seconds} sekund.",
                    attempt=self.failed_silent_attempts,
                    delay_seconds=5,
                )
                self.start_silent_timer(5000)
            else:
                # Trzeci błąd - wylogowujemy, blokujemy system, wyświetlamy autoryzację
                self.log_tr(
                    "log_silent_check_locked",
                    "Trzy nieudane ciche weryfikacje obecności. Wylogowano i zablokowano system.",
                )
                self.root.after(0, self.revoke_access_and_lock)

    def revoke_access_and_lock(self):
        if self._close_pending:
            return
        self.is_authorized = False
        self.current_auth_user = None
        self.failed_silent_attempts = 0

        if getattr(self, "silent_timer_id", None):
            self.cancel_silent_timer()

        if hasattr(self.view, "lock_manager"):
            # Aplikacja wraca na wierzch - ikona w zasobniku nie jest już potrzebna
            self.view.lock_manager.stop_tray()
            # Okno głównego ekranu musi być widoczne ZANIM powstaną czarne nakładki
            # na dodatkowych monitorach (przy ukrytym oknie zostawał sam czarny ekran).
            self._bring_window_to_front()
            self.view.lock_manager.lock_all()

        if hasattr(self.view, "set_authorized_mode"):
            self.view.set_authorized_mode(False)

        try:
            self.view.show_frame("AuthView")
        except Exception:
            pass

    def _bring_window_to_front(self):
        # Przywraca główne okno z ukrycia (tray) na pełny ekran, na wierzch.
        try:
            self.root.deiconify()
            self.root.attributes("-fullscreen", True)
            self.root.attributes("-topmost", True)
            self.root.lift()
            self.root.focus_force()
            self.root.update_idletasks()
        except tk.TclError:
            pass

    def _overlay_watchdog(self):
        # Zabezpieczenie: czarne nakładki na dodatkowych monitorach nie mogą istnieć,
        # gdy główne okno jest ukryte - inaczej użytkownik zostaje z czarnym ekranem
        # bez możliwości odblokowania (do restartu aplikacji).
        if self._closing:
            return
        try:
            lm = getattr(self.view, "lock_manager", None)
            if lm is not None and getattr(lm, "overlays", None) and self._is_minimized():
                lm.unlock_all()
                logger.info("Watchdog: zdjęto nakładki, bo główne okno jest ukryte.")
        except Exception:
            logger.exception("overlay watchdog failed")
        finally:
            if not self._closing:
                try:
                    self.root.after(1000, self._overlay_watchdog)
                except tk.TclError:
                    pass

    def handle_restore_from_tray(self):
        keep_authorized = self.config.getboolean(
            "App", "keep_authorized_on_restore", fallback=False
        )
        if self.is_authorized and not keep_authorized:
            self.revoke_access_and_lock()
            return
        self.view.show_frame("SettingsView")

        if hasattr(self.view, "msg_label"):
            if self.training_running:
                self._show_training_block_message()
            else:
                self.view.msg_label.configure(
                    text=self.ui_i18n.tr("face_missing_logged_out", "Face missing. Signed out."),
                    foreground="red",
                )

        try:
            self.root.deiconify()
            self.root.attributes('-topmost', True)
            self.root.lift()
            self.root.focus_force()
            self.root.update_idletasks()
        except Exception:
            pass
    
    def handle_ui_language_change(self, new_lang: str):
        self.ui_i18n.load_language(new_lang)
        self.root.title(self.ui_i18n.tr("app_title", APP_NAME))
        self.update_pose_ui()
        
    def _cancel_authorization(self):
        self.auth_generation = getattr(self, "auth_generation", 0) + 1
        self.auth_in_progress = False
        self.liveness_in_progress = False
        self.frame_buffer.clear()
        self._face_continuous_start = 0

        auth_after_id = getattr(self, "auth_after_id", None)

        if auth_after_id is not None:
            try:
                self.root.after_cancel(auth_after_id)
            except (ValueError, tk.TclError):
                pass

        self.auth_after_id = None

    def open_fallback_auth_window(self):
        if self.password_prompt_open:
            return

        if self.training_running:
            self._show_training_block_message()
            if getattr(self.view, "current_view", "") != "AuthView":
                messagebox.showwarning(
                    self.ui_i18n.tr("auth_blocked_title", "Authorization blocked"),
                    self._training_block_text(),
                )
            return

        user = safe_user_name(self.view.auth_user_var.get())
        use_totp = self.config.getboolean("App", "auth_key", fallback=True)
        offline_hash = self.config.get("App", "offline_code_hash", fallback="")
        offline_available = bool(offline_hash) and not self.config.getboolean(
            "App", "offline_code_used", fallback=False
        )

        if not user and not offline_available:
            self.view.msg_label.configure(
                text=self.ui_i18n.tr(
                    "select_user_or_offline_code",
                    "Select a user or generate an offline code first.",
                ),
                foreground="red",
            )
            return

        if not use_totp and not offline_available:
            self.view.msg_label.configure(
                text=self.ui_i18n.tr(
                    "one_time_login_disabled",
                    "One-time-code login is disabled in Settings.",
                ),
                foreground="blue",
            )
            return
        
        self._cancel_authorization()
        self.fallback_auth_open = True
        self.is_locked = True
        self._start_authentication_timer()
            
        popup = tk.Toplevel(self.root)
        popup.title(self.ui_i18n.tr("fallback_login_title", "Fallback login"))
        popup.geometry("350x190")
        popup.transient(self.root)
        popup.attributes("-topmost", True)
        popup.lift()
        popup.focus_force()
        popup.wait_visibility()
        popup.grab_set()
        
        def on_close_popup():
            self.fallback_auth_open = False
            self.is_locked = False
            self.ui_interaction_until = time.time() + 1.0
            self._face_continuous_start = 0

            if popup.winfo_exists():
                popup.grab_release()
                popup.destroy()
            
        popup.protocol("WM_DELETE_WINDOW", on_close_popup)
        
        login_target = user or self.ui_i18n.tr("offline_mode", "offline mode")
        tk.Label(
            popup,
            text=self.ui_i18n.tr(
                "fallback_login_user_fmt", "Fallback login:\n{user}"
            ).format(user=login_target),
            font=("Arial", 11, "bold"),
        ).pack(pady=15)
        
        totp_var = tk.StringVar()
        
        tk.Label(
            popup,
            text=self.ui_i18n.tr("enter_2fa_code", "Enter the code from your authenticator app (2FA):"),
        ).pack(pady=(5, 0))
        tk.Entry(popup, textvariable=totp_var).pack(pady=5)
            
        def verify_credentials():
            entered_totp = totp_var.get().strip()

            if not entered_totp:
                self.current_login_attempts += 1
                messagebox.showwarning(
                    self.ui_i18n.tr("warning", "Warning"),
                    self.ui_i18n.tr("enter_2fa_code_error", "Enter your 2FA code."),
                )
                return

            totp_ok = False
            offline_ok = False

            if entered_totp and use_totp:
                global_dir = PROFILE_ROOT.parent
                secret = self.auth_manager.load_secret(global_dir)
                if secret and self.auth_manager.verify_code(secret, entered_totp):
                    totp_ok = True

            if entered_totp and offline_available:
                offline_ok, legacy_hash = verify_credential(entered_totp, offline_hash)

            if totp_ok or offline_ok:
                if offline_ok:
                    self.config.set("App", "offline_code_used", "True")
                    if legacy_hash:
                        self.config.set(
                            "App", "offline_code_hash", hash_credential(entered_totp)
                        )
                self.is_authorized = True
                self.current_auth_user = user
                duration = self._log_authentication_duration()
                attempts = self.current_login_attempts
                self._log_attendance(user, "offline" if offline_ok else "TOTP", duration, attempts, level="n/d")
                method = "offline" if offline_ok else "TOTP"
                self.view.msg_label.configure(
                    text=self.ui_i18n.tr(
                        "fallback_auth_success_fmt",
                        "Fallback authorization succeeded ({method}).",
                    ).format(method=method),
                    foreground="green",
                )
                messagebox.showinfo(
                    self.ui_i18n.tr("success", "Success"),
                    self.ui_i18n.tr(
                        "fallback_auth_success_message",
                        "Successfully signed in using the fallback method.",
                    ),
                )
                self._save_auth_state(user, attempts=0, blocked=False)
                self.current_login_attempts = 0
                
                # Włączenie trybu autoryzowanego w widoku
                if hasattr(self.view, "set_authorized_mode"):
                    self.view.set_authorized_mode(True)

                # Automatyczne przejście do widoku Ustawień
                self.view.show_frame("SettingsView")
                on_close_popup()
                
                # Odliczanie cichej weryfikacji startuje dopiero po minimalizacji (_on_minimize_clicked)
            else:
                messagebox.showerror(
                    self.ui_i18n.tr("error", "Error"),
                    self.ui_i18n.tr("invalid_2fa_code", "Invalid 2FA code."),
                )
            
        tk.Button(
            popup,
            text=self.ui_i18n.tr("btn_login", "Log in"),
            command=verify_credentials,
            bg="green",
            fg="white",
            width=15,
        ).pack(pady=15)

    def _ensure_auth_password(self):
        if (
            getattr(self.view, "current_view", "") != "AuthView"
            or self.password_verified
            or self.password_prompt_open
            or self.fallback_auth_open
            or self.training_running
        ):
            return

        # Sprawdzenie, czy hasło jest wyłączone w configu
        if not self.config.getboolean("App", "password", fallback=True):
            self.password_verified = True
            self.authentication_started_at = time.perf_counter()
            self.ui_interaction_until = time.time() + 0.5
            self.view.msg_label.configure(
                text=self.ui_i18n.tr(
                    "password_disabled_starting_face_auth",
                    "Password protection is disabled. Starting face verification.",
                ),
                foreground="green",
            )
            self._start_authentication_timer()
            return
        
        # Sprawdzenie, czy zapisane hasło jest puste ("")
        saved_password = self.config.get("App", "saved_password", fallback="")
        if not saved_password:
            self.password_verified = True
            self.authentication_started_at = time.perf_counter()
            self.ui_interaction_until = time.time() + 0.5
            self.view.msg_label.configure(
                text=self.ui_i18n.tr(
                    "no_password_starting_face_auth",
                    "No password is set. Starting face verification.",
                ),
                foreground="green",
            )
            self._start_authentication_timer()
            return

        user = safe_user_name(self.view.auth_user_var.get())
        if not user:
            return
        
        self.password_prompt_open = True
        self.view.btn_fallback_auth.configure(state="disabled")
        popup = tk.Toplevel(self.root)
        popup.title(self.ui_i18n.tr("password_before_face_auth", "Password required"))
        popup.geometry("360x180")
        popup.transient(self.root)
        popup.attributes("-topmost", True)
        popup.lift()
        popup.focus_force()
        popup.wait_visibility()
        popup.grab_set()

        tk.Label(
            popup,
            text=self.ui_i18n.tr(
                "enter_user_password_fmt", "Enter the password for user:\n{user}"
            ).format(user=user),
            font=("Arial", 11, "bold"),
        ).pack(pady=15)
        password_var = tk.StringVar()
        password_entry = tk.Entry(popup, textvariable=password_var, show="*", width=28)
        password_entry.pack(pady=5)

        def close_popup():
            self.password_prompt_open = False
            self.view.btn_fallback_auth.configure(state="normal")
            if popup.winfo_exists():
                popup.grab_release()
                popup.destroy()
            if getattr(self.view, "current_view", "") == "AuthView" and not self.password_verified:
                self.root.after(500, self._ensure_auth_password)

        def verify_password():
            saved_password = self.config.get("App", "saved_password", fallback="")
            if not saved_password:
                messagebox.showwarning(
                    self.ui_i18n.tr("warning", "Warning"),
                    self.ui_i18n.tr(
                        "password_not_set",
                        "A password has not been set in Settings yet.",
                    ),
                    parent=popup,
                )
                return
            password_matches, legacy_hash = verify_credential(
                password_var.get(), saved_password
            )
            if not password_matches:
                messagebox.showerror(
                    self.ui_i18n.tr("error", "Error"),
                    self.ui_i18n.tr("invalid_password", "Incorrect password."),
                    parent=popup,
                )
                password_entry.selection_range(0, tk.END)
                password_entry.focus_set()
                return

            if legacy_hash:
                self.config.set(
                    "App", "saved_password", hash_credential(password_var.get())
                )

            self.password_verified = True
            self.current_login_attempts = 0
            self.authentication_started_at = time.perf_counter()
            self.ui_interaction_until = time.time() + 0.5
            self.view.msg_label.configure(
                text=self.ui_i18n.tr(
                    "password_correct_starting_face_auth",
                    "Password accepted. Starting face verification.",
                ),
                foreground="green",
            )
            close_popup()

        popup.protocol("WM_DELETE_WINDOW", close_popup)
        tk.Button(
            popup,
            text=self.ui_i18n.tr("confirm", "Confirm"),
            command=verify_password,
            bg="green",
            fg="white",
            width=15,
        ).pack(pady=15)
        password_entry.bind("<Return>", lambda _event: verify_password())
        password_entry.focus_set()
    
    def handle_log_language_change(self, new_lang: str):
        self.log_i18n.load_language(new_lang)
        self.trainer.i18n = self.log_i18n
        self.log_tr("log_language_changed_fmt", "Zmieniono język logów na: {lang}", lang=new_lang)

    def handle_theme_change(self, new_theme: str):
        self.apply_theme(new_theme)
    
    def _training_block_text(self, registration: bool = False) -> str:
        # Funkcja odpowiedzialna za tekst komunikatu o blokadzie na czas uczenia modelu.
        if registration:
            return self.ui_i18n.tr(
                "registration_blocked_training",
                "ZABLOKOWANA REJESTRACJA: model nie zakończył uczenia się z poprzedniej sesji",
            )
        return self.ui_i18n.tr(
            "auth_blocked_training",
            "ZABLOKOWANA AUTORYZACJA: model nie zakończył uczenia się z poprzedniej sesji",
        )

    def _show_training_block_message(self):
        # Funkcja odpowiedzialna za wyświetlenie informacji o zablokowanej autoryzacji w widoku autoryzacji.
        text = self._training_block_text()
        try:
            if str(self.view.msg_label.cget("text")) != text:
                self.view.msg_label.configure(text=text, foreground="red")
        except tk.TclError:
            pass

    def _authorization_allowed(self) -> bool:
        active_view = getattr(self.view, "current_view", "")
        password_required = self.config.getboolean("App", "password", fallback=True)

        return (
            active_view == "AuthView"
            and not self.is_locked
            and not self.fallback_auth_open
            and not self.auth_in_progress
            and not self.liveness_in_progress
            and not self.training_running
            and (self.password_verified or not password_required)
            and time.time() >= self.ui_interaction_until
        )


    # def _cancel_authorization(self):
    #     self.auth_generation += 1
    #     self.auth_in_progress = False
    #     self.liveness_in_progress = False
    #     self.frame_buffer.clear()
    #     self._face_continuous_start = 0

    #     if self.auth_after_id is not None:
    #         try:
    #             self.root.after_cancel(self.auth_after_id)
    #         except (ValueError, tk.TclError):
    #             pass

    #         self.auth_after_id = None


    def _on_global_click(self, event):
        # Kliknięcie bezpośrednio w obraz kamery nie anuluje procesu.
        if event.widget == getattr(self.view, "video_label_auth", None):
            return

        self._cancel_authorization()

        # Po obsłudze kontrolek trzeba ponownie utrzymać twarz w kamerze.
        self.ui_interaction_until = max(self.ui_interaction_until, time.time() + 1.0)
    
    def handle_view_changed(self, view_name: str):
        self._cancel_authorization()
        self._sync_camera_state()

        if view_name == "RegisterView":
            self.cancel_silent_timer()
            self.auth_in_progress = False
            self.liveness_in_progress = False
            self.fallback_auth_open = False
            self._face_continuous_start = 0
            self.is_locked = False
            self.ui_interaction_until = time.time() + 1.0

        elif view_name == "SettingsView":
            self.view.set_authorized_mode(self.is_authorized)
            # Okno widoczne (nie zminimalizowane) - nic nie odliczamy
            if not self._is_minimized():
                self.cancel_silent_timer()

        elif view_name == "AuthView":
            self.cancel_silent_timer()
            self.is_authorized = False
            self.current_auth_user = None
            self.authentication_started_at = None
            self.password_verified = not self.config.getboolean("App", "password", fallback=True)
            self.last_auth_time = time.time()
            self.ui_interaction_until = time.time() + 1.0
            self.current_login_attempts = 0
            if not self.config.getboolean("App", "password", fallback=True):
                self.password_verified = True
                self.authentication_started_at = time.perf_counter()
                self.ui_interaction_until = time.time() + 0.5
            self._ensure_auth_password()
            if self.training_running:
                self._show_training_block_message()
            
    def log(self, text: str):
        self.view.log(text)
        try:
            logger.info(text)
        except Exception as e:
            final_text = f"Błąd zapisu logu do pliku: {e}"
            self.log_tr("", final_text)

    def _camera_needed(self) -> bool:
        # Kamera jest potrzebna tylko w widoku autoryzacji i rejestracji.
        if getattr(self, "_close_pending", False):
            return False
        view = getattr(self, "view", None)
        return getattr(view, "current_view", "") in {"AuthView", "RegisterView"}

    @staticmethod
    def _configure_capture(cap):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _sync_camera_state(self):
        # Funkcja odpowiedzialna za włączenie kamery w widokach, które jej potrzebują, i wyłączenie w pozostałych.
        if self._camera_needed():
            self._start_camera_async()
        else:
            self.close_camera()

    def _maybe_retry_camera(self):
        # Ponowna próba włączenia kamery (np. gdy była chwilowo zajęta przez cichą weryfikację).
        if not self._camera_needed() or self._camera_opening:
            return
        now = time.time()
        if now >= self._camera_retry_at:
            self._camera_retry_at = now + 2.0
            self._start_camera_async()

    def _start_camera_async(self):
        # Funkcja odpowiedzialna za otwarcie kamery w tle, żeby przełączanie widoku nie zawieszało interfejsu.
        if self.cap is not None or self._camera_opening or not self.cameras:
            return

        idx = self.current_camera_idx
        self._camera_opening = True

        def finish(cap):
            self._camera_opening = False
            if (
                self._closing
                or self.cap is not None
                or not self._camera_needed()
                or not cap.isOpened()
            ):
                if not cap.isOpened():
                    logger.debug("Nie udało się otworzyć kamery %s", idx)
                cap.release()
                return
            self.cap = cap
            self.current_frame = None
            self.frame_buffer.clear()
            self._camera_warmup_until = time.time() + 0.7

        def worker():
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                self._configure_capture(cap)
            try:
                self.root.after(0, finish, cap)
            except (RuntimeError, tk.TclError):
                self._camera_opening = False
                cap.release()

        threading.Thread(target=worker, name="camera-open", daemon=True).start()

    def close_camera(self):
        # Funkcja odpowiedzialna za zwolnienie kamery i wyczyszczenie danych z kamery.
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self.current_frame = None
        self.frame_buffer.clear()
        self._cached_auth_det = (False, None)

        # Wyczyść podgląd, żeby po ponownym włączeniu nie wisiała stara klatka
        view = getattr(self, "view", None)
        for label_name in ("video_label_auth", "video_label_reg"):
            label = getattr(view, label_name, None)
            if label is not None:
                try:
                    label.configure(image="")
                    label.imgtk = None
                except tk.TclError:
                    pass

    def _grab_single_frame(self, idx: int):
        # Funkcja odpowiedzialna za jednorazowe włączenie kamery, zrobienie klatki (po rozgrzewce) i wyłączenie.
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        try:
            if not cap.isOpened():
                return None
            self._configure_capture(cap)
            frame = None
            for _ in range(12):
                ok, f = cap.read()
                if ok and f is not None:
                    frame = f
                time.sleep(0.05)
            return frame
        finally:
            cap.release()

    def open_camera(self, idx: int, show_error: bool = False):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.current_camera_idx = idx
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)

        if not cap.isOpened():
            if show_error:
                title = self.ui_i18n.tr("camera", "Kamera")
                msg = self.ui_i18n.tr(
                    "camera_open_error_fmt",
                    "Nie udało się otworzyć kamery {idx}"
                ).format(idx=idx)
                messagebox.showwarning(title, msg)
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.cap = cap
        self.frame_buffer.clear()
        
    def _load_auth_states(self):
        state_file = BASE / "auth_states.json"
        if state_file.exists():
            try:
                import json
                with open(state_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_auth_state(self, user: str, attempts: int, blocked: bool):
        state_file = BASE / "auth_states.json"
        states = self._load_auth_states()
        if user not in states:
            states[user] = {}
        states[user]["attempts"] = attempts
        states[user]["blocked"] = blocked
        try:
            import json
            with open(state_file, "w") as f:
                json.dump(states, f)
        except Exception as e:
            final_text = f"Błąd zapisu stanu autoryzacji: {e}"
            self.log_tr("", final_text)

    def _save_successful_face_sample(self, user: str, source_image: str | Path, target_root: Path | None = None) -> Path:
        source_path = Path(source_image)
        if not source_path.exists():
            raise FileNotFoundError(f"Brak obrazu do zapisania do profilu: {source_path}")

        user_dir = (target_root or PROFILE_ROOT) / safe_user_name(user)
        user_dir.mkdir(parents=True, exist_ok=True)

        suffix = source_path.suffix or ".jpg"
        stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        candidate = user_dir / f"auth_{stamp}{suffix}"
        counter = 1
        while candidate.exists():
            candidate = user_dir / f"auth_{stamp}_{counter}{suffix}"
            counter += 1

        shutil.copy2(source_path, candidate)
        return candidate
    
    def change_camera(self):
        selected_name = self.view.camera_var.get()
        no_cam_msg = self.ui_i18n.tr("no_camera_detected", "Nie wykryto żadnej kamery")
        
        if selected_name in {
            no_cam_msg,
            "Brak wykrytej kamery",
            "No camera detected",
        }:
            self.log_tr("log_camera_opened_fmt", "Zmieniona kamera: {camera_name}", camera_name=no_cam_msg)
            return

        idx = self.camera_name_to_index.get(selected_name)
        if idx is not None:
            if self.cap is not None:
                self.open_camera(idx, show_error=True)
            else:
                # Kamera jest wyłączona (ustawienia) - zapamiętujemy tylko wybór
                self.current_camera_idx = idx
        self.log_tr("log_camera_opened_fmt", "Zmieniono kamerę na: {camera_name}", camera_name=selected_name)
            
    def lock_app(self, seconds):
        self._cancel_authorization()
        self.is_locked = True

        self.view.msg_label.configure(
            text=self.ui_i18n.tr(
                "locked_for_seconds_fmt", "Locked for {seconds}s"
            ).format(seconds=seconds),
            foreground="red"
        )

        self.root.after(seconds * 1000, self.unlock)


    def unlock(self):
        self.is_locked = False
        self._face_continuous_start = 0
        self.last_auth_time = time.time()

        self.view.msg_label.configure(
            text="",
            foreground="black"
        )

    def _submit_face_detection(self, rgb_frame: np.ndarray):
        with self._face_detection_lock:
            self._face_detection_frame = rgb_frame
        self._face_detection_event.set()

    def _face_detection_worker(self):
        while not self._face_detection_stop.is_set():
            self._face_detection_event.wait()
            if self._face_detection_stop.is_set():
                return

            with self._face_detection_lock:
                rgb_frame = self._face_detection_frame
                self._face_detection_event.clear()

            if rgb_frame is None:
                continue

            try:
                detection_result = self.preview_face_detector.process(rgb_frame)
                face_detected = bool(detection_result.detections)
                bbox_coords = None

                if face_detected:
                    det = max(detection_result.detections, key=lambda d: float(d.score[0]))
                    rb = det.location_data.relative_bounding_box
                    h_f, w_f = rgb_frame.shape[:2]
                    bbox_coords = (
                        int(rb.xmin * w_f),
                        int(rb.ymin * h_f),
                        int(rb.width * w_f),
                        int(rb.height * h_f),
                    )

                self._cached_auth_det = (face_detected, bbox_coords)
            except Exception:
                logger.exception("Face detection worker failed")
                self._cached_auth_det = (False, None)

    def update_camera_frame(self):
        if self._closing:
            return
        try:
            self._update_camera_frame_impl()
        except Exception:
            logger.exception("update_camera_frame failed")
        finally:
            if not self._closing:
                try:
                    self.root.after(30, self.update_camera_frame)
                except tk.TclError:
                    pass

    def _update_camera_frame_impl(self):
        if self._closing or self.root is None or not self.root.winfo_exists():
            return

        if self.cap is None:
            self._maybe_retry_camera()

        if self.cap is not None:
            ok, frame = self.cap.read()

            if ok and frame is not None:
                # Tuż po włączeniu kamery pierwsze klatki bywają czarne/niedoświetlone - pomijamy je
                if time.time() < self._camera_warmup_until:
                    return

                self.current_frame = frame.copy()

                self.frame_counter += 1
                self.frame_buffer.append(frame.copy())
                # if self.frame_counter % 3 == 0:
                #     small = cv2.resize(frame, (320, 240))
                #     self.frame_buffer.append(small.copy())

                active_view = getattr(self.view, "current_view", "AuthView")
                if active_view not in {"AuthView", "RegisterView"}:
                    return

                if active_view == "AuthView":
                    if getattr(self, "auth_in_progress", False) or getattr(self, "liveness_in_progress", False):
                        # Pomiń detekcję twarzy w głównym wątku na czas autoryzacji - oszczędza CPU i zapobiega zacinaniu
                        cv2.putText(frame, self.ui_i18n.tr("verifying", "Weryfikacja w toku..."), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    elif getattr(self, "is_locked", False):
                        cv2.putText(frame, self.ui_i18n.tr("locked", "ZABLOKOWANE"), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    elif self.training_running:
                        # Model jeszcze się uczy - autoryzacja zablokowana
                        cv2.putText(frame, self.ui_i18n.tr("locked", "ZABLOKOWANE"), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        self._show_training_block_message()
                        self._face_continuous_start = 0
                    else:
                        # Optymalizacja CPU: Obliczanie detekcji twarzy co drugą klatkę dla zachowania płynności UI
                        if self.frame_counter % 2 == 0:
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            self._submit_face_detection(rgb_frame)

                        face_detected, bbox_coords = self._cached_auth_det

                        if face_detected and bbox_coords:
                            x, y, bw, bh = bbox_coords
                            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                        
                        # Stabilizacja czasu autoryzacji
                        if face_detected and self._authorization_allowed():
                            if not getattr(self, "_face_continuous_start", 0):
                                self._face_continuous_start = time.time()
                            elif time.time() - self._face_continuous_start > 1.2:
                                self.start_authorization()
                                self._face_continuous_start = 0
                        else:
                            self._face_continuous_start = 0

                    img = self._safe_preview_to_image(frame)
                    if img is None:
                        return
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.view.video_label_auth.imgtk = imgtk
                    self.view.video_label_auth.configure(image=imgtk)

                elif active_view == "RegisterView":
                    # Optymalizacja CPU dla rejestracji
                    if self.frame_counter % 2 == 0:
                        quality_ok = pose_ok = False
                        bbox = None
                        yaw = pitch = roll = 0.0
                        head_detected = False
                        msg = self.ui_i18n.tr("registration_instruction", "Wpisz nową nazwę konta i wciśnij rejestruj")
                        
                        if self.user_dir is not None and self.current_pose_idx < len(POSES):
                            quality_ok, msg, bbox = self.geometry_analyzer.assess_frame_quality(frame, self.current_pose_idx)
                            pose_ok = quality_ok

                            if quality_ok:
                                success, yaw, pitch, roll = self.geometry_analyzer.estimate_head_pose(frame)
                                head_detected = success
                                if success:
                                    pose_ok, pose_msg = self.geometry_analyzer.check_expected_pose(self.current_pose_idx, yaw, pitch, roll)
                                    if not pose_ok:
                                        msg = pose_msg
                                else:
                                    pose_ok = False
                                    msg = self.ui_i18n.tr("cannot_determine_pose", "Nie można określić orientacji głowy")

                            if pose_ok and not self.training_running:
                                self.stable_count += 1
                            else:
                                self.stable_count = 0

                            if (
                                self.stable_count >= AUTO_STABLE_FRAMES
                                and time.time() - self.last_capture_time >= AUTO_CAPTURE_DELAY
                            ):
                                self.last_capture_time = time.time()
                                self.stable_count = 0
                                self.capture_pose(auto=True)

                        elif self.user_dir is not None:
                            # Wszystkie zdjęcia zebrane - zamiast instrukcji rejestracji pokazujemy co dalej
                            pose_ok = True
                            self.stable_count = 0
                            msg = (
                                "Trwa uczenie użytkownika. Proszę czekać..."
                                if self.training_running
                                else self._photos_done_message()
                            )

                        self._cached_reg_data = (pose_ok, msg, bbox, yaw, pitch, roll, head_detected, quality_ok)

                    pose_ok, msg, bbox, yaw, pitch, roll, head_detected, quality_ok = self._cached_reg_data

                    frame = self.geometry_analyzer.draw_face_guide(
                        frame, pose_ok, msg, bbox, yaw, pitch, roll, head_detected,
                        self.user_dir, self.current_pose_idx, self.current_shot_idx, self.stable_count
                    )
                    self.view.pose_label.configure(text=msg, foreground="green" if pose_ok else "red")
                    
                    img = self._safe_preview_to_image(frame)
                    if img is None:
                        return
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.view.video_label_reg.imgtk = imgtk
                    self.view.video_label_reg.configure(image=imgtk)

                elif active_view == "SettingsView":
                    pass


    def start_registration(self):
        if self.training_running:
            messagebox.showwarning(
                self.ui_i18n.tr("registration_blocked_title", "Rejestracja zablokowana"),
                self._training_block_text(registration=True),
            )
            return

        user = safe_user_name(self.view.user_var.get())
        if not user:
            self.view.msg_label.configure(
                text=self.ui_i18n.tr("enter_username_error", "Please enter a username."),
                foreground="red",
            )
            return

        self._cancel_authorization()
        self.fallback_auth_open = False

        self.view.user_var.set(user)
        self.user_dir = PROFILE_ROOT / user

        verifier_path = TRAIN_DIR / "outputs" / f"verifier_{user}.pkl"
        if self.user_dir.exists() and not verifier_path.exists():
            shutil.rmtree(self.user_dir, ignore_errors=True)

        if self.user_dir.exists() and any(self.user_dir.glob("*.jpg")):
            ans = messagebox.askyesno(
                self.ui_i18n.tr("overwrite_registration_title", "Nadpisać rejestrację"),
                self.ui_i18n.tr("overwrite_registration_msg_fmt", "Folder użytkownika już istnieje:\n{path}\n\nUsunąć stare zdjęcia i zacząć od nowa?").format(path=self.user_dir)
            )
            if not ans:
                return
            shutil.rmtree(self.user_dir, ignore_errors=True)

        self.user_dir.mkdir(parents=True, exist_ok=True)
        self.saved_files.clear()
        self.current_pose_idx = 0
        self.current_shot_idx = 0
        self.stable_count = 0
        self.last_capture_time = 0.0
        self.training_running = False

        self.view.capture_btn.configure(state="normal")
        self.view.train_btn.configure(state="disabled")
        self.view.training_status_label.configure(text="")
        self.view.pose_label.configure(
            text=self.ui_i18n.tr("registration_instruction", "Wpisz nową nazwę konta i wciśnij rejestruj"),
            foreground="green",
        )
        self.update_pose_ui()

        self.log_tr("log_registration_started_fmt", f"Rozpoczęto rejestrację użytkownika: {user}")
        self.log_tr("log_photos_dir_fmt", "Folder zdjęć: {path}", path=self.user_dir)

    def _cleanup_live_auth_dir(self):
        live_dir = VERIFICATION_DIR / "outputs" / "_live_auth"
        if not live_dir.exists():
            return
        for child in list(live_dir.iterdir()):
            try:
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink()
            except OSError:
                pass
        try:
            live_dir.rmdir()
        except OSError:
            pass

    def start_authorization(self):
        if getattr(self.view, "current_view", "") != "AuthView":
            self._cancel_authorization()
            return

        if self.training_running:
            self._show_training_block_message()
            return

        if not self._authorization_allowed():
            return
        user = safe_user_name(self.view.auth_user_var.get())
        if not user or self.current_frame is None:
            return

        self._cleanup_live_auth_dir()

        auth_states = self._load_auth_states()
        user_state = auth_states.get(user, {})
        if user_state.get("blocked", False):
            self.view.msg_label.configure(
                text=self.ui_i18n.tr(
                    "account_permanently_locked",
                    "This account is permanently locked due to too many failed attempts. Use an alternative login method.",
                ),
                foreground="red",
            )
            return

        verifier_path = TRAIN_DIR / "outputs" / f"verifier_{user}.pkl"
        if not verifier_path.exists():
            show_text = self.ui_i18n.tr(
                "no_model_msg_fmt",
                "verifier_{user}.pkl not found. Please complete registration and train the model first.",
            ).format(user=user)
            self.view.msg_label.configure(text=show_text, foreground="red")
            return

        if self.is_locked:
            return

        tmp_dir = VERIFICATION_DIR / "outputs" / "_live_auth"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_img = tmp_dir / f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.jpg"
        ok = cv2.imwrite(str(tmp_img), self.current_frame.copy())
        if not ok:
            self.view.msg_label.configure(
                text=self.ui_i18n.tr(
                    "save_frame_failed", "Failed to save the temporary frame."
                ),
                foreground="red",
            )
            return

        if self.authentication_started_at is None:
            self.authentication_started_at = time.perf_counter()
        self.auth_in_progress = True
        self.liveness_in_progress = True
        
        self.auth_generation += 1
        generation = self.auth_generation

        def _liveness_done(liveness_ok, liveness_info):
            self.liveness_in_progress = False

            if not liveness_ok:
                self.auth_in_progress = False
                self.current_login_attempts += 1
                try:
                    if tmp_img.exists():
                        tmp_img.unlink()
                except OSError:
                    pass
                
                self.ui_interaction_until = time.time() + 3.0
                self._face_continuous_start = 0
                
                self.view.msg_label.configure(
                    text=liveness_info.get(
                        "reason",
                        self.ui_i18n.tr(
                            "liveness_failed_msg",
                            "Liveness check failed. Keep your face in frame and blink briefly.",
                        ),
                    ),
                    foreground="red"
                )
                return

            self._continue_authorization(user, tmp_img)

        def start_liveness_worker():
            self.auth_after_id = None

            if generation != self.auth_generation:
                return

            if (
                getattr(self.view, "current_view", "") != "AuthView"
                or self.is_locked
                or self.fallback_auth_open
            ):
                self._cancel_authorization()
                return

            frames = [frame.copy() for frame in self.frame_buffer]

            threading.Thread(
                target=self._run_liveness_async,
                args=(frames, _liveness_done, generation),
                daemon=True
            ).start()


        self.auth_after_id = self.root.after(
            1200,
            start_liveness_worker
        )

    def _run_liveness_async(self, frames, callback, generation):
        local_face_mesh = None
        try:
            local_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.45,
                min_tracking_confidence=0.45,
            )
            checker = LivenessChecker(local_face_mesh, i18n=self.ui_i18n)
            ok, info = checker.check_liveness_frames(frames)
        except Exception as exc:
            ok = False
            info = {"reason": str(exc)}
        finally:
            if local_face_mesh is not None:
                local_face_mesh.close()

        def deliver_result():
            if generation != self.auth_generation:
                return
            if (getattr(self.view, "current_view", "") != "AuthView"
                    or self.is_locked or self.fallback_auth_open):
                self._cancel_authorization()
                return
            callback(ok, info)

        self.root.after(0, deliver_result)        
        
    def _continue_authorization(self, user, tmp_img):
        # Funkcja odpowiedzialna za kontynuację autoryzacji w osobnym wątku, aby uniknąć zfreeze'owania interfejsu.
        
        def _auth_worker():
            try:
                # Blokujące operacje I/O oraz obliczenia ML
                vf = joblib.load(TRAIN_DIR / "outputs" / f"verifier_{user}.pkl")
                model_user = vf.get("user", "")
                
                if user != model_user:
                    # Mismatch użytkownika -> przekieruj powrót na główny wątek
                    self.root.after(0, _auth_ui_mismatch, model_user, user)
                    return

                settings = Settings()
                fpar = FaceParams()
                
                # To jest najcięższa operacja - działamy w wątku więc interfejs płynnie żyje
                emb, info = embed_image(
                    tmp_img, BASE, settings=settings, fpar=fpar, prefix="live_auth",
                    step="normal",
                )
                emb = emb.astype(np.float32)
                emb = emb / (np.linalg.norm(emb) + 1e-8)

                gallery = np.asarray(vf["gallery_embeddings"], dtype=np.float32)
                threshold, used_level, _legacy = resolve_threshold(vf, self.get_security_level())
                self._last_auth_level = used_level
                topk = int(vf.get("topk", 3))

                score_max, score_topk = cosine_scores(emb, gallery, topk=topk)
                decision = score_max >= threshold
                
                # Operacje pomyślnie wykonane, zlecamy uaktualnienie interfejsu
                self.root.after(0, _auth_ui_success, vf, emb, score_max, threshold, decision)
                
            except Exception as e:
                self.root.after(0, _auth_ui_error, str(e))

        def _cleanup_tmp():
            self.auth_in_progress = False
            if tmp_img is not None and tmp_img.exists():
                try:
                    tmp_img.unlink()
                except Exception:
                    pass

        def _auth_ui_mismatch(model_user, requested_user):
            messagebox.showwarning(
                self.ui_i18n.tr("different_user_title", "Inny użytkownik"),
                self.ui_i18n.tr("different_user_msg_fmt", "Verifier jest dla użytkownika: {model_user}\nA wpisano: {user}").format(model_user=model_user, user=requested_user)
            )
            _cleanup_tmp()

        def _auth_ui_error(error_msg):
            messagebox.showerror(self.ui_i18n.tr("auth_error_title", "Błąd autoryzacji"), error_msg)
            self.log_tr("log_auth_error_fmt", "BŁĄD autoryzacji: {exc}", exc=error_msg)
            _cleanup_tmp()

        def _auth_ui_success(vf, emb, score_max, threshold, decision):
            try:
                decision_str = self.ui_i18n.tr("decision_accepted", "AKCEPTACJA") if decision else self.ui_i18n.tr("decision_rejected", "ODRZUCENIE")

                if decision:
                    self._save_auth_state(user, attempts=0, blocked=False)
                    self.is_authorized = True
                    duration = self._log_authentication_duration()
                    attempts = self.current_login_attempts

                    train_after_auth = self.config.getboolean(
                        "App", "train_after_auth", fallback=False
                    )
                    should_retrain = False
                    if train_after_auth and update_user_gallery(
                        vf,
                        emb,
                        user,
                        score_max,
                        threshold,
                        log_fn=self.log,
                        i18n=self.log_i18n,
                    ):
                        saved_sample = self._save_successful_face_sample(
                            user, tmp_img, PROFILE_ROOT
                        )
                        should_retrain = True
                        self.log_tr(
                            "log_face_sample_saved_fmt",
                            "Zapisano świeży obrazek do profilu po autoryzacji: {path}",
                            path=saved_sample,
                        )

                    self._log_attendance(user, "face", duration, attempts, level=getattr(self, "_last_auth_level", self.get_security_level()))
                    self.current_auth_user = user
                    self.log_tr(
                        "log_face_login_fmt",
                        "Zalogowano się po {method} metodą: {user} | czas: {time} | próby: {attempts}",
                        method="twarzą",
                        user=user,
                        time=f"{duration:.2f} s",
                        attempts=attempts,
                    )
                    
                    # Odliczanie cichej weryfikacji startuje dopiero po minimalizacji (_on_minimize_clicked)
                    if should_retrain:
                        self._start_training(user, reason="after_auth")
                    self.current_login_attempts = 0
                    self.view.show_frame("SettingsView")
                else:
                    self.current_login_attempts += 1
                    self.ui_interaction_until = time.time() + 3.0
                    self._face_continuous_start = 0
                    current_states = self._load_auth_states()
                    current_attempts = current_states.get(user, {}).get("attempts", 0) + 1
                    
                    if current_attempts >= 20:
                        self._save_auth_state(user, attempts=current_attempts, blocked=True)
                        self.view.msg_label.configure(
                            text=self.ui_i18n.tr(
                                "account_locked_use_alternative",
                                "Account locked. Use an alternative login method.",
                            ),
                            foreground="red",
                        )
                        decision_str = self.ui_i18n.tr("decision_account_locked", "ACCOUNT LOCKED")
                    elif current_attempts >= 10:
                        self._save_auth_state(user, attempts=current_attempts, blocked=False)
                        self.lock_app(30)
                        self.view.msg_label.configure(
                            text=self.ui_i18n.tr(
                                "authentication_locked_30s",
                                "Too many failed attempts. Locked for 30 seconds.",
                            ),
                            foreground="red",
                        )
                        decision_str = self.ui_i18n.tr(
                            "decision_rejected_locked_fmt",
                            "REJECTED ({attempts} attempts - locked for 30s)",
                        ).format(attempts=current_attempts)
                    else:
                        self._save_auth_state(user, attempts=current_attempts, blocked=False)
                        decision_str = self.ui_i18n.tr(
                            "decision_rejected_attempts_fmt",
                            "REJECTED ({attempts} attempts)",
                        ).format(attempts=current_attempts)

                self.log_tr("log_auth_result_fmt", "AUTORYZACJA [{user}] Score: {score:.4f} | Próg: {threshold:.4f} | Decyzja: {decision}",
                            user=user, score=score_max, threshold=threshold, decision=decision_str)
                messagebox.showinfo(
                    self.ui_i18n.tr("authorization_title", "Autoryzacja"),
                    self.ui_i18n.tr(
                        "authorization_result_fmt", "{decision}\nScore: {score:.4f}"
                    ).format(decision=decision_str, score=score_max)
                )
            finally:
                self.ui_interaction_until = max(self.ui_interaction_until, time.time() + 3.0)
                self._face_continuous_start = 0
                _cleanup_tmp()

        # Uruchomienie całości wewnątrz wątku pobocznego (nie blokuje GUI!)
        threading.Thread(target=_auth_worker, daemon=True).start()
        
    def _photos_done_message(self) -> str:
        # Funkcja zwracająca komunikat po zebraniu zdjęć, z nazwą przycisku uczenia z aktualnego języka.
        btn_text = str(self.view.train_btn.cget("text"))
        return self.ui_i18n.tr(
            "all_photos_saved_press_train",
            "Zdjęcia zapisane. Wciśnij przycisk „{btn}”",
        ).format(btn=btn_text)

    def update_pose_ui(self):
        total_done = len(self.saved_files)
        total_needed = len(POSES) * SHOTS_PER_POSE

        if self.current_pose_idx < len(POSES):
            _, pose_key = POSES[self.current_pose_idx]
            pose_translated = self.ui_i18n.tr(pose_key, pose_key)
            self.view.pose_label.configure(
                text=f"{self.current_pose_idx + 1}/{len(POSES)} - {pose_translated} ({self.current_shot_idx + 1}/{SHOTS_PER_POSE})"
            )
        else:
            done_msg = self._photos_done_message()
            self.view.pose_label.configure(text=done_msg)

        self.view.progress.configure(value=total_done)

    def capture_pose(self, auto: bool = False):
        user = safe_user_name(self.view.user_var.get())
        if self.user_dir is None or self.current_frame is None or self.current_pose_idx >= len(POSES):
            return

        pose_key, _ = POSES[self.current_pose_idx]
        out_path = self.user_dir / f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}{IMG_EXT}"

        ok = cv2.imwrite(str(out_path), self.current_frame.copy())
        if not ok:
            return

        self.saved_files.append(out_path)
        self.log_tr("log_photo_saved_fmt", "Zapisano: {filename}", filename=out_path.name)

        self.current_shot_idx += 1
        if self.current_shot_idx >= SHOTS_PER_POSE:
            self.current_shot_idx = 0
            self.current_pose_idx += 1

        self.update_pose_ui()

        total_needed = len(POSES) * SHOTS_PER_POSE
        if self.current_pose_idx >= len(POSES):
            self.view.capture_btn.configure(state="disabled")
            self.view.train_btn.configure(state="normal")
            done_msg = self._photos_done_message()
            self.view.pose_label.configure(text=done_msg)
            self.view.progress.configure(value=total_needed)

    def train_user(self):
        user = safe_user_name(self.view.user_var.get())
        if user and not self.training_running:
            self._start_training(user, reason="manual")

    def _start_training(self, user: str, reason: str = "manual"):
        self.training_running = True
        self.training_reason = reason
        self.view.train_btn.configure(state="disabled")
        self.view.capture_btn.configure(state="disabled")
        self.view.training_status_label.configure(
            text=self.ui_i18n.tr("training_in_progress", "Training model...")
        )
        self.view.pose_label.configure(
            text=self.ui_i18n.tr("training_wait", "Training in progress. Please wait...")
        )

        # Animacja tekstowa kropkowa dla informacji o uczeniu
        self.training_anim_step = 0
        if self.training_anim_id:
            try:
                self.root.after_cancel(self.training_anim_id)
            except Exception:
                pass
        self.training_anim_id = self.root.after(250, self._spin_training_anim)

        # Rozpoczynamy proces uczenia w osobnym wątku
        t = self.trainer.start_training(user, reason=reason)

        def monitor():
            t.join()  # Czekamy na zakończenie wątku uczenia
            # Zamiast self.root.after, wrzucamy sygnał zakończenia do bezpiecznej kolejki
            self.training_queue.put(("training_finished", None))

        threading.Thread(target=monitor, daemon=True).start()

    def _spin_training_anim(self):
        if not self.training_running:
            self.training_anim_id = None
            return
        self.training_anim_step = (self.training_anim_step + 1) % 4
        dots = "." * self.training_anim_step
        if hasattr(self.view, "training_status_label"):
            self.view.training_status_label.configure(
                text=self.ui_i18n.tr("training_in_progress", "Training model...").rstrip(".") + dots
            )
        self.training_anim_id = self.root.after(250, self._spin_training_anim)

    def drain_training_log(self):
        if not hasattr(self, "_is_showing_dialog"):
            self._is_showing_dialog = False

        try:
            while True:
                msg = self.training_queue.get_nowait()
                
                if isinstance(msg, tuple):
                    status, message = msg
                    
                    # Obsługa sygnału zakończenia uczenia w głównym wątku GUI
                    if status == "training_finished":
                        self.training_running = False
                        if self.training_anim_id:
                            try:
                                self.root.after_cancel(self.training_anim_id)
                            except Exception:
                                pass
                            self.training_anim_id = None
                        self.view.training_status_label.configure(text="")
                        self.view.train_btn.configure(state="normal")
                        self.view.capture_btn.configure(state="normal")
                        self.view.pose_label.configure(
                            text=self.ui_i18n.tr("training_finished", "Training finished.")
                        )

                        # Uczenie skończone, a użytkownik wcześniej poprosił o zamknięcie - zamykamy teraz
                        if self._close_pending:
                            self.root.after(0, self._finalize_close)
                            continue

                        # Zdjęcie blokady autoryzacji, jeśli użytkownik czeka na ekranie logowania
                        if getattr(self.view, "current_view", "") == "AuthView":
                            if str(self.view.msg_label.cget("text")) == self._training_block_text():
                                self.view.msg_label.configure(text="", foreground="black")
                            self.ui_interaction_until = time.time() + 1.0
                            self._ensure_auth_password()
                        continue

                    # Trwa odroczone zamykanie - żadnych okien ani zmian widoku, tylko log
                    if self._close_pending and status in {"success", "error"}:
                        self.log(str(message))
                        continue

                    # Błąd retrainu po autoryzacji (np. za mały próg) to nie błąd rejestracji:
                    # tylko logujemy, nie kasujemy danych i nie przechodzimy do rejestracji.
                    if status == "error" and getattr(self, "training_reason", "manual") == "after_auth":
                        self.log(f"Błąd ponownego uczenia modelu po autoryzacji: {message}")
                        continue

                    if not self._is_showing_dialog:
                        self._is_showing_dialog = True
                        
                        if status == "success":
                            self.view.training_status_label.configure(
                                text=self.ui_i18n.tr("model_ready", "Model ready")
                            )
                            self.view.pose_label.configure(
                                text=self.ui_i18n.tr(
                                    "model_created_saved", "Model created and saved."
                                )
                            )
                            messagebox.showinfo(
                                self.ui_i18n.tr("success", "Success"), message
                            )
                            self.view.show_frame("SettingsView")
                        elif status == "error":
                            self.view.training_status_label.configure(
                                text=self.ui_i18n.tr("training_error", "Training error")
                            )
                            self.view.pose_label.configure(
                                text=self.ui_i18n.tr(
                                    "training_failed", "Training failed."
                                )
                            )
                            messagebox.showerror(
                                self.ui_i18n.tr("registration_error", "Registration error"),
                                message,
                            )
                            if self.user_dir and self.user_dir.exists():
                                shutil.rmtree(self.user_dir, ignore_errors=True)
                            user = safe_user_name(self.view.user_var.get())
                            user_csv = PROFILE_ROOT / user / f"{user}.csv"
                            if user_csv.exists():
                                try:
                                    user_csv.unlink()
                                except OSError:
                                    pass
                            verifier_path = TRAIN_DIR / "outputs" / f"verifier_{user}.pkl"
                            if verifier_path.exists():
                                try:
                                    verifier_path.unlink()
                                except OSError:
                                    pass
                            self.user_dir = None
                            self.saved_files.clear()
                            self.current_pose_idx = 0
                            self.current_shot_idx = 0
                            self.stable_count = 0
                            self.update_pose_ui()
                            self.view.capture_btn.configure(state="normal")
                            self.view.train_btn.configure(state="disabled")
                            self.view.show_frame("RegisterView")
                            
                        self._is_showing_dialog = False
                else:
                    self.log(msg)
        except queue.Empty:
            pass

        if not self._closing:
            try:
                self.root.after(200, self.drain_training_log)
            except tk.TclError:
                pass
           
    def generate_and_show_qr(self):
        global_dir = PROFILE_ROOT.parent
        global_dir.mkdir(parents=True, exist_ok=True)

        secret = self.auth_manager.load_secret(global_dir)
        if not secret:
            secret = self.auth_manager.generate_secret()
            if not self.auth_manager.save_secret(global_dir, secret):
                messagebox.showerror(
                    self.ui_i18n.tr("error", "Error"),
                    self.ui_i18n.tr(
                        "auth_secret_save_failed",
                        "Failed to save the authenticator secret.",
                    ),
                )
                return

        uri = self.auth_manager.get_auth_uri(secret, "Weryfikacja")
        qr_imgtk = self.auth_manager.generate_qr_code(uri)

        self.view.lbl_qr_image.configure(image=qr_imgtk)
        self.view.lbl_qr_image.image = qr_imgtk        

    def generate_offline_code(self):
        code = secrets.token_hex(8).upper()
        code_hash = hash_credential(code)
        self.config.set("App", "offline_code_hash", code_hash)
        self.config.set("App", "offline_code_used", "False")
        messagebox.showinfo(
            self.ui_i18n.tr("offline_code_title", "One-time offline code"),
            self.ui_i18n.tr(
                "offline_code_message_fmt",
                "Save this code somewhere safe. It will only be shown once:\n\n{code}\n\nIt will stop working after one successful use.",
            ).format(code=code),
        )

    def delete_user(self):
        user = self.view.settings_user_var.get()
        
        if not user:
            messagebox.showwarning(
                self.ui_i18n.tr("warning", "Ostrzeżenie"), 
                self.ui_i18n.tr("select_user_to_delete", "Wybierz użytkownika do usunięcia.")
            )
            return
        
        ans = messagebox.askyesno(
            self.ui_i18n.tr("delete_user_title", "Usuwanie użytkownika"),
            self.ui_i18n.tr("delete_user_msg_fmt", "Czy na pewno chcesz usunąć użytkownika '{user}' oraz wszystkie jego dane (zdjęcia i model)?").format(user=user)
        )
        
        if not ans:
            return

        try:
            user_dir = PROFILE_ROOT / user
            if user_dir.exists():
                shutil.rmtree(user_dir, ignore_errors=True)
            
            verifier_path = TRAIN_DIR / "outputs" / f"verifier_{user}.pkl"
            if verifier_path.exists():
                verifier_path.unlink()
            
            self.log_tr(f"log_user_deleted_fmt", "Usunięto użytkownika: {user}")
            messagebox.showinfo(
                self.ui_i18n.tr("success", "Sukces"), 
                self.ui_i18n.tr("user_deleted_success_fmt", "Pomyślnie usunięto użytkownika '{user}'.").format(user=user)
            )
            
            updated_users = self.view._get_registered_users()
            self.view.settings_user_combo.configure(values=updated_users)
            self.view.settings_user_var.set("")
            
            self.view._update_auth_users()
                
        except Exception as e:
            messagebox.showerror(
                self.ui_i18n.tr("error", "Error"),
                self.ui_i18n.tr(
                    "user_delete_error_fmt", "Failed to delete user: {error}"
                ).format(error=e),
            )
            self.log_tr("log_user_delete_error_fmt", "Błąd usuwania użytkownika {user}: {exc}", user=user, exc=e)
            
    def on_close(self):
        # Funkcja odpowiedzialna za zamknięcie aplikacji (Ctrl+F9 / przycisk po zalogowaniu).
        # Gdy model się uczy, zamknięcie jest odroczone do końca uczenia - inaczej ucięłoby logi
        # i zostawiło niedokończony model.
        if self._closing or self._close_pending:
            return

        if self.training_running:
            try:
                messagebox.showinfo(
                    self.ui_i18n.tr("closing_title", "Zamykanie aplikacji"),
                    self.ui_i18n.tr(
                        "closing_after_training_msg",
                        "Model nadal się uczy. Aplikacja zamknie się automatycznie zaraz po zakończeniu uczenia, "
                        "żeby nie przerwać modelu ani logów.",
                    ),
                )
            except tk.TclError:
                pass

            # Uczenie mogło skończyć się w czasie, gdy okno było otwarte
            if self.training_running:
                self._close_pending = True
                self._cancel_authorization()
                self.cancel_silent_timer()
                self.close_camera()
                self.root.withdraw()
                return

        self._finalize_close()

    def _finalize_close(self):
        # Funkcja odpowiedzialna za faktyczne zwolnienie zasobów i zamknięcie aplikacji.
        if self._closing:
            return
        self._closing = True
        self._close_pending = False
        self.cancel_silent_timer()
        if self.cap is not None:
            self.cap.release()

        self._face_detection_stop.set()
        self._face_detection_event.set()
        face_detection_thread = getattr(self, "_face_detection_thread", None)
        if face_detection_thread is not None:
            face_detection_thread.join(timeout=1.0)

        try:
            self.liveness_mesh.close()
        except Exception:
            pass

        try:
            if not face_detection_thread or not face_detection_thread.is_alive():
                self.preview_face_detector.close()
            self.face_detector.close()
            self.face_mesh.close()
        except Exception:
            pass

        # Zabezpieczenie: jeśli proces nie zakończy się sam po zamknięciu okna, kończymy go po 5 s
        def _exit_watchdog():
            time.sleep(5)
            try:
                logging.shutdown()
            except Exception:
                pass
            os._exit(0)

        threading.Thread(target=_exit_watchdog, name="exit-watchdog", daemon=True).start()

        try:
            self.root.destroy()
        except tk.TclError:
            pass