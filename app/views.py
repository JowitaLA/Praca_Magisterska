from __future__ import annotations

import tkinter as tk
import tkinter.messagebox as messagebox
import ttkbootstrap as ttk
from ttkbootstrap.scrolled import ScrolledFrame
from ttkbootstrap.constants import *
from typing import List, Tuple
from pathlib import Path
import ctypes
from resources.config import DATA_DIR
from .system_lock import SystemLockManager
from .config import POSES, SHOTS_PER_POSE, APP_NAME, ICON_PATH
from .credential_hash import hash_credential
from .security_levels import SECURITY_LEVELS, LEVEL_KEYS, DEFAULT_LEVEL, normalize_level


class MainView:
    
    def __init__(self, root: tk.Tk, cameras: List[Tuple[int, str]], lang_manager, callbacks: dict, config_manager):
        self.root = root
        self.cameras = cameras
        self.i18n = lang_manager
        self.callbacks = callbacks
        self.config_manager = config_manager
        
        # 1. Inicjalizacja menedżera blokady ekranów i traya
        self.lock_manager = SystemLockManager(
            root=self.root,
            on_restore_callback=lambda: self.show_frame("AuthView"),
            on_close_callback=self.close_app,
            i18n=self.i18n,
            on_open_settings_callback=self.callbacks.get(
                "on_restore_from_tray",
                lambda: self.show_frame("SettingsView"),
            )
        )
        
        # Wymuszenie pełnego ekranu
        self.root.attributes('-fullscreen', True)
        self.root.attributes("-topmost", True)
        
        self.root.protocol("WM_DELETE_WINDOW", lambda: None)  # Blokuje Alt+F4 oraz sygnały zamknięcia
        self.root.bind("<Escape>", lambda e: "break")         # Blokuje klawisz ESC
        self.root.bind("<F11>", lambda e: "break")            # Blokuje domyślne przełączanie F11
        self.root.bind("<Alt-Tab>", lambda e: "break")        # Próba przechwycenia przełączania okien
        
        # Awaryjne zamknięcie Ctrl+F9
        self.root.bind_all("<Control-F9>", self._on_emergency_close)

        # Kontener na wszystkie widoki
        self.container = ttk.Frame(self.root)
        self.container.pack(fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        self._build_auth_view()
        self._build_settings_view()
        self._build_register_view()

        self.current_view = None
        self.retranslate_ui()
        
        # 2. Aktywacja blokady dodatkowych monitorów i uwięzienie myszy
        self.lock_manager.lock_all()
        
        # Wymuszenie ikony na pasku zadań i w przełączniku Alt+Tab (Windows API)
        try:
            myappid = 'nuraface.biometrics.app.1.0'
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except Exception:
            pass
        
        # Ustawienie ikony okna głównego
        if ICON_PATH.exists():
            try:
                self.root.iconbitmap(str(ICON_PATH))
            except Exception:
                pass
    
    # PRZEŁĄCZANIE WIDOKU
    def show_frame(self, frame_name: str):
        self.current_view = frame_name
        frame = self.frames[frame_name]
        
        if frame_name == "AuthView":
            self._update_auth_users()
        elif frame_name == "SettingsView":
            self._refresh_registered_users()
            if hasattr(self, "security_scale"):
                self._refresh_security_desc(self._security_index())
            
        frame.tkraise()
        
        if "on_view_changed" in self.callbacks:
            self.callbacks["on_view_changed"](frame_name)
    
    def set_authorized_mode(self, is_authorized: bool):
        # Odpinamy na moment oba przyciski nawigacyjne, by zachować odpowiednią kolejność na ekranie
        self.btn_go_register.pack_forget()
        self.btn_go_auth.pack_forget()
        
        if is_authorized or not self._get_registered_users():
            self.protected_settings_frame.pack(fill="both", expand=True, side="top", pady=(0, 10))
            self.btn_go_register.pack(side="left", padx=(0, 10))
        else:
            self.protected_settings_frame.pack_forget()

        # Przyciski "MINIMALIZUJ DO TŁA" i "ZAMKNIJ APLIKACJĘ" tylko dla zalogowanego użytkownika.
        # Poza tym aplikację zamyka wyłącznie skrót Ctrl+F9. Kolejność: minimalizuj na dole, zamknij nad nim.
        self.btn_minimize.pack_forget()
        self.btn_close.pack_forget()
        if is_authorized and self._get_registered_users():
            self.btn_minimize.pack(side="bottom", pady=(0, 5))
            self.btn_close.pack(side="bottom")
            
        # Przycisk "Wróć do autoryzacji" ustawiamy z powrotem, by zawsze był widoczny na samym końcu
        self.btn_go_auth.pack(side="left")
            
    def _get_registered_users(self) -> list[str]:
        profile_dir = DATA_DIR / "profile"
        if not profile_dir.exists():
            return []
        users = [
            d.name for d in profile_dir.iterdir()
            if d.is_dir()
            and d.name != "FairFace"
            and (d / f"{d.name}.csv").is_file()
        ]
        return users

    def _refresh_registered_users(self):
        users = self._get_registered_users()
        self.settings_user_combo.configure(values=users)
        self.settings_user_var.set(users[0] if users else "")
        self.auth_user_combo.configure(values=users)
        self.auth_user_var.set(users[0] if users else "")
    
    def _save_setting(self, key: str, value: str | bool):
        self.config_manager.set("App", key, str(value))
        if key == "autostart":
            self.config_manager.apply_autostart(bool(value))
        # Gwarantuje, że wprowadzane ustawienia zapiszą się fizycznie na dysku
        if hasattr(self.config_manager, "save"):
            self.config_manager.save()

    # ---- Suwak poziomu zabezpieczeń ----
    def _security_index(self) -> int:
        return int(round(float(self.security_scale.get())))

    def _on_security_scale_move(self, value):
        # Przyciąganie suwaka do najbliższego z 3 poziomów
        if self._security_snapping:
            return
        idx = int(round(float(value)))
        self._security_snapping = True
        try:
            self.security_scale.set(idx)
        finally:
            self._security_snapping = False
        self._refresh_security_desc(idx, with_hint=False)

    def _on_security_scale_release(self, event=None):
        self._apply_security_level(self._security_index())

    def _set_security_level_index(self, idx: int):
        self.security_scale.set(idx)
        self._apply_security_level(idx)

    def _apply_security_level(self, idx: int):
        key = LEVEL_KEYS[idx]
        self._refresh_security_desc(idx)
        if key == self._applied_security_key:
            return
        self._applied_security_key = key
        self._save_setting("security_level", key)
        cb = self.callbacks.get("on_security_level_change")
        if cb:
            cb(key)

    def _refresh_security_desc(self, idx: int, with_hint: bool = True):
        key, _name, _far, _desc = SECURITY_LEVELS[idx]
        hint = ""
        cb = self.callbacks.get("get_security_level_hint")
        if not hasattr(self, "lbl_security_desc"):
            return
        if with_hint and cb:
            try:
                hint = cb(key) or ""
            except Exception:
                hint = ""
        description = self.i18n.tr(f"security_description_{key}", _desc)
        self.lbl_security_desc.configure(
            text=description + (("\n" + hint) if hint else "")
        )

    def _on_ui_lang_change(self, event=None):
        new_lang = self.lang_var.get()
        self._save_setting("ui_lang", new_lang)
        if "on_language_changed" in self.callbacks:
            self.callbacks["on_language_changed"](new_lang)
        self.retranslate_ui()

    def _on_log_lang_change(self, event=None):
        new_lang = self.log_lang_var.get()
        self._save_setting("log_lang", new_lang)
        if "on_log_language_changed" in self.callbacks:
            self.callbacks["on_log_language_changed"](new_lang)

    def _on_theme_change(self, event=None):
        labels_to_keys = {
            self.i18n.tr("theme_dark", "Ciemny"): "dark",
            self.i18n.tr("theme_light", "Jasny"): "light",
        }
        new_theme = labels_to_keys.get(self.theme_var.get(), "dark")
        self._save_setting("theme", new_theme)
        if "on_theme_changed" in self.callbacks:
            self.callbacks["on_theme_changed"](new_theme)
            
    # Funkcja do zmiany hasła
    def _on_change_password(self):
        new_pass = self.settings_new_pass_var.get()
        repeat_pass = self.settings_repeat_pass_var.get()

        if not new_pass or not repeat_pass:
            messagebox.showwarning(
                self.i18n.tr("missing_password_title", "Brak danych"),
                self.i18n.tr(
                    "password_fields_required",
                    "Wypełnij oba pola przed zmianą hasła.",
                ),
            )
            return

        if new_pass != repeat_pass:
            messagebox.showerror(
                self.i18n.tr("error", "Błąd"),
                self.i18n.tr("passwords_do_not_match", "Podane hasła nie są identyczne!"),
            )
            return

        # Zapisanie zahashowanego hasła w ustawieniach
        hashed_pass = hash_credential(new_pass)
        self._save_setting("saved_password", hashed_pass)

        # Wyczyszczenie pól
        self.settings_new_pass_var.set("")
        self.settings_repeat_pass_var.set("")
        
        messagebox.showinfo(
            self.i18n.tr("success", "Sukces"),
            self.i18n.tr("password_updated", "Hasło zostało pomyślnie zaktualizowane."),
        )
        
    # ==========================================
    # WIDOK 1: AUTORYZACJA
    # ==========================================
    def _build_auth_view(self):
        frame = ttk.Frame(self.container, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")
        self.frames["AuthView"] = frame

        # MENU
        top_menu = ttk.Frame(frame)
        top_menu.pack(fill="x", pady=(0, 10))

        # Lista użytkowników 
        self.auth_user_var = tk.StringVar()
        self.auth_user_combo = ttk.Combobox(top_menu, textvariable=self.auth_user_var, state="readonly", width=20)
        self.auth_user_combo.pack(side="left", padx=(0, 5))

        # Przycisk do logowania awaryjnego kodem jednorazowym
        self.btn_fallback_auth = ttk.Button(
            top_menu, 
            text="Zaloguj kodem jednorazowym", 
            bootstyle="secondary",
            command=self.callbacks.get("on_fallback_auth_clicked")
        )
        self.btn_fallback_auth.pack(side="left", padx=5)
        
        # Etykieta komunikatów
        self.msg_label = ttk.Label(top_menu, text="", font=("Segoe UI", 10, "bold"))
        self.msg_label.pack(side="left", padx=10)

        # Ustawienia
        self.btn_settings = ttk.Button(top_menu, text="", bootstyle="secondary", command=lambda: self.show_frame("SettingsView"))
        self.btn_settings.pack(side="right")

        # Główny widok
        cam_frame = ttk.Frame(frame)
        cam_frame.pack(fill="both", expand=True)
        self.video_label_auth = ttk.Label(cam_frame, anchor="center")
        self.video_label_auth.pack(fill="both", expand=True)
        
        self.video_label = self.video_label_auth 

    def _update_auth_users(self):
        users = self._get_registered_users()
        if not users:
            self.auth_user_combo.configure(values=[])
            self.auth_user_var.set("")
            # Brak użytkowników – automatycznie przejdź do ustawień
            if self.current_view == "AuthView":
                self.show_frame("SettingsView")
        else:
            self.auth_user_combo.configure(values=users)
            self.auth_user_var.set(users[0])

    # ==========================================
    # WIDOK 2: USTAWIENIA
    # ==========================================
    def _on_emergency_close(self, event=None):
        self.close_app()
        return "break"
    
    def close_app(self):
        # Funkcja odpowiedzialna za odblokowanie systemu i zamknięcie aplikacji.
        # 1. Zwolnij kursor myszy
        self.lock_manager.cleanup()
        
        on_close = self.callbacks.get("on_close")
        if callable(on_close):
            on_close()
        else:
            self.root.destroy()
                    
    def _build_settings_view(self):
        frame = ttk.Frame(self.container, padding=30)
        frame.grid(row=0, column=0, sticky="nsew")
        self.frames["SettingsView"] = frame
        # PRZYPIĘCIE PRZYCISKÓW NA STAŁE DO DOŁU EKRANU
        bottom_frame = ttk.Frame(frame)
        bottom_frame.pack(side="bottom", fill="x", pady=(10, 0))

        nav_frame = ttk.Frame(bottom_frame)
        nav_frame.pack(fill="x", pady=(0, 10))
        
        self.btn_go_register = ttk.Button(nav_frame, text="", bootstyle="success", command=lambda: self.show_frame("RegisterView"))
        self.btn_go_register.pack(side="left", padx=(0, 10))
                
        self.btn_go_auth = ttk.Button(nav_frame, text="", bootstyle="secondary", command=lambda: self.show_frame("AuthView"))
        self.btn_go_auth.pack(side="left")
        
        # Przycisk minimalizacji do zasobnika systemowego - tworzony zawsze,
        # ale pokazywany dopiero po zalogowaniu (patrz set_authorized_mode)
        self.btn_minimize = ttk.Button(
            bottom_frame, 
            text="", 
            command=self.lock_manager.minimize_to_tray, 
            bootstyle="info", 
            width=30
        )
        
        self.btn_close = ttk.Button(
            bottom_frame, 
            text="", 
            command=self.close_app, 
            bootstyle="danger", 
            width=30
        )
        # Przycisk pokazywany tylko po zalogowaniu - patrz set_authorized_mode()
        
        content = ScrolledFrame(frame, autohide=True)
        content.pack(expand=True, fill="both", side="top", pady=(0, 10))

        self.lbl_settings_title = ttk.Label(content, text="", font=("Segoe UI", 20, "bold"))
        self.lbl_settings_title.pack(pady=(0, 24), anchor="w")

        row1 = ttk.Frame(content)
        row1.pack(fill="x", pady=5)
        self.lbl_camera = ttk.Label(row1, width=20, text="")
        self.lbl_camera.pack(side="left")
        
        camera_names = [name for _, name in self.cameras] or [
            self.i18n.tr("no_camera_detected", "Brak wykrytej kamery")
        ]
        saved_camera = self.config_manager.get("App", "camera_name", fallback="Default")
        if saved_camera not in camera_names:
            saved_camera = camera_names[0]
        self.camera_var = tk.StringVar(value=saved_camera)
        
        self.camera_combo = ttk.Combobox(row1, textvariable=self.camera_var, values=camera_names, state="readonly", width=30)
        self.camera_combo.pack(side="left")
        self.camera_combo.bind("<<ComboboxSelected>>", lambda e: [self._save_setting("camera_name", self.camera_var.get()), self.callbacks.get("on_camera_change", lambda: None)()])

        row2 = ttk.Frame(content)
        row2.pack(fill="x", pady=5)
        self.lbl_lang = ttk.Label(row2, width=20, text="")
        self.lbl_lang.pack(side="left")
        
        saved_ui_lang = self.config_manager.get("App", "ui_lang", fallback="pl_PL")
        self.lang_var = tk.StringVar(value=saved_ui_lang)
        self.lang_combo = ttk.Combobox(row2, textvariable=self.lang_var, values=["pl_PL", "en_US"], state="readonly", width=10)
        self.lang_combo.pack(side="left", padx=(0, 10))
        self.lang_combo.bind("<<ComboboxSelected>>", self._on_ui_lang_change)

        self.lbl_log_lang = ttk.Label(row2, text="")
        self.lbl_log_lang.pack(side="left", padx=(10, 5))
        
        saved_log_lang = self.config_manager.get("App", "log_lang", fallback="en_US")
        self.log_lang_var = tk.StringVar(value=saved_log_lang)
        self.log_lang_combo = ttk.Combobox(row2, textvariable=self.log_lang_var, values=["pl_PL", "en_US"], state="readonly", width=10)
        self.log_lang_combo.pack(side="left")
        self.log_lang_combo.bind("<<ComboboxSelected>>", self._on_log_lang_change)

        row_theme = ttk.Frame(content)
        row_theme.pack(fill="x", pady=5)
        self.lbl_theme = ttk.Label(row_theme, width=20, text="")
        self.lbl_theme.pack(side="left")

        saved_theme = self.config_manager.get("App", "theme", fallback="dark")
        theme_labels = [
            self.i18n.tr("theme_dark", "Ciemny"),
            self.i18n.tr("theme_light", "Jasny"),
        ]
        self.theme_var = tk.StringVar(value=theme_labels[1 if saved_theme == "light" else 0])
        self.theme_combo = ttk.Combobox(
            row_theme,
            textvariable=self.theme_var,
            values=theme_labels,
            state="readonly",
            width=15,
        )
        self.theme_combo.pack(side="left")
        self.theme_combo.bind("<<ComboboxSelected>>", self._on_theme_change)

        ttk.Separator(content, orient="horizontal").pack(fill="x", pady=(14, 12))

        # 1. Inicjalizacja chronionej ramki dużo Wcześniej!
        self.protected_settings_frame = ttk.Frame(content)

        # 2. Przenosimy sprawdzanie obecności z ogólnego 'content' do 'self.protected_settings_frame'
        row3 = ttk.Frame(self.protected_settings_frame)
        row3.pack(fill="x", pady=8)
        self.lbl_presence = ttk.Label(row3, text="", width=22)
        self.lbl_presence.pack(side="left")
        
        self.presence_var = tk.StringVar()
        self.presence_combo = ttk.Combobox(
            row3,
            textvariable=self.presence_var,
            state="readonly",
            width=22,
            height=8,
        )
        self.presence_combo.pack(side="left")
        self.presence_combo.bind(
            "<<ComboboxSelected>>",
            lambda e: self._save_setting("presence_check", self.presence_var.get()),
        )

        # --- Poziom zabezpieczeń (docelowy FAR): suwak z 3 poziomami ---
        row_level = ttk.Frame(self.protected_settings_frame)
        row_level.pack(fill="x", pady=(10, 0))
        self.lbl_security_level = ttk.Label(row_level, text="", width=20)
        self.lbl_security_level.pack(side="left", anchor="n")

        saved_level = normalize_level(self.config_manager.get("App", "security_level", fallback=DEFAULT_LEVEL))
        self._applied_security_key = saved_level
        self._security_snapping = False

        slider_box = ttk.Frame(row_level)
        slider_box.pack(side="left")
        self.security_scale = ttk.Scale(
            slider_box, from_=0, to=len(LEVEL_KEYS) - 1, orient="horizontal",
            length=260, command=self._on_security_scale_move,
        )
        # Ustawienie wartości początkowej wywołuje callback suwaka, a etykieta
        # opisu jeszcze nie istnieje - blokujemy go na ten moment.
        self._security_snapping = True
        try:
            self.security_scale.set(LEVEL_KEYS.index(saved_level))
        finally:
            self._security_snapping = False
        self.security_scale.pack(fill="x")
        self.security_scale.bind("<ButtonRelease-1>", self._on_security_scale_release)

        names_row = ttk.Frame(slider_box)
        names_row.pack(fill="x")
        _sticky = ("w", "", "e")
        self.security_name_labels = []
        for i, (_key, _name, _far, _desc) in enumerate(SECURITY_LEVELS):
            names_row.columnconfigure(i, weight=1)
            lbl = ttk.Label(names_row, text="", cursor="hand2")
            lbl.grid(row=0, column=i, sticky=_sticky[i])
            lbl.bind("<Button-1>", lambda e, idx=i: self._set_security_level_index(idx))
            self.security_name_labels.append(lbl)

        self.lbl_security_desc = ttk.Label(self.protected_settings_frame, text="", wraplength=560, justify="left")
        self.lbl_security_desc.pack(fill="x", pady=(2, 5))
        self._refresh_security_desc(LEVEL_KEYS.index(saved_level), with_hint=False)

        row_train_after_auth = ttk.Frame(self.protected_settings_frame)
        row_train_after_auth.pack(fill="x", pady=5)
        train_after_auth = self.config_manager.getboolean(
            "App", "train_after_auth", fallback=False
        )
        self.train_after_auth_var = tk.BooleanVar(value=train_after_auth)
        self.chk_train_after_auth = ttk.Checkbutton(
            row_train_after_auth,
            text="",
            variable=self.train_after_auth_var,
            bootstyle="round-toggle",
            command=lambda: self._save_setting(
                "train_after_auth", self.train_after_auth_var.get()
            ),
        )
        self.chk_train_after_auth.pack(side="left")

        row_keep_authorized = ttk.Frame(self.protected_settings_frame)
        row_keep_authorized.pack(fill="x", pady=5)
        keep_authorized = self.config_manager.getboolean(
            "App", "keep_authorized_on_restore", fallback=False
        )
        self.keep_authorized_on_restore_var = tk.BooleanVar(value=keep_authorized)
        self.chk_keep_authorized_on_restore = ttk.Checkbutton(
            row_keep_authorized,
            text="",
            variable=self.keep_authorized_on_restore_var,
            bootstyle="round-toggle",
            command=lambda: self._save_setting(
                "keep_authorized_on_restore",
                self.keep_authorized_on_restore_var.get(),
            ),
        )
        self.chk_keep_authorized_on_restore.pack(side="left")

        # Od tego momentu rozpoczynają się opcje Autostartu, usuwając w ten sposób lukę widoczności
        row4 = ttk.Frame(self.protected_settings_frame)
        row4.pack(fill="x", pady=5)

        # row3 = ttk.Frame(content)
        # row3.pack(fill="x", pady=5)
        # self.lbl_presence = ttk.Label(row3, text="Sprawdzanie obecności:", width=20)
        # self.lbl_presence.pack(side="left")
        
        # saved_presence = self.config_manager.get("App", "presence_check", fallback="Brak")
        # self.presence_var = tk.StringVar(value=saved_presence)
        # cb_presence = ttk.Combobox(row3, textvariable=self.presence_var, values=["Brak", "Co 1 godzinę", "Co 2 godziny"], state="readonly", width=15)
        # cb_presence.pack(side="left")
        # cb_presence.bind("<<ComboboxSelected>>", lambda e: self._save_setting("presence_check", self.presence_var.get()))

        # self.protected_settings_frame = ttk.Frame(content)
        
        # # 1. Autostart
        # row4 = ttk.Frame(self.protected_settings_frame)
        # row4.pack(fill="x", pady=5)
        saved_autostart = self.config_manager.getboolean("App", "autostart", fallback=False)
        self.autostart_var = tk.BooleanVar(value=saved_autostart)
        self.chk_autostart = ttk.Checkbutton(row4, text="", variable=self.autostart_var, bootstyle="round-toggle", command=lambda: self._save_setting("autostart", self.autostart_var.get()))
        self.chk_autostart.pack(side="left")

        ttk.Separator(self.protected_settings_frame, orient="horizontal").pack(fill="x", pady=20)

        # 2. Zarządzanie kontami
        self.lbl_account_manager = ttk.Label(self.protected_settings_frame, text="", font=("Segoe UI", 14, "bold"))
        self.lbl_account_manager.pack(anchor="w", pady=(0, 10))
        
        row5 = ttk.Frame(self.protected_settings_frame)
        row5.pack(fill="x", pady=5)
        self.settings_user_var = tk.StringVar()
        self.settings_user_combo = ttk.Combobox(row5, textvariable=self.settings_user_var, state="readonly", width=20)
        self.settings_user_combo.pack(side="left", padx=(0, 10))
        self.btn_delete_account = ttk.Button(row5, text="", bootstyle="danger-outline", command=self.callbacks.get("on_delete_user"))        
        self.btn_delete_account.pack(side="left", padx=5)
        
        ttk.Separator(self.protected_settings_frame, orient="horizontal").pack(fill="x", pady=20)
        
        # 3. Zarządzanie weryfikacją awaryjną
        self.lbl_pass_manager = ttk.Label(self.protected_settings_frame, text="", font=("Segoe UI", 14, "bold"))
        self.lbl_pass_manager.pack(anchor="w", pady=(0, 10))
        
        row6 = ttk.Frame(self.protected_settings_frame)
        row6.pack(fill="x", pady=5)
        saved_password = self.config_manager.getboolean("App", "password", fallback=True)
        self.password_var = tk.BooleanVar(value=saved_password)
        self.chk_password = ttk.Checkbutton(row6, text="", variable=self.password_var, bootstyle="round-toggle", command=lambda: self._save_setting("password", self.password_var.get()))
        self.chk_password.pack(side="left")
        
        row7 = ttk.Frame(self.protected_settings_frame)
        row7.pack(fill="x", pady=5)
        saved_auth_key = self.config_manager.getboolean("App", "auth_key", fallback=True)
        self.auth_key_var = tk.BooleanVar(value=saved_auth_key)
        self.chk_auth_key = ttk.Checkbutton(row7, text="", variable=self.auth_key_var, bootstyle="round-toggle", command=lambda: self._save_setting("auth_key", self.auth_key_var.get()))
        self.chk_auth_key.pack(side="left")
        
        # Zmiana hasła
        self.lbl_pass_manager_title_pwd = ttk.Label(self.protected_settings_frame, text="", font=("Segoe UI", 10, "bold"))
        self.lbl_pass_manager_title_pwd.pack(anchor="w", pady=(10, 5))

        inputs_frame = ttk.Frame(self.protected_settings_frame)
        inputs_frame.pack(fill="x", pady=5)
        
        self.lbl_new_password = ttk.Label(inputs_frame, text="")
        self.lbl_new_password.pack(side="left", padx=(0, 2))
        self.settings_new_pass_var = tk.StringVar()
        ttk.Entry(inputs_frame, textvariable=self.settings_new_pass_var, show="*", width=10).pack(side="left", padx=(0, 10))
        
        self.lbl_repeat_password = ttk.Label(inputs_frame, text="")
        self.lbl_repeat_password.pack(side="left", padx=(0, 2))
        self.settings_repeat_pass_var = tk.StringVar()
        ttk.Entry(inputs_frame, textvariable=self.settings_repeat_pass_var, show="*", width=10).pack(side="left", padx=(0, 10))
        
        buttons_frame = ttk.Frame(self.protected_settings_frame)
        buttons_frame.pack(fill="x", pady=5)

        self.btn_change_pass = ttk.Button(
            buttons_frame, 
            text="", 
            bootstyle="info",
            command=self._on_change_password
        )
        self.btn_change_pass.pack(side="left", padx=(0, 5))
        
        # QR kod
        self.lbl_pass_manager_title_qr = ttk.Label(self.protected_settings_frame, text="", font=("Segoe UI", 10, "bold"))
        self.lbl_pass_manager_title_qr.pack(anchor="w", pady=(10, 5))
        
        qr_frame = ttk.Frame(self.protected_settings_frame)
        qr_frame.pack(fill="x", pady=5)
        
        self.btn_generate_qr = ttk.Button(qr_frame, text="", bootstyle="info", command=self.callbacks.get("on_generate_qr"))
        self.btn_generate_qr.pack(side="top", anchor="w", pady=5)

        self.btn_generate_offline_code = ttk.Button(
            qr_frame,
            text="",
            bootstyle="warning",
            command=self.callbacks.get("on_generate_offline_code"),
        )
        self.btn_generate_offline_code.pack(side="top", anchor="w", pady=5)

        self.btn_offline_login = ttk.Button(
            qr_frame,
            text="",
            bootstyle="secondary",
            command=self.callbacks.get("on_fallback_auth_clicked"),
        )
        self.btn_offline_login.pack(side="top", anchor="w", pady=5)
        
        self.lbl_qr_image = ttk.Label(qr_frame)
        self.lbl_qr_image.pack(side="top", anchor="w")
        # ----------------------------------------------------------------------------

        frame.bind("<Visibility>", lambda e: self.settings_user_combo.configure(values=self._get_registered_users()))        


    # ==================== #
    # WIDOK 3: REJESTRACJA #
    # ==================== #
    def _build_register_view(self):
        frame = ttk.Frame(self.container, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")
        self.frames["RegisterView"] = frame

        # Górny panel (Menu) - teraz zawiera dane, przycisk rejestracji, pasek postępu i akcje
        top_reg = ttk.Frame(frame)
        top_reg.pack(fill="x", pady=(0, 10))

        # 1. Dane wejściowe użytkownika po lewej stronie menu
        inputs_frame = ttk.Frame(top_reg)
        inputs_frame.pack(side="left", fill="x")
        
        self.user_var = tk.StringVar()
        ttk.Entry(inputs_frame, textvariable=self.user_var, width=14).pack(side="left", padx=(0, 10))

        self.btn_register = ttk.Button(top_reg, command=self._handle_register_click, bootstyle="success")
        self.btn_register.pack(side="left", padx=(0, 10))

        # 2. Pasek postępu i przyciski umieszczone bezpośrednio w menu górnym
        action_menu_frame = ttk.Frame(top_reg)
        action_menu_frame.pack(side="left", fill="x", expand=True, padx=10)

        total_needed = len(POSES) * SHOTS_PER_POSE
        self.progress = ttk.Progressbar(action_menu_frame, maximum=total_needed, value=0, length=130, bootstyle="success-striped")
        self.progress.pack(side="left", padx=(0, 10))

        self.capture_btn = ttk.Button(action_menu_frame, command=self.callbacks.get("on_capture"), state="disabled", bootstyle="success")
        self.capture_btn.pack(side="left", padx=4)

        self.train_btn = ttk.Button(action_menu_frame, command=self.callbacks.get("on_train"), state="disabled", bootstyle="warning")
        self.train_btn.pack(side="left", padx=4)

        # 3. Przycisk powrotu po prawej stronie menu
        self.btn_cancel_reg = ttk.Button(top_reg, text="", bootstyle="secondary", command=lambda: self.show_frame("SettingsView"))
        self.btn_cancel_reg.pack(side="right")

        self.pose_label = ttk.Label(frame, text="", font=("Segoe UI", 13, "bold"))
        self.training_status_label = ttk.Label(frame, text="", font=("Segoe UI", 10, "bold"))
        self.pose_label.pack(anchor="w")
        self.training_status_label.pack(anchor="w", pady=(4, 0))

        # Główny obszar kamery – zajmuje całą pozostałą przestrzeń pod menu
        cam_container = ttk.Frame(frame)
        cam_container.pack(fill="both", expand=True)

        self.video_label_reg = ttk.Label(cam_container, anchor="center")
        self.video_label_reg.pack(fill="both", expand=True)
    def _handle_register_click(self):
        username = self.user_var.get().strip()
        existing_users = self._get_registered_users()
        profile_dir = DATA_DIR / "profile" / username
        verifier_path = DATA_DIR.parent / "modeling" / "outputs" / f"verifier_{username}.pkl"

        if username.lower() == "fairface":
            tk.messagebox.showerror(
                self.i18n.tr("error", "Błąd"),
                self.i18n.tr(
                    "reserved_username",
                    "Nazwa użytkownika 'FairFace' jest zarezerwowana przez system.",
                ),
            )
            return
        if username in existing_users and verifier_path.exists():
            tk.messagebox.showerror(
                self.i18n.tr("error", "Błąd"),
                self.i18n.tr(
                    "user_already_exists",
                    "Użytkownik o takiej nazwie już istnieje.",
                ),
            )
            return
        if not username:
            tk.messagebox.showerror(
                self.i18n.tr("error", "Błąd"),
                self.i18n.tr("enter_username_error", "Podaj nazwę użytkownika."),
            )
            return

        self.callbacks.get("on_register")()

    def retranslate_ui(self):
        tr = self.i18n.tr
        self.root.title(tr("app_title", APP_NAME))
        self.btn_settings.configure(text=tr("settings_button", "⚙ Ustawienia"))
        self.btn_cancel_reg.configure(text=tr("settings_button", "⚙ Ustawienia"))
        self.btn_register.configure(text=self.i18n.tr("btn_register", "Zarejestruj"))
        self.capture_btn.configure(text=self.i18n.tr("btn_manual_capture", "Zrób zdjęcie ręcznie"))
        self.train_btn.configure(text=self.i18n.tr("btn_train", "Naucz użytkownika"))
        self.pose_label.configure(text=self.i18n.tr("registration_instruction", "Wpisz nową nazwę konta i wciśnij rejestruj"))
        
        self.lbl_settings_title.configure(text=tr("settings_title", "Ustawienia Aplikacji"))
        self.lbl_camera.configure(text=tr("camera_label", "Kamera:"))
        if not self.cameras:
            no_camera_text = tr("no_camera_detected", "Brak wykrytej kamery")
            self.camera_combo.configure(values=[no_camera_text])
            self.camera_var.set(no_camera_text)
        self.lbl_lang.configure(text=tr("language_label", "Język UI:"))
        self.lbl_log_lang.configure(text=tr("log_language_label", "Język logów:"))
        self.lbl_theme.configure(text=tr("theme_label", "Motyw:"))
        self.theme_combo.configure(values=[
            self.i18n.tr("theme_dark", "Ciemny"),
            self.i18n.tr("theme_light", "Jasny"),
        ])
        saved_theme = self.config_manager.get("App", "theme", fallback="dark")
        self.theme_var.set(self.i18n.tr("theme_light", "Jasny") if saved_theme == "light" else self.i18n.tr("theme_dark", "Ciemny"))
        presence_options = (
            ("presence_never", ("Brak", "Never")),
            ("presence_every_minute", ("Co minutę", "Every minute")),
            ("presence_every_5_minutes", ("Co 5 minut", "Every 5 minutes")),
            ("presence_every_10_minutes", ("Co 10 minut", "Every 10 minutes")),
            ("presence_every_30_minutes", ("Co 30 minut", "Every 30 minutes")),
            ("presence_every_hour", ("Co 1 godzinę", "Every hour")),
            ("presence_every_2_hours", ("Co 2 godziny", "Every 2 hours")),
            ("presence_every_5_hours", ("Co 5 godzin", "Every 5 hours")),
        )
        saved_presence = self.config_manager.get("App", "presence_check", fallback="Co 1 godzinę")
        selected_presence_key = next(
            (
                key
                for key, legacy_values in presence_options
                if saved_presence in legacy_values
            ),
            "presence_never",
        )
        self.presence_combo.configure(
            values=[tr(key) for key, _legacy_values in presence_options]
        )
        self.presence_var.set(tr(selected_presence_key))
        self.lbl_presence.configure(text=tr("presence_check_label", "Sprawdzanie obecności:"))
        self.lbl_security_level.configure(text=tr("security_level_label", "Poziom zabezpieczeń:"))
        for label, (level_key, _name, _far, _desc) in zip(
            self.security_name_labels, SECURITY_LEVELS
        ):
            label.configure(text=tr(f"security_{level_key}", _name))
        self.btn_minimize.configure(text=tr("btn_minimize", "MINIMALIZUJ DO TŁA"))
        self.btn_close.configure(text=tr("btn_close_app", "ZAMKNIJ APLIKACJĘ"))
        self.btn_go_register.configure(text=tr("btn_go_register", "Rejestracja Użytkownika"))
        self.btn_go_auth.configure(text=tr("btn_go_auth", "Autoryzacja Użytkownika"))
        self.btn_fallback_auth.configure(text=tr("btn_fallback_auth", "Zaloguj kodem jednorazowym"))
        self.chk_train_after_auth.configure(
            text=tr(
                "train_after_auth",
                "Douczaj się po każdym udanym logowaniu",
            )
        )
        self.chk_keep_authorized_on_restore.configure(
            text=tr(
                "keep_authorized_on_restore",
                "Nie wylogowuj po powrocie do ustawień",
            )
        )
        self.chk_autostart.configure(text=tr("is_autostart", "Uruchamiaj aplikację przy starcie systemu (Autostart)"))
        self.lbl_account_manager.configure(text=tr("account_manager", "Zarządzanie kontami"))
        self.btn_delete_account.configure(text=tr("btn_delete_account", "Usuń konto"))
        self.lbl_pass_manager.configure(text=tr("pass_manager", "Zarządzanie weryfikacją awaryjną"))
        self.chk_password.configure(text=tr("is_pass", "Włącz hasło przed weryfikacją twarzy"))
        self.chk_auth_key.configure(text=tr("is_hotp", "Włącz możliwość weryfikacji za pomocą klucza jednorazowego"))
        self.lbl_pass_manager_title_pwd.configure(text=tr("change_password_label", "Zmień hasło:"))
        self.lbl_new_password.configure(text=tr("new_password_label", "Nowe hasło:"))
        self.lbl_repeat_password.configure(text=tr("repeat_password_label", "Powtórz nowe hasło:"))
        self.btn_change_pass.configure(text=tr("btn_change_pass", "Zmień hasło"))
        self.lbl_pass_manager_title_qr.configure(
            text=tr("authenticator_qr_label", "Kod QR do weryfikacji Authenticatorem:")
        )
        self.btn_generate_qr.configure(text=tr("generate_qr", "Generuj / Pokaż kod QR"))
        self.btn_generate_offline_code.configure(
            text=tr("generate_offline_code", "Wygeneruj jednorazowy kod offline")
        )
        self.btn_offline_login.configure(text=tr("offline_login", "Zaloguj kodem offline"))
        
        self._refresh_security_desc(self._security_index())

    def log(self, text: str):
        pass