from __future__ import annotations

import os
import ctypes
import queue
import webbrowser
from ctypes import wintypes
import tkinter as tk
from screeninfo import get_monitors
import pystray
from .config import ICON_PATH, LOG_ROOT
from PIL import Image, ImageDraw
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .i18n import LanguageManager

class SystemLockManager:
    def __init__(
        self,
        root,
        on_restore_callback,
        on_close_callback,
        on_open_settings_callback=None,
        i18n: LanguageManager | None = None,
    ):
        self.root = root
        self.on_restore = on_restore_callback
        self.on_close = on_close_callback
        self.on_open_settings = on_open_settings_callback
        self.i18n = i18n
        self.overlays = []
        self.tray_icon = None
        self._tray_action_queue = queue.Queue()

    def trap_cursor(self):
        # Funkcja odpowiedzialna za zamknięcie kursora w granicach głównego okna.
        self.root.update_idletasks()
        x, y = self.root.winfo_rootx(), self.root.winfo_rooty()
        w, h = self.root.winfo_width(), self.root.winfo_height()
        rect = wintypes.RECT(x, y, x + w, y + h)
        ctypes.windll.user32.ClipCursor(ctypes.byref(rect))

    def release_cursor(self):
        # Funkcja odpowiedzialna za zwolnienie kursora myszy.
        ctypes.windll.user32.ClipCursor(None)

    def create_overlays(self):
        # Funkcja odpowiedzialna za utworzenie czarnych okien na dodatkowych monitorach.
        self.remove_overlays()
        for m in get_monitors():
            if m.is_primary:
                continue
            overlay = tk.Toplevel(self.root)
            overlay.geometry(f"{m.width}x{m.height}+{m.x}+{m.y}")
            overlay.overrideredirect(True)
            overlay.configure(bg="black")
            overlay.attributes("-topmost", True)
            overlay.protocol("WM_DELETE_WINDOW", lambda: None)
            overlay.bind("<Escape>", lambda e: "break")
            overlay.bind("<Alt-Tab>", lambda e: "break")
            self.overlays.append(overlay)

    def remove_overlays(self):
        # Funkcja odpowiedzialna za usunięcie czarnych okien blokujących.
        for overlay in self.overlays:
            if overlay.winfo_exists():
                overlay.destroy()
        self.overlays.clear()

    def lock_all(self):
        # Funkcja odpowiedzialna za aktywację pełnej blokady ekranów i myszy.
        self.create_overlays()
        self.root.after(100, self.trap_cursor)

    def unlock_all(self):
        # Funkcja odpowiedzialna za wyłączenie blokady ekranów i myszy.
        self.release_cursor()
        self.remove_overlays()

    def _queue_tray_action(self, action):
        # Funkcja odpowiedzialna za kolejkę akcji wykonywanych na głównym wątku Tk.
        if action is None:
            return
        self._tray_action_queue.put(action)

    def _flush_tray_actions(self):
        # Funkcja odpowiedzialna za wykonanie akcji z kolejki na głównym wątku Tk.
        while True:
            try:
                action = self._tray_action_queue.get_nowait()
            except queue.Empty:
                break
            try:
                action()
            finally:
                self._tray_action_queue.task_done()

        if hasattr(self.root, "after"):
            try:
                self.root.after(100, self._flush_tray_actions)
            except RuntimeError:
                pass

    def minimize_to_tray(self):
        # Funkcja odpowiedzialna za ukrycie aplikacji i przeniesienie jej do paska zadań.
        self.unlock_all()
        self.root.withdraw()
        if hasattr(self.root, "after"):
            try:
                self.root.after(100, self._flush_tray_actions)
            except RuntimeError:
                pass
        if ICON_PATH.exists():
            image = Image.open(ICON_PATH)
        else:
            image = Image.new('RGBA', (64, 64), (0, 120, 215, 255))
        # Przy zminimalizowanej aplikacji nie ma opcji zamknięcia - tylko podgląd logów i GitHub
        menu = pystray.Menu(
            pystray.MenuItem(
                self._tr("tray_open", "Otwórz NuraFace"),
                lambda: self._queue_tray_action(self.restore_from_tray),
            ),
            pystray.MenuItem(self._tr("tray_logs", "Wyświetl logi"), lambda: self.open_logs()),
            pystray.MenuItem(self._tr("tray_github", "GitHub"), lambda: self.open_github())
        )
        self.tray_icon = pystray.Icon("NuraFace", image, "NuraFace", menu)
        self.tray_icon.run_detached()

    def _tr(self, key: str, default: str) -> str:
        if self.i18n:
            return self.i18n.tr(key, default)
        return default

    def open_logs(self):
        # Funkcja odpowiedzialna za otwarcie najnowszego pliku logów (lub folderu logów, gdy nie ma plików).
        try:
            LOG_ROOT.mkdir(parents=True, exist_ok=True)
            files = [p for p in LOG_ROOT.iterdir() if p.is_file()]
            target = max(files, key=lambda p: p.stat().st_mtime) if files else LOG_ROOT
            os.startfile(str(target))
        except Exception:
            pass

    def open_github(self):
        # Funkcja odpowiedzialna za otwarcie profilu GitHub w przeglądarce.
        try:
            webbrowser.open("https://github.com/JowitaLA")
        except Exception:
            pass

    def stop_tray(self):
        # Funkcja odpowiedzialna za usunięcie ikony z zasobnika systemowego.
        icon = self.tray_icon
        self.tray_icon = None
        if icon:
            try:
                icon.stop()
            except Exception:
                pass

    def restore_from_tray(self):
        # Funkcja odpowiedzialna za przywrócenie aplikacji z paska systemowego i otwarcie widoku Ustawień.
        if self.tray_icon:
            self.tray_icon.stop()
            self.tray_icon = None

        self.root.deiconify()
        self.root.attributes('-fullscreen', True)
        self.root.attributes("-topmost", True)
        self.lock_all()
        
        # Wywołanie przejścia do Ustawień zamiast domyślnego na restore
        if self.on_open_settings:
            self.on_open_settings()
        elif self.on_restore:
            self.on_restore()
            
    def cleanup(self):
        # Funkcja odpowiedzialna za zdjęcie wszystkich blokad myszy i ekranów oraz zatrzymanie ikony w zasobniku.
        self.unlock_all()
        self.stop_tray()