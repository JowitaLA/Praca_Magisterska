import configparser
import os
import sys
from pathlib import Path

try:
    import winreg
except ImportError:  # pragma: no cover
    winreg = None

from .config import BASE, APP_NAME


def default_settings_path() -> str:
    # Ścieżka zawsze absolutna, niezależna od katalogu roboczego (przy autostarcie
    # z rejestru Windows katalog roboczy bywa inny niż folder aplikacji).
    # Plik leży obok aplikacji (BASE), tak samo jak database/, modeling/ i logs/.
    return str(BASE / "settings.ini")


class ConfigManager:
    def __init__(self, filename=None):
        self.filepath = filename or default_settings_path()
        self.config = configparser.ConfigParser()
        self.load()

    def load(self):
        if os.path.exists(self.filepath):
            self.config.read(self.filepath, encoding="utf-8")
        else:
            # Domyślna konfiguracja
            self.config["App"] = {
                "camera_name": "Default",   # Nazwa kamery
                "theme": "dark",            # Motyw interfejsu
                "ui_lang": "pl_PL",         # Język aplikacji
                "log_lang": "pl_PL",        # Język logów
                "autostart": "True",        # Autostart (aktywacja/dezaktywacja weryfikacji twarzy)
                "password": "False",         # Używanie hasła
                "auth_key": "True",         # Używanie jednorazowego klucza
                "presence_check": "Co 1 godzinę",     # Co ile ma program weryfikować użytkownika
                "train_after_auth": "False",
                "keep_authorized_on_restore": "False",
                "saved_password": "",
                "offline_code_hash": "",
                "offline_code_used": "False"
            }
            self.save_all()

    def get(self, section, key, fallback=None):
        try:
            return self.config.get(section, key, fallback=fallback)
        except Exception:
            return fallback

    def getboolean(self, section, key, fallback=False):
        try:
            return self.config.getboolean(section, key, fallback=fallback)
        except Exception:
            return fallback

    def set(self, section, key, value):
        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config.set(section, key, str(value))
        self.save_all()

    def save_all(self):
        Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            self.config.write(f)

    def resolve_autostart_target(self):
        if getattr(sys, "frozen", False):
            return str(Path(sys.executable).resolve())

        project_root = Path(__file__).resolve().parents[1]
        main_py = project_root / "main.py"
        if main_py.exists():
            return str(main_py)

        return str(Path(sys.executable).resolve())

    def build_autostart_command(self):
        target = self.resolve_autostart_target()
        target_path = Path(target)
        if target_path.suffix.lower() == ".py":
            # pythonw.exe uruchamia skrypt bez okna konsoli
            interpreter = Path(sys.executable)
            pythonw = interpreter.with_name("pythonw.exe")
            if pythonw.exists():
                interpreter = pythonw
            return f'"{interpreter}" "{target_path}"'
        return f'"{target_path}"'

    def apply_autostart(self, enabled=None):
        if winreg is None or os.name != "nt":
            return False

        enabled = self.getboolean("App", "autostart", fallback=False) if enabled is None else bool(enabled)
        run_key = r"Software\Microsoft\Windows\CurrentVersion\Run"
        app_name = APP_NAME

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, run_key, 0, winreg.KEY_READ) as key:
                try:
                    current_value, _ = winreg.QueryValueEx(key, app_name)
                except FileNotFoundError:
                    current_value = None
        except FileNotFoundError:
            current_value = None

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, run_key, 0, winreg.KEY_WRITE) as key:
                if enabled:
                    winreg.SetValueEx(key, app_name, 0, winreg.REG_SZ, self.build_autostart_command())
                else:
                    try:
                        winreg.DeleteValue(key, app_name)
                    except FileNotFoundError:
                        pass
        except OSError:
            return False

        # Zapis do pliku tylko wtedy, gdy wartość rzeczywiście się zmieniła
        if self.config.get("App", "autostart", fallback=None) != str(bool(enabled)):
            self.config["App"]["autostart"] = str(bool(enabled))
            self.save_all()
        return True