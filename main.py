from __future__ import annotations

import multiprocessing
import os
import sys
import threading
import warnings
from pathlib import Path
if getattr(sys, "frozen", False):
    multiprocessing.freeze_support()
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        import runpy

        _script = sys.argv[2]
        sys.argv = [_script] + sys.argv[3:]
        try:
            runpy.run_path(_script, run_name="__main__")
        except SystemExit:
            raise
        except Exception as _exc:
            # Pełny traceback zostaje w logu, a osobna, krótka linia z komunikatem
            # jest odczytywana przez trainer.py i pokazywana użytkownikowi w oknie.
            import traceback

            traceback.print_exc()
            _msg = " ".join(str(_exc).split()) or type(_exc).__name__
            print(f"NURAFACE_ERROR::{type(_exc).__name__}::{_msg}", flush=True)
            sys.exit(1)
        sys.exit(0)

if getattr(sys, "frozen", False):
    base_dir = Path(sys.executable).resolve().parent
else:
    base_dir = Path(__file__).resolve().parent

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_LOG_LEVEL", "2")
os.environ.setdefault("YOLO_CONFIG_DIR", str(base_dir / ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(base_dir / ".matplotlib"))
warnings.filterwarnings("ignore", category=UserWarning, module=r"google\.protobuf\.symbol_database")
warnings.filterwarnings("ignore", message=r".*Feedback manager requires a model with a single signature inference.*")
warnings.filterwarnings("ignore", message=r".*SymbolDatabase.GetPrototype\(\) is deprecated.*")

base_dir.mkdir(exist_ok=True)
for sub in (base_dir / ".ultralytics", base_dir / ".matplotlib"):
    sub.mkdir(exist_ok=True)

import ttkbootstrap as ttk
from app.controller import FaceRegisterApp
import logging
from datetime import datetime
from resources.config import LOG_ROOT

LOG_ROOT.mkdir(parents=True, exist_ok=True)
start_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
log_filename = LOG_ROOT / f"{start_time}.log"

logging.basicConfig(
    filename=log_filename,
    filemode="a",
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    encoding="utf-8"
)

logger = logging.getLogger("FaceRegisterApp")

_instance_mutex = None  # uchwyt musi żyć do końca procesu


def already_running() -> bool:
    # Zabezpieczenie przed dwiema kopiami aplikacji (np. autostart + ręczne uruchomienie),
    # bo obie próbowałyby jednocześnie blokować ekran, mysz i kamerę.
    global _instance_mutex
    if os.name != "nt":
        return False
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    _instance_mutex = kernel32.CreateMutexW(None, False, "NuraFace_single_instance")
    return ctypes.get_last_error() == 183  # ERROR_ALREADY_EXISTS


def _show_error_box(title: str, text: str) -> None:
    # Okno błędu z samego Windows API - działa też wtedy, gdy interfejs Tk jeszcze nie istnieje.
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, text, title, 0x10)  # MB_ICONERROR
    except Exception:
        pass


def install_error_windows(root) -> None:
    # W wersji .exe (bez konsoli) nieobsłużony wyjątek znika bez śladu.
    # Zamiast tego zapisujemy go w logu i pokazujemy krótkie okno z komunikatem.
    # Dotyczy też kodu uruchamianego w tym samym procesie, np. embed_image w weryfikacji.
    def _describe(exc: BaseException) -> str:
        return " ".join(str(exc).split()) or type(exc).__name__

    def _tk_callback_exception(exc_type, exc, tb):
        logger.error("Nieobsłużony wyjątek w interfejsie", exc_info=(exc_type, exc, tb))
        _show_error_box("NuraFace - błąd", f"{exc_type.__name__}: {_describe(exc)}")

    def _thread_exception(args):
        if args.exc_type is SystemExit:
            return
        logger.error("Nieobsłużony wyjątek w wątku", exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
        _show_error_box("NuraFace - błąd", f"{args.exc_type.__name__}: {_describe(args.exc_value)}")

    def _main_exception(exc_type, exc, tb):
        logger.critical("Nieobsłużony wyjątek", exc_info=(exc_type, exc, tb))
        _show_error_box("NuraFace - błąd", f"{exc_type.__name__}: {_describe(exc)}")

    root.report_callback_exception = _tk_callback_exception
    threading.excepthook = _thread_exception
    sys.excepthook = _main_exception


def main():
    if already_running():
        logger.info("NuraFace jest już uruchomiony - zamykam drugą instancję.")
        return
    root = ttk.Window(themename="superhero")
    if getattr(sys, "frozen", False):
        install_error_windows(root)
    app = FaceRegisterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()