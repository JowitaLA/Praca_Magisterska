from __future__ import annotations

import re
import sys
import ctypes                  
import tkinter as tk
from pathlib import Path
from screeninfo import get_monitors
from ctypes import wintypes
from resources.config import DATA_DIR, EXTRACTION_DIR, TRAIN_DIR

# Klucz tytułu w plikach i18n (zamiast sztywnego tekstu)
APP_TITLE_KEY = "app_title"

# Lista pozycji: (kod_pozycji, klucz_tlumaczenia)
POSES = [
    ("front", "pose_front"),
    ("left", "pose_left"),
    ("right", "pose_right"),
    ("up", "pose_up"),
    ("down", "pose_down"),
    ("frontv2", "pose_frontv2"),
]

IMG_EXT = ".jpg"

# Liczba zdjęć wykonywanych dla każdej pozycji
SHOTS_PER_POSE = 10

# Ile kolejnych dobrych klatek musi wystąpić przed automatycznym zdjęciem
AUTO_STABLE_FRAMES = 5

# Minimalna przerwa między automatycznymi zdjęciami (s)
AUTO_CAPTURE_DELAY = 1

# Parametry kontroli ustawienia twarzy
MIN_FACE_WIDTH = 0.12
MAX_FACE_WIDTH = 0.80
MAX_CENTER_OFFSET_X = 0.24
MAX_CENTER_OFFSET_Y = 0.26

# Parametry jakości obrazu
MIN_BRIGHTNESS = 35
MAX_BRIGHTNESS = 235
MIN_BLUR_SCORE = 35

# Progi orientacji głowy
ENFORCE_HEAD_POSE = True

YAW_FRONT_MAX = 25
PITCH_FRONT_MAX = 20

YAW_TURN_MIN = 7
YAW_TURN_MAX = 40

PITCH_TURN_MIN = 8
PITCH_TURN_MAX = 25

MAX_ROLL = 25

# Galerie embeddingów
MAX_GALLERY_EMBEDDINGS = 30
UPDATE_GALLERY_MARGIN = 0.08

# Parametry częstotliwości sprawdzania obecności użytkownika
SILENT_CHECK_INTERVAL_SEC = 300
SILENT_CHECK_RETRY_DELAY_SEC = 2

def project_base() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


BASE = project_base()
PROFILE_ROOT = DATA_DIR / "profile"
LOG_ROOT = BASE / "logs"


def python_exe() -> str:
    venv_py = BASE / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable

ICON_PATH = BASE / "resources" / "app.ico"

def safe_user_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^a-zA-Z0-9_ąćęłńóśźżĄĆĘŁŃÓŚŹŻ-]+", "_", name)
    return name.strip("_")

APP_NAME = "NuraFace"