import sys
from pathlib import Path

if sys.stdout:
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr:
    sys.stderr.reconfigure(encoding="utf-8")

# ZMIENNE GLOBALNE 
# .parent.parent wchodzi dwa poziomy wyżej (z resources/config.py do Magister/)
# BASE = Path(__file__).resolve().parent.parent  # Ścieżka do głównego katalogu projektu (C:\Magister)
if getattr(sys, "frozen", False):
    BASE = Path(sys.executable).resolve().parent
else:
    BASE = Path(__file__).resolve().parent.parent
PY = sys.executable  # Ścieżka do interpretera Pythona
STEP = "\n\n" + ("~" * 140) + "\n\n"  # Separator do czytelności logów
SEPARATOR = "=" * 140
SECTION = "\n" + ("-" * 140) + "\n"
RANDOM_STATE = 339351

# ŚCIEŻKI DO KATALOGÓW Z DANYMI
DATA_DIR = BASE / "database"  # Baza danych z obrazami twarzy (dla aplikacji)
TEST_DATA_DIR = BASE / "test_database"  # Testowa baza danych z obrazami twarzy

# ŚCIEŻKI DO KATALOGÓW MODUŁÓW
EXTRACTION_DIR = BASE / "extraction"        # Katalog z ekstrakcją cech twarzy
TRAIN_DIR = BASE / "modeling"               # Katalog z trenowaniem i ewaluacją
VERIFICATION_DIR = BASE / "verification"    # Katalog z weryfikacją twarzy

# ŚCIEŻKA DO LOGÓW
LOG_ROOT = BASE / "logs"