from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from resources.config import *

LOGS_DIR = BASE / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
RUN_LOG_DIR = LOGS_DIR / RUN_ID
RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)

MASTER_LOG = RUN_LOG_DIR / f"pipeline_{RUN_ID}.log"

DEFAULT_USERS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "11", "16"]

# Wartość argumentu --check dla verify.py: sprawdź w verify.py, jakie wartości przyjmuje
VERIFY_CHECK = "all"

# Pliki wyników zapisywane przez training.py w katalogu użytkownika
USER_OUTPUT_FILES = ["model_report.json", "model_report.csv", "scores.csv",
                     "roc_curve.csv", "det_curve.csv"]


def clean_outputs() -> None:
    safe_targets = [
        EXTRACTION_DIR / "outputs",
        TRAIN_DIR / "outputs",
        VERIFICATION_DIR / "outputs",
    ]
    for p in safe_targets:
        if p.exists():
            if any(part in {"profile", "database"} for part in p.parts):
                continue
            shutil.rmtree(p, ignore_errors=True)
            print(f"Usunięto katalog: {p}")


def iter_verification_files(sub: str | None = None) -> list[Path]:
    ver_dir = TEST_DATA_DIR / "verification"
    if sub:
        ver_dir = ver_dir / sub
    files: list[Path] = []
    if ver_dir.exists():
        for ext in ("*.bmp", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.webp"):
            files.extend(ver_dir.glob(ext))
    return sorted(files)


def run_and_tee(cmd: list[str], stage_name: str, env: dict[str, str]) -> None:
    stage_safe_name = (
        stage_name.lower()
        .replace(" - ", "_")
        .replace(" ", "_")
        .replace(":", "")
        .replace("/", "_")
        .replace("\\", "_")
    )
    stage_log = RUN_LOG_DIR / f"{stage_safe_name}.log"

    with open(MASTER_LOG, "a", encoding="utf-8") as master, open(stage_log, "w", encoding="utf-8") as stage:
        header = f"{STEP}{stage_name}{STEP}CMD: {' '.join(cmd)}\n"
        print(header, end="")
        master.write(header)
        stage.write(header)

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env=env, bufsize=1, encoding="utf-8", errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            master.write(line)
            stage.write(line)

        code = proc.wait()
        footer = f"\n[EXIT CODE] {code}\n"
        print(footer, end="")
        master.write(footer)
        stage.write(footer)
        if code != 0:
            raise subprocess.CalledProcessError(code, cmd)


def save_user_outputs(user: str) -> None:
    """Kopiuje wyniki treningu do osobnego folderu, żeby kolejna osoba ich nie nadpisała."""
    dst = RUN_LOG_DIR / "wyniki" / user
    dst.mkdir(parents=True, exist_ok=True)
    src_dir = Path(TRAIN_DIR) / "outputs" / "users" / user
    for name in USER_OUTPUT_FILES:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst / name)
        else:
            print(f"[UWAGA] Brak pliku {src}")


def run_user(user: str, env: dict[str, str], skip_extraction: bool) -> None:
    if not skip_extraction:
        run_and_tee(
            [PY, str(Path(EXTRACTION_DIR) / "extraction.py"), "all", "--user", user,
             "--step", "test", "--store-outputs", "yes", "--delete-photos", "no",
             "--gaussian-blur", "no", "--median-filter", "no", "--clahe", "no",
             "--face-mask", "no"],
            f"Etap 1 - Ekstrakcja {user}", env)

    run_and_tee(
        [PY, str(Path(TRAIN_DIR) / "training.py"), "--user", user,
         "--step", "test", "--store-outputs", "yes", "--max-frr", "1.0"],
        f"Etap 2 - Modelowanie {user}", env)
    save_user_outputs(user)

    # TODO: dopasować do verify.py (skąd bierze zdjęcia danej osoby)
    run_and_tee(
        [PY, str(Path(VERIFICATION_DIR) / "verify.py"), "--check", VERIFY_CHECK,
         "--user", user, "--step", "test", "--store-outputs", "yes"],
        f"Etap 3 - Weryfikacja {user}", env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", nargs="+", default=DEFAULT_USERS)
    parser.add_argument("--skip-extraction", action="store_true",
                        help="pomiń etap 1 (ekstrakcja uruchomiona już wcześniej)")
    args = parser.parse_args()

    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    env["GLOG_minloglevel"] = "3"
    env["PYTHONPATH"] = str(BASE) + os.pathsep + env.get("PYTHONPATH", "")

    print(f"{STEP}Krok 0: Czyszczenie starych danych...{STEP}")
    # clean_outputs()   # odkomentuj przed przebiegiem pomiarowym (najpierw skopiuj stare outputs)
    print("Czyszczenie zakończone.")

    ver_files = iter_verification_files()
    ver_user_files = iter_verification_files("user")
    ver_impostor_files = iter_verification_files("impostor")
    print(f"Pliki do weryfikacji: korzeń={len(ver_files)}, user={len(ver_user_files)}, impostor={len(ver_impostor_files)}")

    failed: list[tuple[str, str]] = []
    for user in args.users:
        try:
            run_user(user, env, args.skip_extraction)
        except subprocess.CalledProcessError as e:
            failed.append((user, f"kod wyjścia {e.returncode}: {Path(e.cmd[1]).name}"))
            print(f"[BŁĄD] {user}: {failed[-1][1]}, przechodzę do kolejnej osoby")

    print(f"Log zbiorczy zapisano w: {MASTER_LOG}")
    if failed:
        print("Podsumowanie błędów:")
        for user, msg in failed:
            print(f"  {user}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()