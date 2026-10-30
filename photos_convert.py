
import os
from multiprocessing import Pool
from pathlib import Path
from PIL import Image, ImageOps
from resources.config import *

IDS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "11", "16"]
FOLDERS = [TEST_DATA_DIR / "profile", TEST_DATA_DIR / "verification"]
MAX_SIDE = 1920
EXTS = {".jpg", ".jpeg"}


def needs_resize(path: Path) -> bool:
    with Image.open(path) as im:
        return max(im.size) > MAX_SIDE


def convert(src: Path):
    try:
        tmp = src.with_name(src.name + ".tmp")
        with Image.open(src) as im:
            im.draft("RGB", (MAX_SIDE, MAX_SIDE))      # szybkie dekodowanie w mniejszej skali
            im = ImageOps.exif_transpose(im).convert("RGB")
            s = MAX_SIDE / max(im.size)
            if s < 1:
                im = im.resize((round(im.width * s), round(im.height * s)),
                               Image.Resampling.LANCZOS)
            im.save(tmp, format="JPEG", quality=95)
        os.replace(tmp, src)                           # atomowa podmiana oryginału
        return None
    except Exception as e:
        return f"{src}: {e}"


if __name__ == "__main__":
    todo = []
    for base in FOLDERS:
        for uid in IDS:
            folder = base / uid
            if not folder.exists():
                print(f"[UWAGA] brak folderu {folder}")
                continue
            files = [p for p in folder.rglob("*") if p.suffix.lower() in EXTS]
            big = [p for p in files if needs_resize(p)]
            print(f"{base.name}\\{uid}: {len(files)} zdjęć, do zmniejszenia {len(big)}")
            todo.extend(big)

    print(f"\nŁącznie do zmniejszenia: {len(todo)} zdjęć (oryginały zostaną nadpisane)")
    if todo and input("Wpisz TAK, żeby kontynuować: ").strip() == "TAK":
        errors = []
        with Pool() as pool:
            for i, err in enumerate(pool.imap_unordered(convert, todo), 1):
                if err:
                    errors.append(err)
                if i % 100 == 0:
                    print(f"{i}/{len(todo)}")
        print("Gotowe")
        if errors:
            print(f"Błędy ({len(errors)}):")
            for e in errors:
                print("  ", e)
    else:
        print("Przerwano, nic nie zmieniono.")