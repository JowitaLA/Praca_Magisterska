from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


class LanguageManager:
    def __init__(self, lang_dir: Path, default_lang: str = "pl_PL"):
        self.lang_dir = lang_dir
        self.current_lang = default_lang  # Zmienna dynamiczna (z małej litery)
        self.translations: Dict[str, str] = {}
        
        self.load_language(self.current_lang)

    def load_language(self, lang_code: str) -> bool:
        # Funkcja odpowiedzialna za załadowanie pliku JSON dla danego kodu języka.
        json_path = self.lang_dir / f"{lang_code}.json"
        
        if not json_path.exists():
            return False

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                self.translations = json.load(f)
            self.current_lang = lang_code
            return True
        except Exception as e:
            final_text = f"Błąd ładowania języka {lang_code}: {e}"
            self.log_tr("", final_text)
            return False

    def tr(self, key: str, default: str | None = None) -> str:
        # Funkcja odpowiedzialna za pobranie przetłumaczonego tekstu na podstawie klucza.
        return self.translations.get(key, default if default is not None else key)