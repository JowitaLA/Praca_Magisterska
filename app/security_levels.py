from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple
# Poziomy zabezpieczeń weryfikacji twarzy (docelowy FAR) wybierane w ustawieniach.

# (klucz, nazwa w UI, docelowy FAR, opis)
SECURITY_LEVELS = (
    ("high", "Wysoki", 2e-5,
     "FAR 2·10⁻⁵ (1 na 50 000) - poziom wymagany przez Androida. "
     "Najwyższe bezpieczeństwo, ale częstsze odrzucenia właściciela."),
    ("standard", "Standardowy", 1e-4,
     "FAR 1·10⁻⁴ (1 na 10 000) - domyślny kompromis między bezpieczeństwem a wygodą."),
    ("comfort", "Wygodny", 1e-3,
     "FAR 1·10⁻³ (1 na 1 000) - najmniej odrzuceń, ale wyraźnie niższe bezpieczeństwo."),
)

LEVEL_KEYS = tuple(k for k, _, _, _ in SECURITY_LEVELS)
DEFAULT_LEVEL = "standard"


def normalize_level(value: Any) -> str:
    value = str(value or "").strip().lower()
    return value if value in LEVEL_KEYS else DEFAULT_LEVEL


def level_label(key: str) -> str:
    key = normalize_level(key)
    return next(name for k, name, _, _ in SECURITY_LEVELS if k == key)


def level_far(key: str) -> float:
    key = normalize_level(key)
    return next(far for k, _, far, _ in SECURITY_LEVELS if k == key)


def resolve_threshold(vf: Mapping[str, Any], level: str) -> Tuple[float, str, bool]:
    """Zwraca (próg, użyty_poziom, czy_stary_model).

    Nowe modele mają w verifierze słownik "levels" z progiem dla każdego poziomu.
    Stare modele (bez "levels") mają jeden próg skalibrowany na FAR 1e-4 - używamy go
    niezależnie od wybranego poziomu i zwracamy flagę legacy=True.
    """
    level = normalize_level(level)
    floor = float(vf.get("min_operational_threshold", 0.50))
    levels = vf.get("levels")
    if isinstance(levels, Mapping) and level in levels:
        thr = float(levels[level]["threshold"])
        return max(thr, floor), level, False
    thr = float(vf.get("threshold", floor))
    return max(thr, floor), DEFAULT_LEVEL, True


def estimate_frr(vf: Mapping[str, Any], level: str) -> Optional[Tuple[float, int]]:
    """Szacowany odsetek odrzuceń właściciela na zdjęciach testowych (holdout): (FRR, N)."""
    levels = vf.get("levels")
    level = normalize_level(level)
    if not isinstance(levels, Mapping) or level not in levels:
        return None
    entry = levels[level]
    n = int(entry.get("n_holdout", 0))
    if n <= 0:
        return None
    return float(entry.get("frr_holdout", 0.0)), n
