import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "wyniki_h1")
pliki = sorted(folder.glob("*.csv"))
if not pliki:
    sys.exit(f"Brak plikow CSV w folderze: {folder.resolve()}")


def etykieta(plik, df):
    for kandydat in (str(df["user"].iloc[0]), plik.stem):
        if re.fullmatch(r"[pP]\d+", kandydat.strip()):
            return kandydat.strip().upper()
    return plik.stem.upper()


frames = []
for p in pliki:
    df = pd.read_csv(p)
    df["osoba"] = etykieta(p, df)
    frames.append(df)
d = pd.concat(frames, ignore_index=True)
d["t"] = d["time"].astype(str).str.replace(" s", "", regex=False).astype(float)
osoby = sorted(d["osoba"].unique(), key=lambda s: int(re.sub(r"\D", "", s) or 0))
print("Osoby:", osoby)
print(d.groupby(["osoba", "method"]).size().unstack())

C = {"face": "#1f4e79", "TOTP": "#c55a11"}
NAZWY = {"face": "Rozpoznawanie twarzy", "TOTP": "Kod OTP"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
rng = np.random.default_rng(1)

# Rys. 1: rozklad czasu (os logarytmiczna), wszystkie proby
fig, ax = plt.subplots(figsize=(5.2, 4))
for k, m in enumerate(["face", "TOTP"]):
    t = d.loc[d.method == m, "t"]
    ax.boxplot(t, positions=[k], widths=0.45, showfliers=False,
               medianprops=dict(color="black"), boxprops=dict(color=C[m]),
               whiskerprops=dict(color=C[m]), capprops=dict(color=C[m]))
    ax.scatter(k + rng.uniform(-.14, .14, len(t)), t, s=9, alpha=.45, color=C[m])
ticks = [2, 5, 10, 20, 50, 100, 250]
ax.set_yscale("log"); ax.set_yticks(ticks); ax.set_yticklabels(ticks)
ax.set_xticks([0, 1]); ax.set_xticklabels([NAZWY["face"], NAZWY["TOTP"]])
ax.set_ylabel("Czas weryfikacji [s] (skala logarytmiczna)")
fig.tight_layout(); fig.savefig(folder / "czas_weryfikacji_boxplot.png", dpi=300); plt.close(fig)

# Rys. 2: srednia geometryczna czasu w obrebie osoby (wykres par)
g = d.assign(l=np.log(d.t)).groupby(["osoba", "method"]).l.mean().apply(np.exp).unstack()
fig, ax = plt.subplots(figsize=(5.2, 4))
for p in osoby:
    r = g.loc[p]
    col = C["face"] if r["face"] < r["TOTP"] else C["TOTP"]
    ax.plot([0, 1], [r["TOTP"], r["face"]], "-o", color=col, lw=1.5)
    ax.annotate(p, (1.03, r["face"]), va="center", fontsize=9)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Kod OTP", "Rozpoznawanie twarzy"]); ax.set_xlim(-.2, 1.25)
ax.set_ylabel("Średnia geometryczna czasu [s]")
ax.text(.02, .02, "niebieski: szybsza twarz, pomarańczowy: szybszy OTP",
        transform=ax.transAxes, fontsize=8, color="gray")
fig.tight_layout(); fig.savefig(folder / "czas_osoby_pary.png", dpi=300); plt.close(fig)

# Rys. 3: odsetek prob z co najmniej jedna nieudana proba
h = (d.assign(f=d.attempts > 0).groupby(["osoba", "method"]).f.mean().unstack() * 100).loc[osoby]
fig, ax = plt.subplots(figsize=(5.8, 4)); x = np.arange(len(osoby)); w = .38
for k, m in enumerate(["face", "TOTP"]):
    ax.bar(x + (k - .5) * w, h[m], w, color=C[m], label=NAZWY[m])
ax.set_xticks(x); ax.set_xticklabels(osoby)
ax.set_ylabel("Próby z co najmniej jedną nieudaną próbą [%]")
ax.set_ylim(0, 100); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(folder / "czynnosci_dodatkowe.png", dpi=300); plt.close(fig)
print("Zapisano 3 rysunki w:", folder.resolve())
