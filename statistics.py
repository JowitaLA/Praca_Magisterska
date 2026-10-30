import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

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

out = []
def pr(*a):
    line = " ".join(str(x) for x in a)
    print(line); out.append(line)

pr("Osoby:", osoby, "| liczba prob:", len(d))
pr(d.groupby(["osoba", "method"]).size().unstack().to_string())

# statystyki opisowe (tab. czasy_wyniki) 
pr("\n== Statystyki opisowe ==")
for m, g in d.groupby("method"):
    q1, q3 = g.t.quantile([.25, .75])
    pr(f"{m}: n={len(g)} srednia={g.t.mean():.2f} sr.geom={np.exp(np.log(g.t).mean()):.2f} "
       f"odch.std={g.t.std():.2f} mediana={g.t.median():.2f} Q1-Q3={q1:.2f}-{q3:.2f} "
       f"min/max={g.t.min():.2f}/{g.t.max():.2f} | proby z nieudana={int((g.attempts>0).sum())} "
       f"({(g.attempts>0).mean()*100:.1f}%) suma nieudanych={int(g.attempts.sum())} "
       f"srednia na probe={g.attempts.mean():.2f} | >=3 nieudane={int((g.attempts>=3).sum())} max={int(g.attempts.max())}")
    pr(f"   Shapiro (surowe czasy): p={stats.shapiro(g.t).pvalue:.2g}")

# agregacja do poziomu osob 
pm = (d.assign(l=np.log(d.t)).groupby(["osoba", "method"])["l"].mean().unstack()).loc[osoby]
diff = pm["face"] - pm["TOTP"]
pr("\n== Testy na poziomie osob (logarytmy czasu) ==")
pr(f"Shapiro-Wilk (roznice): p={stats.shapiro(diff).pvalue:.3f}")
tt = stats.ttest_rel(pm["face"], pm["TOTP"])
pr(f"t-Student dla prob zaleznych: t({len(diff)-1})={tt.statistic:.2f}, p={tt.pvalue:.3f}")
se = diff.std(ddof=1) / np.sqrt(len(diff)); tc = stats.t.ppf(.975, len(diff) - 1)
pr(f"Iloraz sr. geom. (twarz/OTP)={np.exp(diff.mean()):.2f}, 95% PU: "
   f"{np.exp(diff.mean()-tc*se):.2f}-{np.exp(diff.mean()+tc*se):.2f}")
w = stats.wilcoxon(pm["face"], pm["TOTP"])
pr(f"Wilcoxon (czas): W={w.statistic:.0f}, p={w.pvalue:.3f}")
pr(f"Najmniejsze osiagalne p testu Wilcoxona dla N={len(diff)}: {2/2**len(diff):.4f}")

odsetek = (d.assign(f=d.attempts > 0).groupby(["osoba", "method"]).f.mean().unstack()).loc[osoby]
w2 = stats.wilcoxon(odsetek["face"], odsetek["TOTP"])
pr(f"Wilcoxon (odsetek prob z nieudana proba): W={w2.statistic:.0f}, p={w2.pvalue:.4f}")

# analiza wrazliwosci: U Manna-Whitneya w obrebie osoby + korekta Holma 
p_raw = [stats.mannwhitneyu(g.loc[g.method == "face", "t"], g.loc[g.method == "TOTP", "t"]).pvalue
         for _, g in d.groupby("osoba")]
p_raw = pd.Series(p_raw, index=sorted(d.osoba.unique())).loc[osoby]
order = np.argsort(p_raw.values); holm = np.empty(len(p_raw)); run = 0.0
for rank, idx in enumerate(order):
    run = max(run, (len(p_raw) - rank) * p_raw.values[idx]); holm[idx] = min(1.0, run)

# tabela per osoba (tab. osoby_wyniki) 
pr("\n== Tabela per osoba ==")
med = d.groupby(["osoba", "method"]).t.median().unstack().loc[osoby]
tab = pd.DataFrame({
    "med_twarz": med["face"].round(2), "med_OTP": med["TOTP"].round(2),
    "iloraz_sr_geom": np.exp(diff).round(2), "p_Holm": np.round(holm, 3),
    "nieudane_twarz_%": (odsetek["face"] * 100).round(1),
    "nieudane_OTP_%": (odsetek["TOTP"] * 100).round(1)})
pr(tab.to_string())
pr("\nUwaga: p_Holm to test U Manna-Whitneya w obrebie osoby z korekta Holma.")

(folder / "wyniki_testy.txt").write_text("\n".join(out), encoding="utf-8")
print("\nZapisano:", (folder / "wyniki_testy.txt").resolve())