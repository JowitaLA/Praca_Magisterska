import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import beta

DEFAULT_CSV = Path(__file__).resolve().parent / "verification" / "outputs" / "verification_results.csv"
CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "wyniki_h2")
OUT.mkdir(parents=True, exist_ok=True)

# Kolejność plików w folderze kąta odpowiada kolejności z opisu zbioru:
# ciepłe, zimne, niskie, średnie, jasne. Zdjęcie wzorca (średnie, nr 4) usunięto z kąta 4.
SETTINGS = {1: "ciepłe", 2: "zimne", 3: "niskie", 4: "średnie", 5: "jasne"}
GALLERY_SETTING = 4
FRONTAL_ANGLE = 4
THRESH_FLOOR = 0.5
B = 5000
rng = np.random.default_rng(2026)


def clopper_pearson(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


#  dane 
r = pd.read_csv(CSV)
path = r["check_file"].str.extract(
    r"\\(?P<uid>[^\\]+)\\(?P<zestaw>[^\\]+)\\pos_(?P<pose>\d)\\angle_(?P<angle>\d)\\IMG_(?P<num>\d+)\.JPG$",
    flags=2,
)
assert path.notna().all().all(), "Nie wszystkie ścieżki pasują do wzorca"
r = pd.concat([r, path], axis=1)
for c in ("pose", "angle", "num"):
    r[c] = r[c].astype(int)
r["rej"] = 1 - r["decision"]

r = r.sort_values(["uid", "pose", "angle", "num"])
r["k"] = r.groupby(["uid", "pose", "angle"]).cumcount() + 1
# w kącie frontalnym brakuje zdjęcia wzorca, więc kolejne indeksy przesuwają się o 1
shift = (r["angle"] == FRONTAL_ANGLE) & (r["k"] >= GALLERY_SETTING)
r["setting_i"] = np.where(shift, r["k"] + 1, r["k"])
r["ustawienie"] = r["setting_i"].map(SETTINGS)

n_users = r["uid"].nunique()
print(f"Zdjęć: {len(r)}, osób: {n_users}, zdjęć na osobę: {r.groupby('uid').size().unique()}")
print("Detektory:", r["detector_backend"].value_counts().to_dict())
print("Brak punktów charakterystycznych:", int((r["landmarks_found"] == 0).sum()))
print("Progi decyzyjne:", r.groupby("uid")["threshold"].first().round(3).to_dict())

#  FRR ogółem 
k, n = int(r["rej"].sum()), len(r)
lo, hi = clopper_pearson(k, n)
print(f"\nFRR ogółem: {k}/{n} = {k/n:.1%} (CP 95%: {lo:.1%}-{hi:.1%}; poglądowo)")
per_user = r.groupby("uid")["rej"].agg(["sum", "size", "mean"])
print("FRR na osobę: min {:.1%}, max {:.1%}".format(per_user["mean"].min(), per_user["mean"].max()))
per_user.to_csv(OUT / "frr_na_osobe.csv")


#  bootstrap po osobach 
def cluster_bootstrap(df, by, values, B=B):
    users = df["uid"].unique()
    groups = {u: df[df["uid"] == u] for u in users}
    levels = sorted(df[by].unique())
    res = np.full((B, len(levels)), np.nan)
    for b in range(B):
        sample = pd.concat([groups[u] for u in rng.choice(users, len(users))])
        m = sample.groupby(by)[values].mean().reindex(levels)
        res[b] = m.values
    return levels, np.nanpercentile(res, [2.5, 97.5], axis=0)


levels_a, ci_a = cluster_bootstrap(r, "angle", "rej")
tab_a = r.groupby("angle").agg(n=("rej", "size"), odrzucone=("rej", "sum"), frr=("rej", "mean"),
                                score_sr=("score_max", "mean"), score_med=("score_max", "median"))
tab_a["ci_lo"], tab_a["ci_hi"] = ci_a[0], ci_a[1]
print("\nFRR wg kąta światła:\n", tab_a.round(3))
tab_a.to_csv(OUT / "frr_wg_kata.csv")

#  testy główne 
pa = r.pivot_table(index="uid", columns="angle", values="rej", aggfunc="mean")
fa = stats.friedmanchisquare(*[pa[c] for c in pa.columns])
Wa = fa.statistic / (len(pa) * (pa.shape[1] - 1))

# ustawienie światła: bez kąta frontalnego (brakuje tam zdjęcia 'średnie' - to wzorzec)
r_s = r[r["angle"] != FRONTAL_ANGLE]
ps = r_s.pivot_table(index="uid", columns="setting_i", values="rej", aggfunc="mean")
fs = stats.friedmanchisquare(*[ps[c] for c in ps.columns])
Ws = fs.statistic / (len(ps) * (ps.shape[1] - 1))

p_adj = holm([fa.pvalue, fs.pvalue])
print(f"\nKąt światła   : chi2({pa.shape[1]-1}) = {fa.statistic:.2f}, p = {fa.pvalue:.2e}, Kendall W = {Wa:.2f}, p(Holm) = {p_adj[0]:.2e}")
print(f"Ustawienie    : chi2({ps.shape[1]-1}) = {fs.statistic:.2f}, p = {fs.pvalue:.3f}, Kendall W = {Ws:.2f}, p(Holm) = {p_adj[1]:.3f}")
print("FRR wg ustawienia (bez kąta frontalnego):", (ps.mean() * 100).round(1).rename(SETTINGS).to_dict())

# poza = zmienna kontrolna
pp = r.pivot_table(index="uid", columns="pose", values="rej", aggfunc="mean")
fp = stats.friedmanchisquare(*[pp[c] for c in pp.columns])
print(f"Poza (kontrol.): chi2({pp.shape[1]-1}) = {fp.statistic:.2f}, p = {fp.pvalue:.4f}; FRR: {(pp.mean()*100).round(1).to_dict()}")

# porównania parami: kąt frontalny vs pozostałe (Wilcoxon, Holm)
rows = []
for a in [c for c in pa.columns if c != FRONTAL_ANGLE]:
    d = pa[a] - pa[FRONTAL_ANGLE]
    p = 1.0 if (d == 0).all() else stats.wilcoxon(pa[a], pa[FRONTAL_ANGLE]).pvalue
    rows.append((a, d.mean(), p))
pw = pd.DataFrame(rows, columns=["kat", "roznica_FRR", "p"])
pw["p_holm"] = holm(pw["p"])
print("\nKąt frontalny vs pozostałe:\n", pw.round(4))
pw.to_csv(OUT / "kat_vs_frontalny.csv", index=False)

# punkty charakterystyczne (opisowo)
print("\nFRR wg wykrycia punktów charakterystycznych:\n", r.groupby("landmarks_found")["rej"].agg(["mean", "size"]).round(3))

# ---------------------------------------------------------------- wykresy
plt.rcParams.update({"font.size": 10})

# 1) FRR wg kąta + punkty dla osób
fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)
x = np.arange(len(tab_a))
ax.bar(x, tab_a["frr"] * 100, color="#9aa5b1", zorder=2)
ax.errorbar(x, tab_a["frr"] * 100,
            yerr=[(tab_a["frr"] - tab_a["ci_lo"]) * 100, (tab_a["ci_hi"] - tab_a["frr"]) * 100],
            fmt="none", ecolor="black", capsize=3, zorder=3)
for i, a in enumerate(pa.columns):
    ax.scatter(np.full(len(pa), i) + rng.uniform(-0.18, 0.18, len(pa)), pa[a] * 100,
               s=14, color="#d62728", alpha=0.7, zorder=4,
               label="pojedyncze osoby" if i == 0 else None)
ax.set_xticks(x)
ax.set_xticklabels([f"{a}" + ("\n(frontalny)" if a == FRONTAL_ANGLE else "") for a in tab_a.index])
ax.set_xlabel("Kąt oświetlenia (numer w zbiorze)")
ax.set_ylabel("FRR [%]")
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.legend(loc="upper center", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "frr_wg_kata.png")
plt.close(fig)

# 2) heatmapa FRR: kąt x ustawienie
hm = r.pivot_table(index="angle", columns="setting_i", values="rej", aggfunc="mean") * 100
cnt = r.pivot_table(index="angle", columns="setting_i", values="rej", aggfunc="size")
fig, ax = plt.subplots(figsize=(6.2, 4.6), dpi=200)
im = ax.imshow(hm.values, cmap="Reds", vmin=0, vmax=60, aspect="auto")
ax.set_xticks(range(hm.shape[1]))
ax.set_xticklabels([SETTINGS[c] for c in hm.columns])
ax.set_yticks(range(hm.shape[0]))
ax.set_yticklabels(hm.index)
ax.set_xlabel("Ustawienie światła")
ax.set_ylabel("Kąt oświetlenia")
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        v = hm.values[i, j]
        if np.isnan(v):
            ax.text(j, i, "wzorzec", ha="center", va="center", fontsize=7, color="gray")
        else:
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 35 else "black")
fig.colorbar(im, ax=ax, label="FRR [%]")
fig.tight_layout()
fig.savefig(OUT / "heatmapa_frr.png")
plt.close(fig)

# 3) rozkład score_max wg kąta
fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)
data = [r.loc[r["angle"] == a, "score_max"].values for a in tab_a.index]
ax.boxplot(data, labels=[str(a) for a in tab_a.index], showfliers=True,
           flierprops=dict(markersize=2, alpha=0.5))
ax.axhline(THRESH_FLOOR, color="red", ls="--", lw=1.2, label=f"próg decyzyjny = {THRESH_FLOOR}")
ax.set_xlabel("Kąt oświetlenia (numer w zbiorze)")
ax.set_ylabel("score_max (podobieństwo kosinusowe)")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "score_wg_kata.png")
plt.close(fig)

r.to_csv(OUT / "dane_z_oznaczeniami.csv", index=False)
print(f"\nZapisano wyniki w: {OUT.resolve()}")
