"""
CodeAlpha Data Science Internship - Task 2: Unemployment Analysis with Python
Author: <your name>

Datasets (both from CodeAlpha, CMIE-style monthly state estimates for India):
  A) data/Unemployment in India.csv            May 2019 - Jun 2020, Rural/Urban split per state
  B) data/Unemployment_Rate_upto_11_2020.csv   Jan 2020 - Oct 2020, state totals + zone + coordinates

Key methodological choices
  * National / zone / area rates are WEIGHTED (total unemployed / total labour force),
    not simple averages of state rates, because a small state must not count as much as UP.
    Labour force is reconstructed as  employed / (1 - rate/100).
  * Every time series that sums across states uses a BALANCED PANEL (only states present in
    every month), otherwise the line would move just because states enter/leave the file.
  * The two files are analysed separately and never silently spliced (they disagree for some
    state-months; see the reconciliation section).
"""
import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

warnings.filterwarnings("ignore")
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ style
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlelocation": "left", "font.size": 10, "legend.frameon": False,
})
LOCKDOWN = pd.Timestamp("2020-03-25")   # India's nationwide lockdown began on 25 March 2020
log = []                                  # cleaning log, written to outputs/cleaning_log.txt


def note(msg):
    print(msg)
    log.append(msg)


def wrate(d):
    """Weighted unemployment rate (%) of a group of rows."""
    return d["unemployed"].sum() / d["labour_force"].sum() * 100


def add_derived(d):
    d["labour_force"] = d["employed"] / (1 - d["ur"] / 100)
    d["unemployed"] = d["labour_force"] * d["ur"] / 100
    d["population_15plus"] = d["labour_force"] / (d["lfpr"] / 100)   # implied by LFPR
    return d


def ms(obj):
    """Shift a month-end index to month-start so ticks line up with the month they label."""
    o = obj.copy()
    o.index = o.index.to_period("M").to_timestamp()
    return o


def save(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def mark_lockdown(ax, label=True):
    ax.axvline(LOCKDOWN, color=MUTED, ls="--", lw=1)
    if label:
        ax.text(LOCKDOWN, ax.get_ylim()[1], " Lockdown\n 25 Mar 2020", va="top", ha="left",
                fontsize=8.5, color=MUTED)


# ================================================================== 1. LOAD + CLEAN
note("=== 1. DATA CLEANING ===")
a = pd.read_csv("data/Unemployment in India.csv", encoding="utf-8-sig")
a.columns = a.columns.str.strip()
n_raw = len(a)
a = a.dropna(how="all")
note(f"A: raw rows {n_raw}; dropped {n_raw - len(a)} completely blank rows -> {len(a)} rows")
a.columns = ["state", "date", "freq", "ur", "employed", "lfpr", "area"]
for c in ["state", "freq", "area"]:
    a[c] = a[c].str.strip()                       # headers/values contain stray spaces
a["date"] = pd.to_datetime(a["date"].str.strip(), format="%d-%m-%Y")
assert not a.duplicated(["state", "date", "area"]).any()
zero_a = a[(a.ur == 0)]
note(f"A: {len(zero_a)} rows have an unemployment rate of exactly 0 with positive employment "
     f"(mostly Puducherry/Goa/Sikkim rural) - kept, but treated as suspect small-sample values.")
a = add_derived(a)

b = pd.read_csv("data/Unemployment_Rate_upto_11_2020.csv")
b.columns = ["state", "date", "freq", "ur", "employed", "lfpr", "zone", "lat", "lon"]
# The source file labels the columns the wrong way round: its 'longitude' holds values 10-34
# (India's latitudes) and its 'latitude' holds 71-93 (India's longitudes). The header names
# were assigned above in file order, so lat/lon here already carry the CORRECT meaning:
#   column 8 ('longitude' in source) -> lat ; column 9 ('latitude' in source) -> lon
assert b["lat"].between(8, 37).all() and b["lon"].between(68, 98).all()
note("B: source columns 'longitude'/'latitude' are swapped (values 10-34 / 71-93); relabelled correctly.")
b["state"] = b["state"].str.strip()
b["zone"] = b["zone"].str.strip()
b["date"] = pd.to_datetime(b["date"].str.strip(), format="%d-%m-%Y")
note("B: 'Region' appears twice in the header (state and zone); zone column renamed 'zone'.")
assert not b.duplicated(["state", "date"]).any() and b.isna().sum().sum() == 0
b = add_derived(b)

# balanced panels
months_a = a["date"].nunique()
cnt_a = a.groupby("state").size()
bal_states_a = cnt_a[cnt_a == months_a * 2].index
a_bal = a[a.state.isin(bal_states_a)]
months_b = b["date"].nunique()
cnt_b = b.groupby("state").size()
bal_states_b = cnt_b[cnt_b == months_b].index
b_bal = b[b.state.isin(bal_states_b)]
note(f"A: {a.state.nunique()} states/UTs; {len(bal_states_a)} have every month in both Rural and Urban "
     f"(excluded from totals: {sorted(set(a.state) - set(bal_states_a))})")
note(f"B: {b.state.nunique()} states; {len(bal_states_b)} have all {months_b} months "
     f"(excluded from totals: {sorted(set(b.state) - set(bal_states_b))})")
note(f"A covers {a.date.min():%b %Y} - {a.date.max():%b %Y}; B covers {b.date.min():%b %Y} - {b.date.max():%b %Y}")

# ================================================================== 2. NATIONAL TREND
note("\n=== 2. NATIONAL TREND (weighted, balanced panel) ===")
nat_a = a_bal.groupby("date").apply(wrate).rename("rate_A")
nat_b = b_bal.groupby("date").apply(wrate).rename("rate_B")
simple_b = b_bal.groupby("date")["ur"].mean().rename("simple_mean_B")
print(pd.concat([nat_a, nat_b, simple_b], axis=1).round(2).to_string())

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ms(nat_a).index, nat_a.values, color=BLUE, lw=2, marker="o", ms=5, label="Dataset A (May 2019 - Jun 2020)")
ax.plot(ms(nat_b).index, nat_b.values, color=ORANGE, lw=2, marker="o", ms=5, label="Dataset B (Jan - Oct 2020)")
ax.set_ylim(0, 28)
mark_lockdown(ax)
pk = nat_b.idxmax()
ax.annotate(f"Peak {nat_b.max():.1f}%\n({pk:%b %Y})", (pk.to_period('M').to_timestamp(), nat_b.max()), xytext=(-95, -5),
            textcoords="offset points", fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED))
ax.set_ylabel("Unemployment rate (%)")
ax.set_title("India's unemployment rate jumped from ~8% to ~24% in April 2020, then recovered")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
ax.legend(loc="upper left")
save(fig, "01_national_trend.png")

# ================================================================== 3. COVID PHASES
note("\n=== 3. COVID-19 IMPACT BY PHASE (Dataset B, balanced panel) ===")
phases = {
    "Pre-Covid (Jan-Feb 2020)": ["2020-01-31", "2020-02-29"],
    "Lockdown starts (Mar 2020)": ["2020-03-31"],
    "Lockdown peak (Apr-May 2020)": ["2020-04-30", "2020-05-31"],
    "Unlock begins (Jun 2020)": ["2020-06-30"],
    "Rebound (Jul-Oct 2020)": ["2020-07-31", "2020-08-31", "2020-09-30", "2020-10-31"],
}
rows = []
for name, ds in phases.items():
    d = b_bal[b_bal.date.isin(pd.to_datetime(ds))]
    rows.append({"phase": name, "unemployment_rate_%": round(wrate(d), 2),
                 "avg_LFPR_%": round(d.labour_force.sum() / d.population_15plus.sum() * 100, 2)})
phase_df = pd.DataFrame(rows)
print(phase_df.to_string(index=False))
phase_df.to_csv(f"{OUT}/phase_summary.csv", index=False)

pre_a = a_bal[a_bal.date < "2020-03-01"]
baseline_a = wrate(pre_a)
note(f"Pre-Covid baseline, Dataset A (May 2019 - Feb 2020): {baseline_a:.2f}%")
base_b = wrate(b_bal[b_bal.date < "2020-03-01"])
peak_b = nat_b.max()
note(f"Peak month in Dataset B: {nat_b.idxmax():%b %Y} at {peak_b:.2f}% = {peak_b / base_b:.1f}x the Jan-Feb baseline ({base_b:.2f}%)")
note(f"October 2020: {nat_b.iloc[-1]:.2f}% vs Jan-Feb baseline {base_b:.2f}%")

# jobs
emp = b_bal.groupby("date")["employed"].sum()
emp_idx = emp / emp.iloc[0] * 100
lfpr_nat = b_bal.groupby("date").apply(lambda d: d.labour_force.sum() / d.population_15plus.sum() * 100)
note(f"Employed persons (balanced panel): Jan 2020 {emp.iloc[0]/1e6:.1f} m -> Apr 2020 {emp.loc['2020-04-30']/1e6:.1f} m "
     f"({emp_idx.loc['2020-04-30'] - 100:.1f}%); Oct 2020 {emp.iloc[-1]/1e6:.1f} m (index {emp_idx.iloc[-1]:.1f})")
note(f"Labour force participation: Jan {lfpr_nat.iloc[0]:.1f}% -> Apr {lfpr_nat.loc['2020-04-30']:.1f}% -> Oct {lfpr_nat.iloc[-1]:.1f}%")

fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
axs[0].plot(ms(emp_idx).index, emp_idx.values, color=BLUE, lw=2, marker="o", ms=5)
axs[0].axhline(100, color=MUTED, lw=0.8, ls=":")
axs[0].set_title("Employment index (Jan 2020 = 100)")
axs[0].set_ylim(60, 110)
axs[1].plot(ms(lfpr_nat).index, lfpr_nat.values, color=ORANGE, lw=2, marker="o", ms=5)
axs[1].set_title("Labour force participation rate (%)")
axs[1].set_ylim(30, 48)
for ax in axs:
    mark_lockdown(ax, label=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
save(fig, "02_jobs_and_participation.png")

# ================================================================== 4. RURAL VS URBAN
note("\n=== 4. RURAL vs URBAN (Dataset A, balanced panel) ===")
ru = a_bal.groupby(["date", "area"]).apply(wrate).unstack()
print(ru.round(2).to_string())
pre = a_bal[a_bal.date < "2020-03-01"].groupby("area").apply(wrate)
note(f"Pre-Covid (May19-Feb20) weighted rate: Rural {pre['Rural']:.2f}%, Urban {pre['Urban']:.2f}%")
note(f"Apr 2020: Rural {ru.loc['2020-04-30','Rural']:.2f}%, Urban {ru.loc['2020-04-30','Urban']:.2f}%; "
     f"May 2020: Rural {ru.loc['2020-05-31','Rural']:.2f}%, Urban {ru.loc['2020-05-31','Urban']:.2f}%")
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(ms(ru).index, ru["Urban"], color=BLUE, lw=2, marker="o", ms=5, label="Urban")
ax.plot(ms(ru).index, ru["Rural"], color=ORANGE, lw=2, marker="o", ms=5, label="Rural")
ax.set_ylim(0, 30)
mark_lockdown(ax)
ax.set_ylabel("Unemployment rate (%)")
ax.set_title("Urban unemployment was higher before and during the lockdown; both fell sharply by June")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
ax.legend(loc="upper left")
save(fig, "03_rural_vs_urban.png")

# ================================================================== 5. STATES
note("\n=== 5. STATE-LEVEL IMPACT (Dataset B) ===")
piv = b.pivot_table(index="state", columns="date", values="ur")
piv.columns = [f"{c:%b}" for c in piv.columns]
need = ["Jan", "Feb", "Apr", "May", "Oct"]
sd = piv.dropna(subset=need).copy()
note(f"{len(sd)} states have Jan, Feb, Apr, May and Oct all present (dropped: {sorted(set(piv.index) - set(sd.index))})")
sd["baseline"] = sd[["Jan", "Feb"]].mean(axis=1)
sd["lockdown"] = sd[["Apr", "May"]].mean(axis=1)
sd["oct"] = sd["Oct"]
sd["shock_pp"] = sd["lockdown"] - sd["baseline"]
sd["oct_vs_base_pp"] = sd["oct"] - sd["baseline"]
sd = sd.sort_values("lockdown")
sd[["baseline", "lockdown", "oct", "shock_pp", "oct_vs_base_pp"]].round(2).sort_values("shock_pp", ascending=False)\
  .to_csv(f"{OUT}/state_impact.csv")
top = sd.sort_values("shock_pp", ascending=False)
note("Largest rise (pp) from Jan-Feb baseline to Apr-May average:\n" +
     top[["baseline", "lockdown", "shock_pp"]].head(6).round(1).to_string())
note("States still >3pp above baseline in Oct 2020:\n" +
     sd[sd.oct_vs_base_pp > 3][["baseline", "oct", "oct_vs_base_pp"]].round(1).to_string())
w = stats.wilcoxon(sd["lockdown"], sd["baseline"])
note(f"Paired Wilcoxon (state lockdown avg vs baseline): n={len(sd)}, p={w.pvalue:.2e}; "
     f"{(sd.shock_pp > 0).sum()}/{len(sd)} states rose")
w2 = stats.wilcoxon(sd["oct"], sd["baseline"])
note(f"Paired Wilcoxon (Oct vs baseline): p={w2.pvalue:.3f}; median change {sd.oct_vs_base_pp.median():+.2f} pp")

fig, ax = plt.subplots(figsize=(9, 9))
y = np.arange(len(sd))
ax.hlines(y, sd["baseline"], sd["lockdown"], color=GRID, lw=2, zorder=1)
ax.scatter(sd["baseline"], y, color=BLUE, s=45, zorder=3, edgecolor="white", linewidth=1, label="Jan-Feb 2020 (baseline)")
ax.scatter(sd["lockdown"], y, color=ORANGE, s=45, zorder=3, edgecolor="white", linewidth=1, label="Apr-May 2020 (lockdown avg)")
ax.scatter(sd["oct"], y, color=AQUA, s=45, zorder=3, edgecolor="white", linewidth=1, label="Oct 2020")
ax.set_yticks(y)
ax.set_yticklabels(sd.index)
ax.set_xlabel("Unemployment rate (%)")
ax.set_title(f"Unemployment rose in {(sd.shock_pp > 0).sum()} of {len(sd)} states during lockdown; "
             f"by October {(sd.oct_vs_base_pp < 0).sum()} were below their own baseline", loc="left", fontsize=11)
ax.grid(axis="y", visible=False)
ax.legend(loc="lower right")
save(fig, "04_state_impact_dumbbell.png")

# heatmap state x month
order = piv["Apr"].sort_values(ascending=False).index
hm = piv.loc[order, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]]
fig, ax = plt.subplots(figsize=(9, 10))
im = ax.imshow(hm.values, cmap="Blues", aspect="auto", vmin=0, vmax=60)
ax.set_xticks(range(hm.shape[1])); ax.set_xticklabels(hm.columns)
ax.set_yticks(range(hm.shape[0])); ax.set_yticklabels(hm.index)
ax.grid(False)
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        v = hm.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 32 else INK)
        else:
            ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color=MUTED)
ax.set_title("Unemployment rate (%) by state and month, 2020 (sorted by April)")
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("%")
save(fig, "05_state_month_heatmap.png")

# ================================================================== 6. ZONES
note("\n=== 6. ZONES (Dataset B, balanced panel) ===")
zn = b_bal.groupby(["date", "zone"]).apply(wrate).unstack()
print(zn.round(1).to_string())
zcol = {"East": BLUE, "North": ORANGE, "Northeast": AQUA, "South": YELLOW, "West": MAGENTA}
fig, ax = plt.subplots(figsize=(10, 5))
for z in zn.columns:
    ax.plot(ms(zn).index, zn[z], color=zcol[z], lw=2, marker="o", ms=4.5, label=z)
ax.set_ylim(0, 36)
mark_lockdown(ax)
ax.set_ylabel("Unemployment rate (%)")
ax.set_title("The South and East were hit hardest in April; the North stayed elevated longest")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1))
save(fig, "06_zones.png")

# ================================================================== 7. MAP
apr = b[b.date == "2020-04-30"]
fig, ax = plt.subplots(figsize=(7, 8))
ax.scatter(apr["lon"], apr["lat"], s=apr["ur"] * 14, color=ORANGE, alpha=0.75, edgecolor="white", linewidth=1.5)
for _, r in apr.iterrows():
    big = r.ur >= 25
    ax.annotate(f"{r.state} {r.ur:.0f}%" if big else r.state, (r.lon, r.lat), xytext=(7, 5),
                textcoords="offset points", fontsize=8 if big else 6.5, color=INK if big else MUTED)
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title("April 2020 unemployment by state (bubble size = rate)")
ax.set_aspect("equal")
save(fig, "07_april_map.png")

# ================================================================== 8. SEASONALITY / PATTERNS
note("\n=== 7. SEASONALITY & PATTERNS ===")
pre_series = nat_a[nat_a.index < "2020-03-01"]
note(f"Pre-Covid monthly national rate (A, May19-Feb20): mean {pre_series.mean():.2f}%, "
     f"min {pre_series.min():.2f}% ({pre_series.idxmin():%b %Y}), max {pre_series.max():.2f}% ({pre_series.idxmax():%b %Y}), "
     f"std {pre_series.std():.2f} pp")
yoy = pd.DataFrame({
    "2019": [nat_a.loc["2019-05-31"], nat_a.loc["2019-06-30"]],
    "2020": [nat_a.loc["2020-05-31"], nat_a.loc["2020-06-30"]],
}, index=["May", "Jun"])
d = np.sign(np.diff(pre_series.values))
note(f"Pre-Covid month-to-month direction alternates up/down in {int((d[1:] != d[:-1]).sum())} of {len(d) - 1} consecutive changes "
     f"(amplitude about {np.abs(np.diff(pre_series.values)).mean():.2f} pp per month)")
note("Year-on-year, same month (Dataset A):\n" + yoy.round(2).to_string())
rur_pre = ru.loc[ru.index < "2020-03-01"]
note(f"Rural-Urban gap pre-Covid: urban higher in {(rur_pre['Urban'] > rur_pre['Rural']).sum()} of {len(rur_pre)} months")

fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [2, 1]})
axs[0].plot(ms(pre_series).index, pre_series.values, color=BLUE, lw=2, marker="o", ms=5)
axs[0].axhline(pre_series.mean(), color=MUTED, ls=":", lw=1)
axs[0].text(0.01, 0.97, f"dotted line = mean {pre_series.mean():.1f}%", transform=axs[0].transAxes,
            va="top", fontsize=9, color=MUTED)
axs[0].set_ylim(5, 11)
axs[0].set_title("Pre-Covid rate: flat around 7.6% with a month-to-month zigzag", fontsize=11)
axs[0].set_ylabel("Unemployment rate (%)")
axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%y"))
x = np.arange(2); wd = 0.36
axs[1].bar(x - wd / 2, yoy["2019"], wd, color=BLUE, label="2019")
axs[1].bar(x + wd / 2, yoy["2020"], wd, color=ORANGE, label="2020")
for xi, v in zip(x - wd / 2, yoy["2019"]): axs[1].text(xi, v + 0.4, f"{v:.1f}", ha="center", fontsize=9)
for xi, v in zip(x + wd / 2, yoy["2020"]): axs[1].text(xi, v + 0.4, f"{v:.1f}", ha="center", fontsize=9)
axs[1].set_xticks(x); axs[1].set_xticklabels(yoy.index)
axs[1].set_ylim(0, 30)
axs[1].set_title("Same month, a year apart")
axs[1].grid(axis="x", visible=False)
axs[1].legend(loc="upper left")
save(fig, "08_seasonality_check.png")

# unemployment vs participation (state-months, Dataset B)
b["period"] = np.where(b.date.isin(pd.to_datetime(["2020-04-30", "2020-05-31"])), "Lockdown (Apr-May)", "Other months")
r_all = b["ur"].corr(b["lfpr"])
note(f"Correlation unemployment rate vs LFPR across state-months: r = {r_all:.2f}")
g = b.groupby("period")[["ur", "lfpr"]].mean().round(2)
note("Simple state-month means by period:\n" + g.to_string())
fig, ax = plt.subplots(figsize=(7.5, 5.5))
for p, c in [("Other months", BLUE), ("Lockdown (Apr-May)", ORANGE)]:
    d = b[b.period == p]
    ax.scatter(d["lfpr"], d["ur"], s=28, color=c, alpha=0.8, edgecolor="white", linewidth=0.8, label=p)
ax.set_xlabel("Labour force participation rate (%)"); ax.set_ylabel("Unemployment rate (%)")
ax.set_title("Lockdown months sit at lower participation and higher unemployment", fontsize=11)
ax.legend()
save(fig, "09_unemployment_vs_participation.png")

# ================================================================== 9. RECONCILIATION A vs B
note("\n=== 8. RECONCILIATION OF THE TWO FILES (Jan-Jun 2020 overlap) ===")
ov_a = a[a.date >= "2020-01-01"].groupby(["state", "date"]).apply(
    lambda d: pd.Series({"ur_A": wrate(d), "emp_A": d.employed.sum()})).reset_index()
m = ov_a.merge(b[["state", "date", "ur", "employed"]], on=["state", "date"])
m["diff"] = m["ur_A"] - m["ur"]
note(f"{len(m)} overlapping state-months: correlation {m.ur_A.corr(m.ur):.3f}, "
     f"median |diff| {m['diff'].abs().median():.2f} pp, mean |diff| {m['diff'].abs().mean():.2f} pp, "
     f"{(m['diff'].abs() > 5).sum()} state-months differ by more than 5 pp")
note("Largest disagreements:\n" + m.reindex(m["diff"].abs().sort_values(ascending=False).index)
     .head(5)[["state", "date", "ur_A", "ur", "diff"]].round(1).to_string(index=False))

# ================================================================== SAVE
with open(f"{OUT}/cleaning_and_findings_log.txt", "w") as f:
    f.write("\n".join(log))
summary = {
    "baseline_jan_feb_2020_%": round(base_b, 2),
    "peak_month": f"{nat_b.idxmax():%b %Y}", "peak_rate_%": round(peak_b, 2),
    "oct_2020_rate_%": round(nat_b.iloc[-1], 2),
    "pre_covid_baseline_A_%": round(baseline_a, 2),
    "employment_drop_apr_vs_jan_%": round(emp_idx.loc["2020-04-30"] - 100, 1),
    "states_analysed": int(len(sd)),
}
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
print("\nDone. Outputs in ./outputs")
