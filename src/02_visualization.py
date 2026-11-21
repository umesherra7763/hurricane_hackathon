"""
02_visualization.py
-------------------
Descriptive analytics + 15 unique visualisations (course plots) of the cleaned
hurricane data. Every plot is saved to outputs/figures/figNN_*.png and the
numbers that support each inference are written to outputs/logs/02_visualization.txt
and visualization.json.

Plots: 1 histogram, 2 box plot, 3 quantile plot, 4 Q-Q plot, 5 scatter plot,
6 scatter-plot matrix, 7 correlation heat-map, 8 grouped bar chart, 9 line chart,
10 pie chart, 11 violin plot, 12 geographic track map, 13 parallel coordinates,
14 hexbin density plot, 15 stacked (100 %) bar chart.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st
from pandas.plotting import parallel_coordinates

from common import CLEAN_TRACKS, STORMS, FIG_DIR, Logger, save_json

log = Logger("02_visualization")
res = {}

# ---- consistent styling: Atlantic = blue, Pacific = orange everywhere -------
BASIN_COL = {"Atlantic": "#2a78d6", "Pacific": "#eb6834"}
CAT_COLS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = "Blues"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titleweight": "bold",
    "legend.frameon": False,
})

def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log(f"   -> saved {name}")

df = pd.read_csv(CLEAN_TRACKS, parse_dates=["Datetime"])
storms = pd.read_csv(STORMS, parse_dates=["Start", "End"])
SS_ORDER = ["TD", "TS", "Cat1", "Cat2", "Cat3", "Cat4", "Cat5"]

# =========================================================================
# Descriptive statistics (central tendency + dispersion)
# =========================================================================
log("=========== DESCRIPTIVE STATISTICS ===========")
num = df[["Wind", "Pressure", "Latitude", "Longitude", "Speed_kmh"]]
desc = num.describe().T
desc["median"] = num.median()
desc["mode"] = num.mode().iloc[0]
desc["skew"] = num.skew()
desc["kurtosis"] = num.kurt()
desc["IQR"] = desc["75%"] - desc["25%"]
log(desc.round(2).to_string())
res["desc"] = desc.round(2).to_dict(orient="index")
log("\nPer-basin observation / storm counts:")
log(df.groupby("Basin").agg(observations=("ID", "size"), storms=("ID", "nunique")).to_string())
log("\nStorm-level summary:\n" + storms[["MaxWind", "MinPressure", "DurationH", "TrackKm"]]
    .describe().round(1).to_string())
res["storm_desc"] = storms[["MaxWind", "MinPressure", "DurationH", "TrackKm"]].describe().round(1).to_dict()

# =========================================================================
# 1. HISTOGRAM - distribution of Maximum Wind
# =========================================================================
log("\n[1] Histogram of Maximum Wind")
fig, ax = plt.subplots(figsize=(8, 4.2))
bins = np.arange(10, 195, 5)
for b in ("Atlantic", "Pacific"):
    ax.hist(df.loc[df.Basin == b, "Wind"], bins=bins, alpha=.65, color=BASIN_COL[b],
            label=b, edgecolor="white", linewidth=.6)
for thr, lab in ((34, "TS"), (64, "HU"), (96, "Major")):
    ax.axvline(thr, color="#555", ls="--", lw=.8)
    ax.text(thr + 1, ax.get_ylim()[1] * .92, lab, fontsize=8, color="#555")
ax.set(xlabel="Maximum sustained wind (kt)", ylabel="Number of observations",
       title="Histogram of maximum wind speed")
ax.legend()
save(fig, "fig01_histogram_wind.png")
res["wind_skew"] = round(float(df.Wind.skew()), 2)
res["wind_mode"] = float(df.Wind.mode()[0])
res["pct_below_64"] = round(100 * (df.Wind < 64).mean(), 1)
res["pct_major"] = round(100 * (df.Wind >= 96).mean(), 1)
log(f"   skew={res['wind_skew']}, mode={res['wind_mode']} kt, % obs < 64 kt = {res['pct_below_64']}, "
    f"% obs major (>=96 kt) = {res['pct_major']}")

# =========================================================================
# 2. BOX PLOT - Wind by storm status
# =========================================================================
log("\n[2] Box plot of wind by status")
order = ["LOW", "TD", "SUB", "TS", "EX", "HU"]
fig, ax = plt.subplots(figsize=(8, 4.2))
sns.boxplot(data=df, x="StatusGroup", y="Wind", hue="Basin", order=order,
            palette=BASIN_COL, ax=ax, fliersize=1.5, linewidth=.8)
ax.set(xlabel="Status group", ylabel="Maximum wind (kt)",
       title="Box plot: wind distribution per status (five-number summary)")
save(fig, "fig02_boxplot_status.png")
five = df.groupby("StatusGroup")["Wind"].describe()[["min", "25%", "50%", "75%", "max"]].loc[order]
log(five.to_string())
res["five_num"] = five.to_dict(orient="index")
res["ex_median"] = float(five.loc["EX", "50%"])

# =========================================================================
# 3. QUANTILE PLOT - Minimum Pressure
# =========================================================================
log("\n[3] Quantile plot of minimum pressure")
fig, ax = plt.subplots(figsize=(8, 4.2))
for b in ("Atlantic", "Pacific"):
    x = np.sort(df.loc[(df.Basin == b) & df.Pressure.notna(), "Pressure"].values)
    f = (np.arange(1, len(x) + 1) - 0.5) / len(x)          # f-value
    ax.plot(f, x, color=BASIN_COL[b], lw=2, label=b)
ax.set(xlabel="f-value (fraction of data)", ylabel="Minimum pressure (hPa)",
       title="Quantile plot of central pressure")
ax.legend()
save(fig, "fig03_quantile_pressure.png")
q = df.groupby("Basin")["Pressure"].quantile([.1, .25, .5, .75]).unstack()
log(q.to_string())
res["pressure_quantiles"] = q.to_dict(orient="index")

# =========================================================================
# 4. Q-Q PLOT - Atlantic vs Pacific lifetime-max wind (storm level)
#    (plus a normal-probability Q-Q in a second panel)
# =========================================================================
log("\n[4] Q-Q plot")
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
a = np.sort(storms.loc[storms.Basin == "Atlantic", "MaxWind"].values)
p = np.sort(storms.loc[storms.Basin == "Pacific", "MaxWind"].values)
qs = np.linspace(0.01, 0.99, 99)
qa, qp = np.quantile(a, qs), np.quantile(p, qs)
axs[0].scatter(qa, qp, s=14, color=CAT_COLS[0])
lim = [min(qa.min(), qp.min()), max(qa.max(), qp.max())]
axs[0].plot(lim, lim, color="#888", ls="--", lw=1, label="y = x")
axs[0].set(xlabel="Atlantic quantiles of storm max wind (kt)", ylabel="Pacific quantiles (kt)",
           title="Q-Q: Atlantic vs Pacific")
axs[0].legend()
(osm, osr), (slope, inter, r) = st.probplot(storms["MaxWind"], dist="norm")
axs[1].scatter(osm, osr, s=5, color=CAT_COLS[0])
axs[1].plot(osm, slope * osm + inter, color=CAT_COLS[1], lw=2)
axs[1].set(xlabel="Theoretical normal quantiles", ylabel="Storm max wind (kt)",
           title=f"Normal Q-Q (r = {r:.3f})")
save(fig, "fig04_qq_plot.png")
res["qq_upper_diff"] = round(float(np.mean(qp[80:] - qa[80:])), 1)
res["qq_lower_diff"] = round(float(np.mean(qp[:30] - qa[:30])), 1)
res["normal_qq_r"] = round(float(r), 3)
log(f"   mean(Pacific-Atlantic) at upper 20 % quantiles = {res['qq_upper_diff']} kt; "
    f"lower 30 % = {res['qq_lower_diff']} kt; normal-QQ r = {res['normal_qq_r']}")

# =========================================================================
# 5. SCATTER PLOT - Wind vs Pressure (wind-pressure relationship)
# =========================================================================
log("\n[5] Scatter plot wind vs pressure")
wp = df.dropna(subset=["Pressure"])
fig, ax = plt.subplots(figsize=(8, 4.6))
for b in ("Atlantic", "Pacific"):
    s = wp[wp.Basin == b]
    ax.scatter(s.Pressure, s.Wind, s=3, alpha=.25, color=BASIN_COL[b], label=b, rasterized=True)
lr = st.linregress(wp.Pressure, wp.Wind)
xx = np.linspace(wp.Pressure.min(), wp.Pressure.max(), 50)
ax.plot(xx, lr.intercept + lr.slope * xx, color="#222", lw=1.5,
        label=f"OLS: wind = {lr.intercept:.0f} {lr.slope:+.2f}*P  (r={lr.rvalue:.2f})")
ax.set(xlabel="Minimum central pressure (hPa)", ylabel="Maximum wind (kt)",
       title="Scatter plot: wind-pressure relationship")
ax.legend(markerscale=4)
save(fig, "fig05_scatter_wind_pressure.png")
res["wp_slope"] = round(lr.slope, 3); res["wp_r"] = round(lr.rvalue, 3)
res["wp_intercept"] = round(lr.intercept, 1)
log(f"   wind = {lr.intercept:.1f} + {lr.slope:.3f}*P, r = {lr.rvalue:.3f}, n = {len(wp)}")

# =========================================================================
# 6. SCATTER-PLOT MATRIX (storm level)
# =========================================================================
log("\n[6] Scatter-plot matrix (storm level)")
spm = storms[["MaxWind", "MinPressure", "DurationH", "TrackKm", "GenesisLat", "Basin"]].dropna()
g = sns.pairplot(spm, hue="Basin", palette=BASIN_COL, corner=True,
                 plot_kws={"s": 6, "alpha": .4, "linewidth": 0}, diag_kws={"linewidth": 1},
                 height=1.75)
g.figure.suptitle("Scatter-plot matrix of storm-level attributes", y=1.01, fontweight="bold")
g.figure.savefig(os.path.join(FIG_DIR, "fig06_scatter_matrix.png"), bbox_inches="tight", dpi=130)
plt.close(g.figure)
log("   -> saved fig06_scatter_matrix.png")

# =========================================================================
# 7. CORRELATION HEAT-MAP (storm level)
# =========================================================================
log("\n[7] Correlation heat-map")
cm = storms[["MaxWind", "MinPressure", "DurationH", "TrackKm", "NumObs", "GenesisLat",
             "GenesisLon", "Year", "Max24hIntensification", "Landfall"]].corr()
fig, ax = plt.subplots(figsize=(8, 6.4))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, center=0, ax=ax,
            linewidths=.5, linecolor="white", annot_kws={"size": 8}, cbar_kws={"label": "Pearson r"})
ax.set_title("Correlation heat-map (storm level)")
save(fig, "fig07_correlation_heatmap.png")
log(cm.round(2).to_string())
res["corr"] = cm.round(2).to_dict()

# =========================================================================
# 8. GROUPED BAR CHART - storms per genesis month and basin
# =========================================================================
log("\n[8] Bar chart: storms by genesis month")
mon = storms.groupby(["GenesisMonth", "Basin"]).size().unstack(fill_value=0).reindex(range(1, 13), fill_value=0)
fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(12); w = .4
for i, b in enumerate(("Atlantic", "Pacific")):
    ax.bar(x + (i - .5) * w, mon[b], w * .92, color=BASIN_COL[b], label=b)
ax.set_xticks(x, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
ax.set(ylabel="Number of storms formed", title="Bar chart: storm genesis by month")
ax.legend()
save(fig, "fig08_bar_month.png")
log(mon.T.to_string())
res["month_counts"] = mon.to_dict()
res["peak_month_atl"] = int(mon["Atlantic"].idxmax()); res["peak_month_pac"] = int(mon["Pacific"].idxmax())
res["atl_aug_oct_share"] = round(100 * mon.loc[8:10, "Atlantic"].sum() / mon["Atlantic"].sum(), 1)
res["pac_jul_sep_share"] = round(100 * mon.loc[7:9, "Pacific"].sum() / mon["Pacific"].sum(), 1)

# =========================================================================
# 9. LINE CHART - annual storm counts with 10-year moving average
# =========================================================================
log("\n[9] Line chart: storms per year")
yr = storms.groupby(["Year", "Basin"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(9, 4.2))
for b in ("Atlantic", "Pacific"):
    s = yr[b] if b == "Atlantic" else yr.loc[yr.index >= 1949, b]
    ax.plot(s.index, s.values, color=BASIN_COL[b], lw=.8, alpha=.45)
    ax.plot(s.index, s.rolling(10, center=True).mean(), color=BASIN_COL[b], lw=2.2,
            label=f"{b} (10-yr moving avg)")
ax.axvline(1966, color="#888", ls=":", lw=1)
ax.text(1967, ax.get_ylim()[1] * .93, "satellite era", fontsize=8, color="#555")
ax.set(xlabel="Year", ylabel="Storms per year", title="Line chart: annual storm counts")
ax.legend()
save(fig, "fig09_line_yearly.png")
pre = yr.loc[1851:1900, "Atlantic"].mean(); post = yr.loc[1966:2015, "Atlantic"].mean()
res["atl_mean_1851_1900"] = round(pre, 1); res["atl_mean_1966_2015"] = round(post, 1)
res["pac_mean_1971_2015"] = round(yr.loc[1971:2015, "Pacific"].mean(), 1)
res["pac_mean_1949_1970"] = round(yr.loc[1949:1970, "Pacific"].mean(), 1)
res["max_year_atl"] = int(yr["Atlantic"].idxmax()); res["max_year_atl_n"] = int(yr["Atlantic"].max())
log(f"   Atlantic mean/yr 1851-1900 = {pre:.1f}; 1966-2015 = {post:.1f}; "
    f"busiest Atlantic year {res['max_year_atl']} ({res['max_year_atl_n']} storms); "
    f"Pacific 1949-70 = {res['pac_mean_1949_1970']}, 1971-2015 = {res['pac_mean_1971_2015']}")

# =========================================================================
# 10. PIE CHART - composition of observations by status
# =========================================================================
log("\n[10] Pie chart: status composition")
sc = df["StatusGroup"].value_counts()
fig, ax = plt.subplots(figsize=(6.4, 5))
ax.pie(sc.values, labels=[f"{k}\n{v:,}" for k, v in sc.items()], autopct="%1.1f%%",
       colors=CAT_COLS[:len(sc)], startangle=90, counterclock=False, pctdistance=.78,
       wedgeprops={"edgecolor": "white", "linewidth": 2}, textprops={"fontsize": 9})
ax.set_title("Pie chart: share of 6-hourly observations by status")
ax.grid(False)
save(fig, "fig10_pie_status.png")
res["status_share"] = (100 * sc / sc.sum()).round(1).to_dict()
log(res["status_share"])

# =========================================================================
# 11. VIOLIN PLOT - storm duration by lifetime maximum category
# =========================================================================
log("\n[11] Violin plot: duration by max category")
fig, ax = plt.subplots(figsize=(8, 4.2))
sv = storms.copy(); sv["DurationDays"] = sv.DurationH / 24
sns.violinplot(data=sv, x="MaxCategory", y="DurationDays", order=SS_ORDER, color=CAT_COLS[0],
               inner="quartile", cut=0, linewidth=.8, ax=ax)
ax.set(xlabel="Lifetime maximum Saffir-Simpson category", ylabel="Storm duration (days)",
       title="Violin plot: storm lifetime by peak intensity")
save(fig, "fig11_violin_duration.png")
md = sv.groupby("MaxCategory")["DurationDays"].median().reindex(SS_ORDER).round(1)
log(md.to_string())
res["median_duration_by_cat"] = md.to_dict()

# =========================================================================
# 12. GEOGRAPHIC TRACK MAP - all fixes coloured by intensity
# =========================================================================
log("\n[12] Track map")
fig, ax = plt.subplots(figsize=(10, 4.8))
srt = df.sort_values("Wind")
sca = ax.scatter(srt.Longitude, srt.Latitude, c=srt.Wind, s=.6, cmap="YlOrRd", vmin=20, vmax=150,
                 rasterized=True)
plt.colorbar(sca, ax=ax, label="Maximum wind (kt)", pad=.01)
ax.set(xlim=(-185, 15), ylim=(0, 72), xlabel="Longitude (deg, W negative)", ylabel="Latitude (deg N)",
       title="Track map: every 6-hourly fix coloured by intensity")
ax.set_aspect("equal")
save(fig, "fig12_track_map.png")
hu = df[df.Status == "HU"]
res["hu_lat_mean"] = round(hu.Latitude.mean(), 1)
res["major_lat_p90"] = round(df.loc[df.Wind >= 96, "Latitude"].quantile(.9), 1)
log(f"   mean latitude of HU fixes = {res['hu_lat_mean']}; 90th pct latitude of major fixes = {res['major_lat_p90']}")

# =========================================================================
# 13. PARALLEL COORDINATES - storm profiles by peak category
# =========================================================================
log("\n[13] Parallel coordinates")
pc = storms[["MaxWind", "MinPressure", "DurationH", "TrackKm", "GenesisLat", "MaxCategory"]].dropna()
pc = pc[pc.MaxCategory.isin(["TS", "Cat1", "Cat3", "Cat5"])]
pc = pd.concat([pc[pc.MaxCategory == c].sample(min((pc.MaxCategory == c).sum(), 40), random_state=1)
                for c in ["TS", "Cat1", "Cat3", "Cat5"]])          # ordered weak -> strong
cols = ["MaxWind", "MinPressure", "DurationH", "TrackKm", "GenesisLat"]
pcn = pc.copy()
pcn[cols] = (pc[cols] - pc[cols].min()) / (pc[cols].max() - pc[cols].min())   # min-max scale axes
fig, ax = plt.subplots(figsize=(9, 4.4))
parallel_coordinates(pcn, "MaxCategory", cols=cols, color=CAT_COLS[:4], alpha=.45, lw=1, ax=ax)
ax.set(ylabel="Min-max normalised value", title="Parallel coordinates: storm profiles by peak category")
ax.legend(title="Peak category", loc="upper right")
save(fig, "fig13_parallel_coordinates.png")
log(pc.groupby("MaxCategory")[cols].median().round(0).to_string())
res["pc_medians"] = pc.groupby("MaxCategory")[cols].median().round(0).to_dict(orient="index")

# =========================================================================
# 14. HEXBIN DENSITY PLOT - genesis locations
# =========================================================================
log("\n[14] Hexbin density of genesis points")
fig, ax = plt.subplots(figsize=(10, 4.4))
hb = ax.hexbin(storms.GenesisLon, storms.GenesisLat, gridsize=45, cmap=SEQ, mincnt=1,
               extent=(-185, 0, 0, 50))
plt.colorbar(hb, ax=ax, label="Storms formed per cell")
ax.set(xlabel="Longitude", ylabel="Latitude", title="Hexbin density: where storms are born")
ax.set_aspect("equal")
save(fig, "fig14_hexbin_genesis.png")
res["genesis_lat_median"] = storms.groupby("Basin").GenesisLat.median().round(1).to_dict()
res["genesis_lon_median"] = storms.groupby("Basin").GenesisLon.median().round(1).to_dict()
log(f"   median genesis lat {res['genesis_lat_median']}, lon {res['genesis_lon_median']}")

# =========================================================================
# 15. 100 % STACKED BAR - peak-category mix per era (hurricanes only)
# =========================================================================
log("\n[15] Stacked bar: category mix by period")
hs = storms[storms.MaxCategory.isin(["Cat1", "Cat2", "Cat3", "Cat4", "Cat5"])].copy()
hs["Period"] = pd.cut(hs.Year, [1850, 1900, 1950, 1980, 2000, 2015],
                      labels=["1851-1900", "1901-1950", "1951-1980", "1981-2000", "2001-2015"])
ct = pd.crosstab([hs.Basin, hs.Period], hs.MaxCategory, normalize="index").mul(100)
ct = ct[["Cat1", "Cat2", "Cat3", "Cat4", "Cat5"]]
ct = ct[(ct.index.get_level_values(0) == "Atlantic") |
        (ct.index.get_level_values(1).isin(["1951-1980", "1981-2000", "2001-2015"]))]
fig, ax = plt.subplots(figsize=(9, 4.4))
blues = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
bottom = np.zeros(len(ct))
labels = [f"{b[:3]}\n{p}" for b, p in ct.index]
for c, colr in zip(ct.columns, blues):
    ax.bar(labels, ct[c], bottom=bottom, color=colr, label=c, edgecolor="white", linewidth=1.5)
    bottom += ct[c].values
ax.set(ylabel="% of hurricanes", title="Stacked bar: peak category mix of hurricanes by period")
ax.legend(title="Peak", bbox_to_anchor=(1.01, 1), loc="upper left")
ax.tick_params(axis="x", labelsize=8)
save(fig, "fig15_stacked_category.png")
log(ct.round(1).to_string())
ct["major"] = ct[["Cat3", "Cat4", "Cat5"]].sum(axis=1)
res["major_share_by_period"] = {f"{b} {p}": round(v, 1) for (b, p), v in ct["major"].items()}
log(res["major_share_by_period"])

# ---- extra facts used in the inferences ---------------------------------
res["landfall_by_basin"] = (storms.groupby("Basin").Landfall.mean() * 100).round(1).to_dict()
res["hurricane_rate"] = (storms.groupby("Basin").BecameHurricane.mean() * 100).round(1).to_dict()
res["ex_rate"] = (storms.groupby("Basin").BecameExtratropical.mean() * 100).round(1).to_dict()
log("\nLandfall % by basin:", res["landfall_by_basin"])
log("Storms reaching hurricane %:", res["hurricane_rate"])
log("Storms becoming extratropical %:", res["ex_rate"])
top = storms.nlargest(5, "MaxWind")[["ID", "Name", "Year", "MaxWind", "MinPressure"]]
log("Strongest storms:\n" + top.to_string(index=False))
res["top_storms"] = top.to_dict(orient="records")

save_json(res, "visualization")
log.close()
