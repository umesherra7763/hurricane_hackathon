"""
06_dbscan.py
------------
DBSCAN clustering of storm GENESIS locations (first fix of each storm) to find
the spatial "nurseries" of tropical cyclones and separate them from sporadic,
isolated genesis events (noise).

Why DBSCAN: genesis regions are irregular, elongated shapes of different size,
the number of regions is unknown, and isolated genesis points should be flagged
as noise - exactly what density-based clustering handles (k-means does not).

Parameters (assumed & stated):
  * metric  = haversine (great-circle distance on the Earth, radius 6371 km)
  * MinPts  = 15 storms
  * eps     = chosen from the knee of the sorted 15-distance plot (k-distance graph),
              searched on the main body of the curve (lowest 95 % of distances)
Outputs: k-distance plot, parameter sensitivity grid, cluster map, cluster profile
table, silhouette score, and a dendrogram (Ward linkage) of the cluster profiles.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram

from common import STORMS, FIG_DIR, CLEAN_DIR, Logger, save_json

log = Logger("06_dbscan")
R = 6371.0
MIN_PTS = 15
COLS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

storms = pd.read_csv(STORMS)
X = np.radians(storms[["GenesisLat", "GenesisLon"]].values)       # (lat, lon) in radians
log(f"{len(storms)} genesis points; metric = haversine; MinPts = {MIN_PTS}")

# ---- 1. k-distance graph to choose eps ------------------------------------
nn = NearestNeighbors(n_neighbors=MIN_PTS, metric="haversine").fit(X)
kd = np.sort(nn.kneighbors(X)[0][:, -1] * R)
# knee = point of maximum distance to the chord joining first and last point.
# The top 5 % of k-distances are isolated outliers whose huge values dominate the
# curvature (a knee found on the full curve lands at ~430 km and chains the Atlantic
# and Pacific into ONE cluster), so the knee is searched on the main body (lowest 95 %).
# The knee is stable: cutting at 90 % instead of 95 % gives the same point.
body = kd[:int(0.95 * len(kd))]
x = np.arange(len(body)); y = body
p1, p2 = np.array([0, y[0]]), np.array([len(y) - 1, y[-1]])
chord = (p2 - p1) / np.linalg.norm(p2 - p1)
vec = np.stack([x, y], 1) - p1
dist = np.abs(vec[:, 0] * chord[1] - vec[:, 1] * chord[0])
knee = int(np.argmax(dist))
EPS_KM = float(np.round(kd[knee] / 10) * 10)
log(f"k-distance knee at point {knee}: {kd[knee]:.1f} km -> eps = {EPS_KM:.0f} km")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.arange(len(kd)), kd, color="#2a78d6", lw=2)
ax.axvline(len(body), color="#888", ls=":", lw=1)
ax.text(len(body) - 10, kd.max() * .9, "outlier tail (top 5 %)", ha="right", fontsize=8, color="#555")
ax.axhline(EPS_KM, color="#eb6834", ls="--", lw=1.2, label=f"eps = {EPS_KM:.0f} km (knee)")
ax.scatter([knee], [kd[knee]], color="#eb6834", zorder=3)
ax.set(xlabel="Genesis points sorted by distance", ylabel=f"Distance to {MIN_PTS}-th neighbour (km)",
       title="k-distance graph for choosing eps")
ax.legend(frameon=False); ax.grid(color="#e1e0d9"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "db_kdistance.png"), dpi=150); plt.close(fig)

# ---- 2. Sensitivity of the result to eps / MinPts -----------------------
grid = []
for e in (EPS_KM * .6, EPS_KM * .8, EPS_KM, EPS_KM * 1.2, EPS_KM * 1.5):
    for mp in (10, 15, 25):
        lab = DBSCAN(eps=e / R, min_samples=mp, metric="haversine").fit_predict(X)
        grid.append({"eps_km": round(e), "MinPts": mp, "clusters": len(set(lab)) - (-1 in lab),
                     "noise_%": round(100 * (lab == -1).mean(), 1)})
gdf = pd.DataFrame(grid)
log("\nParameter sensitivity:\n" + gdf.to_string(index=False))

# ---- 3. Final DBSCAN ------------------------------------------------------
db = DBSCAN(eps=EPS_KM / R, min_samples=MIN_PTS, metric="haversine").fit(X)
lab = db.labels_
storms["Cluster"] = lab
n_cl = len(set(lab)) - (-1 in lab)
core = np.zeros(len(lab), bool); core[db.core_sample_indices_] = True
log(f"\nFINAL: eps = {EPS_KM:.0f} km, MinPts = {MIN_PTS} -> {n_cl} clusters, "
    f"{(lab == -1).sum()} noise points ({100 * (lab == -1).mean():.1f} %), "
    f"{core.sum()} core / {((lab != -1) & ~core).sum()} border points")
m = lab != -1
sil = silhouette_score(X[m], lab[m], metric="haversine", sample_size=min(2000, m.sum()), random_state=0)
log(f"Silhouette coefficient (non-noise points, haversine) = {sil:.3f}")

# Cluster profiles (relabel clusters by size so the largest is C0)
order = storms[m].Cluster.value_counts().index.tolist()
remap = {c: i for i, c in enumerate(order)}; remap[-1] = -1
storms["Cluster"] = storms.Cluster.map(remap)
prof = storms[storms.Cluster >= 0].groupby("Cluster").agg(
    storms=("Basin", "size"), atlantic_pct=("Basin", lambda s: round(100 * (s == "Atlantic").mean(), 1)),
    lat_mean=("GenesisLat", "mean"), lon_mean=("GenesisLon", "mean"),
    lat_min=("GenesisLat", "min"), lat_max=("GenesisLat", "max"),
    lon_min=("GenesisLon", "min"), lon_max=("GenesisLon", "max"),
    peak_month=("GenesisMonth", lambda s: int(s.mode()[0])),
    mean_maxwind=("MaxWind", "mean"), hurricane_pct=("BecameHurricane", lambda s: round(100 * s.mean(), 1)),
    major_pct=("MaxWind", lambda s: round(100 * (s >= 96).mean(), 1)),
    landfall_pct=("Landfall", lambda s: round(100 * s.mean(), 1)),
    median_duration_h=("DurationH", "median")).round(1)
noise = storms[storms.Cluster == -1]
log("\nCluster profiles:\n" + prof.to_string())
log(f"\nNoise storms: n={len(noise)}, mean genesis lat={noise.GenesisLat.mean():.1f}, "
    f"hurricane %={100 * noise.BecameHurricane.mean():.1f}, mean max wind={noise.MaxWind.mean():.1f} kt")
storms[["Name", "Basin", "Year", "GenesisLat", "GenesisLon", "MaxWind", "Cluster"]].to_csv(
    f"{CLEAN_DIR}/dbscan_genesis_clusters.csv")

# ---- 4. Cluster map ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4.8))
nz = storms[storms.Cluster == -1]
ax.scatter(nz.GenesisLon, nz.GenesisLat, s=8, color="#b5b4ac", marker="x", lw=.8,
           label=f"noise ({len(nz)})")
for c in range(n_cl):
    s = storms[storms.Cluster == c]
    col = COLS[c % len(COLS)]
    ax.scatter(s.GenesisLon, s.GenesisLat, s=9, color=col, alpha=.75, linewidths=0,
               label=f"C{c} ({len(s)})")
    ax.annotate(f"C{c}", (prof.loc[c, "lon_mean"], prof.loc[c, "lat_mean"]), fontsize=10,
                fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=.2", fc="white", ec=col, lw=1.2))
ax.set(xlim=(-185, 5), ylim=(0, 55), xlabel="Longitude", ylabel="Latitude",
       title=f"DBSCAN genesis clusters (eps = {EPS_KM:.0f} km, MinPts = {MIN_PTS})")
ax.set_aspect("equal"); ax.grid(color="#e1e0d9"); ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8, ncol=1, bbox_to_anchor=(1.01, 1), loc="upper left", markerscale=1.6)
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "db_clusters_map.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

# ---- 5. Level-2 DBSCAN: density varies inside each basin, so each level-1
#         cluster is re-clustered at a finer scale (eps2 = 0.65 * eps) to reveal the
#         individual genesis hot-spots ("multi-scale DBSCAN").
EPS2_KM = round(0.65 * EPS_KM)
storms["SubCluster"] = "noise"
sub_rows = []
for c in range(n_cl):
    idx = storms.index[storms.Cluster == c]
    l2 = DBSCAN(eps=EPS2_KM / R, min_samples=MIN_PTS, metric="haversine").fit_predict(X[idx])
    sizes = pd.Series(l2[l2 >= 0]).value_counts()
    for rank, k in enumerate(sizes.index):
        storms.loc[idx[l2 == k], "SubCluster"] = f"C{c}.{rank}"
    storms.loc[idx[l2 == -1], "SubCluster"] = f"C{c}.diffuse"
sub = storms[storms.SubCluster.str.match(r"^C\d+\.\d+$")]
sprof = sub.groupby("SubCluster").agg(
    storms=("Basin", "size"), lat_mean=("GenesisLat", "mean"), lon_mean=("GenesisLon", "mean"),
    peak_month=("GenesisMonth", lambda s: int(s.mode()[0])), mean_maxwind=("MaxWind", "mean"),
    hurricane_pct=("BecameHurricane", lambda s: round(100 * s.mean(), 1)),
    major_pct=("MaxWind", lambda s: round(100 * (s >= 96).mean(), 1)),
    landfall_pct=("Landfall", lambda s: round(100 * s.mean(), 1)),
    median_duration_h=("DurationH", "median")).round(1)
log(f"\nLevel-2 DBSCAN inside each cluster (eps2 = {EPS2_KM} km, MinPts = {MIN_PTS}):")
log(storms.SubCluster.value_counts().to_string())
log("\nSub-cluster (genesis hot-spot) profiles:\n" + sprof.to_string())

fig, ax = plt.subplots(figsize=(11, 4.8))
rest = storms[~storms.index.isin(sub.index)]
ax.scatter(rest.GenesisLon, rest.GenesisLat, s=7, color="#cfcec6", lw=0, label="diffuse / noise")
for i, (k, s) in enumerate(sub.groupby("SubCluster")):
    col = COLS[i % len(COLS)]
    ax.scatter(s.GenesisLon, s.GenesisLat, s=9, color=col, alpha=.8, lw=0, label=f"{k} ({len(s)})")
    ax.annotate(k, (sprof.loc[k, "lon_mean"], sprof.loc[k, "lat_mean"]), fontsize=8, fontweight="bold",
                ha="center", va="center", bbox=dict(boxstyle="round,pad=.2", fc="white", ec=col, lw=1.2))
ax.set(xlim=(-185, 5), ylim=(0, 55), xlabel="Longitude", ylabel="Latitude",
       title=f"Level-2 DBSCAN: genesis hot-spots (eps = {EPS2_KM} km, MinPts = {MIN_PTS})")
ax.set_aspect("equal"); ax.grid(color="#e1e0d9"); ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left", markerscale=1.6)
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "db_subclusters_map.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

# ---- 6. Dendrogram of the hot-spots (Ward linkage on standardised profiles) --
feat = sprof[["lat_mean", "lon_mean", "peak_month", "mean_maxwind", "hurricane_pct", "landfall_pct",
              "median_duration_h"]]
Z = linkage(StandardScaler().fit_transform(feat), method="ward")
fig, ax = plt.subplots(figsize=(8, 4.2))
dendrogram(Z, labels=[f"{k} (n={sprof.loc[k, 'storms']})" for k in sprof.index], ax=ax,
           color_threshold=0.6 * Z[:, 2].max(), above_threshold_color="#888", leaf_rotation=30)
ax.set(ylabel="Ward linkage distance",
       title="Dendrogram of DBSCAN genesis hot-spots (profile similarity)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "db_dendrogram.png"), dpi=150); plt.close(fig)
log("\nWard linkage matrix (cluster i, cluster j, distance, size):\n" + np.round(Z, 3).__str__())
log("Dendrogram leaf order: " + str(dendrogram(Z, no_plot=True, labels=list(sprof.index))["ivl"]))

save_json({"eps_km": EPS_KM, "min_pts": MIN_PTS, "n_clusters": n_cl, "noise": int((lab == -1).sum()),
           "noise_pct": round(100 * (lab == -1).mean(), 1), "core": int(core.sum()),
           "silhouette": round(float(sil), 3), "profiles": prof.reset_index().to_dict(orient="records"),
           "grid": grid, "eps2_km": EPS2_KM, "sub_profiles": sprof.reset_index().to_dict(orient="records"),
           "noise_hu_pct": round(100 * noise.BecameHurricane.mean(), 1),
           "noise_lat": round(noise.GenesisLat.mean(), 1)}, "dbscan")
log.close()
