"""
01_preprocessing.py
-------------------
DATA PREPROCESSING for the NOAA hurricane database (HURDAT2).

   (i)   DATA CLEANING        - remove / repair noisy, missing and inconsistent data
   (ii)  DATA SELECTION       - keep only the task-relevant attributes
   (iii) DATA TRANSFORMATION  - parse, derive, discretise, aggregate and encode

Outputs (outputs/data):
   clean_tracks.csv   - one row per 6-hourly observation (cleaned + derived features)
   storms.csv         - one row per storm (aggregated)
   transactions.csv   - one item-set per observation for association-rule mining
Log: outputs/logs/01_preprocessing.txt  + preprocessing.json (numbers for report)
"""
import numpy as np
import pandas as pd

from common import (RAW_FILES, CLEAN_TRACKS, STORMS, TRANSACTIONS, Logger, save_json)

log = Logger("01_preprocessing")
stats = {}

RADII_COLS = [f"{lvl} Wind {q}" for lvl in ("Low", "Moderate", "High")
              for q in ("NE", "SE", "SW", "NW")]

# =========================================================================
# 0. LOAD
# =========================================================================
frames = []
for basin, path in RAW_FILES.items():
    d = pd.read_csv(path, dtype={"Date": str, "Time": int})
    d["Basin"] = basin
    frames.append(d)
    log(f"Loaded {basin:8s}: {d.shape[0]:6d} rows x {d.shape[1]} cols, {d['ID'].nunique()} storms")
raw = pd.concat(frames, ignore_index=True)
stats["raw_rows"] = len(raw)
stats["raw_cols"] = raw.shape[1] - 1           # without the added Basin column
stats["raw_storms"] = raw["ID"].nunique()
log(f"Combined raw data: {raw.shape}")

# =========================================================================
# (i) DATA CLEANING
# =========================================================================
log("\n================ (i) DATA CLEANING ================")
df = raw.copy()

# --- 1. Strip the padding white-space present in Name / Event / Status -----
for c in ("ID", "Name", "Event", "Status"):
    df[c] = df[c].astype(str).str.strip()
log("1. Stripped leading/trailing white-space from ID, Name, Event, Status "
    "(e.g. '            UNNAMED' -> 'UNNAMED', ' HU' -> 'HU').")

# --- 2. Sentinel values (-999 / -99) are NOT measurements -> NaN -------------
num_cols = ["Maximum Wind", "Minimum Pressure"] + RADII_COLS
sentinel_counts = {}
for c in num_cols:
    mask = df[c] < 0                         # -999 and -99 are the only negatives
    sentinel_counts[c] = int(mask.sum())
    df.loc[mask, c] = np.nan
log("2. Sentinel values (-999, -99) replaced by NaN:")
for c, n in sentinel_counts.items():
    log(f"     {c:20s}: {n:6d} ({100*n/len(df):5.1f} %)")
stats["sentinels"] = sentinel_counts

# --- 3. Parse coordinates '28.0N' / '94.8W' into signed decimal degrees -----
def parse_coord(s):
    s = str(s).strip()
    val, hemi = float(s[:-1]), s[-1]
    return -val if hemi in ("S", "W") else val

df["Latitude"] = df["Latitude"].map(parse_coord)
df["Longitude"] = df["Longitude"].map(parse_coord)
log("3. Latitude/Longitude strings (e.g. '28.0N','94.8W') converted to signed floats "
    "(S and W negative).")

# --- 4. Build a proper timestamp from Date (YYYYMMDD) + Time (HHMM) --------
df["Datetime"] = pd.to_datetime(df["Date"] + df["Time"].astype(str).str.zfill(4),
                                format="%Y%m%d%H%M", errors="coerce")
bad_dt = int(df["Datetime"].isna().sum())
log(f"4. Date+Time combined into a Datetime; unparsable timestamps: {bad_dt}")
df = df[df["Datetime"].notna()]

# --- 5. Duplicate records ---------------------------------------------------
n0 = len(df)
df = df.drop_duplicates()
exact_dups = n0 - len(df)
n0 = len(df)
df = df.sort_values(["ID", "Datetime"]).drop_duplicates(["ID", "Datetime"], keep="first")
key_dups = n0 - len(df)
log(f"5. Exact duplicate rows removed: {exact_dups}; duplicate (ID, Datetime) fixes removed: {key_dups}")
stats["dups_removed"] = exact_dups + key_dups

# --- 6. Inconsistent categorical codes ---------------------------------------
before = df["Status"].value_counts().to_dict()
STATUS_FIX = {"ET": "EX",    # Pacific file uses ET and EX for the same thing (extratropical)
              "PT": "EX",    # post-tropical -> treated as extratropical remnant
              "ST": "HU"}    # 'ST' = Super Typhoon (Genevieve 2014 after crossing the dateline,
                             #  130-140 kt) -> a hurricane-strength tropical cyclone
df["Status"] = df["Status"].replace(STATUS_FIX)
log(f"6. Status code harmonisation {STATUS_FIX}; raw codes were {before}")
stats["status_fix"] = {k: int(before.get(k, 0)) for k in STATUS_FIX}

# --- 7. Physical range checks (noise): values outside plausible limits ------
#     Wind  : 10 .. 200 kt   (record is 185 kt, Patricia 2015)
#     Press.: 870 .. 1030 hPa (record is 872 hPa, Patricia 2015 / 882 Wilma 2005)
#     Lat   : -90 .. 90 ; Lon must lie in -180 .. 180
rng_wind = df["Maximum Wind"].notna() & ~df["Maximum Wind"].between(10, 200)
rng_pres = df["Minimum Pressure"].notna() & ~df["Minimum Pressure"].between(870, 1030)
df.loc[rng_wind, "Maximum Wind"] = np.nan
df.loc[rng_pres, "Minimum Pressure"] = np.nan
log(f"7. Out-of-range values set to NaN -> wind: {int(rng_wind.sum())}, pressure: {int(rng_pres.sum())}")
# Longitudes such as '359.1W' (AL051952) are typing noise for 0.9E -> repaired by wrapping
bad_lon = ~df["Longitude"].between(-180, 180)
log("   Longitude outside +/-180 (repaired by wrapping into [-180,180]):\n" +
    df.loc[bad_lon, ["ID", "Datetime", "Latitude", "Longitude"]].to_string(index=False))
df.loc[bad_lon, "Longitude"] = (df.loc[bad_lon, "Longitude"] + 180) % 360 - 180
stats["lon_repaired"] = int(bad_lon.sum())
bad_lat = ~df["Latitude"].between(-90, 90)
df = df[~bad_lat]
log(f"   Latitude outside +/-90: {int(bad_lat.sum())}. (Fixes up to 81N exist - extratropical "
    "remnants of AL091951 reaching the Arctic - and are genuine, so they are kept.)")

# --- 8. Missing target attribute: Maximum Wind is the key intensity variable
n0 = len(df)
all_missing = df.groupby("ID")["Maximum Wind"].apply(lambda s: s.isna().all())
df = df[df["Maximum Wind"].notna()]
miss_wind = n0 - len(df)
log(f"8. Rows with missing Maximum Wind dropped: {miss_wind}. {int(all_missing.sum())} storms "
    f"(1967-68 Atlantic depressions, e.g. {', '.join(all_missing[all_missing].index[:3])}) had NO "
    "wind value at all and therefore disappear completely.")
stats["storms_no_wind"] = int(all_missing.sum())
stats["missing_wind_dropped"] = miss_wind

# --- 9. Logical inconsistency between Status and Maximum Wind ---------------
#     By definition  TD < 34 kt ; 34 <= TS <= 63 kt ; HU >= 64 kt
w = df["Maximum Wind"]
incons = (((df["Status"] == "HU") & (w < 64)) |
          ((df["Status"] == "TS") & ((w < 34) | (w > 63))) |
          ((df["Status"] == "TD") & (w >= 34)))
log(f"9. Status/Wind contradictions (e.g. Status=HU but wind<64 kt): {int(incons.sum())} "
    f"-> removed as noisy records")
log("   examples:\n" + df.loc[incons, ["ID", "Datetime", "Status", "Maximum Wind"]]
    .head(6).to_string(index=False))
stats["status_wind_inconsistent"] = int(incons.sum())
df = df[~incons]

# --- 10. Kinematic noise: impossible jumps between consecutive fixes -------
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2 +
         np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))

def add_motion(d):
    d = d.sort_values(["ID", "Datetime"]).copy()
    g = d.groupby("ID")
    d["dt_h"] = g["Datetime"].diff().dt.total_seconds() / 3600
    d["step_km"] = haversine_km(g["Latitude"].shift(), g["Longitude"].shift(),
                                d["Latitude"], d["Longitude"])
    d["Speed_kmh"] = d["step_km"] / d["dt_h"]
    return d

df = add_motion(df)
jump = df["Speed_kmh"] > 150          # tropical cyclones never move > ~110 km/h
log(f"10. Fixes implying a translation speed > 150 km/h (position noise): {int(jump.sum())} -> removed")
stats["speed_noise"] = int(jump.sum())
df = add_motion(df[~jump])             # recompute motion after removal

# --- 11. Outlier inspection (IQR rule) - reported, NOT removed --------------
for c in ("Maximum Wind", "Minimum Pressure"):
    q1, q3 = df[c].quantile([.25, .75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    n_out = int(((df[c] < lo) | (df[c] > hi)).sum())
    log(f"11. IQR outliers in {c}: {n_out} (fences {lo:.1f} .. {hi:.1f}). Kept: these are "
        "genuine extreme storms (e.g. Patricia 2015, Wilma 2005), not errors.")
    stats[f"iqr_outliers_{c}"] = n_out

stats["clean_rows"] = len(df)
stats["clean_storms"] = df["ID"].nunique()
log(f"\nAfter cleaning: {len(df)} rows, {df['ID'].nunique()} storms "
    f"({100*(1-len(df)/len(raw)):.2f} % of rows removed)")

# =========================================================================
# (ii) DATA SELECTION
# =========================================================================
log("\n================ (ii) DATA SELECTION ================")
missing_pct = (df[num_cols].isna().mean() * 100).round(1)
log("Missing % after cleaning:\n" + missing_pct.to_string())
radii_first_year = int(df.loc[df["Low Wind NE"].notna(), "Datetime"].dt.year.min())
log(f"-> 12 wind-radii columns are {missing_pct[RADII_COLS].min()}-{missing_pct[RADII_COLS].max()} % "
    f"missing (only recorded since {radii_first_year}) -> DROPPED.")
log("-> 'Date','Time' replaced by the derived 'Datetime'.")
log("-> 'Event' kept only as a binary 'Landfall' flag (it is blank for 98 % of rows).")
log("-> 'Minimum Pressure' kept (missing mostly before 1979) - used where available.")
stats["missing_pct"] = missing_pct.to_dict()
stats["radii_first_year"] = radii_first_year

# Correlation used to justify that Pressure is redundant-but-useful with Wind
r = df[["Maximum Wind", "Minimum Pressure"]].corr().iloc[0, 1]
log(f"Pearson r(Maximum Wind, Minimum Pressure) = {r:.3f} (strong inverse relation)")
stats["corr_wind_pressure"] = round(float(r), 3)

df["Landfall"] = (df["Event"] == "L").astype(int)
SELECTED = ["ID", "Name", "Basin", "Datetime", "Status", "Latitude", "Longitude",
            "Maximum Wind", "Minimum Pressure", "Landfall", "dt_h", "step_km", "Speed_kmh"]
df = df[SELECTED].rename(columns={"Maximum Wind": "Wind", "Minimum Pressure": "Pressure"})
log(f"Selected attributes ({len(SELECTED)}): {SELECTED}")

# =========================================================================
# (iii) DATA TRANSFORMATION
# =========================================================================
log("\n================ (iii) DATA TRANSFORMATION ================")
df["Year"] = df["Datetime"].dt.year
df["Month"] = df["Datetime"].dt.month
df["Hour"] = df["Datetime"].dt.hour
df["Decade"] = (df["Year"] // 10) * 10
df["Wind_kmh"] = (df["Wind"] * 1.852).round(1)                 # unit conversion kt->km/h
log("a) Temporal features: Year, Month, Hour, Decade; unit conversion Wind kt -> km/h")

# Saffir-Simpson category (discretisation of Wind by domain thresholds)
SS_BINS = [0, 33, 63, 82, 95, 112, 136, 999]
SS_LABELS = ["TD", "TS", "Cat1", "Cat2", "Cat3", "Cat4", "Cat5"]
df["SS_Category"] = pd.cut(df["Wind"], SS_BINS, labels=SS_LABELS).astype(str)
log("b) Discretisation: Wind -> Saffir-Simpson category " + str(dict(zip(SS_LABELS, SS_BINS[1:]))))

# Status grouping (concept hierarchy: raw code -> family)
STATUS_GROUP = {"TD": "TD", "TS": "TS", "HU": "HU", "EX": "EX",
                "SD": "SUB", "SS": "SUB", "LO": "LOW", "WV": "LOW", "DB": "LOW"}
df["StatusGroup"] = df["Status"].map(STATUS_GROUP)
log("c) Concept hierarchy: Status codes -> " + str(STATUS_GROUP))

# Per-storm dynamics
g = df.groupby("ID")
df["HoursSinceGenesis"] = (df["Datetime"] - g["Datetime"].transform("min")).dt.total_seconds() / 3600
dlon = g["Longitude"].diff()
dlon = (dlon + 180) % 360 - 180                                  # dateline-safe
df["dLat"] = g["Latitude"].diff()
df["dLon"] = dlon
df["dWind"] = g["Wind"].diff()
df["Heading"] = np.where(df["dLon"].isna(), np.nan,
                         (np.degrees(np.arctan2(df["dLon"], df["dLat"])) + 360) % 360)
log("d) Derived dynamics: HoursSinceGenesis, dLat, dLon, dWind, Heading (deg), Speed_kmh")

# Min-max normalisation example (used for plots / clustering)
for c in ("Wind", "Latitude", "Longitude"):
    df[c + "_norm"] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())
log("e) Min-max normalisation of Wind, Latitude, Longitude to [0,1] (columns *_norm)")

df.to_csv(CLEAN_TRACKS, index=False)
log(f"Saved {CLEAN_TRACKS}  shape={df.shape}")

# ---- Storm-level aggregation (data reduction / cube roll-up) -------------
first = g.first()
storms = pd.DataFrame({
    "Name": first["Name"],
    "Basin": first["Basin"],
    "Year": first["Year"],
    "GenesisMonth": first["Month"],
    "GenesisLat": first["Latitude"],
    "GenesisLon": first["Longitude"],
    "Start": g["Datetime"].min(),
    "End": g["Datetime"].max(),
    "MaxWind": g["Wind"].max(),
    "MinPressure": g["Pressure"].min(),
    "NumObs": g.size(),
    "TrackKm": g["step_km"].sum(),
    "Landfall": g["Landfall"].max(),
    "BecameHurricane": g["Status"].apply(lambda s: int((s == "HU").any())),
    "BecameExtratropical": g["Status"].apply(lambda s: int((s == "EX").any())),
    "Max24hIntensification": g["dWind"].apply(lambda s: s.rolling(4, min_periods=1).sum().max()),
})
storms["DurationH"] = (storms["End"] - storms["Start"]).dt.total_seconds() / 3600
storms["MaxCategory"] = pd.cut(storms["MaxWind"], SS_BINS, labels=SS_LABELS).astype(str)
storms.index.name = "ID"
storms.to_csv(STORMS)
log(f"f) Aggregation to storm level -> {STORMS}  shape={storms.shape}")
stats["n_storms"] = len(storms)

# ---- Transactions for association-rule mining ----------------------------
# Each cleaned observation becomes a transaction of categorical items
# (discretisation + attribute=value encoding).
def season(m):
    return "Jan-Jul" if m <= 7 else ("Aug-Sep" if m <= 9 else "Oct-Dec")

def lat_band(x):
    return "<15N" if x < 15 else ("15-25N" if x < 25 else ("25-35N" if x < 35 else ">=35N"))

def pres_band(p):
    if pd.isna(p):
        return None
    return "<980" if p < 980 else ("980-999" if p < 1000 else ">=1000")

def era(y):
    return "1851-1949" if y < 1950 else ("1950-1989" if y < 1990 else "1990-2015")

def speed_band(s):
    if pd.isna(s):
        return None
    return "Slow(<15)" if s < 15 else ("Medium(15-30)" if s <= 30 else "Fast(>30)")

items = []
for row in df.itertuples(index=False):
    t = [f"Basin={row.Basin}", f"Season={season(row.Month)}", f"Lat={lat_band(row.Latitude)}",
         f"Status={row.StatusGroup}", f"Era={era(row.Year)}"]
    p = pres_band(row.Pressure)
    if p:
        t.append(f"Pressure={p}")
    s = speed_band(row.Speed_kmh)
    if s:
        t.append(f"Speed={s}")
    if not pd.isna(row.dLon) and row.dLon != 0:
        t.append("Motion=Eastward" if row.dLon > 0 else "Motion=Westward")
    if row.Landfall:
        t.append("Landfall=Yes")
    items.append("|".join(sorted(t)))
pd.DataFrame({"items": items}).to_csv(TRANSACTIONS, index=False)
uniq = sorted({i for t in items for i in t.split("|")})
log(f"g) {len(items)} transactions over {len(uniq)} distinct items -> {TRANSACTIONS}")
log("   items: " + ", ".join(uniq))
log("   example transaction: " + items[100])
stats["n_transactions"] = len(items)
stats["n_items"] = len(uniq)
stats["items"] = uniq

save_json(stats, "preprocessing")
log.close()
