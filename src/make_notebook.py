"""
make_notebook.py
----------------
Assembles Hurricane_Analytics.ipynb from the source scripts in src/ so that the
notebook runs exactly the same code as the .py files. Each script is split into
cells at its section headers; markdown cells with the technique description,
assumptions and interpretations are inserted between them, and every figure a
cell saves is displayed right below it.

Run:  python src/make_notebook.py          (builds the notebook, not executed)
Then: jupyter nbconvert --to notebook --execute --inplace Hurricane_Analytics.ipynb
"""
import ast
import os
import re
import textwrap
import nbformat as nbf

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
OUT = os.path.join(ROOT, "Hurricane_Analytics.ipynb")

cells = []
md = lambda t: cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(t).strip()))
code = lambda t: cells.append(nbf.v4.new_code_cell(t.strip("\n")))


def split_script(name):
    """Return (docstring, [code chunks]) for src/<name>.py.
    The `if __name__ == "__main__":` block is dedented so its sections become cells."""
    src = open(os.path.join(SRC, name + ".py"), encoding="utf-8").read()
    tree = ast.parse(src)
    doc = ast.get_docstring(tree) or ""
    body = src
    if doc:                                              # drop the module docstring
        first = tree.body[0]
        lines = src.splitlines()
        body = "\n".join(lines[first.end_lineno:])
    lines = body.splitlines()
    out, in_main = [], False
    for ln in lines:
        if ln.startswith('if __name__ == "__main__":'):
            in_main = True
            continue
        out.append(ln[4:] if in_main and ln.startswith("    ") else ln)
    # split before a header comment block "# ====" / "# ----" at column 0
    chunks, cur = [], []
    for i, ln in enumerate(out):
        is_hdr = bool(re.match(r"^# ?(-{3,}|={4,})", ln))
        prev_comment = i > 0 and out[i - 1].startswith("#")
        if is_hdr and not prev_comment and any(c.strip() for c in cur):
            chunks.append("\n".join(cur)); cur = []
        cur.append(ln)
    if any(c.strip() for c in cur):
        chunks.append("\n".join(cur))
    # merge chunks that are not valid Python on their own
    merged, buf = [], ""
    for ch in chunks:
        buf = (buf + "\n" + ch) if buf else ch
        try:
            ast.parse(buf)
            merged.append(buf.strip("\n")); buf = ""
        except SyntaxError:
            pass
    if buf:
        merged.append(buf.strip("\n"))
    return doc, merged


def add_script(name, notes=None):
    """Add all cells of a script; after a chunk that saves figures, display them and
    then the interpretation from `notes[figure]` if one is given."""
    notes = notes or {}
    _, chunks = split_script(name)
    for ch in chunks:
        code(ch)
        figs = list(dict.fromkeys(re.findall(r"([A-Za-z0-9_]+\.png)", ch)))
        if figs:
            code("show(" + ", ".join(repr(f) for f in figs) + ")")
            for f in figs:
                if f in notes:
                    md(notes[f])


# ============================================================================
# Title + setup
# ============================================================================
md("""
# Descriptive & Predictive Analytics of the NOAA Hurricane Database (HURDAT2)
**25CS5004 / CS5010 Analytics & Systems of Big Data Practice — End-semester (Nov 2025)**

**Submitted by:** Umesh Chandra (Roll No. CS23I1036)

Dataset: <https://www.kaggle.com/noaa/hurricane-database> — `atlantic.csv` (1851–2015) and `pacific.csv` (1949–2015),
six-hourly position and intensity of every tropical cyclone.

This notebook runs the complete implementation, with outputs and interpretations. Every cell is taken from the
source files in `src/`, so the notebook and the scripts are the same code. The final PDF report
(`report/Hurricane_Analytics_Report.pdf`) is generated in the last section from the outputs of this run.

| Part | Content | Source file |
|---|---|---|
| 1 | Data preprocessing — (i) cleaning, (ii) selection, (iii) transformation | `01_preprocessing.py` |
| 2 | Descriptive statistics + 15 visualisations with inferences | `02_visualization.py` |
| 3 | Association rules with **DIC** | `03_dic.py` |
| 4 | **Decision Tree** classifier | `05_decision_tree.py` |
| 5 | **DBSCAN** clustering (+ dendrogram) | `06_dbscan.py` |
| 6 | Closed frequent itemsets / closed rules with **A-CLOSE** | `04_aclose.py` |
| 7 | PySpark test drive 1 — A-CLOSE on Spark | `07_pyspark_aclose.py` |
| 8 | PySpark test drive 2 — Decision Tree on Spark MLlib | `08_pyspark_decision_tree.py` |
| 9 | Report generation and conclusions | `09_build_report.py` |

**Assumed parameters:** min_support = 5 %, min_confidence = 70 %, DIC block size M = 5000; decision-tree
criterion/depth chosen by 5-fold GroupKFold CV with a storm-wise 70/30 split; DBSCAN with haversine distance,
MinPts = 15, eps from the k-distance knee.
""")
code("""
# ---- Setup: make src/ importable, run from the project root, helper to display saved figures ----
import os, sys, warnings
ROOT = os.path.abspath(".")
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
warnings.filterwarnings("ignore")
from IPython.display import Image, display

def show(*names, width=820):
    \"\"\"Display figures saved by the analysis code in outputs/figures.\"\"\"
    for n in names:
        display(Image(filename=os.path.join(ROOT, "outputs", "figures", n), width=width))

print("Python", sys.version.split()[0], "| project root:", ROOT)
""")

# ============================================================================
# 1. Preprocessing
# ============================================================================
md("""
## 1. Data preprocessing
The raw files contain several kinds of noise that must be handled before any analysis:
`-999`/`-99` sentinel values for missing data, coordinates stored as strings (`28.0N`, `94.8W`), white-space
padding, separate date and time integers, inconsistent status codes between the two basins, and records whose
status contradicts their own wind speed.

* **(i) Data cleaning** — steps 1–11 below (missing values, noise, inconsistencies, outliers).
* **(ii) Data selection** — keep only task-relevant attributes, justified by missing-value % and correlation.
* **(iii) Data transformation** — attribute construction, discretisation, concept hierarchy, normalisation,
  aggregation to storm level and transaction encoding for association-rule mining.
""")
add_script("01_preprocessing")
md("""
**Interpretation of preprocessing**
* 75,242 → **74,895** observations (0.46 % removed) and 2,864 → **2,845** storms. The cleaning removes only
  impossible or contradictory records and keeps genuine extremes (Patricia 2015: 185 kt / 872 hPa).
* Data issues found: the Pacific code **ST = Super Typhoon** (Genevieve 2014, 130–140 kt), which is mapped to HU
  (mapping it to "subtropical" would create impossible 140-kt subtropical storms); `359.1W` longitudes, which are
  typing errors and are repaired; 19 storms from 1967–68 with no wind value at all; 7 status/wind contradictions.
* The 12 wind-radii columns are 83.6 % missing (only recorded since 2004) and are dropped. Pressure is kept
  despite 57.6 % missing values because it is strongly linked to wind (r = −0.944) but physically independent of
  the Status definition.
""")

# ============================================================================
# 2. Visualisation
# ============================================================================
md("""
## 2. Descriptive statistics and visualisation (15 plot types)
Plots used: histogram, box plot, quantile plot, Q-Q plot, scatter plot, scatter-plot matrix, correlation
heat-map, grouped bar chart, line chart, pie chart, violin plot, geographic track map, parallel coordinates,
hexbin density plot and a 100 % stacked bar chart. Atlantic = blue, Pacific = orange throughout.
Each figure is followed by its key inferences.
""")
VIZ = {
    "fig01_histogram_wind.png": """
**Inferences – histogram.** Wind is right-skewed (skew 0.99, mode 30 kt): 70.8 % of all observations are below
hurricane force and only 6.3 % are major hurricanes (≥ 96 kt). 99.99 % of values are multiples of 5 kt (the
reporting resolution), so 5-kt bins are the natural bin width.""",
    "fig02_boxplot_status.png": """
**Inferences – box plot.** HU: median 80 kt, IQR 70–95 kt, outliers up to 185 kt. TS lies strictly within 35–60 kt
and TD ≤ 32 kt, which confirms that the cleaning removed every status/wind contradiction. Atlantic extratropical
systems are much stronger (median 45 kt) than Pacific ones (25 kt): Atlantic storms recurve into the westerlies,
while Pacific storms decay over cold water.""",
    "fig03_quantile_pressure.png": """
**Inferences – quantile plot.** Atlantic pressures are lower at every quantile (10 %: 965 vs 970 hPa; median: 999
vs 1002 hPa). The steep drop below f ≈ 0.05 shows how rare deep systems are: the 5th percentile is 952 hPa and only
4.1 % of readings are below 950 hPa.""",
    "fig04_qq_plot.png": """
**Inferences – Q-Q plots.** Left: Atlantic and Pacific quantiles coincide in the lower part (+1.3 kt) but the
Pacific is higher in the upper 20 % (+4.4 kt), so the most intense Pacific storms are slightly stronger. Right:
the convex curve (r = 0.973) shows that lifetime maximum wind is right-skewed, not normal.""",
    "fig05_scatter_wind_pressure.png": """
**Inferences – scatter plot.** There is a very strong linear inverse relation (r = −0.944): each 1 hPa drop adds
about 1.35 kt. This justifies using pressure as the physical predictor of Status in the decision tree. The spread
widens at low pressure, because the wind-pressure relation depends on storm size and latitude.""",
    "fig06_scatter_matrix.png": """
**Inferences – scatter-plot matrix.** MaxWind–MinPressure is a tight line and DurationH–TrackKm is strongly
positive. Long-lived storms reach higher winds, and Pacific storms form at lower genesis latitudes.""",
    "fig07_correlation_heatmap.png": """
**Inferences – heat-map.** MaxWind–MinPressure r = −0.95. MaxWind–Max24hIntensification r = 0.77: strong storms
get there by rapid intensification. DurationH–NumObs r = 1.00, so NumObs is redundant and is removed. Lower
genesis latitude goes with longer, stronger storms. Year shows no linear trend in storm peaks (r = −0.11).""",
    "fig08_bar_month.png": """
**Inferences – bar chart.** The Atlantic season peaks in September (601 storms), with 76.5 % forming in Aug–Oct.
The Pacific peaks in August (251) and starts earlier (May–July: 404 vs 314 Atlantic storms). Only 28 of 2,845
storms formed outside May–November.""",
    "fig09_line_yearly.png": """
**Inferences – line chart.** Recorded Atlantic activity doubled from 7.6 storms/yr (1851–1900) to 16.2 (1966–2015),
and the Pacific rose from 10.4 to 18.2. The jump around the satellite era shows that much of the rise is improved
detection. The busiest Atlantic year is 2005 (31 storms).""",
    "fig10_pie_status.png": """
**Inferences – pie chart.** TS 37.2 %, HU 28.4 %, TD 22.1 %, EX 6.8 %, LOW 4.4 %, SUB 1.2 %. EX and SUB are small
classes, so the classifier is also evaluated with macro-averaged metrics.""",
    "fig11_violin_duration.png": """
**Inferences – violin plot.** Median lifetime rises steadily with peak category: TD 2.2 days, TS 4.2, Cat1 6.0,
Cat3 9.2, Cat5 11.2. Reaching high intensity requires a long time over warm water.""",
    "fig12_track_map.png": """
**Inferences – track map.** Pacific storms move west at 10–20°N and die over cool water. Atlantic storms recurve
north-east and often become extratropical. Intense fixes are tropical: 90 % of major-hurricane fixes lie south of
30.3°N. Landfall: 32.9 % of Atlantic storms vs 6.4 % of Pacific storms.""",
    "fig13_parallel_coordinates.png": """
**Inferences – parallel coordinates.** Cat5 storms (median 918 hPa, 267 h, 5,665 km track) are separated from
tropical storms (998 hPa, 120 h, 1,874 km) on every axis, and they form closer to the equator (12°N vs 18°N).""",
    "fig14_hexbin_genesis.png": """
**Inferences – hexbin.** There are two dense nurseries: the eastern Pacific off Mexico (median 13.3°N, 106.9°W) and
the Atlantic (19.8°N, 67.5°W). Atlantic genesis is spread from Africa to the Gulf of Mexico, which motivates
density-based clustering (Part 5).""",
    "fig15_stacked_category.png": """
**Inferences – stacked bar.** The share of hurricanes that became major rose from 22.9 % (Atlantic 1851–1900) to
44.0 % (2001–2015), and from 27.5 % to 46.9 % in the Pacific. No Cat5 storm is recorded before 1924. Part of this
shift reflects better measurement (aircraft and satellites).""",
}
add_script("02_visualization", VIZ)

# ============================================================================
# 3. DIC
# ============================================================================
md("""
## 3. Association rules with the DIC algorithm
**DIC (Dynamic Itemset Counting, Brin et al. 1997)** reads the transactions in blocks of *M* and has a "stop" after
every block. Each itemset is a **dashed circle** (being counted, below support), **dashed box** (being counted,
already frequent), **solid circle** (fully counted, infrequent) or **solid box** (fully counted, frequent).
A new itemset starts being counted at the next stop as soon as all its immediate subsets are boxes, instead of
waiting for a full pass as Apriori does. Counting wraps around the file until each itemset has seen all N
transactions.

*Input*: 74,895 transactions (one per observation) over 27 attribute=value items. *Parameters*: min_support 5 %,
min_confidence 70 %, M = 5000. The result is **verified against mlxtend's Apriori**.
""")
add_script("03_dic")
md("""
**Interpretation (DIC)**
* DIC finds the same 566 frequent itemsets as Apriori (verified) in **3.94 passes instead of 6**. Starting
  longer itemsets mid-pass saves about two database scans.
* `{Status=EX} ⇒ {Lat≥35N}` (confidence 82 %, lift 6.1): extratropical transition happens poleward of 35°N,
  while the storm is carried east by the westerlies (`{Motion=Eastward, Status=EX} ⇒ {Lat≥35N}`, lift 6.4).
* `{Lat≥35N, Speed>30} ⇒ {Motion=Eastward}` (96 %): storms speed up after recurving.
* `{Pressure<980} ⇒ {Status=HU}` (94 %; 98 % for westward-moving storms): a deep central pressure almost always
  means a hurricane.
* `{Era=1990-2015, Status=TD} ⇒ {Pressure≥1000}` (87 %): modern depressions are measured, and they are weak lows.
""")

# ============================================================================
# 4. Decision tree
# ============================================================================
md("""
## 4. Decision Tree classifier — predicting the storm status
* **Target:** Status ∈ {TD, TS, HU, EX} of each observation.
* **Features:** Latitude, Longitude, Month, Basin, Pressure, 6-h pressure tendency, speed, heading, hours since
  genesis. **Maximum Wind is excluded**, because Status is defined from wind thresholds (that would be target
  leakage).
* **Validation:** consecutive fixes of one storm are nearly identical, so the train/test split is done **by storm**
  (GroupShuffleSplit 70/30), and depth/criterion are tuned with **5-fold GroupKFold** CV.
* **Outputs:** accuracy, precision/recall/F1, Cohen's kappa, confusion matrix, feature importance, tree diagram
  and **IF-THEN rules**.
""")
add_script("05_decision_tree")
md("""
**Interpretation (Decision Tree)**
* Test accuracy **0.810** on storms never seen in training (kappa 0.725, macro-F1 0.782), against a majority-class
  baseline of 0.405. The CV estimate (0.819) matches the test score, so the model generalises.
* HU is recognised best (F1 0.90) and EX is hardest (F1 0.66). Errors occur almost only between neighbouring
  intensity classes (TD↔TS, TS↔HU).
* Pressure carries 76 % of the importance and latitude 13 %. The rules read like forecaster heuristics, e.g.
  *IF Latitude ≤ 38.8 AND Pressure ≤ 979.5 THEN HU* (99 %), *IF Latitude > 50.1 AND Pressure ≤ 988.5 THEN EX* (100 %).
* Training accuracy keeps rising with depth while CV accuracy levels off after depth 8, so deeper trees over-fit.
""")

# ============================================================================
# 5. DBSCAN
# ============================================================================
md("""
## 5. DBSCAN clustering of storm genesis locations
Genesis regions have irregular shapes and an unknown number, and isolated genesis events should be treated as
noise. DBSCAN handles all of this; k-means does not. Each storm's first fix is clustered with the **haversine**
(great-circle) distance. MinPts = 15, and eps comes from the knee of the k-distance graph, taken on the main body
of the curve (the top 5 % outlier tail would push the knee to ≈430 km and merge the two oceans into one cluster).
A second, finer DBSCAN pass inside each cluster finds the genesis hot-spots, and a **dendrogram** (Ward linkage)
groups the hot-spots by their profiles.
""")
add_script("06_dbscan")
md("""
**Interpretation (DBSCAN)**
* eps = 230 km gives 2 clusters, 13.3 % noise and a silhouette of 0.447. Without being given any basin label,
  DBSCAN separates the **Atlantic (100 % Atlantic)** and **Pacific (0.4 % Atlantic)** nurseries: the low-density
  gap over Central America forms a natural boundary.
* Noise storms form further north (mean 22.3°N) and become hurricanes less often (41.7 %).
* The Cape Verde hot-spot **C0.4** (≈21.5°W) produces the longest-lived (282 h) and most intense (42.6 % major)
  storms, with few landfalls. The **Gulf of Mexico C0.0** produces short, weak storms, but 63 % make landfall.
* The dendrogram shows three families: the deep-tropical Atlantic main development region, the eastern Pacific,
  and near-land western Atlantic / Caribbean / Gulf genesis.
""")

# ============================================================================
# 6. A-CLOSE
# ============================================================================
md("""
## 6. Closed frequent itemsets and closed rules with A-CLOSE
**A-CLOSE (Pasquier et al. 1999)** mines *generators* level by level. A candidate is pruned if it is infrequent,
or if its support equals that of one of its subsets (then it lies in that subset's closure). One final pass
computes the closures h(g) = ∩{t : g ⊆ t} of the generators from level i−1 onwards. The closures form the set of
**closed frequent itemsets (CFIs)**. Rules: **exact** g ⇒ h(g)∖g (100 %) and **approximate** g ⇒ C∖g for closed
C ⊃ h(g). The result is verified against the closed sets derived from DIC's frequent itemsets.
""")
add_script("04_aclose")
md("""
**Interpretation (A-CLOSE)**
* There are 563 CFIs, identical to the brute-force closed sets (verified). Only 3 of the 566 frequent itemsets are
  not closed: the items rarely imply one another, so the data are **sparse** and closed itemsets compress little
  here. A-CLOSE pays off on dense, correlated data.
* The 3 exact rules say that every pre-1950 record that is late-season or north of 35°N is Atlantic, because the
  Pacific record only starts in 1949.
* The closed basis (292 rules) keeps the strongest DIC rules (EX ⇒ Lat≥35N, Pressure<980 ⇒ HU) while dropping the
  redundant ones.
""")

# ============================================================================
# 7-8. PySpark
# ============================================================================
md("""
## 7. PySpark test drive 1 — A-CLOSE on Spark
The A-CLOSE driver is the **same function** as in Part 6. Only the data access changes: the transactions become a
cached, partitioned Spark DataFrame of 64-bit item bit-masks. One distributed aggregation per level computes
`SUM((mask & c) == c)` (support) and `BIT_AND(mask)` (closure) for all candidates in parallel.
On this Windows machine Spark's Python workers cannot start (Microsoft-Store Python 3.13, WinError 10038), so
only JVM-side Spark SQL is used. To emulate growing data the transactions are replicated ×1 … ×32 (up to 2.4 M),
and every run must return the identical closed itemsets.

*Runtime: about 10 minutes (the sequential baseline at ×8 alone takes ~2 minutes).*
""")
add_script("07_pyspark_aclose")
md("""
**Interpretation (Spark A-CLOSE)** — see the timing table above.
* On the original 75 k transactions Spark is *slower* than plain Python, because every level launches distributed
  jobs with a fixed scheduling and code-generation cost.
* As the data grow, the sequential time grows linearly while Spark's grows much more slowly. Spark becomes
  several times faster at 600 k transactions and handles 2.4 M transactions in about a minute, with
  **identical closed itemsets** in every run.
* At ×8 data the run is still dominated by per-job overhead, so adding cores gives only a modest, well-below-linear
  gain. Much of the gain there comes from executing the counting in Spark's compiled JVM engine.
""")
md("""
## 8. PySpark test drive 2 — Decision Tree on Spark MLlib
Same task, features, storm-wise split and hyper-parameters as Part 4 (gini, depth 8, min 20 per leaf).
1. Model quality: Spark MLlib vs scikit-learn on the same test storms.
2. Scalability: the training set is replicated ×1 … ×100 (17.6 k → 1.76 M rows).
3. Core scaling with 1, 2, 4 and 8 cores.

*Runtime: about 6 minutes.*
""")
add_script("08_pyspark_decision_tree")
md("""
**Interpretation (Spark decision tree)** — see the benchmark above.
* MLlib reaches the same accuracy as scikit-learn (≈0.81–0.82). Binning continuous features (maxBins = 64) costs
  nothing here.
* For small data scikit-learn (optimised C, in memory) is much faster, because Spark pays a fixed job overhead.
  The gap closes as data grow, and at ~1.76 M rows Spark is faster.
* Spark training time drops when cores are added, and on a cluster the same code scales to data that do not fit
  on one machine.
""")

# ============================================================================
# 9. Report + conclusions
# ============================================================================
md("""
## 9. Generate the PDF report
The report reads every number from the outputs written by the cells above (`outputs/logs/*.json`,
`outputs/data/*.csv`), so it always matches this notebook run.
""")
add_script("09_build_report")
md("""
## 10. Conclusions
* **Preprocessing** turned noisy raw data (sentinel codes, string coordinates, inconsistent and contradictory
  records) into a clean dataset while losing only 0.46 % of rows, and uncovered genuine data errors.
* **Descriptive analytics**: intensity is right-skewed, the wind–pressure law is tight (r = −0.944), the seasons
  peak in September (Atlantic) and August (Pacific), detection was biased before the satellite era, and intense
  storms are long-lived and form near the equator.
* **DIC** matched Apriori exactly with about 2 fewer scans, and its rules capture the storm life cycle.
* **Decision tree**: 81 % accuracy on unseen storms from pressure and position alone, with readable IF-THEN rules.
* **DBSCAN** separated the two basins without labels and found genesis hot-spots with very different risk
  profiles (Cape Verde: long and intense; Gulf of Mexico: short but landfalling).
* **A-CLOSE** gave a verified, lossless condensed representation and a non-redundant rule basis.
* **PySpark**: Spark carries a fixed overhead on small data, but its advantage grows with volume for both
  A-CLOSE and the MLlib decision tree, without any change in results. That is the performance gain from
  big-data support.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, OUT)
n_code = sum(c.cell_type == "code" for c in cells)
print(f"Notebook written: {OUT}  ({len(cells)} cells, {n_code} code)")
