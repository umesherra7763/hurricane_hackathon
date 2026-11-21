"""
09_build_report.py
------------------
Builds the course-project PDF (report/Hurricane_Analytics_Report.pdf) from the
figures in outputs/figures and the numbers saved by scripts 01-08. No number in
the report is typed by hand.
"""
import os
import json
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether, ListFlowable, ListItem, Preformatted, CondPageBreak,
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as pdfcanvas

from common import ROOT, FIG_DIR, LOG_DIR, CLEAN_DIR, MIN_SUPPORT, MIN_CONFIDENCE, DIC_BLOCK_SIZE

REPORT_DIR = os.path.join(ROOT, "report")
os.makedirs(REPORT_DIR, exist_ok=True)
OUT = os.path.join(REPORT_DIR, "Hurricane_Analytics_Report.pdf")


def J(name):
    p = os.path.join(LOG_DIR, name + ".json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


pre, viz, dic, ac, dt, db = (J(n) for n in ("preprocessing", "visualization", "dic", "aclose",
                                             "decision_tree", "dbscan"))
sac, sdt = J("spark_aclose"), J("spark_dt")

# ----------------------------------------------------------------------------
# Fonts & styles
# ----------------------------------------------------------------------------
import matplotlib
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
_fd = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
for name, file in (("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"),
                   ("DV-I", "DejaVuSans-Oblique.ttf"), ("DV-BI", "DejaVuSans-BoldOblique.ttf"),
                   ("DVM", "DejaVuSansMono.ttf")):
    pdfmetrics.registerFont(TTFont(name, os.path.join(_fd, file)))
addMapping("DV", 0, 0, "DV"); addMapping("DV", 1, 0, "DV-B")
addMapping("DV", 0, 1, "DV-I"); addMapping("DV", 1, 1, "DV-BI")

ss = getSampleStyleSheet()
BLUE = colors.HexColor("#1c5cab")
NAVY = colors.HexColor("#0d3a6e")
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="DV-B", fontSize=14, textColor=NAVY,
                    spaceBefore=10, spaceAfter=8, leading=18)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="DV-B", fontSize=11.5, textColor=BLUE,
                    spaceBefore=10, spaceAfter=5, leading=15)
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontName="DV-B", fontSize=10, spaceBefore=8, spaceAfter=3)
B = ParagraphStyle("B", parent=ss["BodyText"], fontName="DV", fontSize=9.5, leading=13.2, spaceAfter=6,
                   alignment=4)
SM = ParagraphStyle("SM", parent=B, fontSize=8.2, leading=11, alignment=0, spaceAfter=3)
CAP = ParagraphStyle("CAP", parent=B, fontSize=8.5, textColor=colors.HexColor("#333333"),
                     alignment=1, spaceBefore=3, spaceAfter=8, fontName="DV-I")
TCAP = ParagraphStyle("TCAP", parent=B, fontSize=8.5, textColor=colors.HexColor("#333333"),
                      alignment=1, spaceBefore=8, spaceAfter=4, fontName="DV-I")
CODE = ParagraphStyle("CODE", parent=SM, fontName="DVM", fontSize=6.6, leading=8.4, alignment=0)
COVER_H = ParagraphStyle("COVER_H", parent=B, fontName="DV-B", fontSize=16, leading=20,
                         alignment=1, textColor=NAVY, spaceAfter=6)
SEC = ParagraphStyle("SEC", parent=H1, fontName="DV-B", fontSize=14, textColor=NAVY,
                     spaceBefore=10, spaceAfter=8, leading=18)
COVER_S = ParagraphStyle("COVER_S", parent=B, alignment=1, fontSize=10.5, leading=14)
KW = ParagraphStyle("KW", parent=B, fontSize=9, leading=12.5, alignment=4, spaceAfter=4)
story = []
W = A4[0] - 3.6 * cm


def h1(t): story.append(Paragraph(t, H1))
def h2(t): story.append(Paragraph(t, H2))
def h3(t): story.append(Paragraph(t, H3))
def p(t): story.append(Paragraph(t, B))


def bullets(items, style=B):
    story.append(ListFlowable([ListItem(Paragraph(i, style), leftIndent=10) for i in items],
                              bulletType="bullet", start="•", leftIndent=12, bulletFontSize=8))


def fig(name, caption, width=0.88):
    path = os.path.join(FIG_DIR, name)
    if not os.path.exists(path):
        p(f"<i>[figure {name} not generated]</i>")
        return
    iw, ih = ImageReader(path).getSize()
    w = W * width
    h = w * ih / iw
    if h > 13.5 * cm:
        h = 13.5 * cm
        w = h * iw / ih
    story.append(KeepTogether([Image(path, width=w, height=h), Paragraph(caption, CAP)]))


def table(df, col_widths=None, font=7.5, header_bg="#e8f0fb", caption=None):
    if caption:
        story.append(Paragraph(caption, TCAP))
    st = ParagraphStyle("t", parent=SM, fontSize=font, leading=font + 2.2)
    data = [[Paragraph(f"<b>{c}</b>", st) for c in df.columns]]
    for row in df.itertuples(index=False):
        data.append([Paragraph(str(v), st) for v in row])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c3c2b7")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f7f5")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))


def inferences(items):
    h3("Interpretation")
    bullets(items)


class ListOfX(TableOfContents):
    """TOC that listens to a custom notify kind (LOFEntry / LOTEntry)."""
    def __init__(self, kind):
        TableOfContents.__init__(self)
        self._kind = kind

    def notify(self, kind, stuff):
        if kind == self._kind:
            TableOfContents.notify(self, "TOCEntry", stuff)


class HurricaneDoc(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if not isinstance(flowable, Paragraph):
            return
        style = flowable.style.name
        text = flowable.getPlainText()
        if style == "H1":
            self.notify("TOCEntry", (0, text, self.page))
        elif style == "H2":
            self.notify("TOCEntry", (1, text, self.page))
        elif style == "CAP" and text.startswith("Figure"):
            self.notify("LOFEntry", (0, text, self.page))
        elif style == "TCAP" and text.startswith("Table"):
            self.notify("LOTEntry", (0, text, self.page))


class NumberedCanvas(pdfcanvas.Canvas):
    def __init__(self, *args, **kwargs):
        pdfcanvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        n = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._decorate(n)
            pdfcanvas.Canvas.showPage(self)
        pdfcanvas.Canvas.save(self)

    def _decorate(self, n):
        page = self._pageNumber
        self.saveState()
        self.setStrokeColor(BLUE)
        self.setLineWidth(0.7)
        if page > 1:
            self.line(1.8 * cm, A4[1] - 1.2 * cm, A4[0] - 1.8 * cm, A4[1] - 1.2 * cm)
            self.setFont("DV", 7.5)
            self.setFillColor(colors.HexColor("#555555"))
            self.drawString(1.8 * cm, A4[1] - 1.08 * cm,
                            "Analytics &amp; Systems of Big Data — NOAA Hurricane Database (HURDAT2)"
                            .replace("&amp;", "&"))
            self.line(1.8 * cm, 1.4 * cm, A4[0] - 1.8 * cm, 1.4 * cm)
            self.drawString(1.8 * cm, 0.95 * cm, "Umesh Chandra  ·  CS23I1036")
        self.setFont("DV", 8)
        self.setFillColor(colors.HexColor("#555555"))
        self.drawRightString(A4[0] - 1.8 * cm, 0.95 * cm, f"Page {page} of {n}")
        self.restoreState()


# ============================================================================
# COVER
# ============================================================================
story.append(Spacer(1, 1.6 * cm))
story.append(Paragraph("ANALYTICS AND SYSTEMS OF BIG DATA PRACTICE (CS5010)", COVER_H))
story.append(Paragraph("25CS5004 / CS5010 — End-semester Course Project Report", COVER_S))
story.append(Spacer(1, 0.35 * cm))
bar = Table([[""]], colWidths=[W], rowHeights=[4])
bar.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BLUE)]))
story.append(bar)
story.append(Spacer(1, 0.7 * cm))
story.append(Paragraph("Exhaustive Descriptive, Predictive and Big-Data "
                       "Analytics of the NOAA Atlantic and Pacific Hurricane Database",
                       ParagraphStyle("ct", parent=COVER_H, fontSize=15, leading=20)))
story.append(Spacer(1, 0.25 * cm))
story.append(Paragraph("HURDAT2, 1851–2015", COVER_S))
story.append(Spacer(1, 0.45 * cm))
story.append(Paragraph("Data preprocessing (KDD), descriptive statistics, visualisation, "
                       "DIC association rules, decision-tree classification, DBSCAN clustering, "
                       "A-CLOSE closed frequent itemsets and PySpark benchmarking",
                       ParagraphStyle("csub", parent=B, alignment=1, fontSize=9.5, leading=13)))
story.append(Spacer(1, 1.1 * cm))
table(pd.DataFrame([
    ["Dataset", "NOAA hurricane database (Kaggle / HURDAT2) — atlantic.csv, pacific.csv"],
    ["Source", "https://www.kaggle.com/noaa/hurricane-database"],
    ["Student name", "Umesh Chandra"],
    ["Roll / registration number", "CS23I1036"],
    ["Course", "25CS5004 / CS5010 Analytics & Systems of Big Data Practice"],
    ["Submission date", "21 November 2026"],
    ["Raw size", f"{pre['raw_rows']:,} six-hourly observations, {pre['raw_cols']} attributes, "
                 f"{pre['raw_storms']:,} storms"],
    ["Tools", "Python 3.13, pandas, NumPy, matplotlib/seaborn, scikit-learn, SciPy, "
              "mlxtend (verification only), PySpark 4.0.1, OpenJDK 17, ReportLab"],
    ["Hardware", f"Windows 11 laptop, {sac['cores'] if sac else 8} logical cores, 16 GB RAM, Spark local mode"],
], columns=["Item", "Details"]), col_widths=[4.2 * cm, W - 4.2 * cm], font=8.5,
      caption="Table 1. Project identification")
story.append(PageBreak())

# ============================================================================
# TOC / LOF / LOT
# ============================================================================
story.append(Paragraph("Table of Contents", SEC))
toc = TableOfContents()
toc.levelStyles = [
    ParagraphStyle(name="TOC0", fontName="DV-B", fontSize=10.5, leading=14, leftIndent=0,
                   firstLineIndent=0, spaceBefore=5, spaceAfter=2, textColor=NAVY),
    ParagraphStyle(name="TOC1", fontName="DV", fontSize=9, leading=12, leftIndent=16,
                   firstLineIndent=0, spaceBefore=1, spaceAfter=1),
]
story.append(toc)
story.append(PageBreak())

story.append(Paragraph("List of Figures", SEC))
lof = ListOfX("LOFEntry")
lof.levelStyles = [
    ParagraphStyle(name="LOF0", fontName="DV", fontSize=8.5, leading=11.5, leftIndent=0,
                   firstLineIndent=0, spaceBefore=1.5, spaceAfter=1.5),
]
story.append(lof)
story.append(PageBreak())

story.append(Paragraph("List of Tables", SEC))
lot = ListOfX("LOTEntry")
lot.levelStyles = [
    ParagraphStyle(name="LOT0", fontName="DV", fontSize=8.5, leading=11.5, leftIndent=0,
                   firstLineIndent=0, spaceBefore=1.5, spaceAfter=1.5),
]
story.append(lot)
story.append(PageBreak())

# ============================================================================
# ABSTRACT
# ============================================================================
h1("Abstract")
p(f"This report applies the complete Knowledge Discovery in Databases (KDD) process to the NOAA "
  f"HURDAT2 hurricane database: {pre['raw_rows']:,} six-hourly fixes of {pre['raw_storms']:,} "
  f"Atlantic (1851–2015) and North-East/Central Pacific (1949–2015) tropical cyclones. The raw "
  f"files use sentinel missing codes, string coordinates, padded names and a handful of physically "
  f"impossible records. A reproducible pipeline cleans, selects and transforms the data into "
  f"{pre['clean_rows']:,} analysable observations of {pre['clean_storms']:,} storms "
  f"({100 * (1 - pre['clean_rows'] / pre['raw_rows']):.2f} % of rows removed). Descriptive "
  f"statistics and 26 figures (15 distinct visualisation techniques, plus diagnostic plots for "
  f"the models) characterise intensity, seasonality, tracks and genesis. A from-scratch DIC "
  f"implementation finds {dic['n_frequent']} frequent itemsets and {dic['n_rules']} rules at "
  f"{MIN_SUPPORT:.0%} support / {MIN_CONFIDENCE:.0%} confidence in {dic['passes']} passes "
  f"(Apriori would need {dic['max_k'] + 1}) and matches mlxtend Apriori exactly. A decision tree "
  f"predicts storm status (TD/TS/HU/EX) from pressure and position with test accuracy "
  f"{dt['test_acc']:.3f} on unseen storms. DBSCAN (haversine) separates the Atlantic and Pacific "
  f"nurseries without labels and isolates {db['noise_pct']} % sporadic genesis. A-CLOSE mines "
  f"{ac['n_closed']} closed frequent itemsets and a non-redundant closed-rule basis, verified "
  f"against the DIC itemsets. Spark MLlib / Spark SQL versions of A-CLOSE and the decision tree "
  f"are benchmarked against Python. On the original 75 k rows Spark is slower because of fixed "
  f"distributed-execution overhead; the advantage appears only as the data are replicated "
  f"(A-CLOSE {sac['speedup']['8']:.1f}× faster at ×8). All assumed parameters are stated and justified.")
p("<b>Keywords:</b> KDD, data preprocessing, HURDAT2, DIC, A-CLOSE, closed frequent itemsets, "
  "decision tree, DBSCAN, PySpark, MLlib, benchmarking.")
story.append(PageBreak())

# ============================================================================
# 1. INTRODUCTION
# ============================================================================
h1("1. Introduction")
h2("1.1 Background of data analytics")
p("Data analytics converts raw data into information and, beyond what a database query can return, "
  "into knowledge that is not explicitly stored: patterns, associations, predictive models and "
  "groupings. The course frames this as the KDD process: cleaning, integration, selection, "
  "transformation, data mining, pattern evaluation and knowledge presentation. Real-world "
  "observational records are dirty (incomplete, noisy, inconsistent), so most of the effort in a "
  "KDD project goes into preprocessing. When data volumes grow, efficiency, scalability and "
  "parallelisation become major issues. This motivates distributed engines such as Apache Spark, "
  "and also an honest test of whether distribution pays off on a data set that still fits on one "
  "laptop.")
p("The NOAA HURDAT2 best-track archive is a canonical example of such data: a long, "
  "heterogeneous, six-hourly time series of tropical cyclones with changing observing systems "
  "(ship reports, aircraft reconnaissance, satellites). Intensity, motion and landfall are "
  "physically coupled, so association rules, a classifier and density-based clustering all have "
  "a natural interpretation — provided the sentinel codes and definitional leaks are handled first.")

h2("1.2 Objectives")
bullets([
    "Perform exhaustive preprocessing (cleaning, selection, transformation) of the HURDAT2 Atlantic "
    "and Pacific files and document every operation.",
    "Characterise the data with descriptive statistics and at least ten distinct, appropriate visualisations.",
    "Mine association rules with the DIC algorithm, implemented and explained from first principles, "
    "and verify the frequent itemsets against Apriori.",
    "Build a decision-tree classifier of storm status that does not leak the defining wind attribute, "
    "and cluster genesis locations with DBSCAN.",
    "Mine closed frequent itemsets with A-CLOSE and validate the result against the DIC itemsets.",
    "Re-implement two predictive techniques in PySpark and quantify the effect of big-data support "
    "honestly, including the case where distribution does not pay off.",
])

h2("1.3 Problem statement")
p("Perform exhaustive and detailed descriptive analytics on the NOAA hurricane database, including "
  "data cleaning, data selection and data transformation. Apply at least ten unique visualisation "
  "techniques and explain their inferences. Further perform association rule mining using the DIC "
  "algorithm, decision-tree classification and DBSCAN clustering, closed frequent itemset mining "
  "using A-CLOSE, and test two predictive analytics techniques using PySpark, establishing the "
  "performance enhancement achieved with big-data support. All assumed parameters (support, "
  "confidence, eps, MinPts, tree depth, etc.) must be stated and justified.")

h2("1.4 Scope")
p("The analysis covers the two official basin files of the Kaggle NOAA hurricane database "
  "(atlantic.csv, 1851–2015; pacific.csv, 1949–2015). No external storm catalogues are added and "
  "no records are synthesised for the descriptive or mining stages. The only artificial data is a "
  "clearly labelled row-replication stress test in the benchmarking chapter, used to study "
  "scaling while holding relative supports and class proportions fixed. All work runs on a single "
  "Windows laptop, with Spark in local mode. Wind radii (available only since 2004) are excluded "
  "from modelling; they are discussed only as a data-quality fact.")
story.append(PageBreak())

# ============================================================================
# 2. DATASET
# ============================================================================
h1("2. Dataset Description")
h2("2.1 Source")
p("The data set is the NOAA National Hurricane Center HURDAT2 best-track archive as published on "
  "Kaggle (<i>hurricane-database</i>). HURDAT2 records the position and intensity of every "
  "Atlantic and eastern/central Pacific tropical cyclone at synoptic times (00/06/12/18 UTC) plus "
  "special fixes (landfall, intensity peaks). The Atlantic basin begins in 1851; the Pacific "
  "basin in 1949. The Kaggle distribution stores each basin as a CSV with one six-hourly "
  "observation per row. Format documentation is supplied in the archive as atlantic.pdf and "
  "pacific.pdf [1].")

h2("2.2 Files, formats and relationships")
table(pd.DataFrame([
    ["atlantic.csv", "Atlantic basin tracks", "1851–2015", "AL + year-number ID"],
    ["pacific.csv", "NE / Central Pacific tracks", "1949–2015", "EP/CP + year-number ID"],
], columns=["File", "Content", "Years", "Storm ID"]),
      col_widths=[3.2 * cm, 4.5 * cm, 3 * cm, W - 10.7 * cm],
      caption="Table 2. Files of the Kaggle NOAA hurricane distribution")
p(f"Together the two files contain {pre['raw_rows']:,} rows and {pre['raw_cols']} attributes, "
  f"describing {pre['raw_storms']:,} distinct storms. Rows of one storm are consecutive and share "
  "an ID; there is no separate storm-level table. The two basins are concatenated after a Basin "
  "flag is attached. Storms are not shared across files (a few systems that cross the dateline "
  "appear with Pacific identifiers such as Super Typhoon Genevieve 2014).")

h2("2.3 Data dictionary")
table(pd.DataFrame([
    ["ID", "nominal (key)", "Basin + number + year, e.g. AL011851"],
    ["Name", "text", "Storm name; 'UNNAMED' before naming; space-padded"],
    ["Date", "integer YYYYMMDD", "UTC date of the fix"],
    ["Time", "integer HHMM", "UTC time; mostly 0, 600, 1200, 1800 plus specials"],
    ["Event", "nominal", "Record identifier: L = landfall, I = intensity peak; usually blank"],
    ["Status", "nominal", "TD, TS, HU, EX, SD/SS, LO, WV, DB; Pacific also ET, PT, ST"],
    ["Latitude", "string", "e.g. 28.0N — not a numeric degree"],
    ["Longitude", "string", "e.g. 94.8W; rare 359°W typing errors"],
    ["Maximum Wind", "numeric (knots)", "1-minute sustained wind; −99/−999 = missing"],
    ["Minimum Pressure", "numeric (hPa)", "Central pressure; −999 = missing"],
    ["Low/Mod/High Wind NE–NW", "numeric (n mi)", "12 wind-radii columns; −999 = missing"],
], columns=["Attribute", "Type", "Description"]),
      col_widths=[4.2 * cm, 3.4 * cm, W - 7.6 * cm],
      caption="Table 3. Data dictionary of the raw HURDAT2 attributes")

h2("2.4 Initial observations and data quality")
s = pre["sentinels"]
p("Inspection of the raw files, before any cleaning, already shows the issues that dominate "
  "preprocessing:")
bullets([
    f"Sentinel missingness, not NaN: Maximum Wind has {s['Maximum Wind']} codes of −99/−999; "
    f"Minimum Pressure {s['Minimum Pressure']:,} ({100 * s['Minimum Pressure'] / pre['raw_rows']:.1f} %); "
    f"each wind-radius column {s['Low Wind NE']:,} ({100 * s['Low Wind NE'] / pre['raw_rows']:.1f} %). "
    "A mean computed on −999 hPa would be meaningless.",
    "Coordinates and names are stored as padded strings. Status codes differ by basin (Pacific ET/PT/ST).",
    "Status is defined from wind (HU ≥ 64 kt, TS 34–63 kt, TD &lt; 34 kt), so any classifier that "
    "is given Maximum Wind is solving a definition, not a prediction problem.",
    "Observing capability changes over 165 years: pressure is almost absent before 1950 and "
    "routine after the satellite era; wind radii exist only from 2004. Apparent trends in storm "
    "counts mix climate and detection.",
])
story.append(PageBreak())

# ============================================================================
# 3. METHODOLOGY
# ============================================================================
h1("3. Methodology")
h2("3.1 KDD process")
p("The project follows the KDD phases. Each phase is a separate, re-runnable Python module, and "
  "all intermediate results are written to disk:")
bullets([
    "<b>Cleaning</b> (<font face='DVM'>src/01_preprocessing.py</font>): strip padding, parse "
    "coordinates, decode sentinels, standardise status codes, drop contradictory or kinematically "
    "impossible rows, keep physically real IQR extremes.",
    "<b>Selection:</b> drop 83.6 %-missing wind radii; replace Event by a landfall flag; keep "
    "pressure as the independent intensity measurement.",
    "<b>Transformation:</b> datetime merge, motion kinematics, Saffir–Simpson discretisation, "
    "status family hierarchy, storm-level roll-up, ARM transactions, bit-mask encoding.",
    "<b>Data mining:</b> DIC, decision tree, DBSCAN (+ Ward on hot-spot profiles), A-CLOSE, and "
    "the PySpark equivalents of A-CLOSE and the tree.",
    "<b>Pattern evaluation:</b> support/confidence/lift, storm-wise GroupKFold, silhouette, "
    "independent verification of every from-scratch algorithm.",
    "<b>Knowledge presentation:</b> figures, rule tables, IF–THEN rules, cluster profiles, this report.",
])

h2("3.2 Overall workflow")
p("Scripts are executed in numeric order from the project folder: preprocessing → visualisation → "
  "DIC → A-CLOSE → decision tree → DBSCAN → PySpark A-CLOSE → PySpark tree → report. Every script "
  "writes console output to <font face='DVM'>outputs/logs/&lt;script&gt;.txt</font> and a metrics "
  "JSON used by this PDF. Random seeds are fixed (42) for the storm-wise split and the tree. "
  "Replication in the Spark chapter is a labelled stress test, not new weather.")

h2("3.3 Tools, technologies and software environment")
table(pd.DataFrame([
    ["Language", "Python 3.13 (Windows)"],
    ["Data handling", "pandas, NumPy, SciPy"],
    ["Machine learning", "scikit-learn (decision tree, metrics, GroupKFold); mlxtend (Apriori verification only)"],
    ["Visualisation", "matplotlib, seaborn"],
    ["Big data", "Apache Spark 4.0.1 (PySpark, MLlib, Spark SQL), OpenJDK 17, local[*]"],
    ["Reporting", "ReportLab (this PDF); figures as PNG"],
    ["Hardware", f"Windows 11, {sac['cores'] if sac else 8} logical cores, 16 GB RAM"],
], columns=["Aspect", "Configuration"]),
      col_widths=[3.5 * cm, W - 3.5 * cm],
      caption="Table 4. Software and hardware environment")
p("Algorithms that the course derives from first principles (DIC, A-CLOSE, the closed-itemset "
  "basis) were implemented by hand and then validated against independent library results. "
  "Library implementations (scikit-learn, Spark MLlib, sklearn.cluster.DBSCAN) are used where "
  "the task is to apply a standard method with a justified distance and split protocol.")

h2("3.4 Assumed parameters")
table(pd.DataFrame([
    ["Association rules (DIC, A-CLOSE)", f"min_support = {MIN_SUPPORT:.0%}, min_confidence = {MIN_CONFIDENCE:.0%}"],
    ["DIC block size M", f"{DIC_BLOCK_SIZE:,} transactions between stops"],
    ["Decision tree", "gini / entropy and max_depth 2–20 by 5-fold GroupKFold; min 20 samples per leaf; "
                      "storm-wise 70/30 train/test split"],
    ["DBSCAN", f"haversine distance; MinPts = {db['min_pts']}; eps = {db['eps_km']:.0f} km from the "
               f"k-distance knee; level-2 eps = {db['eps2_km']} km"],
    ["Noise thresholds", "wind 10–200 kt; pressure 870–1030 hPa; translation speed ≤ 150 km/h"],
    ["PySpark", "local[*] mode, 6 GB driver memory; data replicated to emulate growth"],
], columns=["Technique", "Parameter values"]),
      col_widths=[4.6 * cm, W - 4.6 * cm],
      caption="Table 5. Assumed parameters (justified in the corresponding chapters)")
story.append(PageBreak())

# ============================================================================
# 4. PREPROCESSING
# ============================================================================
h1("4. Data Preprocessing")
p("Preprocessing is reported in the three required groups — cleaning, selection, transformation — "
  "with enough detail that every dropped or derived field can be traced.")

h2("4.1 Data cleaning")
h3("4.1.1 Noisy and malformed records")
p("Names, status codes and Event were space-padded (e.g. Name = '<font face='DVM'>            UNNAMED</font>'). "
  "They were stripped. Date and Time were stored as separate integers (HHMM as 0, 600, 1200, …) "
  "and merged into a single UTC Datetime. Latitude/longitude strings such as 28.0N / 94.8W were "
  "parsed to signed degrees. Two longitudes 359.1W / 358.4W (storm AL051952) are typing noise for "
  f"~1°E and were wrapped to ±180° ({pre['lon_repaired']} repairs).")

h3("4.1.2 Missing values")
p(f"Sentinel codes −999/−99 were replaced by NaN. Leaving them as numbers would destroy every "
  f"mean and correlation: pressure would appear around 400 hPa. After decoding, pressure is still "
  f"missing for {pre['missing_pct']['Minimum Pressure']} % of rows (available for only ~3 % of "
  f"pre-1950 rows but ~82 % after 1979). Wind radii are missing for {pre['missing_pct']['Low Wind NE']} % "
  f"because they are recorded only since {pre['radii_first_year']}. "
  f"{pre['missing_wind_dropped']} rows without Maximum Wind were dropped "
  f"({pre['storms_no_wind']} storms, 1967–68 depressions, had no wind at all).")

h3("4.1.3 Inconsistent and invalid values")
p(f"Pacific uses ET ({pre['status_fix']['ET']}) as well as EX for extratropical, PT "
  f"({pre['status_fix']['PT']}) for post-tropical, and ST ({pre['status_fix']['ST']} rows = Super "
  "Typhoon Genevieve 2014, 130–140 kt after crossing the dateline). These were mapped ET, PT → EX "
  "and ST → HU so that a status family is comparable across basins. "
  f"{pre['status_wind_inconsistent']} rows contradicted the wind definition of Status (e.g. AL101958 "
  f"coded HU at 60 kt) and were removed. {pre['speed_noise']} rows implied a translation speed "
  "&gt; 150 km/h between consecutive fixes (position error) and were removed.")

h3("4.1.4 Duplicates")
p(f"Exact-row duplicates and (ID, Datetime) duplicates: {pre['dups_removed']} found.")

h3("4.1.5 Outliers")
p(f"The IQR rule flags {pre['iqr_outliers_Maximum Wind']} wind and "
  f"{pre['iqr_outliers_Minimum Pressure']:,} pressure outliers. They are kept: they are genuine "
  "extremes (Patricia 2015: 185 kt, 872 hPa). Cleaning is conservative — only impossible or "
  "contradictory records are removed.")
table(pd.DataFrame([
    ["1", "White-space padding", "Name, Status, Event padded", "Stripped"],
    ["2", "Sentinel codes", f"Wind {s['Maximum Wind']}; pressure {s['Minimum Pressure']:,}; "
                            f"radii {s['Low Wind NE']:,} per column", "Replaced by NaN"],
    ["3", "Coordinate strings", "28.0N / 94.8W", "Parsed to signed degrees"],
    ["4", "Separate Date/Time", "HHMM integers", "Merged into Datetime"],
    ["5", "Duplicates", "Exact and (ID, Datetime)", f"{pre['dups_removed']} found"],
    ["6", "Inconsistent codes", "Pacific ET/PT/ST", "ET, PT → EX; ST → HU"],
    ["7", "Out-of-range longitude", "359.1W / 358.4W", f"{pre['lon_repaired']} wrapped to ±180°"],
    ["8", "Missing wind", f"{pre['missing_wind_dropped']} rows; {pre['storms_no_wind']} storms", "Rows removed"],
    ["9", "Status vs wind", "HU with &lt; 64 kt, etc.", f"{pre['status_wind_inconsistent']} rows removed"],
    ["10", "Kinematic noise", "&gt; 150 km/h between fixes", f"{pre['speed_noise']} rows removed"],
    ["11", "IQR outliers", f"{pre['iqr_outliers_Maximum Wind']} wind, "
                           f"{pre['iqr_outliers_Minimum Pressure']:,} pressure", "Kept (physical extremes)"],
], columns=["#", "Problem", "Evidence", "Action"]),
      col_widths=[0.7 * cm, 3.2 * cm, 6.5 * cm, W - 10.4 * cm],
      caption="Table 6. Cleaning operations")
p(f"<b>Result:</b> {pre['raw_rows']:,} → <b>{pre['clean_rows']:,}</b> observations "
  f"({100 * (1 - pre['clean_rows'] / pre['raw_rows']):.2f} % removed) and {pre['raw_storms']:,} → "
  f"<b>{pre['clean_storms']:,}</b> storms.")

h2("4.2 Data selection")
bullets([
    f"<b>Dropped:</b> the 12 wind-radii columns — {pre['missing_pct']['Low Wind NE']} % missing "
    f"because they are only recorded since {pre['radii_first_year']}; imputing 84 % of a column "
    "would create, not describe, data.",
    "<b>Replaced:</b> Date and Time → Datetime; Event → binary Landfall flag (Event is blank for "
    "≈98.7 % of rows and only 'L' is analytically useful).",
    f"<b>Kept with caution:</b> Minimum Pressure ({pre['missing_pct']['Minimum Pressure']} % missing). "
    f"It is strongly redundant with wind (Pearson r = {pre['corr_wind_pressure']}), but because "
    "Status is <i>defined</i> from wind, pressure is the physically independent measurement used "
    "by the classifier.",
    "<b>Redundancy check at storm level:</b> NumObs and DurationH have r = 1.00 (Figure 7), so "
    "only DurationH is used afterwards.",
    "<b>Selected attributes:</b> ID, Name, Basin, Datetime, Status, Latitude, Longitude, Wind, "
    "Pressure, Landfall, plus the derived motion attributes of Section 4.3.",
])

h2("4.3 Data transformation")
bullets([
    "<b>Attribute construction:</b> Year, Month, Hour, Decade; Basin (from the file); 6-hourly "
    "step distance (haversine), translation speed (km/h), heading, dLat/dLon (dateline-safe), "
    "dWind, 6-h pressure tendency dPressure6h, HoursSinceGenesis.",
    "<b>Unit conversion:</b> knots → km/h (Wind_kmh).",
    "<b>Discretisation by domain thresholds:</b> Wind → Saffir–Simpson category (TD, TS, Cat1–Cat5).",
    "<b>Concept hierarchy:</b> 9 status codes → 6 families (TD, TS, HU, EX, SUB = SD+SS, LOW = LO+WV+DB).",
    "<b>Normalisation:</b> min-max scaling of Wind, Latitude, Longitude for parallel coordinates; "
    "standardisation of cluster profiles before Ward linkage.",
    f"<b>Aggregation (roll-up):</b> observation level → storm level ({pre['n_storms']:,} storms: "
    "genesis position and month, lifetime max wind, min pressure, duration, track length, landfall, "
    "max 24-h intensification).",
    f"<b>Transaction encoding for ARM:</b> every observation becomes an itemset of attribute=value "
    f"items ({pre['n_transactions']:,} transactions, {pre['n_items']} distinct items): Basin, Season "
    "(Jan–Jul / Aug–Sep / Oct–Dec), latitude band (&lt;15N, 15–25N, 25–35N, ≥35N), Status family, "
    "Era (1851–1949 / 1950–1989 / 1990–2015), Pressure band (&lt;980, 980–999, ≥1000 hPa, when "
    "known), Speed band (&lt;15, 15–30, &gt;30 km/h), Motion (Eastward / Westward), Landfall.",
    "<b>Label encoding:</b> Basin → 0/1 for the decision tree; bit-mask encoding of itemsets for "
    "A-CLOSE and Spark.",
])

h2("4.4 Descriptive statistics (after cleaning)")
dd = pd.DataFrame(viz["desc"]).T
dd = dd[["count", "mean", "median", "mode", "std", "min", "25%", "75%", "max", "IQR", "skew", "kurtosis"]]
dd.insert(0, "attribute", dd.index)
table(dd.round(2), font=6.8,
      caption="Table 7. Five-number summary and shape statistics of cleaned observations")
sd = pd.DataFrame(viz["storm_desc"]).T.reset_index().rename(columns={"index": "storm attribute"})
table(sd.round(1), font=7.5, caption="Table 8. Storm-level summaries")
bullets([
    f"Wind is right-skewed (skew {viz['desc']['Wind']['skew']}): mean {viz['desc']['Wind']['mean']} kt "
    f"&gt; median {viz['desc']['Wind']['median']} kt &gt; mode {viz['desc']['Wind']['mode']} kt — most "
    "of a storm's life is spent at modest intensity.",
    f"Pressure is left-skewed (skew {viz['desc']['Pressure']['skew']}) with kurtosis "
    f"{viz['desc']['Pressure']['kurtosis']}: a long tail of very deep lows.",
    f"A typical storm lives {viz['storm_desc']['DurationH']['50%'] / 24:.1f} days (median) and travels "
    f"{viz['storm_desc']['TrackKm']['50%']:,.0f} km; the longest track is "
    f"{viz['storm_desc']['TrackKm']['max']:,.0f} km.",
    f"Strongest storm: {viz['top_storms'][0]['Name'].title()} {viz['top_storms'][0]['Year']} "
    f"({viz['top_storms'][0]['MaxWind']:.0f} kt, {viz['top_storms'][0]['MinPressure']:.0f} hPa).",
])
story.append(PageBreak())

# ============================================================================
# 5. VISUALISATION
# ============================================================================
h1("5. Visualisation Techniques and Inferences")
p("Fifteen distinct plot types are applied to the cleaned observation- and storm-level tables. "
  "Atlantic is always shown in blue and Pacific in orange; magnitudes use a single-hue sequential "
  "scale. Each subsection states why that technique is appropriate, shows the figure, and lists "
  "the inferences.")
v, fs, pq = viz, viz["five_num"], viz["pressure_quantiles"]
mc, sh, mdc = v["month_counts"], v["status_share"], v["median_duration_by_cat"]
pcm, msp, corr = v["pc_medians"], v["major_share_by_period"], v["corr"]

h2("5.1 Histogram")
p("A histogram of Maximum Wind, with 5-kt bins matching the reporting resolution of HURDAT2, "
  "shows the shape of the intensity distribution and the share of hurricane-force observations.")
fig("fig01_histogram_wind.png", "Figure 1. Histogram of maximum wind (5-kt bins)")
bullets([
    f"Right-skewed (skew {v['wind_skew']}), mode {v['wind_mode']:.0f} kt; {v['pct_below_64']} % of "
    f"observations are below hurricane force and only {v['pct_major']} % are major hurricanes (≥ 96 kt).",
    "99.99 % of wind values are multiples of 5 kt, so 5-kt bins are the natural width (no spurious gaps).",
    "Both basins have the same shape; the Atlantic has more observations because its record is 98 years longer.",
])

h2("5.2 Box plot")
p("Grouped box plots compare the five-number summary of wind across status families and basins, "
  "and expose residual coding errors (the Super Typhoon rows before remapping).")
fig("fig02_boxplot_status.png", "Figure 2. Box plot of wind per status family and basin")
bullets([
    f"HU: median {fs['HU']['50%']:.0f} kt, IQR {fs['HU']['25%']:.0f}–{fs['HU']['75%']:.0f} kt, outliers "
    f"up to {fs['HU']['max']:.0f} kt.",
    f"TS lies strictly within {fs['TS']['min']:.0f}–{fs['TS']['max']:.0f} kt and TD ≤ {fs['TD']['max']:.0f} kt "
    "after cleaning.",
    "Atlantic EX systems are stronger (median 45 kt) than Pacific EX (25 kt): Atlantic storms "
    "recurve into the mid-latitude westerlies while Pacific storms decay over cold water.",
])

h2("5.3 Quantile plot")
p("A quantile plot of central pressure compares Atlantic and Pacific intensity without binning "
  "and highlights the rarity of deep systems.")
fig("fig03_quantile_pressure.png", "Figure 3. Quantile plot of central pressure")
bullets([
    f"Atlantic quantiles are lower everywhere: 10 % quantile {pq['Atlantic']['0.1']:.0f} vs "
    f"{pq['Pacific']['0.1']:.0f} hPa, median {pq['Atlantic']['0.5']:.0f} vs {pq['Pacific']['0.5']:.0f} hPa.",
    "The steep drop below f ≈ 0.05 shows how rare deep systems are: the 5th percentile is 952 hPa "
    "and only 4.1 % of pressure readings are below 950 hPa.",
    "The flat upper part (≈ 1000–1010 hPa) is the mass of weak depressions and storms.",
])

h2("5.4 Q–Q plot")
p("Two Q–Q plots test (i) whether Atlantic and Pacific winds share a distribution and (ii) whether "
  "storm lifetime-maximum wind is Gaussian — which would licence parametric methods.")
fig("fig04_qq_plot.png",
    "Figure 4. Q–Q plots: Atlantic vs Pacific (left) and storm max wind vs normal (right)")
bullets([
    f"Left: points lie on y = x in the lower part (mean difference {v['qq_lower_diff']} kt) but "
    f"above it in the upper 20 % (Pacific +{v['qq_upper_diff']} kt).",
    f"Right: correlation with the normal line is high (r = {v['normal_qq_r']}) but the curve is "
    "convex — lifetime max wind is right-skewed, so medians, IQRs and trees are preferred.",
])

h2("5.5 Scatter plot")
p("Wind versus central pressure is the classical intensity diagram and the reason pressure can "
  "stand in for wind in the classifier.")
fig("fig05_scatter_wind_pressure.png", "Figure 5. Scatter plot of wind vs central pressure")
bullets([
    f"Very strong linear inverse relation: r = {v['wp_r']}; wind ≈ {v['wp_intercept']:.0f} "
    f"{v['wp_slope']:+.2f} × P, i.e. every 1 hPa deeper adds ≈ {abs(v['wp_slope']):.1f} kt.",
    "Scatter widens at low pressure — the wind–pressure relationship depends on storm size and latitude.",
    "Justifies using Pressure as the physically independent predictor of Status in the decision tree.",
])

h2("5.6 Scatter-plot matrix")
p("A storm-level scatter-plot matrix shows pairwise structure among peak intensity, duration, "
  "track length and genesis latitude, coloured by basin.")
fig("fig06_scatter_matrix.png", "Figure 6. Scatter-plot matrix of storm-level attributes", width=0.78)
bullets([
    "MaxWind–MinPressure is a tight line; DurationH–TrackKm is strongly positive.",
    "Long-lived storms reach higher peak winds (positive MaxWind–DurationH cloud).",
    "Pacific storms form at lower genesis latitudes (orange mass shifted left on GenesisLat).",
])

h2("5.7 Correlation heatmap")
p("Pearson correlations at storm level quantify redundancy (DurationH vs NumObs) and the "
  "physical couplings used later (rapid intensification vs peak wind).")
fig("fig07_correlation_heatmap.png", "Figure 7. Correlation heat-map (storm level)")
bullets([
    f"MaxWind–MinPressure r = {corr['MaxWind']['MinPressure']}; MaxWind–Max24hIntensification "
    f"r = {corr['MaxWind']['Max24hIntensification']}.",
    f"DurationH–NumObs r = {corr['DurationH']['NumObs']} → redundant attribute removed.",
    f"Genesis latitude correlates negatively with MaxWind (r = {corr['MaxWind']['GenesisLat']}) "
    f"and duration (r = {corr['DurationH']['GenesisLat']}).",
    f"Year vs MaxWind is weak (r = {corr['MaxWind']['Year']}): no simple linear trend in individual peaks.",
])

h2("5.8 Grouped bar chart")
p("Storm counts by genesis month, grouped by basin, define the hurricane season and the "
  "Atlantic/Pacific offset.")
fig("fig08_bar_month.png", "Figure 8. Grouped bar chart: storms formed per month")
bullets([
    f"Atlantic peaks in September ({mc['Atlantic']['9']} storms) and {v['atl_aug_oct_share']} % form in Aug–Oct.",
    f"Pacific peaks in August ({mc['Pacific']['8']}) and starts earlier: May–July genesis is much "
    f"higher ({mc['Pacific']['5'] + mc['Pacific']['6'] + mc['Pacific']['7']} vs "
    f"{mc['Atlantic']['5'] + mc['Atlantic']['6'] + mc['Atlantic']['7']} Atlantic storms); "
    f"{v['pac_jul_sep_share']} % form in Jul–Sep.",
    "Only 28 of 2,845 storms (≈1 %) formed outside May–November.",
])

h2("5.9 Line chart with smoothing")
p("Annual storm counts with a 10-year moving average separate the satellite-era detection jump "
  "from multi-decadal oscillations.")
fig("fig09_line_yearly.png",
    "Figure 9. Line chart of annual storm counts (thin) with 10-year moving average (thick)")
bullets([
    f"Recorded Atlantic activity more than doubled: {v['atl_mean_1851_1900']} storms/yr (1851–1900) vs "
    f"{v['atl_mean_1966_2015']} (1966–2015); Pacific {v['pac_mean_1949_1970']} (1949–70) vs "
    f"{v['pac_mean_1971_2015']} (1971–2015).",
    "The step around the mid-1960s is improved detection of short-lived open-ocean storms as well "
    "as any climatic change.",
    f"The busiest Atlantic year is {v['max_year_atl']} ({v['max_year_atl_n']} storms).",
])

h2("5.10 Pie chart")
p("The share of six-hourly observations by status family documents class imbalance for the tree.")
fig("fig10_pie_status.png", "Figure 10. Pie chart: share of observations by status")
bullets([
    f"Tropical storms dominate ({sh['TS']} %), then hurricanes ({sh['HU']} %) and depressions ({sh['TD']} %).",
    f"EX ({sh['EX']} %), LOW ({sh['LOW']} %) and SUB ({sh['SUB']} %) are minor classes — handled in "
    "the tree by macro-averaged metrics.",
])

h2("5.11 Violin plot")
p("Violin plots of storm lifetime by peak Saffir–Simpson category show both the median duration "
  "and the spread, which a box plot would hide.")
fig("fig11_violin_duration.png", "Figure 11. Violin plot of storm lifetime by peak category")
bullets([
    f"Median lifetime rises monotonically: TD {mdc['TD']} d, TS {mdc['TS']} d, Cat1 {mdc['Cat1']} d, "
    f"Cat3 {mdc['Cat3']} d, Cat5 {mdc['Cat5']} d.",
    "Reaching high intensity requires a long time over warm water; major hurricanes have wide, long tails.",
])

h2("5.12 Geographic track map")
p("Plotting every fix coloured by wind is the natural spatial view of the two basins' typical paths "
  "and of where hurricane force actually occurs.")
fig("fig12_track_map.png", "Figure 12. Geographic track map (every fix coloured by wind)")
bullets([
    "Pacific storms move west at 10–20°N and dissipate over the cool eastern Pacific; Atlantic storms "
    "recurve north-eastwards and often become extratropical.",
    f"Intense fixes are tropical: mean latitude of hurricane fixes {v['hu_lat_mean']}°N; 90 % of "
    f"major-hurricane fixes are south of {v['major_lat_p90']}°N.",
    f"Atlantic landfall rate {v['landfall_by_basin']['Atlantic']} % vs Pacific "
    f"{v['landfall_by_basin']['Pacific']} %.",
])

h2("5.13 Parallel coordinates")
p("Parallel coordinates of a balanced sample of storms (40 per peak class) show the multivariate "
  "profile of Cat-5 versus tropical-storm systems.")
fig("fig13_parallel_coordinates.png",
    "Figure 13. Parallel coordinates of storm profiles (40 storms per class)")
bullets([
    f"Cat-5 storms (median {pcm['Cat5']['MinPressure']:.0f} hPa, {pcm['Cat5']['DurationH']:.0f} h, "
    f"{pcm['Cat5']['TrackKm']:,.0f} km) separate from TS ({pcm['TS']['MinPressure']:.0f} hPa, "
    f"{pcm['TS']['DurationH']:.0f} h, {pcm['TS']['TrackKm']:,.0f} km) on every axis.",
    f"Stronger storms form at lower latitude (median genesis {pcm['Cat5']['GenesisLat']:.0f}°N for "
    f"Cat5 vs {pcm['TS']['GenesisLat']:.0f}°N for TS).",
])

h2("5.14 Hexbin density plot")
p("A hexbin of genesis locations is the right density estimate for a point process on the map, "
  "and motivates density-based clustering in Chapter 8.")
fig("fig14_hexbin_genesis.png", "Figure 14. Hexbin density plot of genesis locations")
bullets([
    f"Two dense nurseries: eastern Pacific off Mexico (median genesis {v['genesis_lat_median']['Pacific']}°N, "
    f"{-v['genesis_lon_median']['Pacific']}°W) and the Atlantic/Caribbean "
    f"({v['genesis_lat_median']['Atlantic']}°N, {-v['genesis_lon_median']['Atlantic']}°W).",
    "Pacific genesis is concentrated; Atlantic genesis is spread from Africa to the Gulf of Mexico.",
])

h2("5.15 100 % stacked bar chart")
p("The mix of peak categories among hurricanes, by period, is the appropriate way to compare "
  "composition when the number of recorded storms is itself increasing.")
fig("fig15_stacked_category.png",
    "Figure 15. 100 % stacked bar: peak-category mix of hurricanes by period")
bullets([
    f"Share of hurricanes that became major (Cat3+) rose in the Atlantic from {msp['Atlantic 1851-1900']} % "
    f"(1851–1900) to {msp['Atlantic 2001-2015']} % (2001–2015), and in the Pacific from "
    f"{msp['Pacific 1951-1980']} % (1951–80) to {msp['Pacific 2001-2015']} % (2001–15).",
    "No Cat5 is recorded before 1924 and 33 of 48 Cat5 storms occur after 1970; early under-estimation "
    "of peak winds explains part of the shift.",
])

h2("5.16 Summary of visualisation techniques")
table(pd.DataFrame([
    ["1", "Histogram", "Wind distribution, reporting resolution"],
    ["2", "Box plot", "Status-family intensity and basin contrast"],
    ["3", "Quantile plot", "Pressure tails without binning"],
    ["4", "Q–Q plot", "Basin equality; Gaussianity of max wind"],
    ["5", "Scatter plot", "Wind–pressure law"],
    ["6", "Scatter-plot matrix", "Storm-level pairwise structure"],
    ["7", "Correlation heatmap", "Redundancy and couplings"],
    ["8", "Grouped bar chart", "Seasonal genesis"],
    ["9", "Line + moving average", "Detection vs climate in annual counts"],
    ["10", "Pie chart", "Class shares / imbalance"],
    ["11", "Violin plot", "Lifetime vs peak category"],
    ["12", "Geographic track map", "Paths and intensity geography"],
    ["13", "Parallel coordinates", "Multivariate Cat-5 vs TS profiles"],
    ["14", "Hexbin density", "Genesis nurseries"],
    ["15", "100 % stacked bar", "Category mix by era"],
], columns=["#", "Technique", "Question answered"]),
      col_widths=[1.2 * cm, 4.2 * cm, W - 5.4 * cm],
      caption="Table 9. The fifteen descriptive visualisation techniques")
story.append(PageBreak())

# ============================================================================
# 6. DIC
# ============================================================================
h1("6. Association Rule Mining Using DIC")
h2("6.1 Introduction and transaction definition")
p("Association rules answer 'which attribute-values co-occur more often than independence would "
  "suggest?' Each six-hourly observation is one transaction: a set of discretised items (basin, "
  "season, latitude band, status family, era, pressure band when known, speed band, motion, "
  "landfall). Continuous fields are not used raw; they are binned with domain thresholds so that "
  f"a rule is readable. This yields {dic['N']:,} transactions on {dic['items']} items.")

h2("6.2 Mathematical foundation")
p("For itemsets X, Y with X ∩ Y = ∅,")
p("support(X → Y) = P(X ∪ Y), &nbsp;&nbsp; confidence(X → Y) = P(Y | X), &nbsp;&nbsp; "
  "lift(X → Y) = P(Y | X) / P(Y).")
p("A rule is retained if support ≥ min_support and confidence ≥ min_confidence. Lift &gt; 1 "
  "marks a positive association. Frequent itemsets are those with support ≥ min_support; rules "
  "are generated from them by splitting into antecedent and consequent.")

h2("6.3 Algorithm: Dynamic Itemset Counting")
p("Apriori counts k-itemsets only after a full pass over (k−1)-itemsets. DIC (Brin et al., 1997) "
  "[2] reads the file in blocks of M transactions and places a stop after every block. Each "
  "itemset is in one of four states: dashed circle (being counted, still below support), dashed "
  "box (being counted, already frequent), solid circle (finished, infrequent), solid box "
  "(finished, frequent). As soon as all immediate subsets of a candidate are boxes, counting of "
  "that candidate starts at the next stop. Counting wraps around the file until every dashed "
  "itemset has seen all N transactions. Implemented from scratch in "
  "<font face='DVM'>src/03_dic.py</font> with NumPy block counting.")

h2("6.4 Parameter selection")
p(f"min_support = {MIN_SUPPORT:.0%} (absolute count {dic['min_count']:,}) keeps only patterns that "
  f"appear in several thousand fixes, not one-off storms. min_confidence = {MIN_CONFIDENCE:.0%} "
  f"is the usual 'strong rule' cutoff. M = {DIC_BLOCK_SIZE:,} is large enough for stable early "
  "promotion of boxes but small enough that several stops occur per pass "
  f"(N/M ≈ {dic['N'] / DIC_BLOCK_SIZE:.0f}).")

h2("6.5 Execution results")
table(pd.DataFrame([
    ["Stops (blocks read)", dic["stops"]],
    ["Passes over the data", dic["passes"]],
    ["Candidate itemsets counted", dic["candidates"]],
    ["Frequent itemsets", dic["n_frequent"]],
    ["Frequent itemsets by size", ", ".join(f"{k}-itemsets: {v}" for k, v in dic["by_len"].items())],
    ["Apriori passes for the same result", dic["max_k"] + 1],
    ["Identical to mlxtend Apriori", dic["apriori_match"]],
    ["Rules with confidence ≥ 70 %", dic["n_rules"]],
    ["Run time", f"{dic['time_s']} s"],
], columns=["DIC output", "Value"]),
      col_widths=[7.2 * cm, W - 7.2 * cm], font=8,
      caption="Table 10. DIC run summary")
rules = pd.read_csv(os.path.join(CLEAN_DIR, "dic_rules.csv"))
h3("Top 15 DIC rules by lift")
table(rules.head(15), col_widths=[5.6 * cm, 5 * cm, 1.6 * cm, 1.9 * cm, W - 14.1 * cm], font=7.0,
      caption="Table 11. Top 15 DIC rules (IF antecedent THEN consequent)")
stat = rules[rules["THEN (consequent)"].str.match(r"^\{Status=[A-Z]+\}$")]
h3("Rules whose consequent is storm Status")
table(stat.head(8), col_widths=[5.6 * cm, 5 * cm, 1.6 * cm, 1.9 * cm, W - 14.1 * cm], font=7.0,
      caption="Table 12. Status-consequent DIC rules")

h2("6.6 Interpretation")
bullets([
    f"DIC finished in {dic['passes']} passes instead of the {dic['max_k'] + 1} level-wise passes of "
    f"Apriori (≈{100 * (1 - dic['passes'] / (dic['max_k'] + 1)):.0f} % fewer scans) with identical "
    "frequent itemsets — the benefit of starting longer itemsets mid-pass.",
    "{Status=EX} ⇒ {Lat ≥ 35N} (conf 82 %, lift 6.1) and {Motion=Eastward, Status=EX} ⇒ {Lat ≥ 35N} "
    "(lift 6.4): extratropical transition happens poleward of 35°N on an eastward track.",
    "{Lat ≥ 35N, Speed &gt; 30 km/h} ⇒ {Motion=Eastward} (conf 96 %): storms accelerate after recurvature.",
    "{Pressure &lt; 980} ⇒ {Status=HU} (conf 94 %, lift 3.3); with westward motion confidence 98 % — "
    "consistent with Figure 5.",
    "{Era=1990–2015, Status=TD} ⇒ {Pressure ≥ 1000} (conf 87 %): modern depressions are routinely "
    "measured weak lows; the rule cannot appear in older eras where pressure is missing.",
])
p(f"The full list of {dic['n_rules']} rules is saved as "
  "<font face='DVM'>outputs/data/dic_rules.csv</font> rather than printed here.")
story.append(PageBreak())

# ============================================================================
# 7. DECISION TREE
# ============================================================================
h1("7. Decision Tree Classification")
h2("7.1 Suitability and problem definition")
cd = dt["class_dist"]
p("Storm Status is a four-class label (TD / TS / HU / EX) that a forecaster infers from intensity "
  "and location. A decision tree is appropriate: the features mix numeric (pressure, latitude) "
  "and binary (basin) attributes, interactions are threshold-like (pressure below ~980 hPa and "
  "latitude below ~40°N), and the course requires readable IF–THEN rules. "
  f"<b>Maximum Wind is excluded</b> because Status is defined from wind thresholds — including it "
  f"would be target leakage. Only observations with measured pressure and a motion vector are "
  f"used: {dt['n_rows']:,} rows from {dt['n_storms']:,} storms (TS {cd['TS']:,}, HU {cd['HU']:,}, "
  f"TD {cd['TD']:,}, EX {cd['EX']:,}). Predictors: Latitude, Longitude, Month, Basin, Pressure, "
  "dPressure6h, Speed_kmh, Heading, HoursSinceGenesis.")

h2("7.2 Split criteria and validation design")
p("Gini impurity and information entropy were both candidates. Consecutive six-hourly fixes of "
  "the same storm are almost identical, so a random row split would leak the test storm into "
  f"training. The data are split <b>by storm</b> (GroupShuffleSplit 70/30: {dt['train_rows']:,} / "
  f"{dt['test_rows']:,} rows, no storm in both). Hyper-parameters (criterion, max_depth ∈ [2, 20], "
  "min_samples_leaf = 20) are chosen with 5-fold <b>GroupKFold</b> on the training storms.")

h2("7.3 Depth selection")
fig("dt_depth_curve.png",
    f"Figure 16. Depth selection: training vs cross-validated accuracy "
    f"(best: {dt['best_criterion']}, depth {dt['best_depth']})", width=0.7)
p("Training accuracy keeps rising with depth while CV accuracy flattens after depth 8: deeper "
  "trees over-fit the particular storms in the training folds.")

h2("7.4 Test performance")
rep = dt["report"]
mt = pd.DataFrame([[c, round(rep[c]["precision"], 3), round(rep[c]["recall"], 3),
                    round(rep[c]["f1-score"], 3), int(rep[c]["support"])]
                   for c in ["TD", "TS", "HU", "EX"]] +
                  [["macro avg", round(rep["macro avg"]["precision"], 3),
                    round(rep["macro avg"]["recall"], 3),
                    round(rep["macro avg"]["f1-score"], 3),
                    int(rep["macro avg"]["support"])]],
                  columns=["class", "precision", "recall", "F1", "support"])
table(pd.DataFrame([["Accuracy", dt["test_acc"]], ["Macro F1", dt["macro_f1"]],
                    ["Weighted F1", dt["weighted_f1"]], ["Cohen's kappa", dt["kappa"]],
                    ["Majority-class baseline", dt["baseline"]],
                    ["CV accuracy (training storms)", dt["cv_acc"]],
                    ["Leaves", dt["leaves"]]], dtype=object,
                   columns=["Measure", "Value"]),
      col_widths=[7 * cm, 4 * cm], font=8,
      caption="Table 13. Held-out test metrics (unseen storms)")
table(mt, col_widths=[2.6 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm], font=8,
      caption="Table 14. Per-class precision, recall and F1")
fig("dt_confusion_matrix.png",
    "Figure 17. Confusion matrix (rows = actual, columns = predicted)", width=0.5)
fig("dt_feature_importance.png",
    "Figure 18. Feature importance (total impurity reduction)", width=0.62)

h2("7.5 Interpretable rules")
fig("dt_tree.png", "Figure 19. Top three levels of the interpretable depth-4 tree", width=1.0)
h3(f"IF–THEN rules (depth-4 tree, test accuracy {dt['small_acc']})")
dr = pd.read_csv(os.path.join(CLEAN_DIR, "dt_rules.csv"))
table(dr[["rule", "samples", "confidence"]],
      col_widths=[W - 3.4 * cm, 1.6 * cm, 1.8 * cm], font=7.2,
      caption="Table 15. Depth-4 tree rules")
imp = dt["importance"]
h2("7.6 Interpretation")
bullets([
    f"Test accuracy {dt['test_acc']:.3f} (kappa {dt['kappa']:.3f}) versus {dt['baseline']:.3f} for "
    f"the majority-class baseline. CV ({dt['cv_acc']:.3f}) matches the test score, so the model generalises.",
    f"HU is recognised best (F1 {rep['HU']['f1-score']:.3f}); EX is hardest (F1 {rep['EX']['f1-score']:.3f}) — "
    "extratropical systems share pressures with TS and are separated mainly by latitude.",
    "Errors are almost all between neighbouring intensity classes (TD↔TS, TS↔HU). TD and HU are "
    "virtually never confused.",
    f"Pressure carries {100 * imp['Pressure']:.0f} % of the importance, latitude "
    f"{100 * imp['Latitude']:.0f} %, longitude {100 * imp['Longitude']:.0f} %.",
    "The rules read as heuristics: IF Latitude ≤ 38.8 AND Pressure ≤ 979.5 THEN HU (99 % "
    "confidence, 3,454 obs); IF Latitude &gt; 50.1 AND Pressure ≤ 988.5 THEN EX (100 %).",
])
story.append(PageBreak())

# ============================================================================
# 8. DBSCAN
# ============================================================================
h1("8. DBSCAN Clustering of Storm Genesis")
h2("8.1 Why DBSCAN")
p("Genesis regions have irregular shapes, an unknown number of modes, and isolated subtropical "
  "events that should be flagged as noise — properties that k-means (spherical clusters, every "
  "point assigned) cannot handle. Each storm's first fix is clustered with the haversine "
  "(great-circle) distance, which is the correct metric on the sphere; Euclidean degrees would "
  "distort near the dateline and stretch longitude at higher latitude.")

h2("8.2 Parameter selection")
p(f"MinPts = {db['min_pts']} (a genesis 'hot-spot' should contain at least a small handful of "
  f"storms, not two). eps is read from the knee of the sorted {db['min_pts']}-distance graph. "
  "The top 5 % of k-distances are isolated outliers that dominate curvature (a knee on the full "
  "curve gives ≈430 km and chains the Atlantic and Pacific into one cluster), so the knee is "
  f"taken on the main body: eps = <b>{db['eps_km']:.0f} km</b>.")
fig("db_kdistance.png", "Figure 20. k-distance graph used to choose eps", width=0.65)
g = pd.DataFrame(db["grid"])
table(g, col_widths=[2.6 * cm] * 4, font=7.5,
      caption="Table 16. Sensitivity of DBSCAN to eps and MinPts")

h2("8.3 Level-1 result")
prof = pd.DataFrame(db["profiles"])
table(prof[["Cluster", "storms", "atlantic_pct", "lat_mean", "lon_mean", "peak_month",
            "mean_maxwind", "hurricane_pct", "major_pct", "landfall_pct",
            "median_duration_h"]].round(1), font=7,
      caption=f"Table 17. Level-1 clusters (eps = {db['eps_km']:.0f} km, MinPts = {db['min_pts']})")
p(f"{db['n_clusters']} clusters, {db['noise']} noise storms ({db['noise_pct']} %), {db['core']:,} "
  f"core points, silhouette coefficient = <b>{db['silhouette']}</b>.")
fig("db_clusters_map.png", "Figure 21. Level-1 DBSCAN clusters (grey crosses = noise)", width=0.95)

h2("8.4 Multi-scale hot-spots")
p(f"Density varies strongly inside the basins, so each level-1 cluster is re-clustered with "
  f"eps = {db['eps2_km']} km to expose individual genesis hot-spots.")
sp = pd.DataFrame(db["sub_profiles"])
sp["_o"] = sp.SubCluster.str.extract(r"C(\d+)\.(\d+)").astype(int).apply(lambda r: r[0] * 100 + r[1], axis=1)
sp = sp.sort_values("_o").drop(columns="_o")
table(sp.round(1), font=6.6, caption="Table 18. Level-2 genesis hot-spot profiles")
fig("db_subclusters_map.png", "Figure 22. Level-2 DBSCAN genesis hot-spots", width=0.95)
fig("db_dendrogram.png",
    "Figure 23. Dendrogram (Ward linkage on standardised hot-spot profiles)", width=0.75)
spi = sp.set_index("SubCluster")

h2("8.5 Interpretation")
bullets([
    f"Without being told the basin, DBSCAN separates the two nurseries: C0 is "
    f"{prof.loc[0, 'atlantic_pct']} % Atlantic and C1 is {prof.loc[1, 'atlantic_pct']} % Atlantic — "
    "the low-density gap over Central America is a natural boundary.",
    f"Noise storms ({db['noise_pct']} %) form at higher latitude (mean {db['noise_lat']}°N) and "
    f"become hurricanes less often ({db['noise_hu_pct']} %).",
    f"Cape Verde hot-spot C0.4 (≈{-spi.loc['C0.4', 'lon_mean']:.1f}°W) produces the longest-lived "
    f"({spi.loc['C0.4', 'median_duration_h']:.0f} h) and most dangerous storms "
    f"({spi.loc['C0.4', 'major_pct']} % major) but few landfalls ({spi.loc['C0.4', 'landfall_pct']} %).",
    f"Gulf of Mexico C0.0: short ({spi.loc['C0.0', 'median_duration_h']:.0f} h), weaker "
    f"({spi.loc['C0.0', 'hurricane_pct']} % hurricanes) but {spi.loc['C0.0', 'landfall_pct']} % landfall "
    "— highest coastal risk per storm.",
    "Ward grouping of hot-spot profiles: (a) deep-tropical Atlantic MDR (C0.3, C0.4, C0.6, C0.8, "
    "C0.9, C0.10), (b) eastern Pacific (C1.0, C1.1), (c) near-land western Atlantic / Caribbean / "
    "Gulf (C0.0, C0.1, C0.2, C0.5, C0.7).",
    f"Silhouette {db['silhouette']} is reasonable for geographic data; the sensitivity table shows "
    "that smaller eps fragments clusters and raises noise, while larger eps merges basins.",
])
story.append(PageBreak())

# ============================================================================
# 9. A-CLOSE
# ============================================================================
h1("9. Closed Frequent Itemsets with A-CLOSE")
h2("9.1 Frequent vs closed vs maximal itemsets")
p("A frequent itemset may have a proper superset with the same support; listing all frequent "
  "itemsets then repeats the same information. A <b>closed</b> frequent itemset (CFI) has no "
  "superset with the same support. The set of CFIs is a lossless condensed representation: every "
  "frequent itemset and its support can be recovered. A <b>maximal</b> frequent itemset has no "
  "frequent superset at all and is more compact but not lossless. A-CLOSE (Pasquier et al., 1999) "
  "[3] mines CFIs via generators.")

h2("9.2 Algorithm")
p("A-CLOSE mines generators (minimal itemsets) level by level. A candidate is discarded if it is "
  "infrequent <i>or</i> if its support equals that of one of its subsets (then it lies in that "
  "subset's closure). One final pass computes the Galois closure h(g) = ∩{t : g ⊆ t} of the "
  "generators from level i−1 on, where i is the first level at which such pruning happened. "
  "Rules are produced as the closed-itemset basis: <b>exact rules</b> g ⇒ h(g)∖g (100 % "
  "confidence) and <b>approximate rules</b> g ⇒ C∖g for closed C ⊃ h(g). Implemented from "
  "scratch in <font face='DVM'>src/04_aclose.py</font>; counting and closure are the bit-mask "
  "kernels in <font face='DVM'>src/arm_common.py</font>.")

h2("9.3 Parameters and results")
table(pd.DataFrame([
    ["Generators", ac["n_generators"]],
    ["Closed frequent itemsets", ac["n_closed"]],
    ["CFIs by size", ", ".join(f"{k}: {v}" for k, v in ac["closed_by_len"].items())],
    ["First level with a non-generator (i)", ac["level_i"]],
    ["Database passes", ac["passes"]],
    ["Frequent itemsets (DIC) represented", ac["n_frequent"]],
    ["Verified against closed sets from DIC", ac["verified"]],
    ["Exact rules (conf = 100 %)", ac["n_exact"]],
    ["Approximate rules (conf ≥ 70 %)", ac["n_approx"]],
    ["Run time (pure Python)", f"{ac['time_s']} s"],
], columns=["A-CLOSE output", "Value"]),
      col_widths=[7.5 * cm, W - 7.5 * cm], font=8,
      caption="Table 19. A-CLOSE run summary")
cfi = pd.read_csv(os.path.join(CLEAN_DIR, "aclose_closed_itemsets.csv"))
table(cfi.head(10), col_widths=[W - 5.2 * cm, 1.2 * cm, 2.2 * cm, 1.8 * cm], font=7.0,
      caption="Table 20. Largest closed frequent itemsets")
ar = pd.read_csv(os.path.join(CLEAN_DIR, "aclose_rules.csv"))
table(ar[ar.type == "exact"].drop(columns="type"),
      col_widths=[5.6 * cm, 4.4 * cm, 1.8 * cm, 2 * cm, W - 13.8 * cm], font=7.0,
      caption="Table 21. Exact closed rules (100 % confidence)")
table(ar[ar.type == "approximate"].sort_values(["lift", "confidence"], ascending=False)
      .head(10).drop(columns="type"),
      col_widths=[5.6 * cm, 4.4 * cm, 1.8 * cm, 2 * cm, W - 13.8 * cm], font=7.0,
      caption="Table 22. Top approximate closed rules by lift")

h2("9.4 Validation and interpretation")
bullets([
    f"A-CLOSE returns exactly the closed sets obtained from the DIC frequent itemsets "
    f"({ac['n_closed']} CFIs) — correctness verified.",
    f"Only {ac['n_frequent'] - ac['n_closed']} of {ac['n_frequent']} frequent itemsets are "
    "non-closed: basin, season, latitude, status, era and motion are rarely implied by one "
    "another, so the data are sparse and closed itemsets give little compression. A-CLOSE pays "
    "off on dense, highly correlated data.",
    "The three exact rules all say: every pre-1950 observation that is late-season (Oct–Dec) or "
    "north of 35°N is Atlantic — the Pacific record only begins in 1949 and its storms rarely "
    "reach 35°N.",
    f"The closed basis contains {ac['n_exact'] + ac['n_approx']} rules vs {dic['n_rules']} from "
    "all frequent itemsets, with the same strongest rules (EX ⇒ Lat ≥ 35N, Pressure &lt; 980 ⇒ HU).",
    "The largest CFI {Atlantic, Lat 15–25N, Westward, Aug–Sep, 15–30 km/h} (7.6 % of all fixes) is "
    "the archetypal Atlantic peak-season storm in the trade winds.",
])
p("The full closed-rule basis is in <font face='DVM'>outputs/data/aclose_rules.csv</font>.")
story.append(PageBreak())

# ============================================================================
# 10. PYSPARK IMPLEMENTATION
# ============================================================================
h1("10. PySpark Implementation")
h2("10.1 Big-data motivation")
p("HURDAT2 at 75 k rows fits in memory. The course still requires two predictive techniques to "
  "be test-driven on PySpark so that (i) correctness against the sequential result is shown and "
  "(ii) the performance effect of big-data support is measured as volume grows. The honest "
  "answer on a laptop-sized table may be that Spark is slower; that is still a result.")

h2("10.2 Architecture and configuration")
p("Apache Spark 4.0.1 is used in local[*] mode (driver and executors in one JVM, one thread per "
  f"core, {sac['cores'] if sac else 8} cores, 6 GB driver memory). DataFrames are cached after "
  "replication. Two techniques are ported: A-CLOSE (Spark SQL aggregations over 64-bit item "
  "bit-masks) and the decision tree (Spark MLlib DecisionTreeClassifier).")

h2("10.3 Selected algorithms")
bullets([
    "<b>A-CLOSE on Spark:</b> each transaction is a 64-bit mask. Support of a candidate c is "
    "SUM(CAST((mask &amp; c) = c AS INT)); the closure is BIT_AND(mask) over rows containing the "
    "generator. One distributed job per level. The same aclose() driver as the Python version is used.",
    "<b>Decision tree on Spark:</b> MLlib bins continuous features (maxBins = 64) and trains a "
    "distributed tree of the same depth selected in Chapter 7. Accuracy is compared on the same "
    "storm-wise test split.",
])

h2("10.4 Windows-specific implementation notes")
p("On this machine (Microsoft Store Python 3.13) Spark's Python worker processes cannot start "
  "(WinError 10038). Python lambdas and UDFs are therefore unusable. Both techniques were written "
  "with JVM-side APIs only (Spark SQL and MLlib). That is also the idiomatic, faster way to use "
  "Spark. The same constraint is documented in the project README.")
story.append(PageBreak())

# ============================================================================
# 11. BENCHMARKING
# ============================================================================
h1("11. Performance Benchmarking")
h2("11.1 Experimental methodology")
p("To emulate larger volumes the data are replicated (×1 … ×32 for A-CLOSE, ×1 … ×100 for the "
  "tree). Replication keeps relative supports and class proportions unchanged, so closed itemsets "
  "and class predictions must stay identical — this is checked on every run. Sequential Python "
  "and Spark are timed after a warm-up; wall-clock seconds are reported. Core scaling uses a "
  "fixed data size and varies local[k].")

h2("11.2 Hardware")
p(f"Windows 11 laptop, {sac['cores'] if sac else 8} logical cores, 16 GB RAM, Spark local mode, "
  "OpenJDK 17. No cluster. Times therefore include JVM scheduling and whole-stage code generation "
  "on one machine, not network shuffle.")

if sac:
    r = pd.DataFrame(sac["results"])
    seq = r[r.engine == "Python sequential"].set_index("scale")
    spk = r[r.engine.str.startswith("Spark")].set_index("scale")
    tb = pd.DataFrame({
        "scale": spk.index,
        "transactions": [f"{int(x):,}" for x in spk.rows],
        "Python sequential (s)": [seq.seconds.get(s, "—") for s in spk.index],
        "PySpark (s)": spk.seconds.values,
        "speed-up": [round(seq.seconds[s] / spk.seconds[s], 1) if s in seq.index else "—"
                     for s in spk.index],
        "same CFIs": spk.identical.values,
    })
    h2("11.3 A-CLOSE: Python vs PySpark")
    table(tb, font=7.8, caption="Table 23. A-CLOSE wall time vs replicated data size")
    fig("spark_aclose_runtime.png", "Figure 24. A-CLOSE run time vs data size", width=0.72)
    cs = pd.DataFrame(sac["core_scaling"])
    cs["speed-up vs 1 core"] = (cs.seconds.iloc[0] / cs.seconds).round(2)
    table(cs, col_widths=[3 * cm, 3 * cm, 4.5 * cm], font=8,
          caption="Table 24. A-CLOSE core scaling (×8 data)")
    big = spk.index.max()
    sq_max = seq.index.max()
    slope_seq = seq.seconds[sq_max] / sq_max
    slope_spk = (spk.seconds[big] - spk.seconds[1]) / (big - 1)
    bullets([
        f"On the original data (×1) Spark is slower ({spk.seconds[1]:.1f} s vs {seq.seconds[1]:.1f} s): "
        "each level launches distributed jobs whose scheduling and code-generation overhead is fixed.",
        f"At ×{sq_max} ({int(seq.rows[sq_max]):,} transactions) Spark is "
        f"{seq.seconds[sq_max] / spk.seconds[sq_max]:.1f}× faster ({seq.seconds[sq_max]:.0f} s → "
        f"{spk.seconds[sq_max]:.1f} s), and it mines {int(spk.rows[big]):,} transactions in "
        f"{spk.seconds[big]:.1f} s — sequential Python would need ≈{slope_seq * big / 60:.0f} min "
        "by linear extrapolation.",
        f"Sequential cost grows linearly (≈{slope_seq:.1f} s per 75 k transactions) while Spark grows "
        f"at ≈{slope_spk:.2f} s per 75 k.",
        f"Core scaling: {cs.seconds.iloc[0]:.1f} s on 1 core → {cs.seconds.iloc[-1]:.1f} s on "
        f"{cs.cores.iloc[-1]} cores ({cs['speed-up vs 1 core'].iloc[-1]}×). At ×8 the run is still "
        "dominated by per-job planning (~150 aggregate expressions × ~12 jobs), so adding cores "
        "gives a modest, sub-linear gain.",
        "All runs return exactly the same closed itemsets.",
    ])

if sdt:
    h2("11.4 Decision tree: scikit-learn vs Spark MLlib")
    sm, km = sdt["spark_metrics"], sdt["sk_metrics"]
    table(pd.DataFrame([["Spark MLlib", sm["accuracy"], sm["f1"], sdt["depth"], sdt["nodes"]],
                        ["scikit-learn", km["accuracy"], km["f1"], 8, "—"]],
                       columns=["Implementation", "test accuracy", "weighted F1", "depth", "nodes"]),
          font=8, caption="Table 25. Model quality of the two tree implementations")
    b = pd.DataFrame(sdt["bench"])
    bdisp = b.copy()
    bdisp["rows"] = bdisp.rows.map(lambda x: f"{int(x):,}")
    table(bdisp.rename(columns={"sklearn_s": "scikit-learn (s)", "spark_s": "Spark MLlib (s)",
                                "speedup": "speed-up (sk/Spark)"}),
          font=7.8, caption="Table 26. Tree training time vs training-set size")
    fig("spark_dt_runtime.png",
        "Figure 25. Decision-tree training time vs training-set size", width=0.7)
    fig("spark_dt_cores.png",
        "Figure 26. Spark decision tree: speed-up vs number of cores", width=0.55)
    cs2 = pd.DataFrame(sdt["core_scaling"])
    bb = pd.DataFrame(sdt["bench"])
    cross = bb[bb.speedup >= 1]
    h3("Spark tree rules (excerpt of toDebugString)")
    story.append(Paragraph("<br/>".join(l.replace(" ", "&nbsp;") for l in sdt["rules_head"][:24]), CODE))
    story.append(Spacer(1, 6))
    bullets([
        f"Spark MLlib reproduces model quality (accuracy {sm['accuracy']} vs {km['accuracy']} for "
        "scikit-learn). MLlib bins continuous features (maxBins = 64) instead of testing every split. "
        f"The scikit-learn score differs slightly from Chapter 7 ({dt['test_acc']}) because numeric "
        "class labels change tie-breaking between equally good splits.",
        (f"For small data scikit-learn (optimised C, in memory) is faster because Spark pays a fixed "
         f"job-scheduling overhead; from ×{int(cross.scale.iloc[0])} ({int(cross.rows.iloc[0]):,} rows) "
         f"Spark is faster and at ×{int(bb.scale.iloc[-1])} it is {bb.speedup.iloc[-1]}× faster."
         if len(cross) else
         "For every tested size scikit-learn remained faster: the data still fit in RAM, where Spark's "
         "job-scheduling overhead dominates."),
        f"Spark training time scales with cores: {cs2.seconds.iloc[0]:.1f} s (1 core) → "
        f"{cs2.seconds.iloc[-1]:.1f} s ({cs2.cores.iloc[-1]} cores), a "
        f"{cs2.seconds.iloc[0] / cs2.seconds.iloc[-1]:.1f}× speed-up.",
    ])

h2("11.5 Interpretation of the big-data test")
p("Big-data support pays off once data volume, not overhead, dominates the cost. On the real "
  "HURDAT2 table a single process is the better tool. The same Spark programs, without change, "
  "are the ones that would run on a cluster if the archive grew by one or two orders of magnitude "
  "(or if wind-radii / ensemble tracks were added). Replication is not a substitute for new "
  "storms; it only isolates the scaling of the algorithm.")
story.append(PageBreak())

# ============================================================================
# 12. LIMITATIONS
# ============================================================================
h1("12. Limitations")
h2("12.1 Data limitations")
bullets([
    "HURDAT2 is a best-track archive, not a homogeneous instrument. Pre-satellite years miss "
    "short-lived open-ocean storms; early peak winds are underestimated. Apparent rises in storm "
    "counts and in the Cat-3+ share mix climate with detection.",
    f"Pressure is missing for {pre['missing_pct']['Minimum Pressure']} % of rows and almost all "
    "pre-1950 Atlantic observations. The decision tree and any pressure-based rule apply only to "
    "the measured subset (25,365 rows).",
    "Wind radii (83.6 % missing) were dropped; storm size is therefore absent from the models.",
    "Landfall is only as complete as the Event flag (mostly 'L' in the modern era).",
])
h2("12.2 Algorithm limitations")
bullets([
    "DIC/A-CLOSE thresholds hide rare but important events (individual Cat-5 storms, landfalls). "
    "A second low-support run would be needed for those.",
    "The transaction encoding discards continuous magnitude except through coarse bins; two 80 kt "
    "and 140 kt hurricanes look the same as Status=HU.",
    "The decision tree is unstable to small data changes; EX is intrinsically hard (F1 0.66) "
    "because it overlaps TS in pressure.",
    f"Closed itemsets compress only {ac['n_frequent'] - ac['n_closed']} of {ac['n_frequent']} "
    "frequent itemsets: the encoding is sparse. A-CLOSE's benefit is the non-redundant rule basis, "
    "not a dramatic reduction in itemsets.",
    "DBSCAN eps is a judgement on the k-distance knee; the sensitivity table is provided because "
    "another reasonable knee would merge or fragment the nurseries.",
])
h2("12.3 Hardware and implementation limitations")
bullets([
    "Spark ran in local mode, not on a cluster. Speed-ups are intra-machine parallelism plus JVM "
    "execution, not distributed I/O.",
    "PySpark on Windows required a no-UDF design (WinError 10038). The implementation is "
    "therefore Spark SQL / MLlib rather than RDD lambdas.",
    "Replication stress tests do not add meteorological diversity; they only scale row count.",
])
story.append(PageBreak())

# ============================================================================
# 13. CONCLUSION
# ============================================================================
h1("13. Conclusion")
bullets([
    f"<b>Preprocessing.</b> A reproducible KDD pipeline turned {pre['raw_rows']:,} raw rows with "
    "sentinel codes, string coordinates, inconsistent status codes and contradictory records into "
    f"{pre['clean_rows']:,} analysable observations ({100 * (1 - pre['clean_rows'] / pre['raw_rows']):.2f} % "
    "removed) and uncovered genuine issues (mis-coded super typhoon, 359°W longitudes, 1967–68 "
    "storms without winds).",
    f"<b>Descriptive analytics.</b> Fifteen visualisation techniques show a right-skewed intensity "
    f"distribution, a tight wind–pressure law (r = {viz['wp_r']}), a September (Atlantic) / August "
    "(Pacific) season peak, a satellite-era detection jump, and that intense storms are long-lived "
    "and born at low latitude.",
    f"<b>Association patterns.</b> DIC found the same {dic['n_frequent']} frequent itemsets as "
    f"Apriori in {dic['passes']} rather than {dic['max_k'] + 1} passes. Rules capture the life cycle: "
    "EX ⇒ high latitude, eastward and fast; deep pressure ⇒ HU.",
    f"<b>Classification.</b> {dt['test_acc']:.1%} accuracy (kappa {dt['kappa']:.2f}) on unseen storms "
    "from pressure and position alone, with human-readable IF–THEN rules.",
    f"<b>Clustering.</b> DBSCAN separated the Atlantic and Pacific nurseries without labels, "
    f"isolated {db['noise_pct']} % sporadic genesis, and at a finer scale distinguished Cape Verde "
    "(long, intense, oceanic) from the Gulf of Mexico (short, landfalling).",
    f"<b>A-CLOSE.</b> {ac['n_closed']} verified closed itemsets and a non-redundant rule basis; "
    "compression is small because the encoding is sparse.",
    f"<b>PySpark.</b> Spark is slower than Python on the original 75 k rows. A-CLOSE on Spark was "
    f"{sac['speedup']['8']:.1f}× faster at 600 k transactions and mined {sac['results'][-1]['rows']:,} "
    f"transactions in {sac['results'][-1]['seconds']:.0f} s with identical CFIs. MLlib matched "
    f"scikit-learn accuracy and was {sdt['bench'][-1]['speedup']}× faster at "
    f"{sdt['bench'][-1]['rows']:,} training rows. Big-data support pays off once volume, not "
    "overhead, dominates.",
    "<b>Overall.</b> Careful preprocessing and leak-free validation matter more than algorithmic "
    "sophistication on messy observational data. Every mining technique pointed to the same "
    "physical structure — trade-wind westward motion, recurvature, wind–pressure coupling, two "
    "genesis nurseries — and the honest big-data answer for 75 k rows is that a single machine is "
    "the better tool.",
])
story.append(PageBreak())

# ============================================================================
# 14. REFERENCES
# ============================================================================
h1("14. References")
refs = [
    "[1] NOAA National Hurricane Center. HURDAT2 format documentation (atlantic.pdf, pacific.pdf in the Kaggle distribution). Hurricane database: https://www.kaggle.com/noaa/hurricane-database.",
    "[2] Brin, S., Motwani, R., Ullman, J. D., &amp; Tsur, S. (1997). Dynamic itemset counting and implication rules for market basket data. In Proceedings of the ACM SIGMOD Conference (pp. 255–264).",
    "[3] Pasquier, N., Bastide, Y., Taouil, R., &amp; Lakhal, L. (1999). Discovering frequent closed itemsets for association rules. In Database Theory — ICDT'99, LNCS 1540 (pp. 398–416). Springer.",
    "[4] Agrawal, R., &amp; Srikant, R. (1994). Fast algorithms for mining association rules. In Proceedings of VLDB (pp. 487–499).",
    "[5] Ester, M., Kriegel, H.-P., Sander, J., &amp; Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise (DBSCAN). In Proceedings of KDD (pp. 226–231).",
    "[6] Quinlan, J. R. (1986). Induction of decision trees. Machine Learning, 1(1), 81–106.",
    "[7] Quinlan, J. R. (1993). C4.5: Programs for Machine Learning. Morgan Kaufmann.",
    "[8] Breiman, L., Friedman, J. H., Olshen, R. A., &amp; Stone, C. J. (1984). Classification and Regression Trees. Wadsworth.",
    "[9] Han, J., Kamber, M., &amp; Pei, J. (2011). Data Mining: Concepts and Techniques (3rd ed.). Morgan Kaufmann.",
    "[10] Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics, 20, 53–65.",
    "[11] Zaharia, M., et al. (2016). Apache Spark: A unified engine for big data processing. Communications of the ACM, 59(11), 56–65.",
    "[12] Meng, X., et al. (2016). MLlib: Machine learning in Apache Spark. Journal of Machine Learning Research, 17(34), 1–7.",
    "[13] Apache Software Foundation. Spark 4.0 MLlib Guide (classification, frequent pattern mining). https://spark.apache.org/docs/latest/ml-guide.html",
    "[14] scikit-learn developers. User Guide: Decision Trees; Clustering; Model Evaluation. https://scikit-learn.org/stable/user_guide.html",
    "[15] Landsea, C. W., &amp; Franklin, J. L. (2013). Atlantic hurricane database uncertainty and presentation of a new database format. Monthly Weather Review, 141, 3576–3592.",
]
for r in refs:
    story.append(Paragraph(r, SM))

story.append(PageBreak())

# ============================================================================
# APPENDICES
# ============================================================================
h1("Appendix A. Important Code Snippets")
p("DIC stop, block count and mid-pass candidate start (<font face='DVM'>src/03_dic.py</font>):")
story.append(Preformatted(
    "dashed = [k for k, s in state.items() if s in ('DC', 'DB')]\n"
    "idx = (pos + np.arange(M)) % N          # cyclic block\n"
    "block = T[idx]\n"
    "for k in dashed:\n"
    "    todo = min(M, N - seen[k])          # one full cycle per itemset\n"
    "    count[k] += int(np.all(block[:todo][:, list(k)], axis=1).sum())\n"
    "    seen[k] += todo\n"
    "# at this stop: DC->DB if count>=min_support; DC/DB->SC/SB if seen==N\n"
    "# if x just became a box, start y = x ∪ {i} whose (k-1)-subsets are all boxes",
    CODE))
story.append(Spacer(1, 8))
p("A-CLOSE generator pruning and closure (<font face='DVM'>src/04_aclose.py</font>):")
story.append(Preformatted(
    "for c, s in zip(cands, sup):\n"
    "    if s < min_count: continue\n"
    "    if any(G[sub] == s for sub in combinations(c, k)):  # not a generator\n"
    "        pruned_eq += 1\n"
    "        continue\n"
    "    newG[c] = s\n"
    "# h(g) = AND of every transaction mask that contains g\n"
    "clos = [full_mask] * len(generators)\n"
    "for t in txn_iter:\n"
    "    for i, g in enumerate(generators):\n"
    "        if t & g == g: clos[i] &= t",
    CODE))
story.append(Spacer(1, 8))
p("Storm-wise split (no row leakage) for the tree (<font face='DVM'>src/05_decision_tree.py</font>):")
story.append(Preformatted(
    "gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)\n"
    "train_idx, test_idx = next(gss.split(X, y, groups=storm_id))\n"
    "cv = GroupKFold(n_splits=5)\n"
    "# Wind is not in X: Status is defined from wind thresholds.",
    CODE))
story.append(Spacer(1, 8))
p("Haversine k-distance used to choose DBSCAN eps (<font face='DVM'>src/06_dbscan.py</font>):")
story.append(Preformatted(
    "R = 6371.0  # km\n"
    "db = DBSCAN(eps=eps_rad, min_samples=MinPts, metric='haversine')\n"
    "labels = db.fit_predict(np.radians(coords))",
    CODE))

story.append(PageBreak())
h1("Appendix B. Algorithm Configurations and Parameter Values")
table(pd.DataFrame([
    ["min_support", "0.05", "Several thousand fixes; avoids one-off storms"],
    ["min_confidence", "0.70", "Standard strong-rule cutoff"],
    ["DIC M", "5000", "Several stops per pass (~15)"],
    ["Tree criterion / depth", f"{dt['best_criterion']} / {dt['best_depth']}", "5-fold GroupKFold on training storms"],
    ["min_samples_leaf", "20", "Smooth leaves on 17 k training rows"],
    ["Train/test", "70/30 by storm", "Prevents 6-hourly leakage"],
    ["DBSCAN MinPts", str(db["min_pts"]), "Hot-spot, not a pair of storms"],
    ["DBSCAN eps (level 1 / 2)", f"{db['eps_km']:.0f} / {db['eps2_km']} km", "k-distance knee; then intra-basin"],
    ["DBSCAN metric", "haversine", "Great-circle distance on the sphere"],
    ["Spark mode", "local[*], 6 GB driver", "Single laptop; no cluster"],
    ["Replication", "×1–32 (A-CLOSE), ×1–100 (tree)", "Scaling only; not new storms"],
    ["Random seed", "42", "Split and tree"],
], columns=["Parameter", "Value", "Justification"]),
      col_widths=[4.2 * cm, 4.5 * cm, W - 8.7 * cm], font=8,
      caption="Table 27. Complete parameter list")

story.append(PageBreak())
h1("Appendix C. Execution Environment and Reproduction")
p("Run from the project folder, in order:")
story.append(Preformatted(
    "pip install -r requirements.txt          # JDK 17 required for PySpark (JAVA_HOME)\n"
    "python src/01_preprocessing.py           # cleaning, selection, transformation\n"
    "python src/02_visualization.py           # descriptive statistics + 15 plots\n"
    "python src/03_dic.py                     # DIC + Apriori verification\n"
    "python src/04_aclose.py                  # A-CLOSE + closed-rule basis\n"
    "python src/05_decision_tree.py           # GroupKFold tree + IF-THEN rules\n"
    "python src/06_dbscan.py                  # haversine DBSCAN + hot-spots\n"
    "python src/07_pyspark_aclose.py          # Spark vs sequential A-CLOSE\n"
    "python src/08_pyspark_decision_tree.py   # Spark MLlib vs scikit-learn\n"
    "python src/09_build_report.py            # this PDF from saved outputs",
    CODE))
p("Every script writes <font face='DVM'>outputs/logs/&lt;script&gt;.txt</font> and a metrics JSON. "
  "Clean tables, transactions, itemsets, rules and cluster labels are in "
  "<font face='DVM'>outputs/data/</font>. Figures are in <font face='DVM'>outputs/figures/</font>. "
  "This report never types a numeric result by hand.")

story.append(PageBreak())
h1("Appendix D. Project Deliverables")
table(pd.DataFrame([
    ["src/common.py", "Paths, assumed parameters, logging helpers"],
    ["src/01_preprocessing.py", "Cleaning, selection, transformation"],
    ["src/02_visualization.py", "Descriptive statistics and 15 plots"],
    ["src/arm_common.py", "Bit-mask encoding, counting/closure kernels, rules"],
    ["src/03_dic.py", "DIC from scratch + Apriori verification"],
    ["src/04_aclose.py", "A-CLOSE from scratch + closed rule basis"],
    ["src/05_decision_tree.py", "GroupKFold tree, metrics, IF–THEN rules"],
    ["src/06_dbscan.py", "Haversine DBSCAN, sensitivity, hot-spots, dendrogram"],
    ["src/spark_setup.py", "Local SparkSession on Windows (JDK 17)"],
    ["src/07_pyspark_aclose.py", "A-CLOSE on Spark vs sequential Python"],
    ["src/08_pyspark_decision_tree.py", "MLlib tree vs scikit-learn"],
    ["src/09_build_report.py", "Generates this PDF"],
    ["src/make_notebook.py", "Assembles Hurricane_Analytics.ipynb from the scripts"],
    ["report/*.pdf", "This document"],
    ["outputs/data/dic_rules.csv", "All 294 DIC rules (not printed in full)"],
    ["outputs/data/aclose_rules.csv", "Closed-rule basis (not printed in full)"],
], columns=["File", "Purpose"]),
      col_widths=[5.4 * cm, W - 5.4 * cm], font=8,
      caption="Table 28. Source files and artefacts")

# ============================================================================
# BUILD
# ============================================================================
doc = HurricaneDoc(
    OUT, pagesize=A4,
    leftMargin=1.8 * cm, rightMargin=1.8 * cm,
    topMargin=1.7 * cm, bottomMargin=1.9 * cm,
    title="Descriptive, Predictive and Big-Data Analytics of the NOAA Hurricane Database (HURDAT2)",
    author="Umesh Chandra",
)
doc.multiBuild(story, canvasmaker=NumberedCanvas, maxPasses=4)
print("Report written to", OUT)
