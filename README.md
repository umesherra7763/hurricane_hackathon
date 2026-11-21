# Hurricane Database – Descriptive & Predictive Analytics (CS5010 End-sem)

Dataset: <https://www.kaggle.com/noaa/hurricane-database> (`hurricane/atlantic.csv`, `hurricane/pacific.csv`)

Final report: **`report/Hurricane_Analytics_Report.pdf`**

Notebook with all code, outputs and interpretations: **`Hurricane_Analytics.ipynb`**
(HTML copy for viewing without Jupyter: `Hurricane_Analytics.html`). The notebook is assembled from
the scripts in `src/` by `src/make_notebook.py`, so both run identical code.

## Layout
```
hurricane/        raw Kaggle files
src/              source code (run in numeric order)
outputs/data/     cleaned data, transactions, itemsets, rules, cluster labels (CSV)
outputs/figures/  all plots (PNG)
outputs/logs/     console output of every script (*.txt) + metrics used by the report (*.json)
report/           generated PDF
```

## How to run (from this folder)
```
pip install -r requirements.txt          # needs a JDK 17 for PySpark (JAVA_HOME)
python src/01_preprocessing.py           # (i) cleaning (ii) selection (iii) transformation
python src/02_visualization.py           # descriptive statistics + 15 plots
python src/03_dic.py                     # DIC association rules (verified vs Apriori)
python src/04_aclose.py                  # A-CLOSE closed itemsets + closed rules
python src/05_decision_tree.py           # Decision tree classifier + IF-THEN rules
python src/06_dbscan.py                  # DBSCAN genesis clustering + dendrogram
python src/07_pyspark_aclose.py          # PySpark test drive 1: A-CLOSE on Spark
python src/08_pyspark_decision_tree.py   # PySpark test drive 2: MLlib decision tree
python src/09_build_report.py            # builds the PDF from the saved outputs
```

## Assumed parameters
min_support 5 %, min_confidence 70 %, DIC block size 5000; decision tree depth/criterion by
GroupKFold CV with a storm-wise 70/30 split; DBSCAN haversine, MinPts 15, eps from k-distance knee.

Note: on Windows with the Microsoft-Store Python 3.13, Spark's Python workers cannot start, so both
Spark programs use JVM-side Spark SQL / MLlib APIs only (no Python UDFs or RDD lambdas).
