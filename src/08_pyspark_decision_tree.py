"""
08_pyspark_decision_tree.py
---------------------------
PySpark test-drive #2: Decision Tree classifier with Spark MLlib.

Same task, features, storm-wise train/test split and hyper-parameters as
05_decision_tree.py (gini, max depth 8, min 20 samples per leaf):
  1. Quality check  - Spark MLlib tree vs scikit-learn tree on the same test storms.
  2. Big-data run   - the training set is replicated x1 .. x100 (17.6 k -> 1.76 M rows)
                      and the training time of scikit-learn (single core, in-memory)
                      is compared with Spark MLlib local[*] (all cores, partitioned).
  3. Core scaling   - Spark training time at the largest size with 1, 2, 4, 8 cores.
MLlib's tree learner runs in the JVM, so no Python workers are required.
"""
import importlib
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier as SkTree
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score

from common import FIG_DIR, RANDOM_STATE, Logger, save_json
from spark_setup import get_spark
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import functions as F

dt05 = importlib.import_module("05_decision_tree")
FEATURES, CLASSES = dt05.FEATURES, dt05.CLASSES
log = Logger("08_pyspark_decision_tree")
DEPTH, MIN_LEAF = 8, 20
LABEL = {c: float(i) for i, c in enumerate(CLASSES)}

d = dt05.build_dataset()
X, y, groups = d[FEATURES].values, d["Status"].values, d["ID"].values
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE).split(X, y, groups))
train_pd = d.iloc[tr][FEATURES + ["Status"]].assign(label=lambda x: x.Status.map(LABEL)).drop(columns="Status")
test_pd = d.iloc[te][FEATURES + ["Status"]].assign(label=lambda x: x.Status.map(LABEL)).drop(columns="Status")
log(f"Train rows {len(train_pd)}, test rows {len(test_pd)} (same storm-wise split as 05_decision_tree.py)")

def assembler():
    """Created after a SparkSession exists (MLlib objects need a live JVM)."""
    return VectorAssembler(inputCols=FEATURES, outputCol="features")


def spark_tree():
    return DecisionTreeClassifier(labelCol="label", featuresCol="features", impurity="gini",
                                  maxDepth=DEPTH, minInstancesPerNode=MIN_LEAF, maxBins=64, seed=RANDOM_STATE)


def replicated(spark, base, k, parts):
    df = (base.crossJoin(spark.range(k).withColumnRenamed("id", "rep")).drop("rep")
          .repartition(parts).cache())
    n = df.count()
    return df, n


# ======================================================================
# 1. Quality: Spark vs scikit-learn on the same split
# ======================================================================
spark = get_spark("dt-spark", cores="*")
cores = spark.sparkContext.defaultParallelism
train_sdf = assembler().transform(spark.createDataFrame(train_pd)).select("features", "label").cache()
test_sdf = assembler().transform(spark.createDataFrame(test_pd)).select("features", "label").cache()
train_sdf.count(); test_sdf.count()

model = spark_tree().fit(train_sdf)
pred = model.transform(test_sdf)
ev = lambda m: MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                metricName=m).evaluate(pred)
spark_metrics = {m: round(ev(m), 4) for m in ("accuracy", "f1", "weightedPrecision", "weightedRecall")}
sk = SkTree(criterion="gini", max_depth=DEPTH, min_samples_leaf=MIN_LEAF, random_state=RANDOM_STATE)
sk.fit(train_pd[FEATURES].values, train_pd.label.values)
skp = sk.predict(test_pd[FEATURES].values)
sk_metrics = {"accuracy": round(accuracy_score(test_pd.label, skp), 4),
              "f1": round(f1_score(test_pd.label, skp, average="weighted"), 4)}
model_depth, model_nodes = model.depth, model.numNodes          # read while the JVM is alive
log(f"\nSpark MLlib tree : {spark_metrics}  (depth={model_depth}, nodes={model_nodes})")
log(f"scikit-learn tree: {sk_metrics}")

# per-class recall + confusion matrix from Spark predictions
pp = pred.select("label", "prediction").toPandas()
cm = pd.crosstab(pp.label.map(dict(enumerate(CLASSES))), pp.prediction.map(dict(enumerate(CLASSES))),
                 rownames=["actual"], colnames=["predicted"]).reindex(index=CLASSES, columns=CLASSES, fill_value=0)
log("Spark confusion matrix:\n" + cm.to_string())

imp = pd.Series(model.featureImportances.toArray(), index=FEATURES).sort_values(ascending=False).round(4)
log("Spark feature importance:\n" + imp.to_string())

# IF-THEN rules: Spark's debug string with real feature names
dbg = model.toDebugString
for i, f in reversed(list(enumerate(FEATURES))):
    dbg = dbg.replace(f"feature {i} ", f"{f} ")
for c, v in LABEL.items():
    dbg = dbg.replace(f"Predict: {v}", f"Predict: {c}")
lines = dbg.splitlines()
log(f"\nSpark tree rules (first 40 of {len(lines)} lines):\n" + "\n".join(lines[:40]))

# ======================================================================
# 2. Scalability: replicate training data, time fit()
# ======================================================================
SCALES = [1, 5, 10, 25, 50, 100]
spark_tree().fit(train_sdf)                                  # JVM warm-up
bench = []
for k in SCALES:
    big, n = replicated(spark, train_sdf, k, 2 * cores)
    t0 = time.perf_counter(); spark_tree().fit(big); ts = time.perf_counter() - t0
    big.unpersist()
    Xb = np.tile(train_pd[FEATURES].values, (k, 1)); yb = np.tile(train_pd.label.values, k)
    t0 = time.perf_counter()
    SkTree(criterion="gini", max_depth=DEPTH, min_samples_leaf=MIN_LEAF, random_state=RANDOM_STATE).fit(Xb, yb)
    tk = time.perf_counter() - t0
    bench.append({"scale": k, "rows": n, "sklearn_s": round(tk, 2), "spark_s": round(ts, 2),
                  "speedup": round(tk / ts, 2)})
    log(f"x{k:<4d} rows={n:>9,d}  scikit-learn={tk:7.2f}s  Spark local[{cores}]={ts:7.2f}s  "
        f"speed-up={tk / ts:5.2f}")
spark.stop()

# ======================================================================
# 3. Core scaling at the largest size
# ======================================================================
scaling = []
for c in (1, 2, 4, cores):
    sp = get_spark(f"dt-{c}", cores=c)
    base = assembler().transform(sp.createDataFrame(train_pd)).select("features", "label").cache()
    base.count()
    spark_tree().fit(base)                                   # warm-up
    big, n = replicated(sp, base, SCALES[-1], max(2 * c, 4))
    t0 = time.perf_counter(); spark_tree().fit(big); t = time.perf_counter() - t0
    scaling.append({"cores": c, "seconds": round(t, 2)})
    log(f"[core scaling] x{SCALES[-1]} ({n:,} rows), local[{c}] -> {t:.2f}s")
    sp.stop()
sc = pd.DataFrame(scaling); sc["speedup_vs_1core"] = (sc.seconds.iloc[0] / sc.seconds).round(2)
log("\nCore scaling:\n" + sc.to_string(index=False))
b = pd.DataFrame(bench)
log("\nBenchmark:\n" + b.to_string(index=False))

# ---------------- plots ---------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(b.rows / 1e6, b.sklearn_s, "o-", color="#eb6834", lw=2, label="scikit-learn (1 core, in-memory)")
ax.plot(b.rows / 1e6, b.spark_s, "o-", color="#2a78d6", lw=2, label=f"Spark MLlib local[{cores}]")
ax.set(xlabel="Training rows (millions)", ylabel="Training time (s)",
       title="Decision tree training time vs data size")
ax.legend(frameon=False); ax.grid(color="#e1e0d9"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIG_DIR}/spark_dt_runtime.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(sc.cores, sc.speedup_vs_1core, "o-", color="#2a78d6", lw=2, label="measured")
ax.plot(sc.cores, sc.cores, "--", color="#888", lw=1, label="ideal (linear)")
ax.set(xlabel="Spark cores (local[n])", ylabel="Speed-up vs 1 core",
       title=f"Spark decision tree: core scaling (x{SCALES[-1]} data)")
ax.set_xticks(sc.cores)
ax.legend(frameon=False); ax.grid(color="#e1e0d9"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIG_DIR}/spark_dt_cores.png", dpi=150); plt.close(fig)

save_json({"spark_metrics": spark_metrics, "sk_metrics": sk_metrics, "depth": model_depth,
           "nodes": model_nodes, "importance": imp.to_dict(), "bench": bench, "core_scaling": scaling,
           "cores": cores, "cm": cm.values.tolist(), "rules_head": lines[:40]}, "spark_dt")
log.close()
