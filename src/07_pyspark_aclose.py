"""
07_pyspark_aclose.py
--------------------
PySpark test-drive #1: A-CLOSE closed-itemset mining on Spark.

The A-CLOSE driver (generator levels, pruning, closure phase) is the SAME function
used in 04_aclose.py; only the data-access back-end changes:

  sequential : support counts / closures computed by a Python loop over all
               transactions (arm_common.count_kernel / closure_kernel)
  Spark      : transactions are a cached, partitioned Spark DataFrame of 64-bit
               item bitmasks. ONE distributed aggregation job per level computes
                 count(c)   = SUM( (mask & c) == c )            for every candidate c
                 closure(g) = BIT_AND( mask  WHERE (mask & g) == g ) for every generator g
               executed in parallel by all cores (Catalyst + whole-stage code-gen).
               (Spark's Python workers cannot run on this Windows/Python 3.13 build,
               so the job is expressed with DataFrame operations that run in the JVM.)

Big-data experiment: the 74,895 transactions are replicated x1 .. x32 (up to 2.4 M)
to emulate growing data volume; relative supports are unchanged, so every run must
return the identical set of closed itemsets - this is verified.
"""
import importlib
import math
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import MIN_SUPPORT, FIG_DIR, Logger, save_json, load_transactions
from arm_common import build_vocab, encode, count_kernel, closure_kernel, mask_to_items
from spark_setup import get_spark
from pyspark.sql import functions as F

aclose = importlib.import_module("04_aclose").aclose
log = Logger("07_pyspark_aclose")
CHUNK = 150                      # aggregate expressions per Spark job (keeps code-gen small)


def spark_backend(df, full_mask, stats):
    """Return (count_fn, closure_fn) running on the cached Spark DataFrame `df`."""
    m = F.col("mask")

    def count_fn(cands):
        stats["jobs"] += 1
        out = []
        for i in range(0, len(cands), CHUNK):
            part = cands[i:i + CHUNK]
            exprs = [F.sum(F.when(m.bitwiseAND(F.lit(c)) == F.lit(c), 1).otherwise(0)).alias(f"c{j}")
                     for j, c in enumerate(part)]
            row = df.agg(*exprs).collect()[0]
            out += [int(row[f"c{j}"] or 0) for j in range(len(part))]
        return out

    def closure_fn(gens):
        stats["jobs"] += 1
        out = []
        for i in range(0, len(gens), CHUNK):
            part = gens[i:i + CHUNK]
            exprs = [F.bit_and(F.when(m.bitwiseAND(F.lit(g)) == F.lit(g), m)).alias(f"g{j}")
                     for j, g in enumerate(part)]
            row = df.agg(*exprs).collect()[0]
            out += [int(row[f"g{j}"]) if row[f"g{j}"] is not None else full_mask
                    for j in range(len(part))]
        return out

    return count_fn, closure_fn


def run_sequential(enc, I, min_count):
    full = (1 << I) - 1
    t0 = time.perf_counter()
    _, closed, _, _ = aclose(lambda c: count_kernel(enc, c), lambda g: closure_kernel(enc, g, full),
                             I, min_count, log=lambda *a: None)
    return time.perf_counter() - t0, closed


def run_spark(spark, base_sdf, scale, I, min_count, partitions):
    df = (base_sdf.crossJoin(spark.range(scale).withColumnRenamed("id", "rep"))
          .select("mask").repartition(partitions).cache())
    n = df.count()                                   # materialise the cache (not timed)
    stats = {"jobs": 0}
    cnt, clo = spark_backend(df, (1 << I) - 1, stats)
    t0 = time.perf_counter()
    _, closed, _, _ = aclose(cnt, clo, I, min_count, log=lambda *a: None)
    t = time.perf_counter() - t0
    df.unpersist()
    return t, closed, n


if __name__ == "__main__":
    txns = load_transactions()
    items, index = build_vocab(txns)
    I, N = len(items), len(txns)
    enc = encode(txns, index)
    log(f"Base data: {N} transactions, {I} items, min_support = {MIN_SUPPORT:.0%}")

    SEQ_SCALES = [1, 2, 4, 8]
    SPARK_SCALES = [1, 2, 4, 8, 16, 32]
    results = []

    # ---------------- sequential (pure Python, single core) ---------------
    seq_closed = {}
    for s in SEQ_SCALES:
        mc = math.ceil(MIN_SUPPORT * N * s)
        t, closed = run_sequential(enc * s, I, mc)
        seq_closed[s] = closed
        results.append({"engine": "Python sequential", "scale": s, "rows": N * s, "seconds": round(t, 2),
                        "n_closed": len(closed)})
        log(f"[sequential] x{s:<3d} rows={N * s:>9,d}  time={t:7.2f}s  CFIs={len(closed)}")

    # ---------------- Spark local[*] (all cores) --------------------------
    spark = get_spark("aclose-spark", cores="*")
    cores = spark.sparkContext.defaultParallelism
    base = spark.createDataFrame(pd.DataFrame({"mask": np.array(enc, dtype="int64")})).cache()
    base.count()
    run_spark(spark, base, 1, I, math.ceil(MIN_SUPPORT * N), 2 * cores)          # JVM warm-up
    spark_closed = {}
    for s in SPARK_SCALES:
        mc = math.ceil(MIN_SUPPORT * N * s)
        t, closed, n = run_spark(spark, base, s, I, mc, 2 * cores)
        spark_closed[s] = closed
        ok = (closed == seq_closed[s]) if s in seq_closed else (
            {k: v // s for k, v in closed.items()} == seq_closed[1])
        results.append({"engine": f"Spark local[{cores}]", "scale": s, "rows": n, "seconds": round(t, 2),
                        "n_closed": len(closed), "identical": ok})
        log(f"[spark {cores} cores] x{s:<3d} rows={n:>9,d}  time={t:7.2f}s  CFIs={len(closed)}  "
            f"identical to sequential = {ok}")
    spark.stop()

    # ---------------- core scaling at a fixed size (x8) --------------------
    scaling = []
    for c in (1, 2, 4, cores):
        sp = get_spark(f"aclose-{c}", cores=c)
        b = sp.createDataFrame(pd.DataFrame({"mask": np.array(enc, dtype="int64")})).cache(); b.count()
        run_spark(sp, b, 1, I, math.ceil(MIN_SUPPORT * N), 2 * c)                # warm-up
        t, closed, n = run_spark(sp, b, 8, I, math.ceil(MIN_SUPPORT * N * 8), max(2 * c, 4))
        scaling.append({"cores": c, "seconds": round(t, 2)})
        log(f"[core scaling] x8 data, local[{c}] -> {t:.2f}s")
        sp.stop()

    res = pd.DataFrame(results)
    log("\nRESULTS\n" + res.to_string(index=False))
    seq = res[res.engine == "Python sequential"].set_index("scale").seconds
    spk = res[res.engine.str.startswith("Spark")].set_index("scale").seconds
    speed = (seq / spk).dropna().round(2)
    log("\nSpeed-up (sequential / Spark) by scale:\n" + speed.to_string())
    sc = pd.DataFrame(scaling)
    sc["speedup_vs_1core"] = (sc.seconds.iloc[0] / sc.seconds).round(2)
    log("\nCore scaling (x8 data):\n" + sc.to_string(index=False))

    # ---------------- plots ---------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(seq.index * N / 1e6, seq.values, "o-", color="#eb6834", lw=2, label="Python sequential (1 core)")
    ax.plot(spk.index * N / 1e6, spk.values, "o-", color="#2a78d6", lw=2, label=f"PySpark local[{cores}]")
    for x, y in zip(spk.index * N / 1e6, spk.values):
        ax.annotate(f"{y:.1f}s", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    for x, y in zip(seq.index * N / 1e6, seq.values):
        ax.annotate(f"{y:.0f}s", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax.set(xlabel="Transactions (millions)", ylabel="A-CLOSE run time (s)",
           title="A-CLOSE: sequential Python vs PySpark")
    ax.legend(frameon=False); ax.grid(color="#e1e0d9"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/spark_aclose_runtime.png", dpi=150); plt.close(fig)

    save_json({"results": results, "speedup": speed.to_dict(), "core_scaling": scaling, "cores": cores,
               "base_rows": N}, "spark_aclose")
    log.close()
