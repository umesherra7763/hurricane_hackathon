"""
03_dic.py
---------
Association Rule Mining with the DIC (Dynamic Itemset Counting) algorithm
(Brin, Motwani, Ullman & Tsur, SIGMOD 1997) - implemented from scratch.

Idea: the database is read in blocks of M transactions. Unlike Apriori (which
starts counting k-itemsets only after a full pass for (k-1)-itemsets), DIC can
START counting a new candidate at any block boundary ("stop") as soon as all its
immediate subsets are known/suspected to be frequent. Each itemset is in one of
four states:
    DC  dashed circle : being counted, currently below min-support (suspected small)
    DB  dashed box    : being counted, already reached min-support (suspected large)
    SC  solid circle  : counted over ALL transactions, infrequent
    SB  solid box     : counted over ALL transactions, frequent
An itemset is counted cyclically (wrapping around the file) until it has seen all
N transactions. The algorithm stops when no dashed itemsets remain.

Parameters (assumed & stated): min_support = 5 %, min_confidence = 70 %, M = 5000.
Verification: the frequent itemsets are compared with mlxtend's Apriori.
"""
import time
from itertools import combinations
import numpy as np
import pandas as pd

from common import (MIN_SUPPORT, MIN_CONFIDENCE, DIC_BLOCK_SIZE, CLEAN_DIR, Logger, save_json,
                    load_transactions)
from arm_common import generate_rules, rules_to_frame, fmt_set

log = Logger("03_dic")


def dic(T, min_count, M, item_names=None, trace=None):
    """
    T          : boolean numpy matrix (N transactions x I items)
    min_count  : absolute minimum support count
    M          : block size (transactions between stops)
    returns    : dict {tuple(sorted item idx): support count} of frequent itemsets,
                 number of passes over the data, number of candidates ever counted
    """
    N, I = T.shape
    # state tables, keyed by sorted tuple of item indices
    count, seen, state = {}, {}, {}
    for j in range(I):                              # all 1-itemsets start as dashed circles
        key = (j,)
        count[key], seen[key], state[key] = 0, 0, "DC"
    boxes = set()                                   # DB or SB itemsets
    pos, read, stop_no = 0, 0, 0

    def try_supersets(x):
        """x just became a box: start counting every superset x+{i} whose
        immediate subsets are all boxes (DB or SB)."""
        for i in range(I):
            if i in x:
                continue
            y = tuple(sorted(x + (i,)))
            if y in state:
                continue
            if all(s in boxes for s in combinations(y, len(y) - 1)):
                count[y], seen[y], state[y] = 0, 0, "DC"

    while True:
        dashed = [k for k, s in state.items() if s in ("DC", "DB")]
        if not dashed:
            break
        # ---- read the next block of M transactions (cyclic) ---------------
        idx = (pos + np.arange(M)) % N
        block = T[idx]
        for k in dashed:
            todo = min(M, N - seen[k])               # never count past one full cycle
            rows = block[:todo]
            count[k] += int(np.all(rows[:, list(k)], axis=1).sum())
            seen[k] += todo
        pos = (pos + M) % N
        read += M
        stop_no += 1
        # ---- state transitions at this stop -------------------------------
        newly_boxed = []
        for k in dashed:
            if state[k] == "DC" and count[k] >= min_count:
                state[k] = "DB"
                boxes.add(k)
                newly_boxed.append(k)
        for k in dashed:                             # completed a full cycle -> solid
            if seen[k] >= N:
                state[k] = "SB" if state[k] == "DB" else "SC"
        for k in newly_boxed:
            try_supersets(k)
        if trace is not None:
            sizes = pd.Series([len(k) for k, s in state.items() if s in ("DC", "DB")]).value_counts().to_dict()
            trace.append({"stop": stop_no, "passes": round(read / N, 3),
                          "DC": sum(s == "DC" for s in state.values()),
                          "DB": sum(s == "DB" for s in state.values()),
                          "SC": sum(s == "SC" for s in state.values()),
                          "SB": sum(s == "SB" for s in state.values()),
                          "active_by_size": sizes})
    freq = {k: count[k] for k, s in state.items() if s == "SB"}
    return freq, read / N, len(state)


if __name__ == "__main__":
    txns = load_transactions()
    items = sorted({i for t in txns for i in t})
    idx = {it: j for j, it in enumerate(items)}
    N = len(txns)
    T = np.zeros((N, len(items)), dtype=bool)
    for r, t in enumerate(txns):
        for it in t:
            T[r, idx[it]] = True
    min_count = int(np.ceil(MIN_SUPPORT * N))
    log(f"Transactions N = {N}, items I = {len(items)}")
    log(f"Parameters: min_support = {MIN_SUPPORT:.0%} (count {min_count}), "
        f"min_confidence = {MIN_CONFIDENCE:.0%}, block size M = {DIC_BLOCK_SIZE}")

    trace = []
    t0 = time.perf_counter()
    freq_idx, passes, n_cand = dic(T, min_count, DIC_BLOCK_SIZE, trace=trace)
    t_dic = time.perf_counter() - t0
    freq = {frozenset(items[j] for j in k): v for k, v in freq_idx.items()}

    log(f"\nDIC finished in {t_dic:.2f} s after {len(trace)} stops = {passes:.2f} passes over the data")
    log(f"Candidates counted: {n_cand};  frequent itemsets: {len(freq)}")
    by_len = pd.Series([len(k) for k in freq]).value_counts().sort_index()
    log("Frequent itemsets by size: " + by_len.to_dict().__str__())
    log("\nState evolution at selected stops (DC/DB = still counting, SC/SB = finished):")
    tr = pd.DataFrame(trace)
    log(tr.iloc[np.unique(np.linspace(0, len(tr) - 1, 12).astype(int))].to_string(index=False))

    # ---- verification against Apriori (mlxtend) ------------------------------
    from mlxtend.frequent_patterns import apriori
    t0 = time.perf_counter()
    ap = apriori(pd.DataFrame(T, columns=items), min_support=min_count / N, use_colnames=True)
    t_ap = time.perf_counter() - t0
    ap_dict = {frozenset(s): int(round(sup * N)) for s, sup in zip(ap.itemsets, ap.support)}
    same = ap_dict == freq
    log(f"\nVerification vs mlxtend Apriori: {len(ap_dict)} itemsets, identical result = {same}")
    max_k = max(len(k) for k in freq)
    log(f"Apriori needs {max_k + 1} level-wise passes (one per itemset size + final empty check)"
        f" vs DIC {passes:.2f} passes -> DIC saves {max_k + 1 - passes:.2f} database scans")

    # ---- frequent itemsets output --------------------------------------------
    fi = pd.DataFrame([{"itemset": fmt_set(k), "size": len(k), "support_count": v,
                        "support": round(v / N, 4)} for k, v in freq.items()])
    fi = fi.sort_values(["size", "support"], ascending=[True, False])
    fi.to_csv(f"{CLEAN_DIR}/dic_frequent_itemsets.csv", index=False)
    log("\nTop frequent itemsets of size >= 3:")
    log(fi[fi["size"] >= 3].sort_values("support", ascending=False).head(12).to_string(index=False))

    # ---- association rules --------------------------------------------------
    rules = generate_rules(freq, N, MIN_CONFIDENCE)
    rdf = rules_to_frame(rules)
    rdf.to_csv(f"{CLEAN_DIR}/dic_rules.csv", index=False)
    log(f"\n{len(rules)} association rules with confidence >= {MIN_CONFIDENCE:.0%} (saved dic_rules.csv)")
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
    log("\nTop 20 rules by lift:")
    log(rdf.head(20).to_string(index=False))
    # interesting rules with a Status consequent (excluding trivial ones)
    st = rdf[rdf["THEN (consequent)"].str.match(r"^\{Status=[A-Z]+\}$")]
    log("\nTop 15 rules predicting the storm Status (single consequent):")
    log(st.sort_values(["lift", "confidence"], ascending=False).head(15).to_string(index=False))

    save_json({"N": N, "items": len(items), "min_count": min_count, "time_s": round(t_dic, 2),
               "passes": round(passes, 2), "stops": len(trace), "candidates": n_cand,
               "n_frequent": len(freq), "by_len": {int(k): int(v) for k, v in by_len.items()},
               "apriori_match": same, "apriori_time_s": round(t_ap, 2), "max_k": max_k,
               "n_rules": len(rules)}, "dic")
    log.close()
