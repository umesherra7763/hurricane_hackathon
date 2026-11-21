"""
arm_common.py
-------------
Helpers shared by DIC (03), A-CLOSE (04) and the PySpark A-CLOSE (08):
  * bitmask encoding of transactions (an itemset = a Python int, item j = bit j)
  * support counting / closure kernels written in plain Python (the SAME kernels
    are executed sequentially and inside Spark partitions, so the benchmark is fair)
  * association-rule generation from frequent itemsets
"""
from itertools import combinations


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
def build_vocab(transactions):
    """Sorted list of items and item -> bit index map."""
    items = sorted({i for t in transactions for i in t})
    return items, {it: k for k, it in enumerate(items)}


def to_mask(itemset, index):
    m = 0
    for it in itemset:
        m |= 1 << index[it]
    return m


def mask_to_items(mask, items):
    return frozenset(items[k] for k in range(len(items)) if mask >> k & 1)


def encode(transactions, index):
    return [to_mask(t, index) for t in transactions]


# ---------------------------------------------------------------------------
# Kernels (pure Python on an iterable of int transactions)
# ---------------------------------------------------------------------------
def count_kernel(txn_iter, candidates):
    """Return a list: number of transactions containing each candidate mask."""
    counts = [0] * len(candidates)
    for t in txn_iter:
        for i, c in enumerate(candidates):
            if t & c == c:
                counts[i] += 1
    return counts


def closure_kernel(txn_iter, generators, full_mask):
    """For each generator g return AND of all transactions containing g
    (= Galois closure h(g)); full_mask if no transaction contains g."""
    clos = [full_mask] * len(generators)
    for t in txn_iter:
        for i, g in enumerate(generators):
            if t & g == g:
                clos[i] &= t
    return clos


# ---------------------------------------------------------------------------
# Rule generation from a dict {frozenset itemset: support_count}
# ---------------------------------------------------------------------------
def generate_rules(freq, n, min_conf):
    """All rules X -> Y with X u Y frequent, X,Y non-empty, conf >= min_conf.
    Returns list of dicts sorted by lift then confidence."""
    rules = []
    for itemset, sup in freq.items():
        if len(itemset) < 2:
            continue
        for r in range(1, len(itemset)):
            for lhs in combinations(sorted(itemset), r):
                lhs = frozenset(lhs)
                rhs = itemset - lhs
                conf = sup / freq[lhs]
                if conf >= min_conf:
                    lift = conf / (freq[rhs] / n)
                    rules.append({"antecedent": lhs, "consequent": rhs, "support": sup / n,
                                  "confidence": conf, "lift": lift})
    rules.sort(key=lambda r: (-r["lift"], -r["confidence"]))
    return rules


def fmt_set(s):
    return "{" + ", ".join(sorted(s)) + "}"


def rules_to_frame(rules):
    import pandas as pd
    return pd.DataFrame([{"IF (antecedent)": fmt_set(r["antecedent"]),
                          "THEN (consequent)": fmt_set(r["consequent"]),
                          "support": round(r["support"], 4),
                          "confidence": round(r["confidence"], 4),
                          "lift": round(r["lift"], 3)} for r in rules])


def apriori_gen(level_sets, k):
    """Classic Apriori join + prune on a collection of sorted k-tuples -> (k+1)-tuples."""
    level = sorted(level_sets)
    lset = set(level)
    out = []
    for i in range(len(level)):
        for j in range(i + 1, len(level)):
            a, b = level[i], level[j]
            if a[:k - 1] != b[:k - 1]:
                break                    # sorted: no further joins with prefix a[:k-1]
            cand = a + (b[-1],)
            if all(sub in lset for sub in combinations(cand, k)):
                out.append(cand)
    return out
