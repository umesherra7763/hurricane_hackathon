"""
04_aclose.py
------------
Closed Frequent Itemsets (CFIs) and closed association rules with the A-CLOSE
algorithm (Pasquier, Bastide, Taouil & Lakhal, ICDT 1999) - implemented from scratch.

A-CLOSE works on GENERATORS (minimal / key itemsets):
  1. G1 = frequent 1-itemsets.
  2. Level k -> k+1: candidates = Apriori-join of G_k (every k-subset must be in G_k);
     count support; drop infrequent; drop candidate c if supp(c) == supp(s) for a
     k-subset s in G_k (then c is NOT a generator - it lies in the closure of s).
     The first level where such a drop happens is recorded as level i.
  3. Closure phase (one extra pass): for every generator g of level >= i-1,
     closure h(g) = intersection of all transactions containing g.
     Generators of lower levels are already closed (h(g) = g).
  4. FC = { h(g) }  with supp(h(g)) = supp(g).

Rules (closed-itemset based, Bastide et al. 2000 min-max basis):
  * EXACT rules      g -> h(g)\\g            (confidence 100 %)
  * APPROXIMATE rules g -> C\\g for closed C strictly containing h(g), conf = supp(C)/supp(g)

The support-counting / closure kernels come from arm_common.py and are the same
ones run inside Spark partitions in 08_pyspark_aclose.py.
Parameters: min_support = 5 %, min_confidence = 70 %.
Verification: CFIs are compared with the closed sets derived from DIC's frequent itemsets.
"""
import time
from itertools import combinations
import pandas as pd

from common import MIN_SUPPORT, MIN_CONFIDENCE, CLEAN_DIR, Logger, save_json, load_transactions
from arm_common import (build_vocab, encode, count_kernel, closure_kernel, apriori_gen,
                        mask_to_items, fmt_set)


def aclose(count_fn, closure_fn, n_items, min_count, log=print):
    """
    Generic A-CLOSE driver. The data access is abstracted:
      count_fn(list_of_masks)   -> list of support counts
      closure_fn(list_of_masks) -> list of closure masks
    so the same driver runs sequentially (Python) or distributed (Spark).
    Itemsets are handled as sorted tuples of item indices internally.
    Returns generators {tuple: support}, closed {mask: support}, gen_closure {tuple: mask}, level_i
    """
    tup2mask = lambda t: sum(1 << j for j in t)
    # ---- level 1 --------------------------------------------------------
    c1 = [(j,) for j in range(n_items)]
    sup = count_fn([tup2mask(c) for c in c1])
    G = {c: s for c, s in zip(c1, sup) if s >= min_count}
    generators = dict(G)
    log(f"   level 1: {n_items} candidates -> {len(G)} frequent generators")
    level_i, k = None, 1
    # ---- levels k+1 -----------------------------------------------------
    while G:
        cands = apriori_gen(G.keys(), k)
        if not cands:
            break
        sup = count_fn([tup2mask(c) for c in cands])
        newG, pruned_eq, infreq = {}, 0, 0
        for c, s in zip(cands, sup):
            if s < min_count:
                infreq += 1
                continue
            # not a generator if it has the same support as one of its k-subsets
            if any(G[sub] == s for sub in combinations(c, k)):
                pruned_eq += 1
                if level_i is None:
                    level_i = k + 1
                continue
            newG[c] = s
        log(f"   level {k + 1}: {len(cands)} candidates -> {infreq} infrequent, "
            f"{pruned_eq} non-generators (same support as a subset), {len(newG)} generators")
        generators.update(newG)
        G, k = newG, k + 1
    # ---- closure computation -------------------------------------------
    if level_i is None:
        need = []                                    # every generator is closed
    else:
        need = [g for g in generators if len(g) >= level_i - 1]
    log(f"   first level with a non-generator: i = {level_i}; computing closures of "
        f"{len(need)} generators (levels >= {level_i - 1 if level_i else '-'}) in one pass")
    clos_masks = closure_fn([tup2mask(g) for g in need]) if need else []
    gen_closure = {g: tup2mask(g) for g in generators}
    gen_closure.update(dict(zip(need, clos_masks)))
    closed = {}
    for g, m in gen_closure.items():
        closed[m] = generators[g]
    return generators, closed, gen_closure, level_i


def closed_rules(generators, gen_closure, closed, n, items, min_conf):
    """Exact + approximate rules derived from generators and closed itemsets."""
    exact, approx = [], []
    sup_closed = closed
    for g, sg in generators.items():
        gm = sum(1 << j for j in g)
        hg = gen_closure[g]
        lhs = mask_to_items(gm, items)
        if hg != gm:                                         # exact rule
            rhs = mask_to_items(hg & ~gm, items)
            exact.append({"antecedent": lhs, "consequent": rhs, "support": sg / n,
                          "confidence": 1.0, "type": "exact"})
        for cm, sc in sup_closed.items():                    # approximate rules
            if cm != hg and cm & hg == hg:                   # C strictly contains h(g)
                conf = sc / sg
                if conf >= min_conf:
                    approx.append({"antecedent": lhs, "consequent": mask_to_items(cm & ~gm, items),
                                   "support": sc / n, "confidence": conf, "type": "approximate"})
    return exact, approx


if __name__ == "__main__":
    log = __import__("common").Logger("04_aclose")
    txns = load_transactions()
    items, index = build_vocab(txns)
    N, I = len(txns), len(items)
    enc = encode(txns, index)
    full = (1 << I) - 1
    import math
    min_count = math.ceil(MIN_SUPPORT * N)
    log(f"N = {N} transactions, {I} items; min_support = {MIN_SUPPORT:.0%} (count {min_count}), "
        f"min_confidence = {MIN_CONFIDENCE:.0%}")

    passes = {"n": 0}
    def count_fn(c):
        passes["n"] += 1
        return count_kernel(enc, c)
    def closure_fn(g):
        passes["n"] += 1
        return closure_kernel(enc, g, full)

    t0 = time.perf_counter()
    gens, closed, gen_clos, lvl_i = aclose(count_fn, closure_fn, I, min_count, log)
    t_ac = time.perf_counter() - t0
    log(f"\nA-CLOSE finished in {t_ac:.2f} s using {passes['n']} database passes")
    log(f"Generators: {len(gens)};  Closed frequent itemsets (CFIs): {len(closed)}")
    by_len = pd.Series([bin(m).count("1") for m in closed]).value_counts().sort_index()
    log(f"CFIs by size: {by_len.to_dict()}")

    # ---- verification: closed sets from DIC frequent itemsets -------------
    fi = pd.read_csv(f"{CLEAN_DIR}/dic_frequent_itemsets.csv")
    freq = {frozenset(s.strip("{}").split(", ")): c for s, c in zip(fi.itemset, fi.support_count)}
    brute_closed = {s for s, c in freq.items()
                    if not any(len(t) == len(s) + 1 and s < t and freq[t] == c for t in freq)}
    ac_sets = {mask_to_items(m, items): s for m, s in closed.items()}
    ok = set(ac_sets) == brute_closed and all(ac_sets[s] == freq[s] for s in ac_sets)
    log(f"Verification: closed itemsets derived from DIC/Apriori frequent itemsets = {len(brute_closed)}; "
        f"A-CLOSE result identical = {ok}")
    log(f"Compression: {len(freq)} frequent itemsets are losslessly represented by {len(closed)} CFIs "
        f"({100 * (1 - len(closed) / len(freq)):.1f} % fewer)")

    cfi = pd.DataFrame([{"closed_itemset": fmt_set(s), "size": len(s), "support_count": c,
                         "support": round(c / N, 4)} for s, c in ac_sets.items()])
    cfi = cfi.sort_values(["size", "support"], ascending=[False, False])
    cfi.to_csv(f"{CLEAN_DIR}/aclose_closed_itemsets.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
    log("\nLargest closed frequent itemsets:")
    log(cfi.head(12).to_string(index=False))

    gdf = pd.DataFrame([{"generator": fmt_set(mask_to_items(sum(1 << j for j in g), items)),
                         "closure": fmt_set(mask_to_items(gen_clos[g], items)), "support_count": s}
                        for g, s in gens.items() if gen_clos[g] != sum(1 << j for j in g)])
    log(f"\nGenerators whose closure is larger than themselves: {len(gdf)} (first 10)")
    log(gdf.sort_values("support_count", ascending=False).head(10).to_string(index=False))

    # ---- closed association rules ---------------------------------------
    exact, approx = closed_rules(gens, gen_clos, closed, N, items, MIN_CONFIDENCE)
    def lift(r):   # consequent is a subset of a frequent itemset -> its support is known
        return r["confidence"] / (freq[frozenset(r["consequent"])] / N)
    for r in exact + approx:
        r["lift"] = lift(r)
    rows = [{"type": r["type"], "IF (antecedent)": fmt_set(r["antecedent"]),
             "THEN (consequent)": fmt_set(r["consequent"]), "support": round(r["support"], 4),
             "confidence": round(r["confidence"], 4), "lift": round(r["lift"], 3)}
            for r in exact + approx]
    rdf = pd.DataFrame(rows)
    rdf.to_csv(f"{CLEAN_DIR}/aclose_rules.csv", index=False)
    log(f"\nClosed-itemset rule basis: {len(exact)} exact rules (conf = 100 %) + "
        f"{len(approx)} approximate rules (conf >= {MIN_CONFIDENCE:.0%}) = {len(rdf)} rules "
        f"(DIC produced {pd.read_csv(f'{CLEAN_DIR}/dic_rules.csv').shape[0]} rules from all frequent itemsets)")
    log("\nEXACT rules (100 % confidence), by support:")
    log(rdf[rdf.type == "exact"].sort_values("support", ascending=False).head(15)
        .drop(columns="type").to_string(index=False))
    log("\nTop APPROXIMATE rules by lift:")
    log(rdf[rdf.type == "approximate"].sort_values(["lift", "confidence"], ascending=False).head(15)
        .drop(columns="type").to_string(index=False))

    save_json({"time_s": round(t_ac, 2), "passes": passes["n"], "n_generators": len(gens),
               "n_closed": len(closed), "closed_by_len": {int(k): int(v) for k, v in by_len.items()},
               "level_i": lvl_i, "verified": ok, "n_frequent": len(freq),
               "n_exact": len(exact), "n_approx": len(approx)}, "aclose")
    log.close()
