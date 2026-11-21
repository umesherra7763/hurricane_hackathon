"""
05_decision_tree.py
-------------------
Decision Tree classifier: predict the storm STATUS of an observation
(TD = tropical depression, TS = tropical storm, HU = hurricane, EX = extratropical)
from its position, season, pressure and motion.

Design decisions (stated in the report):
  * 'Maximum Wind' is EXCLUDED from the features - Status is defined from wind
    thresholds, so using it would be target leakage.
  * Only observations with a measured central pressure and a known motion vector
    are used (pressure is missing for most pre-1979 records).
  * Train/test split is done BY STORM (GroupShuffleSplit, 70/30) - consecutive 6-h
    fixes of one storm are nearly identical, so a random row split would leak.
  * Split criterion (entropy = information gain as in ID3/C4.5, or gini as in CART)
    and max_depth are chosen by 5-fold GroupKFold cross-validation on the training
    storms (the selected model is gini, depth 8).
Outputs: metrics, confusion matrix, feature importance, tree diagram, IF-THEN rules.
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, cohen_kappa_score)

from common import CLEAN_TRACKS, CLEAN_DIR, FIG_DIR, RANDOM_STATE, Logger, save_json

log = Logger("05_decision_tree")
CLASSES = ["TD", "TS", "HU", "EX"]
FEATURES = ["Latitude", "Longitude", "Month", "BasinCode", "Pressure", "dPressure6h",
            "Speed_kmh", "Heading", "HoursSinceGenesis"]


def build_dataset():
    df = pd.read_csv(CLEAN_TRACKS, parse_dates=["Datetime"])
    df = df.sort_values(["ID", "Datetime"])
    # 6-hour pressure tendency (transformation: differencing within a storm)
    df["dPressure6h"] = df.groupby("ID")["Pressure"].diff()
    df["BasinCode"] = (df["Basin"] == "Pacific").astype(int)          # label encoding
    d = df[df["Status"].isin(CLASSES)].dropna(subset=FEATURES)
    return d


if __name__ == "__main__":
    d = build_dataset()
    X, y, groups = d[FEATURES].values, d["Status"].values, d["ID"].values
    log(f"Modelling set: {len(d)} observations from {d.ID.nunique()} storms; features = {FEATURES}")
    log("Class distribution: " + str(d.Status.value_counts().to_dict()))

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    tr, te = next(gss.split(X, y, groups))
    log(f"Train: {len(tr)} rows / {len(set(groups[tr]))} storms;  Test: {len(te)} rows / "
        f"{len(set(groups[te]))} storms (no storm appears in both)")

    # ---- model selection: depth x criterion via GroupKFold CV ------------
    cv = GroupKFold(n_splits=5)
    depths = list(range(2, 21))
    curve = []
    for crit in ("entropy", "gini"):
        for dep in depths:
            m = DecisionTreeClassifier(criterion=crit, max_depth=dep, min_samples_leaf=20,
                                       random_state=RANDOM_STATE)
            sc = cross_val_score(m, X[tr], y[tr], groups=groups[tr], cv=cv, scoring="accuracy")
            m.fit(X[tr], y[tr])
            curve.append({"criterion": crit, "depth": dep, "cv_acc": sc.mean(), "cv_std": sc.std(),
                          "train_acc": accuracy_score(y[tr], m.predict(X[tr]))})
    cdf = pd.DataFrame(curve)
    best = cdf.loc[cdf.cv_acc.idxmax()]
    log("\nCross-validation (5-fold GroupKFold) - selected rows:")
    log(cdf[cdf.depth.isin([2, 3, 4, 6, 8, 10, 12, 15, 20])].round(4).to_string(index=False))
    log(f"Best: criterion={best.criterion}, max_depth={int(best.depth)}, CV accuracy={best.cv_acc:.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    e = cdf[cdf.criterion == best.criterion]
    ax.plot(e.depth, e.train_acc, color="#eb6834", lw=2, marker="o", ms=4, label="training accuracy")
    ax.plot(e.depth, e.cv_acc, color="#2a78d6", lw=2, marker="o", ms=4, label="5-fold CV accuracy")
    ax.fill_between(e.depth, e.cv_acc - e.cv_std, e.cv_acc + e.cv_std, color="#2a78d6", alpha=.15)
    ax.axvline(best.depth, color="#888", ls="--", lw=1)
    ax.set(xlabel="max_depth", ylabel="Accuracy", title=f"Decision tree: depth selection ({best.criterion})")
    ax.legend(frameon=False); ax.grid(color="#e1e0d9"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "dt_depth_curve.png"), dpi=150); plt.close(fig)

    # ---- final model -----------------------------------------------------
    clf = DecisionTreeClassifier(criterion=best.criterion, max_depth=int(best.depth),
                                 min_samples_leaf=20, random_state=RANDOM_STATE)
    t0 = time.perf_counter(); clf.fit(X[tr], y[tr]); t_fit = time.perf_counter() - t0
    pred = clf.predict(X[te])
    acc = accuracy_score(y[te], pred)
    mf1 = f1_score(y[te], pred, average="macro")
    wf1 = f1_score(y[te], pred, average="weighted")
    kappa = cohen_kappa_score(y[te], pred)
    base = pd.Series(y[tr]).value_counts(normalize=True).max()
    log(f"\nFINAL MODEL on held-out test storms: accuracy={acc:.4f}, macro-F1={mf1:.4f}, "
        f"weighted-F1={wf1:.4f}, Cohen kappa={kappa:.4f}  (majority-class baseline={base:.4f}); "
        f"leaves={clf.get_n_leaves()}, fit time={t_fit:.3f}s")
    rep = classification_report(y[te], pred, labels=CLASSES, digits=4)
    log("\nClassification report (precision / recall / F1 per class):\n" + rep)
    cm = confusion_matrix(y[te], pred, labels=CLASSES)
    log("Confusion matrix (rows = actual, cols = predicted) " + str(CLASSES) + "\n" + str(cm))

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES,
                cbar=False, ax=ax, linewidths=1, linecolor="white")
    ax.set(xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix (test acc = {acc:.3f})")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "dt_confusion_matrix.png"), dpi=150); plt.close(fig)

    imp = pd.Series(clf.feature_importances_, index=FEATURES).sort_values()
    log(f"\nFeature importance (total {best.criterion} impurity reduction):\n" + imp[::-1].round(4).to_string())
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(imp.index, imp.values, color="#2a78d6", height=.6)
    ax.set(xlabel="Importance", title="Decision tree feature importance")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color="#e1e0d9")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "dt_feature_importance.png"), dpi=150); plt.close(fig)

    # ---- interpretable shallow tree for IF-THEN rules ------------------------
    small = DecisionTreeClassifier(criterion=best.criterion, max_depth=4, min_samples_leaf=50,
                                   random_state=RANDOM_STATE).fit(X[tr], y[tr])
    s_acc = accuracy_score(y[te], small.predict(X[te]))
    log(f"\nInterpretable depth-4 tree: test accuracy = {s_acc:.4f}, leaves = {small.get_n_leaves()}")
    log("\nTree structure (export_text):\n" + export_text(small, feature_names=FEATURES, decimals=1))

    fig, ax = plt.subplots(figsize=(22, 9))
    plot_tree(small, feature_names=FEATURES, class_names=list(small.classes_), filled=True,
              rounded=True, fontsize=7, impurity=False, proportion=True, max_depth=3, ax=ax)
    ax.set_title("Decision tree (top 3 levels of the depth-4 interpretable tree)", fontsize=14)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "dt_tree.png"), dpi=130); plt.close(fig)

    # extract IF-THEN rules (one per leaf)
    t = small.tree_
    rules = []

    def simplify(bounds):
        """Collapse the path conditions to one tight interval per feature,
        e.g. 'P > 988.5 AND P > 1005.5' -> 'Pressure > 1005.5'."""
        parts = []
        for f in FEATURES:
            if f not in bounds:
                continue
            lo, hi = bounds[f]
            if lo is not None and hi is not None:
                parts.append(f"{lo:.1f} < {f} <= {hi:.1f}")
            elif lo is not None:
                parts.append(f"{f} > {lo:.1f}")
            else:
                parts.append(f"{f} <= {hi:.1f}")
        return " AND ".join(parts)

    def walk(node, bounds):
        if t.children_left[node] == -1:
            dist = t.value[node][0]
            k = int(np.argmax(dist))
            rules.append({"rule": "IF " + simplify(bounds) + f" THEN Status = {small.classes_[k]}",
                          "samples": int(t.n_node_samples[node]),
                          "confidence": round(float(dist[k] / dist.sum()), 3)})
            return
        f, thr = FEATURES[t.feature[node]], float(t.threshold[node])
        lo, hi = bounds.get(f, (None, None))
        walk(t.children_left[node], {**bounds, f: (lo, thr if hi is None else min(hi, thr))})
        walk(t.children_right[node], {**bounds, f: (thr if lo is None else max(lo, thr), hi)})
    walk(0, {})
    rdf = pd.DataFrame(rules).sort_values("samples", ascending=False)
    rdf.to_csv(f"{CLEAN_DIR}/dt_rules.csv", index=False)
    pd.set_option("display.max_colwidth", 200); pd.set_option("display.width", 250)
    log(f"\nIF-THEN rules extracted from the depth-4 tree ({len(rdf)} leaves), by coverage:")
    for r in rdf.itertuples():
        log(f"  [{r.samples:5d} obs, conf {r.confidence:.2f}] {r.rule}")

    save_json({"n_rows": len(d), "n_storms": int(d.ID.nunique()), "train_rows": len(tr), "test_rows": len(te),
               "class_dist": d.Status.value_counts().to_dict(), "best_criterion": best.criterion,
               "best_depth": int(best.depth), "cv_acc": round(best.cv_acc, 4), "test_acc": round(acc, 4),
               "macro_f1": round(mf1, 4), "weighted_f1": round(wf1, 4), "kappa": round(kappa, 4),
               "baseline": round(base, 4), "leaves": int(clf.get_n_leaves()), "fit_s": round(t_fit, 3),
               "small_acc": round(s_acc, 4), "importance": imp[::-1].round(4).to_dict(),
               "report": classification_report(y[te], pred, labels=CLASSES, output_dict=True),
               "cm": cm.tolist()}, "decision_tree")
    log.close()
