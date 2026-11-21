"""
common.py
---------
Shared paths, constants and helper functions used by every script in this
project (NOAA HURDAT2 Atlantic + North-East/Central Pacific hurricane database).

All scripts are run from the project root:
    python src/01_preprocessing.py
"""
import os
import json
import time
from contextlib import contextmanager

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "hurricane")             # raw Kaggle files
OUT_DIR = os.path.join(ROOT, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
LOG_DIR = os.path.join(OUT_DIR, "logs")
CLEAN_DIR = os.path.join(OUT_DIR, "data")
for _d in (OUT_DIR, FIG_DIR, LOG_DIR, CLEAN_DIR):
    os.makedirs(_d, exist_ok=True)

RAW_FILES = {"Atlantic": os.path.join(DATA_DIR, "atlantic.csv"),
             "Pacific": os.path.join(DATA_DIR, "pacific.csv")}

CLEAN_TRACKS = os.path.join(CLEAN_DIR, "clean_tracks.csv")      # observation level
STORMS = os.path.join(CLEAN_DIR, "storms.csv")                  # storm level
TRANSACTIONS = os.path.join(CLEAN_DIR, "transactions.csv")      # ARM item-sets

# --------------------------------------------------------------------------
# Assumed parameters (stated in the report)
# --------------------------------------------------------------------------
MIN_SUPPORT = 0.05        # 5 % of transactions
MIN_CONFIDENCE = 0.70     # 70 %
DIC_BLOCK_SIZE = 5000     # M : number of transactions between DIC "stops"
RANDOM_STATE = 42


class Logger:
    """Prints to console AND writes to outputs/logs/<name>.txt so that every
    output quoted in the report can be traced back to a file."""

    def __init__(self, name):
        self.path = os.path.join(LOG_DIR, name + ".txt")
        self.f = open(self.path, "w", encoding="utf-8")

    def __call__(self, *args):
        msg = " ".join(str(a) for a in args)
        print(msg)
        self.f.write(msg + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def save_json(obj, name):
    """Save a metrics dictionary to outputs/logs/<name>.json (used by the report)."""
    with open(os.path.join(LOG_DIR, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(name):
    with open(os.path.join(LOG_DIR, name + ".json"), encoding="utf-8") as f:
        return json.load(f)


@contextmanager
def timer(store, key):
    """Context manager that records wall-clock seconds into store[key]."""
    t0 = time.perf_counter()
    yield
    store[key] = time.perf_counter() - t0


def load_transactions():
    """Return the ARM transactions as a list of frozensets of item strings."""
    import pandas as pd
    df = pd.read_csv(TRANSACTIONS)
    return [frozenset(s.split("|")) for s in df["items"]]
