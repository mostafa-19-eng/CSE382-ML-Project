import json, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.preprocessing import encode_month

DATA = Path("data/H2_cleaned.csv")
OUT  = Path("outputs/clustering")
SEED = 42

# feature lists -- hardcoded from measured skewness (|skew| > 1 gets log1p)
LOG1P = ["LeadTime", "StaysInWeekNights", "PreviousCancellations",
         "PreviousBookingsNotCanceled", "DaysInWaitingList",
         "RequiredCarParkingSpaces", "TotalOfSpecialRequests"]
SCALE = ["StaysInWeekendNights", "TotalGuests"]
OHE   = ["MarketSegment", "DistributionChannel", "CustomerType", "Meal"]
BIN   = ["IsRepeatedGuest"]
MONTH = ["ArrivalDateMonth"]

NUMERIC      = LOG1P + SCALE
ALL_FEATURES = LOG1P + SCALE + OHE + BIN + MONTH

FORBIDDEN = [
    "IsCanceled", "DepositType", "AssignedRoomType", "RoomMismatch", "BookingChanges",
    "Country", "Agent", "Company", "ADR",
    "ArrivalDateYear", "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth",
]
assert not (set(ALL_FEATURES) & set(FORBIDDEN)), "Leakage: forbidden column in feature list"


def build_preprocessor():
    return ColumnTransformer([
        ("log1p", Pipeline([("f", FunctionTransformer(np.log1p)), ("s", StandardScaler())]), LOG1P),
        ("scale", StandardScaler(), SCALE),
        ("ohe",   OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse_output=False), OHE),
        ("bin",   "passthrough", BIN),
        ("month", FunctionTransformer(encode_month, validate=False), MONTH),
    ])


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA)
    for c in df.select_dtypes("object").columns:
        df[c] = df[c].str.strip()

    Xt = build_preprocessor().fit_transform(df[ALL_FEATURES])

    t0 = time.perf_counter()
    km = KMeans(n_clusters=4, n_init=10, random_state=SEED).fit(Xt)
    fit_time = time.perf_counter() - t0
    labels = km.labels_

    sil = silhouette_score(Xt, labels, sample_size=10000, random_state=SEED)
    db  = davies_bouldin_score(Xt, labels)

    print(f"k=4  n_features={Xt.shape[1]}  Silhouette={sil:.4f}  "
          f"DB={db:.4f}  Inertia={km.inertia_:.2f}  fit={fit_time:.3f}s")
    for c, n in zip(*np.unique(labels, return_counts=True)):
        print(f"  Cluster {c}: {n:,}  ({n/len(labels):.1%})")

    # cluster labels (no outcome columns)
    pd.DataFrame({"row_index": df.index, "cluster": labels}).to_csv(
        OUT / "cluster_labels.csv", index=False)

    results = {
        "algorithm": "KMeans", "n_clusters": 4, "n_init": 10, "random_state": SEED,
        "n_samples": len(labels), "n_features": int(Xt.shape[1]),
        "silhouette": round(float(sil), 4), "silhouette_sample_size": 10000,
        "davies_bouldin": round(float(db), 4), "inertia": round(float(km.inertia_), 2),
        "fit_time_s": round(fit_time, 4),
    }
    (OUT / "baseline_results.json").write_text(json.dumps(results, indent=2) + "\n")

    # print numeric means per cluster (no file saved)
    prof = df[NUMERIC].assign(cluster=labels).groupby("cluster")[NUMERIC].mean()
    prof.loc["overall"] = df[NUMERIC].mean()
    print("\nNumeric means per cluster:")
    print(prof.round(2).to_string())

    # print audit table -- outcome columns used AFTER clustering only
    df["cluster"] = labels
    df["is_nonrefund"] = (df["DepositType"] == "Non Refund").astype(int)
    audit = df.groupby("cluster").agg(
        n=("cluster", "count"),
        mean_ADR=("ADR", "mean"),
        cancel_rate=("IsCanceled", "mean"),
        nonrefund_share=("is_nonrefund", "mean"),
    ).round(3)
    print("\nAudit (post-hoc -- outcome columns NOT in clustering):")
    print(audit.to_string())

    from src.clustering.clustering_plots import make_plots
    make_plots(df, Xt, labels, km, OUT)


if __name__ == "__main__":
    main()
