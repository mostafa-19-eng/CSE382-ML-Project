import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from src.clustering.clustering_baseline import ALL_FEATURES, FORBIDDEN, SEED, DATA, build_preprocessor


def test_no_leakage():
    assert not (set(ALL_FEATURES) & set(FORBIDDEN)), "forbidden column in ALL_FEATURES"


def test_determinism():
    df = pd.read_csv(DATA)
    for c in df.select_dtypes("object").columns:
        df[c] = df[c].str.strip()
    Xt = build_preprocessor().fit_transform(df.sample(5000, random_state=SEED)[ALL_FEATURES])
    l1 = KMeans(4, n_init=10, random_state=SEED).fit_predict(Xt)
    l2 = KMeans(4, n_init=10, random_state=SEED).fit_predict(Xt)
    assert np.array_equal(l1, l2), "labels differ across two runs"


if __name__ == "__main__":
    test_no_leakage(); test_determinism(); print("All tests passed.")
