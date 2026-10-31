"""M1 - Baseline: TF-IDF word + character n-grams with one logistic-regression classifier per dimension.
Run:  python src/baseline.py   (trains and saves models/baseline.joblib)
"""
import time

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from config import DATA_PROCESSED, MODELS, DIMENSIONS, UNKNOWN, SEED
from preprocess import normalise


class Baseline:
    def __init__(self):
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=100000, sublinear_tf=True)
        self.clfs = {}

    def _X(self, texts, fit=False):
        texts = [normalise(t) for t in texts]
        if fit:
            return hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()
        return hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()

    def fit(self, df):
        X = self._X(df["text"].tolist(), fit=True)
        for d in DIMENSIONS:
            m = (df[d] != UNKNOWN).values
            clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced", random_state=SEED)
            clf.fit(X[m], df.loc[m, d].astype(int).values)
            self.clfs[d] = clf
        return self

    def predict_proba(self, texts):
        X = self._X(list(texts))
        return np.column_stack([self.clfs[d].predict_proba(X)[:, 1] for d in DIMENSIONS])


def main():
    train = pd.read_csv(DATA_PROCESSED / "train.csv")
    t0 = time.time()
    import baseline as module  # pickle the class as baseline.Baseline, not __main__.Baseline
    model = module.Baseline().fit(train)
    print(f"Baseline trained on {len(train)} items in {time.time() - t0:.1f} s")
    joblib.dump(model, MODELS / "baseline.joblib")


if __name__ == "__main__":
    main()
