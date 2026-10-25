import os
import joblib
import numpy as np
import pandas as pd
from features import (extract, legacy_features, text_norm, red_flags, lookalike_brand,
                       is_popular, logit)

ART = "artifacts"


class Detector:
    def __init__(self, path=os.path.join(ART, "bundle.joblib")):
        b = joblib.load(path)
        self.A, self.B, self.C, self.S = b["A"], b["B"], b["C"], b["S"]
        self.thr = b["thr"]
        self.cols, self.lcols = b["cols"], b["lcols"]
        self.baseline = pd.Series(b["baseline"])

    def level(self, p):
        return "High" if p >= self.thr else "Medium" if p >= self.thr * 0.5 else "Low"

    def _probs(self, urls):
        urls = [str(u) for u in urls]
        R = pd.DataFrame([extract(u) for u in urls])[self.cols]
        L = pd.DataFrame([legacy_features(u) for u in urls])[self.lcols]
        pA = self.A.predict_proba(R)[:, 1]
        pB = self.B.predict_proba([text_norm(u) for u in urls])[:, 1]
        pC = self.C.predict_proba(L)[:, 1]
        final = self.S.predict_proba(logit(np.c_[pA, pB, pC]))[:, 1]
        popular = np.array([is_popular(u) and lookalike_brand(u) is None for u in urls])
        final = np.where(popular, np.minimum(final, 0.03), final)
        return R, pA, pB, pC, final, popular

    def score_many(self, urls):
        urls = [str(u).strip() for u in urls if str(u).strip()]
        _, pA, pB, pC, final, popular = self._probs(urls)
        return pd.DataFrame({
            "url": urls,
            "risk": np.round(final, 4),
            "level": [self.level(p) for p in final],
            "feature_model": np.round(pA, 4),
            "text_model": np.round(pB, 4),
            "pooled_model": np.round(pC, 4),
            "known_domain": popular,
        })

    def explain(self, row):
        """What-if explanation: how much does the feature model's score drop when each
        feature is replaced by a typical legitimate value?"""
        rows = [row.copy()]
        for c in self.cols:
            r = row.copy()
            r[c] = self.baseline[c]
            rows.append(r)
        pm = self.A.predict_proba(pd.DataFrame(rows)[self.cols])[:, 1]
        contrib = pd.Series(pm[0] - pm[1:], index=self.cols)
        return contrib[contrib.abs() > 1e-3].sort_values(key=abs, ascending=False)

    def analyze(self, url):
        R, pA, pB, pC, final, popular = self._probs([url])
        row = R.iloc[0]
        return {
            "risk": float(final[0]),
            "level": self.level(final[0]),
            "feature_model": float(pA[0]),
            "text_model": float(pB[0]),
            "pooled_model": float(pC[0]),
            "known_domain": bool(popular[0]),
            "flags": red_flags(url),
            "contrib": self.explain(row),
            "features": row,
        }