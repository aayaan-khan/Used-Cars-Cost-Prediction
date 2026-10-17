"""Conformalized quantile regression (CQR) classes used for the prediction intervals.

These live in a module, not in the notebook, so that the saved models in ``models/`` can be
unpickled from anywhere (the Deal Score demo, a script, another notebook). A class defined inside
a notebook is pickled as ``__main__.<Class>`` and cannot be loaded outside that notebook.

All three classes work on log(price) and share the same fit/predict protocol:

    model.fit(X_prop, y_prop, X_cal, y_cal)   # proper-training set, then calibration set
    lower, upper = model.predict(X)           # interval limits, in log(price)

``base_lo`` / ``base_hi`` are unfitted quantile-regression pipelines (alpha 0.05 and 0.95), so the
nominal coverage is 90%. The calibration set is held out by car model (see the notebook, Section 6).
"""

import numpy as np
import pandas as pd
from sklearn.base import clone


def compute_cqr_adjustment(y_cal, lo_cal, hi_cal, alpha=0.10):
    """Finite-sample conformal adjustment Q: the ceil((n+1)(1-alpha))/n quantile of the CQR scores."""
    scores = np.maximum(lo_cal - np.asarray(y_cal), np.asarray(y_cal) - hi_cal)
    n = len(scores)
    q_level = np.clip(np.ceil((n + 1) * (1 - alpha)) / n, 0.0, 1.0)
    return float(np.quantile(scores, q_level, method="higher"))


class MondrianCQR:
    """Group-conditional CQR.

    Bins are formed on the PREDICTED mid-price (lo+hi)/2 from the quantile models on the
    calibration fold. No true price is used at fit or predict time, so it is deployable without labels.
    """

    def __init__(self, base_lo, base_hi, alpha=0.10, n_groups=5):
        self.base_lo = base_lo
        self.base_hi = base_hi
        self.alpha = alpha
        self.n_groups = n_groups
        self.q_adjs_ = {}
        self.bin_edges_ = None

    def fit(self, X_prop, y_prop, X_cal, y_cal):
        self.m_lo_ = clone(self.base_lo).fit(X_prop, y_prop)
        self.m_hi_ = clone(self.base_hi).fit(X_prop, y_prop)
        lo = self.m_lo_.predict(X_cal)
        hi = self.m_hi_.predict(X_cal)
        mid = (lo + hi) / 2.0                       # predicted log-mid-price
        bins, edges = pd.qcut(mid, q=self.n_groups, labels=False,
                              retbins=True, duplicates="drop")
        self.bin_edges_ = edges
        scores = np.maximum(lo - np.asarray(y_cal), np.asarray(y_cal) - hi)
        for g in range(len(edges) - 1):
            sg = scores[bins == g]
            ng = len(sg)
            if ng > 10:
                ql = np.clip(np.ceil((ng + 1) * (1 - self.alpha)) / ng, 0.0, 1.0)
                self.q_adjs_[g] = float(np.quantile(sg, ql, method="higher"))
            else:
                self.q_adjs_[g] = compute_cqr_adjustment(
                    np.asarray(y_cal), lo, hi, self.alpha)
        return self

    def predict(self, X):
        lo = self.m_lo_.predict(X)
        hi = self.m_hi_.predict(X)
        mid = (lo + hi) / 2.0
        bins = np.digitize(mid, self.bin_edges_[1:-1])
        fb = float(np.median(list(self.q_adjs_.values())))
        adj = np.array([self.q_adjs_.get(b, fb) for b in bins])
        return lo - adj, hi + adj


class FuelMondrianCQR:
    """Fuel-conditional CQR: separate adjustment for Diesel and Petrol.

    CNG/LPG (too few rows) use the global adjustment.
    """

    def __init__(self, base_lo, base_hi, alpha=0.10):
        self.base_lo = base_lo
        self.base_hi = base_hi
        self.alpha = alpha
        self.q_adjs_ = {}
        self.global_q_ = None

    def fit(self, X_prop, y_prop, X_cal, y_cal):
        self.m_lo_ = clone(self.base_lo).fit(X_prop, y_prop)
        self.m_hi_ = clone(self.base_hi).fit(X_prop, y_prop)
        lo = self.m_lo_.predict(X_cal)
        hi = self.m_hi_.predict(X_cal)
        y = np.asarray(y_cal)
        scores = np.maximum(lo - y, y - hi)
        n = len(scores)
        ql = np.clip(np.ceil((n + 1) * (1 - self.alpha)) / n, 0.0, 1.0)
        self.global_q_ = float(np.quantile(scores, ql, method="higher"))
        fuels = X_cal["fuel"].values
        for f in ["Diesel", "Petrol"]:
            mask = (fuels == f)
            s_f = scores[mask]
            nf = len(s_f)
            if nf > 10:
                ql_f = np.clip(np.ceil((nf + 1) * (1 - self.alpha)) / nf, 0.0, 1.0)
                self.q_adjs_[f] = float(np.quantile(s_f, ql_f, method="higher"))
            else:
                self.q_adjs_[f] = self.global_q_
        return self

    def predict(self, X):
        lo = self.m_lo_.predict(X)
        hi = self.m_hi_.predict(X)
        fuels = X["fuel"].values
        adjs = np.array([self.q_adjs_.get(f, self.global_q_) for f in fuels])
        return lo - adjs, hi + adjs


class LocallyAdaptiveCQR:
    """Locally adaptive CQR (the selected interval model, candidate C in notebook Section 6c).

    Conformity scores are divided by a per-listing difficulty sigma(x): a regressor (``diff_model``,
    an unfitted pipeline) trained on the proper-training set to predict the absolute residual.
    Hard-to-price listings therefore get wider intervals and easy ones narrower intervals.
    """

    def __init__(self, base_lo, base_hi, alpha=0.10, diff_model=None):
        if diff_model is None:
            raise ValueError("diff_model is required: pass an unfitted pipeline that regresses |residual| "
                             "(the notebook defines it as DIFF_MODEL).")
        self.base_lo = base_lo
        self.base_hi = base_hi
        self.alpha = alpha
        self.diff_model = diff_model
        self.q_val_ = None

    def fit(self, X_prop, y_prop, X_cal, y_cal):
        self.m_lo_ = clone(self.base_lo).fit(X_prop, y_prop)
        self.m_hi_ = clone(self.base_hi).fit(X_prop, y_prop)
        p_lo_p = self.m_lo_.predict(X_prop)
        p_hi_p = self.m_hi_.predict(X_prop)
        p_mid_p = (p_lo_p + p_hi_p) / 2.0
        res_p = np.abs(np.asarray(y_prop) - p_mid_p)
        self.diff_model_ = clone(self.diff_model).fit(X_prop, res_p)
        cal_lo = self.m_lo_.predict(X_cal)
        cal_hi = self.m_hi_.predict(X_cal)
        sig_c = np.clip(self.diff_model_.predict(X_cal), 1e-3, None)
        y = np.asarray(y_cal)
        norm_scores = np.maximum(cal_lo - y, y - cal_hi) / sig_c
        n = len(norm_scores)
        ql = np.clip(np.ceil((n + 1) * (1 - self.alpha)) / n, 0.0, 1.0)
        self.q_val_ = float(np.quantile(norm_scores, ql, method="higher"))
        return self

    def predict(self, X):
        lo = self.m_lo_.predict(X)
        hi = self.m_hi_.predict(X)
        sig = np.clip(self.diff_model_.predict(X), 1e-3, None)
        adj = self.q_val_ * sig
        return lo - adj, hi + adj
