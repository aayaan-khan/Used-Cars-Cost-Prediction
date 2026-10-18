"""Deal Score computation module.

Provides a reusable function `deal_score(row_features, listed_price, model_counts=None, model_key=None)` that:
1. Predicts the fair market price using the LightGBM point model.
2. Predicts the 90% conformalized prediction interval from the selected candidate model (LocallyAdaptiveCQR).
3. Computes the log gap, percentage gap, interval-normalised position, Deal Score (0-100), and label.

Two routes (notebook Section 13). A car model with at least K_ROUTE listings in the training data takes the
"known-model" route: a point model that knows the car model's identity and a range calibrated on random
listings. Every other car model, and every call without a ``model_key``, takes the "general" route: the
baseline point model and a range calibrated on held-out car models, which stays honest for unfamiliar models.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# The interval model is pickled as src.conformal.LocallyAdaptiveCQR, so that module must be
# importable before joblib.load() runs. Importing it here is what makes the artifact portable.
try:
    from . import conformal  # noqa: F401  (normal case: imported as src.deal_score)
    from . import deal_validation as _dv
except ImportError:          # run as a plain script from src/: put the repo root on the path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import conformal  # noqa: F401
    from src import deal_validation as _dv

# Paths to models (git-ignored; regenerate by running the notebook, see README)
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
POINT_MODEL_PATH = MODEL_DIR / "lightgbm_point_model.joblib"
INTERVAL_MODEL_PATH = MODEL_DIR / "c_adaptive_cqr_model.joblib"
ROUTED_MODEL_PATH = MODEL_DIR / "known_model_routed.joblib"      # optional: without it every listing takes the general route

_point_artifact = None
_interval_artifact = None
_routed_artifact = None


def _load_artifacts():
    global _point_artifact, _interval_artifact
    if _point_artifact is None:
        if not POINT_MODEL_PATH.exists():
            raise FileNotFoundError(f"Point model artifact not found at {POINT_MODEL_PATH}. "
                                    "Run the notebook (Section 7) to create it.")
        _point_artifact = joblib.load(POINT_MODEL_PATH)
    if _interval_artifact is None:
        if not INTERVAL_MODEL_PATH.exists():
            raise FileNotFoundError(f"Interval model artifact not found at {INTERVAL_MODEL_PATH}. "
                                    "Run the notebook (Section 6, Cell 33) to create it.")
        _interval_artifact = joblib.load(INTERVAL_MODEL_PATH)

def _load_routed():
    """The known-model artifact, or None if the notebook has not been run since Section 13 was added."""
    global _routed_artifact
    if _routed_artifact is None and ROUTED_MODEL_PATH.exists():
        _routed_artifact = joblib.load(ROUTED_MODEL_PATH)
    return _routed_artifact


def route_for(model_key=None, model_counts=None):
    """Choose the route for one listing. Returns a dict with route ('known-model' or 'general'), the point model,
    the interval model, the feature list and the listing counts used (``model_counts`` if given, else the artifact's)."""
    _load_artifacts()
    routed = _load_routed()
    counts = model_counts if model_counts is not None else (routed["model_counts"] if routed is not None else None)
    if routed is not None and model_key is not None and counts is not None and int(counts.get(model_key, 0)) >= routed["k_route"]:
        return {"route": "known-model", "point_model": routed["point_model"], "interval_model": routed["interval_model"],
                "features": routed["features"], "counts": counts}
    return {"route": "general", "point_model": _point_artifact["model"], "interval_model": _interval_artifact["model"],
            "features": _point_artifact["features"], "counts": counts}


def _as_frame(row_features):
    """Turn a dict, Series or DataFrame of listing features into a DataFrame (shared with src/explain.py)."""
    if isinstance(row_features, dict):
        return pd.DataFrame([row_features])
    if isinstance(row_features, pd.Series):
        return row_features.to_frame().T.infer_objects()   # a transposed Series is all-object; restore numeric dtypes
    if isinstance(row_features, pd.DataFrame):
        return row_features.copy()
    raise TypeError("row_features must be a dict, pd.Series, or pd.DataFrame")


def deal_score(row_features, listed_price, model_counts=None, model_key=None):
    """Compute Deal Score (0-100), fair price range, gap %, and label for a vehicle listing.

    Parameters
    ----------
    row_features : dict or pd.DataFrame or pd.Series
        Feature values for the vehicle. Must contain:
        ['age', 'log_km', 'log_engine', 'log_power', 'seats', 'mileage_val',
         'fuel', 'transmission', 'owner', 'seller_type', 'brand']
        (the saved artifact stores the exact list, so extra keys are ignored; log_km = log1p(km_driven),
        log_engine = log(engine_cc), log_power = log(power_bhp), age = 2020 - year)
    listed_price : float or int
        Seller asking price in INR.
    model_counts : pd.Series or dict, optional
        Number of development listings per car model. Optional: the saved routed artifact carries the counts
        of the final model, which is what the demo should use. Together with ``model_key`` it decides the
        route and lets the function warn about rare or unknown car models.
    model_key : str, optional
        The car model of this listing (brand + model, as in the data), used to look up ``model_counts``.

    Returns
    -------
    dict
        Dictionary containing:
        - listed_price: asking price in INR
        - predicted_price: estimated fair median price in INR
        - lower_bound: 90% interval lower bound in INR
        - upper_bound: 90% interval upper bound in INR
        - gap_log: log(listed_price) - log(predicted_price)
        - gap_pct: percentage difference relative to predicted price ((listed - pred) / pred * 100)
        - half_width_log: interval log half-width (uncertainty normaliser)
        - position: interval-normalised position (gap_log / half_width_log)
        - deal_score: continuous score from 0 to 100 (50 = fair price, >50 = bargain, <50 = overpriced)
        - label: 'Underpriced', 'Fair', or 'Overpriced'
        - graded_label: 5-level label ('Deep Underpriced', 'Moderate Underpriced', 'Fair Price',
                        'Moderate Overpriced', 'Deep Overpriced')
        - low_confidence: True if any warning applies (rare/unknown car model, old car, missing specs;
                        thresholds in src/deal_validation.py, validated in notebook Section 12)
        - confidence_reasons: list of the warnings that apply (empty if none)
        - route: 'known-model' or 'general' (see the module docstring)
    """
    _load_artifacts()
    
    X = _as_frame(row_features)

    r = route_for(model_key, model_counts)                     # known car model -> identity model and its own range; else general
    if r["route"] == "known-model":
        X = X.assign(model_key=model_key)
    X_input = X[r["features"]]
    
    # Point prediction
    pred_log = float(r["point_model"].predict(X_input)[0])
    pred_price = float(np.exp(pred_log))
    
    # Interval prediction
    lo_log_arr, hi_log_arr = r["interval_model"].predict(X_input)
    lo_log = float(lo_log_arr[0])
    hi_log = float(hi_log_arr[0])
    lo_price = float(np.exp(lo_log))
    hi_price = float(np.exp(hi_log))
    
    # Deal metrics
    listed = float(listed_price)
    listed_log = float(np.log(listed))
    gap_log = listed_log - pred_log
    gap_pct = float((np.exp(gap_log) - 1.0) * 100.0)
    
    half_width_log = float((hi_log - lo_log) / 2.0)
    position = float(gap_log / half_width_log) if half_width_log > 0 else 0.0
    
    # Deal Score 0-100: monotone decreasing transform of position
    # (bargains have negative position -> deal score > 50; overpriced listings have positive position -> deal score < 50)
    score_val = float(np.clip(50.0 - 25.0 * position, 0.0, 100.0))
    
    # 3-level label
    if listed < lo_price:
        label = "Underpriced"
    elif listed > hi_price:
        label = "Overpriced"
    else:
        label = "Fair"
        
    # 5-level graded label
    if position <= -1.0:
        graded_label = "Deep Underpriced"
    elif position <= -0.33:
        graded_label = "Moderate Underpriced"
    elif position < 0.33:
        graded_label = "Fair Price"
    elif position < 1.0:
        graded_label = "Moderate Overpriced"
    else:
        graded_label = "Deep Overpriced"
        
    # Confidence warnings (additive: they do not change any number above)
    age = float(X["age"].iloc[0])
    specs_missing = bool(X[["log_engine", "log_power", "seats"]].isna().any(axis=1).iloc[0])
    n_model = int(r["counts"].get(model_key, 0)) if (r["counts"] is not None and model_key is not None) else None
    reasons = _dv.confidence_reasons(age, specs_missing, n_model)

    return {
        "listed_price": listed,
        "predicted_price": round(pred_price, 2),
        "lower_bound": round(lo_price, 2),
        "upper_bound": round(hi_price, 2),
        "gap_log": round(gap_log, 4),
        "gap_pct": round(gap_pct, 2),
        "half_width_log": round(half_width_log, 4),
        "position": round(position, 4),
        "deal_score": round(score_val, 1),
        "label": label,
        "graded_label": graded_label,
        "low_confidence": bool(reasons),
        "confidence_reasons": reasons,
        "route": r["route"],
    }
