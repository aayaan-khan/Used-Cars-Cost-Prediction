"""Validation tools for the Deal Score (notebook Section 12).

Everything here works on OUT-OF-FOLD arrays, so the same functions can be re-run after the point model or the
intervals change (Step 12). Nothing here fits a model.

Conventions
-----------
* All prices are handled in log space; ``lo``/``hi`` are the interval limits in log(price), ``pred`` the point
  prediction in log(price), ``listed`` the listed price in log(price).
* The labels reproduce ``src/deal_score.py``: 3-level = outside the interval; 5-level = interval-normalised position
  with cut-offs +-0.33 and +-1.0, where position = (log listed - log predicted) / interval half-width.

Usage example
-------------
    from src.deal_validation import inject_and_label, detection_table, confidence_flags
    lab = inject_and_label(listed_log, lo, hi, 1.2)            # 'Over' / 'Fair' / 'Under' for every listing
    tab = detection_table(listed_log, lo, hi, {"all": slice(None)}, [0.8, 1.2])
"""

import numpy as np
import pandas as pd

# ---- thresholds fixed in advance (see notebook Section 12.1) ----------------------------------------------------
RARE_MODEL_MIN_LISTINGS = 10      # fewer listings of the same car model in the training data -> "rare model"
OLD_CAR_AGE = 13                  # years (2020 minus model year) at or above which a car counts as old
GRADED_CUTS = (-1.0, -0.33, 0.33, 1.0)   # position cut-offs of the 5-level label (same as src/deal_score.py)


def three_level(listed_log, lo, hi):
    """'Under' if listed below the interval, 'Over' if above, else 'Fair' (same rule as deal_score)."""
    listed_log, lo, hi = np.asarray(listed_log), np.asarray(lo), np.asarray(hi)
    return np.where(listed_log < lo, "Under", np.where(listed_log > hi, "Over", "Fair"))


def position(listed_log, pred, lo, hi):
    """Interval-normalised position (gap in log units divided by the interval half-width)."""
    half = (np.asarray(hi) - np.asarray(lo)) / 2.0
    return (np.asarray(listed_log) - np.asarray(pred)) / half


def five_level(pos):
    """5-level graded label from the position (cut-offs in GRADED_CUTS)."""
    a, b, c, d = GRADED_CUTS
    pos = np.asarray(pos)
    return np.select([pos <= a, pos <= b, pos < c, pos < d],
                     ["Deep under", "Moderate under", "Fair", "Moderate over"], default="Deep over")


def inject_and_label(listed_log, lo, hi, multiplier):
    """Multiply every listed price by ``multiplier`` and label it with the SAME intervals."""
    return three_level(np.asarray(listed_log) + np.log(multiplier), lo, hi)


def detection_rate(listed_log, lo, hi, multiplier, mask=None):
    """Share of listings flagged in the correct direction after the price change (None if the segment is empty)."""
    lab = inject_and_label(listed_log, lo, hi, multiplier)
    target = "Over" if multiplier > 1 else "Under"
    hit = (lab == target)
    if mask is not None:
        hit = hit[np.asarray(mask)]
    return float(100 * hit.mean()) if len(hit) else np.nan


def detection_table(listed_log, lo, hi, segments, multipliers):
    """Detection rate (%) per multiplier (rows) and segment (columns); ``segments`` maps name -> boolean mask."""
    out = {}
    for name, mask in segments.items():
        m = np.ones(len(listed_log), bool) if isinstance(mask, slice) else np.asarray(mask)
        out[name] = [detection_rate(listed_log, lo, hi, k, m) for k in multipliers]
    return pd.DataFrame(out, index=[f"x{k:.2f}" for k in multipliers])


def min_detectable(table, thresholds=(50, 80)):
    """Smallest price drop (multiplier < 1) and rise (> 1) in the table detected at >= each threshold."""
    rows = {}
    mult = np.array([float(i[1:]) for i in table.index])
    for col in table.columns:
        r = {}
        for t in thresholds:
            for side, sel in (("cut", mult < 1), ("rise", mult > 1)):
                cand = table.loc[sel, col]
                ok = cand[cand >= t]
                if side == "cut":      # closest to 1 from below that reaches the threshold
                    r[f"{side} >= {t}%"] = f"x{max(float(i[1:]) for i in ok.index):.2f}" if len(ok) else "not reached"
                else:
                    r[f"{side} >= {t}%"] = f"x{min(float(i[1:]) for i in ok.index):.2f}" if len(ok) else "not reached"
        rows[col] = r
    return pd.DataFrame(rows).T


def confidence_flags(age, specs_missing, n_model_listings):
    """The three low-confidence components for each listing.

    n_model_listings = how many listings of the same car model the model was trained on (0 if the model is unknown).
    Returns a DataFrame of booleans: rare_model, old_car, specs_missing (and unknown_model for n == 0).
    """
    n = np.asarray(n_model_listings)
    return pd.DataFrame({"unknown_model": n == 0,
                         "rare_model": (n < RARE_MODEL_MIN_LISTINGS),
                         "old_car": np.asarray(age) >= OLD_CAR_AGE,
                         "specs_missing": np.asarray(specs_missing).astype(bool)})


def confidence_reasons(age, specs_missing, n_model_listings=None):
    """Human-readable reasons for ONE listing (used by deal_score and the demo app)."""
    reasons = []
    if n_model_listings is not None:
        if n_model_listings == 0:
            reasons.append("car model not in the development data")
        elif n_model_listings < RARE_MODEL_MIN_LISTINGS:
            reasons.append(f"rare car model (fewer than {RARE_MODEL_MIN_LISTINGS} listings in the data)")
    if age >= OLD_CAR_AGE:
        reasons.append(f"old car ({OLD_CAR_AGE}+ years)")
    if specs_missing:
        reasons.append("engine, power or seats missing")
    return reasons


def error_by_flag(ape, flag):
    """Median absolute % error of flagged vs unflagged listings, their ratio and the flagged share."""
    ape, flag = np.asarray(ape), np.asarray(flag).astype(bool)
    f, u = np.median(ape[flag]) if flag.any() else np.nan, np.median(ape[~flag]) if (~flag).any() else np.nan
    return {"flagged share %": round(100 * flag.mean(), 1), "n flagged": int(flag.sum()),
            "median error flagged %": round(f, 1), "median error unflagged %": round(u, 1),
            "ratio": round(f / u, 2) if u == u and u > 0 else np.nan}
