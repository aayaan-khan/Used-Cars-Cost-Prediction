"""Explain a fair-price prediction: which listing features pushed the price up or down, and by how much.

The point model predicts log(price) and is a pipeline (imputer + one-hot encoder + LightGBM), so its
raw SHAP values are spread over about 50 encoded columns (one per brand, per fuel type, per missing-value
flag, ...). This module adds the SHAP values of the columns that belong to the same listing feature, so the
explanation is read in the terms a buyer uses: age, kilometres, brand, fuel and so on.

Because the model works on log(price), contributions are *multiplicative*:

    predicted price = base price x factor_1 x factor_2 x ...        (exact; factor = exp(SHAP value))

``base_price`` is the model's average prediction over the development data, i.e. what it would say about a listing
it knows nothing about. A factor of 0.82 for ``age`` means "the age of this car lowers the price by 18% relative
to that average listing, given the other features".

What the explanation is, and is not: it describes how the **model** arrives at its price. It is not a causal
claim (paying for a newer car does not "cause" a higher price here; age, power and brand are correlated and
SHAP shares credit among correlated features), and it explains the fair price only. How far the *asking*
price sits from that fair price is the Deal Score (``src/deal_score.py``).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import shap

try:
    from . import deal_score as _ds       # normal case: imported as src.explain
except ImportError:                       # run as a plain script from src/: put the repo root on the path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import deal_score as _ds

# Order shown in tables and figures; the readable name of each driver.
DRIVERS = ["age", "log_km", "log_engine", "log_power", "seats", "mileage_val", "specs_missing",
           "fuel", "transmission", "owner", "seller_type", "brand", "model_key"]
DRIVER_NAMES = {"age": "Age", "log_km": "Kilometres driven", "log_engine": "Engine size", "log_power": "Power",
                "seats": "Seats", "mileage_val": "Rated mileage", "specs_missing": "Specs missing in listing",
                "fuel": "Fuel", "transmission": "Transmission", "owner": "Owner history",
                "seller_type": "Seller type", "brand": "Brand", "model_key": "Car model"}

_explainer_cache = {}


def feature_group(encoded_name, features):
    """Map an encoded column of the pipeline (e.g. 'cat__brand_Tata') to the listing feature it comes from."""
    name = encoded_name.split("__", 1)[1]
    if name.startswith("missingindicator_"):       # the imputer's "this value was missing" flags
        return "specs_missing"
    for f in features:
        if name == f or name.startswith(f + "_"):
            return f
    raise ValueError(f"cannot map encoded column {encoded_name!r} to a feature")


def driver_contributions(pipeline, X, features):
    """SHAP values of ``pipeline`` for the rows of ``X``, summed per listing feature.

    Returns (contributions, base_log): a DataFrame with one column per driver (log-price units; the columns
    plus ``base_log`` add up to the model's log prediction for each row) and the base value in log-price.
    """
    cols, model = pipeline.named_steps["cols"], pipeline.named_steps["model"]
    names = list(cols.get_feature_names_out())
    groups = np.array([feature_group(n, features) for n in names])
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(cols.transform(X[features]))
    drivers = [d for d in DRIVERS if d != "model_key" or "model_key" in features]        # 'Car model' only for the identity model
    out = pd.DataFrame({d: values[:, groups == d].sum(axis=1) if (groups == d).any() else np.zeros(len(X))
                        for d in drivers}, index=X.index)
    return out, float(np.ravel(explainer.expected_value)[0])


def _describe(driver, row):
    """Human-readable 'Age: 6 years' style text for one driver of one listing."""
    v = row[driver] if driver in row.index else None
    if driver in ("age", "log_km", "log_engine", "log_power", "seats", "mileage_val") and pd.isna(v):
        return "unknown"                                  # a missing spec: the model uses the training median
    if driver == "age":         return f"{v:.0f} years old"
    if driver == "log_km":      return f"{np.expm1(v):,.0f} km"
    if driver == "log_engine":  return f"{np.exp(v):,.0f} cc"
    if driver == "log_power":   return f"{np.exp(v):.0f} bhp"
    if driver == "seats":       return f"{v:.0f} seats"
    if driver == "mileage_val": return f"{v:.1f} km/l"
    if driver == "specs_missing":
        return "yes" if row[["log_engine", "log_power", "seats", "mileage_val"]].isna().any() else "no"
    return str(v)


def explain_listing(row_features, model_key=None, model_counts=None):
    """Explain the fair-price prediction for ONE listing.

    Parameters: the same ``row_features``, ``model_key`` and ``model_counts`` as ``deal_score.deal_score``, so the
    explanation follows the same route (known-model or general) and explains the same fair price. On the
    known-model route the model also knows the car model's identity, which shows up as the "Car model" driver.

    Returns a dict with ``base_price`` (INR, the average listing), ``predicted_price`` (INR, the fair price) and
    ``drivers``: a DataFrame sorted by size of effect with the columns ``driver``, ``value`` (readable),
    ``log_effect``, ``factor`` (price multiplier) and ``effect_pct`` (= (factor - 1) x 100).
    ``base_price`` x the product of all ``factor`` equals ``predicted_price`` (up to rounding).
    """
    _ds._load_artifacts()
    X = _ds._as_frame(row_features)
    if len(X) != 1:
        raise ValueError("explain_listing explains one listing at a time")
    r = _ds.route_for(model_key, model_counts)
    if r["route"] == "known-model":
        X = X.assign(model_key=model_key)
    contrib, base_log = driver_contributions(r["point_model"], X, r["features"])
    row = X.iloc[0]
    log_eff = contrib.iloc[0]
    shown = [d for d in DRIVERS if d != "model_key" or r["route"] == "known-model"]      # the general route does not use the identity
    table = pd.DataFrame({"driver": [DRIVER_NAMES[d] for d in shown],
                          "value": [_describe(d, row) for d in shown],
                          "log_effect": [float(log_eff[d]) for d in shown]})
    table["factor"] = np.exp(table.log_effect)
    table["effect_pct"] = (table.factor - 1.0) * 100.0
    table = table.reindex(table.log_effect.abs().sort_values(ascending=False).index).reset_index(drop=True)
    return {"base_price": float(np.exp(base_log)),
            "predicted_price": float(np.exp(base_log + log_eff.sum())),
            "route": r["route"],
            "drivers": table}


def format_explanation(explanation, top=5):
    """One-line-per-driver text for the largest ``top`` drivers, e.g. for the demo app."""
    lines = [f"Average listing: INR {explanation['base_price']:,.0f}"]
    for _, r in explanation["drivers"].head(top).iterrows():
        lines.append(f"  {r.driver} ({r.value}): {r.effect_pct:+.0f}%")
    lines.append(f"Fair price for this listing: INR {explanation['predicted_price']:,.0f}")
    return "\n".join(lines)
