"""Logic behind the demo app, kept out of the Streamlit file so it can be tested without a browser.

``assess(inputs)`` turns what a person types into a listing, runs the Deal Score and the SHAP explanation,
and returns plain-language text next to the numbers. Nothing here fits or changes a model.
"""

import json
from pathlib import Path
import numpy as np

try:
    from . import deal_score as ds, explain as xp
except ImportError:                                   # run as a plain script: put the repo root on the path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import deal_score as ds, explain as xp

REF_YEAR = 2020                                       # the data ends in 2020, so the market it describes is the 2020 market
CATALOG_PATH = Path(__file__).resolve().parent.parent / "app" / "model_catalog.json"
FUELS = ["Diesel", "Petrol", "CNG", "LPG"]
TRANSMISSIONS = ["Manual", "Automatic"]
OWNERS = ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"]
SELLERS = ["Individual", "Dealer", "Trustmark Dealer"]

VERDICTS = {"Deep Underpriced": ("Priced far below the fair range", "good"),
            "Moderate Underpriced": ("Priced below the fair price", "good"),
            "Fair Price": ("In line with the fair price", "fair"),
            "Moderate Overpriced": ("Priced above the fair price", "high"),
            "Deep Overpriced": ("Priced far above the fair range", "high")}


def load_catalog(path=CATALOG_PATH):
    """The per-car-model catalogue written by notebook Section 15: listing counts and typical specs."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run the notebook (Section 15) to create it.")
    cat = json.loads(p.read_text())
    cat["by_key"] = {m["model_key"]: m for m in cat["models"]}
    return cat


def fmt_inr(x):
    """INR in lakh (1 lakh = 100,000) for amounts from 1 lakh up, otherwise plain rupees."""
    return f"₹{x / 1e5:,.2f} lakh" if x >= 1e5 else f"₹{x:,.0f}"


def _num(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def build_row(i):
    """Form inputs -> the feature dict the models expect. Missing specs become NaN (the models impute them)."""
    eng, pw, seats, mil = (_num(i.get(k)) for k in ("engine_cc", "power_bhp", "seats", "mileage"))
    return {"age": REF_YEAR - int(i["year"]), "log_km": float(np.log1p(float(i["km"]))),
            "log_engine": np.log(eng) if eng else np.nan, "log_power": np.log(pw) if pw else np.nan,
            "seats": seats if seats else np.nan, "mileage_val": mil if mil else np.nan,
            "fuel": i["fuel"], "transmission": i["transmission"], "owner": i["owner"],
            "seller_type": i["seller_type"], "brand": i["brand"]}


def check_inputs(i):
    """Raise ValueError for impossible input; return a list of soft warnings for unusual input."""
    warnings = []
    if not (1995 <= int(i["year"]) <= REF_YEAR):
        raise ValueError(f"Model year must be between 1995 and {REF_YEAR}: the data describes the {REF_YEAR} market.")
    if float(i["km"]) < 0:
        raise ValueError("Kilometres driven cannot be negative.")
    if float(i["asking_price"]) <= 0:
        raise ValueError("The asking price must be positive.")
    for k, name in (("engine_cc", "Engine size"), ("power_bhp", "Power")):
        v = _num(i.get(k))
        if v is not None and v <= 0:
            raise ValueError(f"{name} must be positive (or leave it unknown).")
    if float(i["km"]) > 400_000:
        warnings.append("More than 400,000 km is far outside the data; treat the result with extra care.")
    if not (30_000 <= float(i["asking_price"]) <= 12_000_000):
        warnings.append("The asking price is outside the range of the data (₹0.3 lakh to ₹120 lakh).")
    return warnings


def assess(i, catalog=None):
    """Run everything for one listing. ``i`` keys: brand, model_key (None = a car model not in the list), year, km, fuel,
    transmission, owner, seller_type, engine_cc, power_bhp, seats, mileage (specs may be None), asking_price."""
    warnings = check_inputs(i)
    row, key, price = build_row(i), (i.get("model_key") or None), float(i["asking_price"])
    out = ds.deal_score(row, price, model_key=key)
    ex = xp.explain_listing(row, model_key=key)
    headline, tone = VERDICTS[out["graded_label"]]
    known = out["route"] == "known-model"
    seen = None if catalog is None or key is None else catalog["by_key"].get(key, {}).get("n")
    if known:
        range_note = (f"The system has seen {seen if seen else 'many'} listings of this car model, so the range is calibrated on listings "
                      "like it: in development it contained the asking price 90.6% of the time.")
    else:
        range_note = ("The system has little or no data on this car model, so it uses a wider range built for unfamiliar models. "
                      "For such models the range contained the asking price 84% to 88% of the time (final test: 84.1%), not the nominal 90%.")
    gap = out["gap_pct"]
    sentence = (f"The asking price is {fmt_inr(price)}, about {abs(gap):.0f}% {'above' if gap > 0 else 'below'} the model's fair price of {fmt_inr(out['predicted_price'])}."
                if abs(gap) >= 0.5 else f"The asking price is {fmt_inr(price)}, essentially equal to the model's fair price of {fmt_inr(out['predicted_price'])}.")
    return {"score": out, "explanation": ex, "headline": headline, "tone": tone, "sentence": sentence, "range_note": range_note,
            "warnings": warnings, "known_route": known, "confidence": "Low" if out["low_confidence"] else "Standard"}
