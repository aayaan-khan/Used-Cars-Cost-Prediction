"""Tests for the demo logic (src/demo.py) and a smoke test of the Streamlit app.
Need the notebook outputs (models/*.joblib, app/model_catalog.json); skipped if they are missing."""
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
if not (ROOT / "app" / "model_catalog.json").exists() or not (ROOT / "models" / "known_model_routed.joblib").exists():
    pytest.skip("run the notebook first (creates models/ and app/model_catalog.json)", allow_module_level=True)

from src import demo

BASE = dict(brand="Maruti", model_key="Maruti Swift", year=2016, km=60000, fuel="Diesel", transmission="Manual",
            owner="First Owner", seller_type="Individual", engine_cc=1248, power_bhp=75, seats=5, mileage=22, asking_price=650000)


def test_assess_known_model():
    r = demo.assess(BASE)
    assert r["known_route"] and r["confidence"] == "Standard"
    assert 0 <= r["score"]["deal_score"] <= 100
    assert r["score"]["lower_bound"] < r["score"]["predicted_price"] < r["score"]["upper_bound"]


def test_cheaper_asking_price_scores_higher():
    lo, hi = demo.assess({**BASE, "asking_price": 450000}), demo.assess({**BASE, "asking_price": 900000})
    assert lo["score"]["deal_score"] > hi["score"]["deal_score"]


def test_unknown_model_uses_general_route():
    r = demo.assess({**BASE, "model_key": None})
    assert not r["known_route"]


def test_missing_specs_gives_low_confidence():
    r = demo.assess({**BASE, "engine_cc": None, "power_bhp": None, "seats": None, "mileage": None})
    assert r["confidence"] == "Low"


@pytest.mark.parametrize("bad", [{"year": 1980}, {"year": 2025}, {"km": -1}, {"asking_price": 0}, {"engine_cc": -5}, {"asking_price": float("nan")},
                                 {"asking_price": float("inf")}, {"km": float("nan")}, {"fuel": "Hydrogen"}, {"owner": "Nobody"}, {"seats": 0}, {"year": "abc"}])
def test_bad_inputs_raise(bad):
    with pytest.raises(ValueError):
        demo.assess({**BASE, **bad})


def test_every_catalogue_model_scores():
    import math
    for m in demo.load_catalog()["models"]:
        r = demo.assess({**BASE, "brand": m["brand"], "model_key": m["model_key"], "fuel": m["fuel"], "transmission": m["transmission"],
                         "engine_cc": m["engine_cc"], "power_bhp": m["power_bhp"], "seats": m["seats"], "mileage": m["mileage_val"],
                         "year": max(min(m["year_max"], demo.REF_YEAR), demo.MIN_YEAR)})
        s = r["score"]
        assert all(math.isfinite(s[k]) for k in ("deal_score", "predicted_price", "lower_bound", "upper_bound")), m["model_key"]
        assert s["lower_bound"] <= s["predicted_price"] <= s["upper_bound"], m["model_key"]


def test_app_every_brand_and_model():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    for brand in at.selectbox(key="brand").options:
        at.selectbox(key="brand").select(brand).run()
        assert not at.exception, brand
        for label in at.selectbox(key="model_label").options:
            at.selectbox(key="model_label").select(label).run()
            assert not at.exception, (brand, label)


def test_app_smoke():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception
    at.selectbox(key="preset").select("Honda City, 2014, petrol, 90,000 km").run()
    assert not at.exception and at.session_state.year == 2014
    at.checkbox(key="specs_known").uncheck().run()
    assert not at.exception
