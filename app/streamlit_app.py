"""Used Car Deal Score: demo app.   Run from the repo root:   streamlit run app/streamlit_app.py

Needs the files the notebook creates (models/*.joblib and app/model_catalog.json): run the notebook first.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from src import demo, deal_score as ds

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
OTHER = "Other / not in the list"

st.set_page_config(page_title="Used Car Deal Score", page_icon=None, layout="wide")


@st.cache_resource
def load_everything():
    cat = demo.load_catalog()
    ds._load_artifacts()
    return cat


try:
    CAT = load_everything()
except FileNotFoundError as e:
    st.error(f"{e}\n\nRun the notebook first (see the README): it creates the models and the catalogue this app uses.")
    st.stop()

BRANDS = sorted({m["brand"] for m in CAT["models"]})
BY_BRAND = {b: sorted([m for m in CAT["models"] if m["brand"] == b], key=lambda m: -m["n"]) for b in BRANDS}


def label_of(m):
    return f"{m['model_key'].split(' ', 1)[1] if ' ' in m['model_key'] else m['model_key']} ({m['n']} listings)"


def model_options(brand):
    return [label_of(m) for m in BY_BRAND.get(brand, [])] + [OTHER]


def key_of(brand, label):
    for m in BY_BRAND.get(brand, []):
        if label_of(m) == label:
            return m["model_key"]
    return None


def prefill():
    """When the car model changes, fill the specs with that model's typical values (the user can still edit them)."""
    m = CAT["by_key"].get(key_of(st.session_state.brand, st.session_state.model_label))
    if m:
        for k, v in (("engine", m["engine_cc"]), ("power", m["power_bhp"]), ("seats", m["seats"]), ("mileage", m["mileage_val"])):
            if v is not None:
                st.session_state[k] = float(v)
        st.session_state.fuel, st.session_state.transmission = m["fuel"], m["transmission"]
        st.session_state.year = int(min(max(m["year_max"] - 2, m["year_min"], demo.MIN_YEAR), demo.REF_YEAR))


def on_brand():
    st.session_state.model_label = model_options(st.session_state.brand)[0]
    prefill()


PRESETS = {"Maruti Swift Dzire, 2015, diesel, 70,000 km": ("Maruti", "Swift Dzire", 2015, 70000, 5.2),
           "Hyundai Grand i10, 2017, petrol, 35,000 km": ("Hyundai", "Grand i10", 2017, 35000, 5.6),
           "Honda City, 2014, petrol, 90,000 km": ("Honda", "City", 2014, 90000, 4.3),
           "Mahindra XUV500, 2016, diesel, 80,000 km": ("Mahindra", "XUV500", 2016, 80000, 9.8)}


def apply_preset():
    name = st.session_state.preset
    if name == "(choose an example)":
        return
    brand, model, year, km, price = PRESETS[name]
    st.session_state.brand = brand
    st.session_state.model_label = next((l for l in model_options(brand) if l.startswith(model + " (")), model_options(brand)[0])
    prefill()
    st.session_state.year, st.session_state.km, st.session_state.price_lakh = year, km, price


if "brand" not in st.session_state:
    st.session_state.brand = "Maruti"
    st.session_state.model_label = next((l for l in model_options("Maruti") if l.startswith("Swift (")), model_options("Maruti")[0])
    st.session_state.update(year=2016, km=60000, fuel="Petrol", transmission="Manual", owner="First Owner", seller="Individual",
                            engine=1197.0, power=81.8, seats=5.0, mileage=19.0, price_lakh=6.8, specs_known=True)
    prefill()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("About this demo")
    st.write("Is a used-car asking price fair? The model predicts a fair price and a range from the listing's features, "
             "and the Deal Score says where the asking price sits relative to that range.")
    st.markdown("**How accurate is it?** On listings of car models the system knows well, the median price error was about 11% in development testing. "
                "On car models it has never seen, the sealed final test gave **19%**.")
    st.markdown("**Read this first**\n"
                "- The data are **2020 asking prices** from CarDekho, not sale prices and not today's market.\n"
                "- Model years run to 2020; age means 2020 minus the model year.\n"
                "- A score is a comparison with similar listings, not a verdict on the car's condition.\n"
                "- Not financial advice.")
    st.caption("Source code, tests and the full evaluation are in the repository README and notebook.")

# ------------------------------------------------------------------ inputs
st.title("Used Car Deal Score")
st.caption("India, 2020 asking prices. Change anything and the result updates.")
left, right = st.columns([5, 6], gap="large")

with left:
    st.selectbox("Start from an example", ["(choose an example)"] + list(PRESETS), key="preset", on_change=apply_preset)
    c1, c2 = st.columns(2)
    c1.selectbox("Brand", BRANDS, key="brand", on_change=on_brand)
    c2.selectbox("Car model", model_options(st.session_state.brand), key="model_label", on_change=prefill)
    c3, c4 = st.columns(2)
    c3.number_input("Model year", min_value=demo.MIN_YEAR, max_value=demo.REF_YEAR, step=1, key="year")
    c4.number_input("Kilometres driven", min_value=0, max_value=1_000_000, step=1000, key="km")
    c5, c6 = st.columns(2)
    c5.selectbox("Fuel", demo.FUELS, key="fuel")
    c6.selectbox("Transmission", demo.TRANSMISSIONS, key="transmission")
    c7, c8 = st.columns(2)
    c7.selectbox("Owner", demo.OWNERS, key="owner")
    c8.selectbox("Seller", demo.SELLERS, key="seller")
    st.checkbox("I know the engine, power, seats and mileage", key="specs_known",
                help="Untick it and the model fills the gaps with typical values (and shows a low-confidence note).")
    if st.session_state.specs_known:
        c9, c10 = st.columns(2)
        c9.number_input("Engine (cc)", min_value=300.0, max_value=7000.0, step=10.0, key="engine")
        c10.number_input("Power (bhp)", min_value=20.0, max_value=700.0, step=1.0, key="power")
        c11, c12 = st.columns(2)
        c11.number_input("Seats", min_value=2.0, max_value=14.0, step=1.0, key="seats")
        c12.number_input("Rated mileage (km/l)", min_value=3.0, max_value=40.0, step=0.5, key="mileage")
    st.number_input("Asking price (₹ lakh)", min_value=0.1, max_value=200.0, step=0.1, format="%.2f", key="price_lakh")

s = st.session_state
inputs = {"brand": s.brand, "model_key": key_of(s.brand, s.model_label), "year": s.year, "km": s.km, "fuel": s.fuel,
          "transmission": s.transmission, "owner": s.owner, "seller_type": s.seller, "asking_price": s.price_lakh * 1e5,
          "engine_cc": s.engine if s.specs_known else None, "power_bhp": s.power if s.specs_known else None,
          "seats": s.seats if s.specs_known else None, "mileage": s.mileage if s.specs_known else None}

# ------------------------------------------------------------------ result
with right:
    try:
        R = demo.assess(inputs, CAT)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    sc, ex = R["score"], R["explanation"]
    for w in R["warnings"]:
        st.warning(w)
    colour = {"good": BLUE, "fair": MUTED, "high": ORANGE}[R["tone"]]
    st.markdown(
        f"""<div style="border:1px solid {colour}55;border-left:6px solid {colour};border-radius:8px;padding:14px 18px;margin-bottom:10px">
        <div style="font-size:0.85rem;opacity:.7">Deal Score (50 = fair, higher = better for the buyer)</div>
        <div style="font-size:2.6rem;font-weight:700;line-height:1.1;color:{colour}">{sc['deal_score']:.0f}<span style="font-size:1.1rem;opacity:.6"> / 100</span></div>
        <div style="font-size:1.15rem;font-weight:600">{R['headline']}</div></div>""", unsafe_allow_html=True)
    st.markdown(
        f"""<div style="position:relative;height:12px;border-radius:6px;background:linear-gradient(90deg,{ORANGE}cc,#9a9a9a66 50%,{BLUE}cc);margin:4px 0 2px 0">
        <div style="position:absolute;left:calc({sc['deal_score']:.1f}% - 2px);top:-4px;width:4px;height:20px;background:{INK};border:1px solid #fff;border-radius:2px"></div></div>
        <div style="display:flex;justify-content:space-between;font-size:0.75rem;opacity:.65;margin-bottom:10px"><span>overpriced</span><span>fair</span><span>underpriced</span></div>""",
        unsafe_allow_html=True)
    st.write(R["sentence"])
    def card(title, value, big=False):
        return (f'<div style="flex:1;min-width:0"><div style="font-size:0.8rem;opacity:.65">{title}</div>'
                f'<div style="font-size:{1.5 if big else 1.15}rem;font-weight:600;line-height:1.3">{value}</div></div>')
    st.markdown('<div style="display:flex;gap:16px;margin:6px 0">'
                + card("Fair price", demo.fmt_inr(sc["predicted_price"]), True)
                + card("Fair range", f"{demo.fmt_inr(sc['lower_bound'])} to {demo.fmt_inr(sc['upper_bound'])}")
                + card("Asking price", demo.fmt_inr(sc["listed_price"]), True) + "</div>", unsafe_allow_html=True)
    st.caption(R["range_note"])
    if R["confidence"] == "Low":
        st.warning("**Low confidence.** Treat this as a rough guide: " + "; ".join(sc["confidence_reasons"]) + ".")
    else:
        st.success("**Standard confidence.** No known weak spot applies (rare car model, car 13+ years old, or missing specs). "
                   "Standard still means a wide range: asking prices of the very same car differ by about 12%.")

    st.subheader("Why this fair price")
    d = ex["drivers"].head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ax.barh([f"{r.driver}: {r.value}" for r in d.itertuples()], d.effect_pct, color=[BLUE if v >= 0 else ORANGE for v in d.effect_pct], height=0.62)
    ax.axvline(0, color=MUTED, linewidth=1)
    lim = max(float(d.effect_pct.abs().max()) + 8, 12)
    for yy, v in enumerate(d.effect_pct):
        ax.text(v + (0.8 if v >= 0 else -0.8), yy, f"{v:+.0f}%", va="center", ha="left" if v >= 0 else "right", fontsize=8.5)
    ax.set_xlim(-lim, lim)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("Effect on the fair price vs the average listing (%)  blue raises, orange lowers", fontsize=8)
    ax.tick_params(labelsize=8.5)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
    st.caption("These effects describe how the model prices cars; they are not causes. Correlated features (age, kilometres, power) share credit.")

    with st.expander("What the labels and the range mean"):
        st.markdown(
            "- **Fair range:** where the asking price of a similar listing is expected to fall. It is wide on purpose: it is built to contain the asking price "
            "90% of the time for known car models, and about 84% to 88% of the time for car models the system has never seen.\n"
            "- **Deal Score:** 50 means the asking price equals the fair price; each step away is measured against the width of the range, so the same gap "
            "counts for less on a hard-to-price car. The score only notices large gaps: a 10% markup is barely distinguishable from an ordinary listing.\n"
            "- **Where it is weaker:** very new cars, car models with few listings, cars 13+ years old, and listings with missing specs.")
