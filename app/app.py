import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

BASE = Path(__file__).resolve().parent.parent
st.set_page_config(page_title="Delivery Time Predictor", page_icon="🛵", layout="wide")


@st.cache_resource
def load():
    m = joblib.load(BASE / "models" / "delivery_time_model.joblib")
    o = json.load(open(BASE / "models" / "options.json"))
    return m, o


model, opt = load()

# ---------- constants ----------
INK, TEAL, MUTED = "#16313A", "#1F6F78", "#9DB8BB"
FAST, MID, SLOW = "#2E8B6A", "#B27A00", "#C0392B"
MAE = opt["mae"]
TMIN, TMAX = opt.get("t_min", 10), opt.get("t_max", 54)
# test-set results from 02_modeling.ipynb (update if you retrain)
R2, WITHIN5 = opt.get("r2", 0.82), opt.get("within5", 80.8)


def ordered(vals, preferred):
    head = [v for v in preferred if v in vals]
    return head + [v for v in vals if v not in head]


OPTS = {
    "traffic": ordered(opt["Road_traffic_density"], ["Low", "Medium", "High", "Jam"]),
    "weather": ordered(opt["Weather_conditions"], ["Sunny", "Cloudy", "Windy", "Fog", "Stormy", "Sandstorms"]),
    "vehicle": opt["Type_of_vehicle"],
    "vcond": opt["Vehicle_condition"],
    "multi": opt["multiple_deliveries"],
    "otype": opt["Type_of_order"],
    "festival": opt["Festival"],
    "city": opt["City"],
    "day": ordered(opt["day_of_week"], ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
}
AGE_MIN, AGE_MAX = opt["age_range"]
DIST_MAX = float(opt["dist_max"])


def pick(v, options):
    return v if v in options else options[0]


DEFAULTS = {
    "dist": 6.0, "traffic": pick("Medium", OPTS["traffic"]), "weather": pick("Sunny", OPTS["weather"]),
    "age": int(np.clip(30, AGE_MIN, AGE_MAX)), "rating": 4.7,
    "vehicle": pick("motorcycle", OPTS["vehicle"]), "vcond": pick("1", OPTS["vcond"]),
    "multi": pick("1", OPTS["multi"]), "otype": pick("Snack", OPTS["otype"]),
    "festival": pick("No", OPTS["festival"]), "city": pick("Urban", OPTS["city"]),
    "hour": 19, "day": pick("Friday", OPTS["day"]),
}
LABELS = {
    "dist": "Distance (km)", "traffic": "Traffic", "weather": "Weather", "age": "Courier age",
    "rating": "Courier rating", "vehicle": "Vehicle", "vcond": "Vehicle condition",
    "multi": "Other deliveries", "otype": "Order type", "festival": "Festival day",
    "city": "City type", "hour": "Order hour", "day": "Day",
}
PRESETS = {
    "Quiet sunny day": {"dist": 3.0, "traffic": "Low", "weather": "Sunny", "multi": "0"},
    "Rush hour in a storm": {"dist": 12.0, "traffic": "Jam", "weather": "Stormy", "multi": "2"},
    "Long trip, clear roads": {"dist": 18.0, "traffic": "Low", "weather": "Sunny", "multi": "0"},
}

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_values(changes):
    for k, v in changes.items():
        if k in OPTS and v not in OPTS[k]:
            continue
        st.session_state[k] = v


def save_scenario(name, values):
    st.session_state["scn_" + name] = values


# ---------- model helpers ----------
def make_row(s):
    return {
        "Delivery_person_Age": s["age"], "Delivery_person_Ratings": s["rating"],
        "distance_km": s["dist"], "order_hour": s["hour"],
        "Vehicle_condition": str(s["vcond"]), "multiple_deliveries": str(s["multi"]),
        "is_weekend": "1" if s["day"] in ("Saturday", "Sunday") else "0",
        "day_of_week": s["day"], "Weather_conditions": s["weather"],
        "Road_traffic_density": s["traffic"], "Type_of_order": s["otype"],
        "Type_of_vehicle": s["vehicle"], "Festival": s["festival"], "City": s["city"],
    }


def predict(rows):
    return model.predict(pd.DataFrame(rows))


def sweep(s, key, values):
    return predict([make_row({**s, key: v}) for v in values])


def band(minutes):
    if minutes <= 20:
        return FAST, "Quick"
    if minutes <= 32:
        return MID, "Typical"
    return SLOW, "Slow"


def pct(v):
    return float(np.clip((v - TMIN) / (TMAX - TMIN), 0, 1) * 100)


# ---------- styling ----------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700&family=DM+Sans:wght@400;500;700&display=swap');
.stApp {{ font-family: 'DM Sans', system-ui, sans-serif; }}
.block-container {{ padding-top: 2rem; max-width: 1150px; }}
.page-title {{ font-family: 'Bricolage Grotesque', 'DM Sans', sans-serif; font-weight: 700; font-size: 2.2rem;
  line-height: 1.15; color: {INK}; margin: 0 0 .25rem 0; }}
.page-sub {{ color: #4B6670; margin: 0 0 1.2rem 0; max-width: 60ch; }}
.est {{ background: #fff; border: 1px solid #D5E1E1; border-left: 8px solid var(--band); border-radius: 10px;
  padding: 1.1rem 1.4rem 1.2rem 1.4rem; }}
.est-top {{ display: flex; justify-content: space-between; align-items: baseline; color: #4B6670; font-size: .95rem; }}
.est-word {{ color: var(--band); font-weight: 700; }}
.est-num {{ font-family: 'Bricolage Grotesque', 'DM Sans', sans-serif; font-weight: 700; font-size: 4.6rem;
  line-height: 1; color: var(--band); margin: .35rem 0 .2rem 0; }}
.est-num span {{ font-size: 1.4rem; color: {INK}; font-weight: 500; }}
.est-range {{ color: {INK}; font-size: .98rem; margin-bottom: .9rem; }}
.track {{ position: relative; height: 10px; border-radius: 5px; background: #DDE6E6; }}
.track .range {{ position: absolute; top: 0; height: 10px; border-radius: 5px; background: var(--band); opacity: .4; }}
.track .mark {{ position: absolute; top: -7px; width: 4px; height: 24px; margin-left: -2px; border-radius: 2px; background: {INK}; }}
.scale {{ display: flex; justify-content: space-between; color: #6B8088; font-size: .8rem; margin-top: .45rem; }}
.small-note {{ color: #4B6670; font-size: .9rem; }}
</style>
""",
    unsafe_allow_html=True,
)


def result_card(pred, title="Estimated delivery time"):
    color, word = band(pred)
    lo, hi = max(pred - MAE, 0), pred + MAE
    left, width = pct(lo), pct(hi) - pct(lo)
    html = (
        f'<div class="est" style="--band:{color}">'
        f'<div class="est-top"><span>{title}</span><span class="est-word">{word}</span></div>'
        f'<div class="est-num">{pred:.0f}<span> min</span></div>'
        f'<div class="est-range">Likely range {lo:.0f} to {hi:.0f} min (average model error {MAE:.1f} min)</div>'
        f'<div class="track"><div class="range" style="left:{left}%;width:{width}%"></div>'
        f'<div class="mark" style="left:{pct(pred)}%"></div></div>'
        f'<div class="scale"><span>{TMIN} min</span><span>{TMAX} min</span></div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


st.markdown(
    '<h1 class="page-title">How long will this delivery take?</h1>'
    '<p class="page-sub">Change any input and the estimate updates straight away. '
    "The model learned from about 45,000 food delivery orders.</p>",
    unsafe_allow_html=True,
)

tab_predict, tab_drivers, tab_compare, tab_about = st.tabs(
    ["Predict", "What changes the time", "Compare two orders", "About the model"]
)

# ---------- Predict ----------
with tab_predict:
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.caption("Quick starts")
        pc = st.columns(len(PRESETS) + 1)
        for col, (name, ch) in zip(pc, PRESETS.items()):
            col.button(name, on_click=apply_values, args=(ch,), key="p_" + name)
        pc[-1].button("Reset", on_click=apply_values, args=(DEFAULTS,), key="p_reset")

        st.subheader("Route and conditions")
        st.slider("Distance, restaurant to customer (km)", 1.0, DIST_MAX, step=0.5, key="dist")
        st.radio("Road traffic", OPTS["traffic"], horizontal=True, key="traffic")
        st.radio("Weather", OPTS["weather"], horizontal=True, key="weather")

        st.subheader("Courier")
        c1, c2 = st.columns(2)
        c1.slider("Age", AGE_MIN, AGE_MAX, key="age")
        c2.slider("Rating", 1.0, 6.0, step=0.1, key="rating")
        c3, c4 = st.columns(2)
        c3.selectbox("Vehicle", OPTS["vehicle"], key="vehicle")
        c4.selectbox("Vehicle condition (0 to 3)", OPTS["vcond"], key="vcond")

        with st.expander("More order details (small effect)"):
            e1, e2 = st.columns(2)
            e1.selectbox("Other deliveries on the same trip", OPTS["multi"], key="multi")
            e2.selectbox("Order type", OPTS["otype"], key="otype")
            e3, e4 = st.columns(2)
            e3.selectbox("Festival day", OPTS["festival"], key="festival")
            e4.selectbox("City type", OPTS["city"], key="city")
            e5, e6 = st.columns(2)
            e5.slider("Order hour (0 to 23)", 0, 23, key="hour")
            e6.selectbox("Day", OPTS["day"], key="day")

    S = {k: st.session_state[k] for k in DEFAULTS}
    pred = float(predict([make_row(S)])[0])

    with right:
        result_card(pred)
        st.write("")
        slower = sweep(S, "traffic", OPTS["traffic"])
        worst = OPTS["traffic"][int(np.argmax(slower))]
        best_t = OPTS["traffic"][int(np.argmin(slower))]
        m1, m2 = st.columns(2)
        m1.metric(f"If traffic were {best_t}", f"{slower.min():.0f} min", f"{slower.min() - pred:+.0f} min", delta_color="inverse")
        m2.metric(f"If traffic were {worst}", f"{slower.max():.0f} min", f"{slower.max() - pred:+.0f} min", delta_color="inverse")
        st.markdown('<p class="small-note">Open the next tabs to see how each input moves the estimate.</p>', unsafe_allow_html=True)

# ---------- Drivers ----------
with tab_drivers:
    st.markdown("Each chart changes **one input** and keeps everything else as set on the Predict tab.")
    color_now, _ = band(pred)

    xs = [float(x) for x in np.arange(1, DIST_MAX + 0.01, 1.0)]
    ys = sweep(S, "dist", xs)
    d1 = pd.DataFrame({"km": xs, "minutes": ys, "lo": ys - MAE, "hi": ys + MAE})
    yenc = alt.Y("minutes:Q", title="Predicted minutes", scale=alt.Scale(zero=False))
    base = alt.Chart(d1).encode(x=alt.X("km:Q", title="Distance (km)"))
    area = base.mark_area(opacity=0.18, color=TEAL).encode(y=alt.Y("lo:Q", title="Predicted minutes"), y2="hi:Q")
    line = base.mark_line(color=INK, strokeWidth=2.5).encode(y=yenc, tooltip=["km", alt.Tooltip("minutes:Q", format=".1f")])
    now = alt.Chart(pd.DataFrame({"km": [S["dist"]], "minutes": [pred]})).mark_point(
        filled=True, size=160, color=color_now, stroke="white", strokeWidth=2
    ).encode(x="km:Q", y="minutes:Q")
    st.markdown("#### Distance")
    st.altair_chart((area + line + now).properties(height=280, width="container"))

    def bars(key, values):
        out = sweep(S, key, values)
        d = pd.DataFrame({"option": values, "minutes": out, "current": [v == S[key] for v in values]})
        b = alt.Chart(d).encode(y=alt.Y("option:N", sort=values, title=None), x=alt.X("minutes:Q", title="Predicted minutes"))
        bar = b.mark_bar(cornerRadiusEnd=3).encode(
            color=alt.condition("datum.current", alt.value(INK), alt.value(MUTED)),
            tooltip=["option", alt.Tooltip("minutes:Q", format=".1f")],
        )
        txt = b.mark_text(align="left", dx=4, color=INK).encode(text=alt.Text("minutes:Q", format=".0f"))
        return (bar + txt).properties(height=30 * len(values) + 30, width="container")

    g1, g2 = st.columns(2, gap="large")
    with g1:
        st.markdown("#### Traffic")
        st.altair_chart(bars("traffic", OPTS["traffic"]))
    with g2:
        st.markdown("#### Weather")
        st.altair_chart(bars("weather", OPTS["weather"]))
    st.caption("Dark bar = your current choice.")

# ---------- Compare ----------
with tab_compare:
    st.markdown("Set up an order on the Predict tab, save it as **A**, change something, then save it as **B**.")
    b1, b2, b3 = st.columns([1, 1, 3])
    b1.button("Save current as A", on_click=save_scenario, args=("A", dict(S)), key="saveA")
    b2.button("Save current as B", on_click=save_scenario, args=("B", dict(S)), key="saveB")
    A, B = st.session_state.get("scn_A"), st.session_state.get("scn_B")
    ca, cb = st.columns(2, gap="large")
    pa = float(predict([make_row(A)])[0]) if A else None
    pb = float(predict([make_row(B)])[0]) if B else None
    with ca:
        result_card(pa, "Order A") if A else st.info("Nothing saved as A yet.")
    with cb:
        result_card(pb, "Order B") if B else st.info("Nothing saved as B yet.")
    if A and B:
        diff = pb - pa
        if abs(diff) < 0.5:
            st.success("A and B have about the same estimate.")
        else:
            st.markdown(f"### Order B is **{abs(diff):.0f} min {'slower' if diff > 0 else 'faster'}** than order A")
        rows = [(LABELS[k], str(A[k]), str(B[k])) for k in DEFAULTS if A[k] != B[k]]
        if rows:
            st.table(pd.DataFrame(rows, columns=["What changed", "A", "B"]).set_index("What changed"))

# ---------- About ----------
with tab_about:
    st.markdown("### How good are the estimates?")
    k1, k2, k3 = st.columns(3)
    k1.metric("Average error", f"{MAE:.1f} min")
    k2.metric("Orders within 5 min", f"{WITHIN5:.0f}%")
    k3.metric("R²", f"{R2:.2f}")
    st.markdown(
        "These numbers come from orders the model never saw during training. "
        "A model that always guesses the average would be off by about 7.6 min."
    )
    fi = BASE / "figures" / "feature_importance.png"
    if fi.exists():
        try:
            img = Image.open(fi)
            img.load()
            st.markdown("### What the model relies on most")
            st.image(img)
        except Exception:
            st.caption("The importance chart could not be read. Re-run the importance cell in 02_modeling.ipynb to recreate it.")
    st.markdown("### Good to know")
    st.markdown(
        "- Traffic, weather, distance and the courier's profile matter most. Day of week and order type barely matter.\n"
        "- The data comes from Indian cities, so estimates for other places may be off.\n"
        "- Courier age, rating and vehicle condition are inputs, so this suits a dispatcher or restaurant more than a customer.\n"
        "- Built by Nak for the AUPP Data Science Club final project."
    )