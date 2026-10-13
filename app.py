#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
EV Price Intelligence
Interactive price-estimation tool for electric vehicles, powered by a
scikit-learn model. Run with:  streamlit run app.py
"""

import html
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EV Price Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_FILENAME = "ev_price_model.joblib"


def find_model_path() -> Path:
    """Look next to app.py; fall back to the working directory when
    __file__ is unavailable (notebooks, Colab, exec-based runners)."""
    try:
        base = Path(__file__).resolve().parent
    except NameError:
        base = Path.cwd()

    for folder in (base, Path.cwd()):
        candidate = folder / MODEL_FILENAME
        if candidate.is_file():
            return candidate
    return base / MODEL_FILENAME


MODEL_PATH = find_model_path()

# ---------------------------------------------------------------------------
# Styling: light, engineering-spec look. One accent colour, one typeface.
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root {
    --paper: #f4f6f8;
    --card: #ffffff;
    --ink: #0f1b2a;
    --muted: #5b6b7e;
    --line: #dbe2ea;
    --accent: #007f6d;
    --accent-soft: #e2f4f0;
    --focus: #0b5fff;
}

html, body, .stApp, [class*="css"] {
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
    font-variant-numeric: tabular-nums;
}
.stApp { background: var(--paper); color: var(--ink); }
.block-container { max-width: 1200px; padding: 2.2rem 1.5rem 3rem; }
h1, h2, h3, h4, p, label, li, span, .stMarkdown { color: var(--ink); }
footer, #MainMenu { visibility: hidden; }

/* ---------- Header ---------- */
.app-header { border-bottom: 1px solid var(--line); padding-bottom: 22px; margin-bottom: 8px; }
.app-header h1 {
    font-size: clamp(28px, 3.6vw, 40px); font-weight: 700;
    letter-spacing: -0.8px; margin: 0 0 8px; line-height: 1.15;
}
.app-header p { color: var(--muted); font-size: 16px; line-height: 1.6; max-width: 720px; margin: 0; }

/* ---------- Section titles ---------- */
.sec-title { font-size: 17px; font-weight: 600; margin: 26px 0 2px; }
.sec-sub { color: var(--muted); font-size: 13.5px; margin-bottom: 14px; }
.group-title {
    font-size: 15px; font-weight: 600; margin: 8px 0 12px;
    padding-left: 10px; border-left: 3px solid var(--accent);
}

/* ---------- Inputs ---------- */
[data-testid="stForm"] {
    background: var(--card); border: 1px solid var(--line);
    border-radius: 10px; padding: 26px 28px;
}
[data-testid="stWidgetLabel"] p { color: var(--ink) !important; font-weight: 500 !important; font-size: 13.5px !important; }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, div[data-baseweb="base-input"] {
    background: #fff !important; border: 1px solid #c3cdd9 !important;
    border-radius: 6px !important; min-height: 42px;
}
div[data-baseweb="select"] *, div[data-baseweb="input"] input, input { color: var(--ink) !important; }
div[data-baseweb="select"] > div:focus-within, div[data-baseweb="input"] > div:focus-within {
    border-color: var(--focus) !important; box-shadow: 0 0 0 2px #0b5fff33 !important;
}
.stNumberInput button { background: #eef2f6 !important; color: var(--ink) !important; border-color: #c3cdd9 !important; }

/* ---------- Buttons ---------- */
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
    min-height: 50px; border-radius: 6px; font-weight: 600; font-size: 15px;
}
.stFormSubmitButton > button {
    background: var(--ink); color: #fff !important; border: 1px solid var(--ink);
}
.stFormSubmitButton > button:hover { background: var(--accent); border-color: var(--accent); }
.stFormSubmitButton > button:focus-visible, .stDownloadButton > button:focus-visible {
    outline: 3px solid var(--focus); outline-offset: 2px;
}
.stDownloadButton > button { background: #fff; color: var(--ink) !important; border: 1px solid #c3cdd9; }
.stDownloadButton > button:hover { border-color: var(--accent); color: var(--accent) !important; }

/* ---------- Result ---------- */
.result {
    background: var(--ink); color: #fff; border-radius: 10px;
    padding: 30px 32px; margin: 8px 0 18px;
}
.result .lbl { color: #9fb3c8; font-size: 14px; font-weight: 500; }
.result .price { font-size: clamp(40px, 6vw, 64px); font-weight: 700; letter-spacing: -1.5px; margin: 6px 0 4px; color: #fff; }
.result .band { color: #7fe3d1; font-size: 15px; font-weight: 500; }
.result .note { color: #9fb3c8; font-size: 13px; margin-top: 14px; line-height: 1.55; }

.kpi { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 18px 20px; height: 100%; }
.kpi .k { color: var(--muted); font-size: 13px; font-weight: 500; }
.kpi .v { font-size: 26px; font-weight: 700; margin-top: 6px; letter-spacing: -0.4px; }
.kpi .h { color: var(--muted); font-size: 12px; margin-top: 4px; }

/* ---------- Spec position bars ---------- */
.bar-row { margin-bottom: 14px; }
.bar-head { display: flex; justify-content: space-between; font-size: 13.5px; margin-bottom: 5px; }
.bar-head span:last-child { color: var(--muted); }
.track { height: 8px; background: #e4eaf0; border-radius: 4px; position: relative; }
.fill { height: 8px; background: var(--accent); border-radius: 4px; }
.scale { display: flex; justify-content: space-between; color: var(--muted); font-size: 11.5px; margin-top: 3px; }

.panel { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 22px 24px; }
.panel p, .panel li { color: var(--ink); font-size: 14.5px; line-height: 1.65; }

.tbl-wrap { overflow-x: auto; background: var(--card); border: 1px solid var(--line); border-radius: 10px; margin-bottom: 8px; }
table.tbl { border-collapse: collapse; width: 100%; font-size: 13.5px; color: var(--ink); }
table.tbl th { background: var(--ink); color: #fff; text-align: left; font-weight: 600; padding: 9px 12px; white-space: nowrap; }
table.tbl td { padding: 8px 12px; border-top: 1px solid var(--line); white-space: nowrap; }
table.tbl tbody tr:nth-child(even) td { background: #f7f9fb; }
button[data-baseweb="tab"] { font-weight: 600; }
[data-testid="stExpander"] { background: var(--card); border-radius: 10px; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 10px; }
@media (max-width: 768px) { [data-testid="stForm"] { padding: 18px 16px; } .result { padding: 22px 20px; } }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Model loading (cached so it only runs once per server session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_bundle(path: str, modified: float) -> dict:
    """`modified` is part of the cache key, so a retrained model file reloads automatically."""
    bundle = joblib.load(path)

    if not isinstance(bundle, dict):
        raise ValueError("The model file must contain a dictionary bundle.")

    missing = {"model", "feature_columns"} - bundle.keys()
    if missing:
        raise ValueError(f"Bundle is missing: {', '.join(sorted(missing))}.")

    if not hasattr(bundle["model"], "predict"):
        raise ValueError("The stored model has no predict() method.")

    cols = bundle["feature_columns"]
    if not isinstance(cols, (list, tuple)) or not cols or not all(isinstance(c, str) for c in cols):
        raise ValueError("feature_columns must be a non-empty list of strings.")
    if len(cols) != len(set(cols)):
        raise ValueError("feature_columns contains duplicates.")

    if not isinstance(bundle.get("hierarchy") or {}, dict):
        raise ValueError("hierarchy must be a dictionary.")
    for key in ("categorical_options", "numeric_ranges", "numeric_defaults"):
        if not isinstance(bundle.get(key, {}), dict):
            raise ValueError(f"{key} must be a dictionary.")
    return bundle


if not MODEL_PATH.is_file():
    st.error("Model file not found. Place `ev_price_model.joblib` in the same folder as `app.py`.")
    st.stop()

try:
    bundle = load_bundle(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
except Exception as err:  # noqa: BLE001 - show any load problem clearly
    st.error("The model could not be loaded. It may be corrupted or built with a different library version.")
    with st.expander("Technical details"):
        st.code(f"{type(err).__name__}: {err}")
    st.stop()

model = bundle["model"]
feature_columns = list(bundle["feature_columns"])
categorical_options = bundle.get("categorical_options", {})
numeric_ranges = bundle.get("numeric_ranges", {})
numeric_defaults = bundle.get("numeric_defaults", {})
metrics = bundle.get("metrics", {})  # optional: {"mae": ..., "rmse": ..., "r2": ..., "n_train": ...}


def estimator_name(m) -> str:
    steps = getattr(m, "steps", None)
    return type(steps[-1][1]).__name__ if steps else type(m).__name__


# ---------------------------------------------------------------------------
# Labels and form layout
# ---------------------------------------------------------------------------
LABELS = {
    "brand": "Brand",
    "model": "Model",
    "year": "Model year",
    "variant": "Trim / variant",
    "battery_capacity_kwh": "Battery capacity (kWh)",
    "range_miles": "Range (miles)",
    "charging_speed_kw": "Peak charging speed (kW)",
    "acceleration_0_60_mph": "0-60 mph (seconds)",
    "top_speed_mph": "Top speed (mph)",
    "horsepower": "Horsepower (hp)",
    "drive_type": "Drivetrain",
    "seating_capacity": "Seats",
    "body_type": "Body style",
    "cargo_volume_cubic_ft": "Cargo volume (cu ft)",
    "weight_kg": "Weight (kg)",
    "safety_rating": "Safety rating",
    "autopilot_level": "Driver-assist level",
    "country_of_origin": "Country of origin",
    "annual_sales_units": "Annual sales (units)",
    "customer_rating": "Customer rating",
}

SECTIONS = {
    "Vehicle identity": ["brand", "model", "year", "variant"],
    "Performance and battery": [
        "battery_capacity_kwh", "range_miles", "charging_speed_kw",
        "acceleration_0_60_mph", "top_speed_mph", "horsepower",
    ],
    "Design and practicality": [
        "drive_type", "seating_capacity", "body_type",
        "cargo_volume_cubic_ft", "weight_kg",
    ],
    "Safety and market": [
        "safety_rating", "autopilot_level", "country_of_origin",
        "annual_sales_units", "customer_rating",
    ],
}

# Any model feature not listed above still gets an input, so nothing is silently dropped.
listed = {c for cols in SECTIONS.values() for c in cols}
extra = [c for c in feature_columns if c not in listed]
if extra:
    SECTIONS["Additional specifications"] = extra


def label_for(col: str) -> str:
    return LABELS.get(col, col.replace("_", " ").capitalize())


def numeric_spec(col: str):
    """Return (min, max, default, is_integer) for a numeric feature."""
    info = numeric_ranges.get(col, {})
    default = float(numeric_defaults.get(col, info.get("median", 0.0)))
    lo = float(info.get("min", 0.0))
    hi = float(info.get("max", max(default, 1.0)))
    if lo == hi:
        hi = lo + 1.0
    default = min(max(default, lo), hi)
    is_int = all(float(v).is_integer() for v in (default, lo, hi))
    return lo, hi, default, is_int


def default_value(col: str):
    """Typical value for a feature: first category, or the numeric default."""
    if col in categorical_options:
        opts = categorical_options[col]
        return opts[0] if opts else None
    _, _, default, is_int = numeric_spec(col)
    return int(default) if is_int else default


# Brand -> model -> variant relationships (optional bundle key "hierarchy").
_hier = bundle.get("hierarchy") or {}
HIER_COLS = list(_hier.get("columns", []))
if not HIER_COLS or not all(c in feature_columns for c in HIER_COLS):
    HIER_COLS = []
HIER_TREE = _hier.get("tree", {})


def options_for(col: str, chosen: dict) -> list:
    """Dropdown options for a column, narrowed by the parents already chosen
    (for example, only the selected brand's models)."""
    base = list(categorical_options.get(col, []))
    if col not in HIER_COLS or HIER_COLS.index(col) == 0:
        return base
    node = HIER_TREE
    for parent in HIER_COLS[:HIER_COLS.index(col)]:
        value = chosen.get(parent)
        if not isinstance(node, dict) or value not in node:
            return base
        node = node[value]
    opts = list(node.keys()) if isinstance(node, dict) else list(node)
    return opts or base


def render_input(col: str, key: str, bias: float = 0.0, chosen: dict = None):
    """Draw the right widget for a feature and return its value.
    bias shifts the starting numeric value: +0.4 moves 40% of the way to the
    maximum, -0.4 moves 40% of the way to the minimum."""
    label = label_for(col)

    if col in categorical_options:
        options = options_for(col, chosen if chosen is not None else {})
        if not options:
            st.warning(f"No options available for {label}.")
            return None
        # If the parent changed and the old choice no longer exists, reset to the first option.
        if key in st.session_state and st.session_state[key] not in options:
            st.session_state[key] = options[0]
        value = st.selectbox(label, options=options, key=key)
        if chosen is not None:
            chosen[col] = value
        return value

    lo, hi, default, is_int = numeric_spec(col)
    start = default + bias * (hi - default) if bias >= 0 else default + bias * (default - lo)
    start = min(max(start, lo), hi)

    if is_int:
        return st.number_input(label, min_value=int(lo), max_value=int(hi),
                               value=int(round(start)), step=1, key=key)
    return st.number_input(label, min_value=lo, max_value=hi, value=float(start),
                           step=0.1, format="%.2f", key=key)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
INK, ACCENT, MUTED, LINE = "#0f1b2a", "#007f6d", "#5b6b7e", "#dbe2ea"
SERIES = [INK, ACCENT, "#d9822b"]
NUM_FEATURES = [c for c in feature_columns if c not in categorical_options and c in numeric_ranges]


def style_fig(fig: go.Figure, height: int = 400) -> go.Figure:
    fig.update_layout(
        template="simple_white", height=height, margin=dict(l=8, r=8, t=36, b=8),
        font=dict(family="IBM Plex Sans, sans-serif", color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.2),
    )
    fig.update_xaxes(gridcolor=LINE)
    fig.update_yaxes(gridcolor=LINE)
    return fig


def kpi(label: str, value: str, hint: str = "", size: int = 26) -> str:
    h = f'<div class="h">{html.escape(hint)}</div>' if hint else ""
    return (f'<div class="kpi"><div class="k">{html.escape(label)}</div>'
            f'<div class="v" style="font-size:{size}px">{html.escape(value)}</div>{h}</div>')


def title(text: str, sub: str = "") -> None:
    st.markdown(f'<div class="sec-title">{html.escape(text)}</div>'
                + (f'<div class="sec-sub">{html.escape(sub)}</div>' if sub else ""),
                unsafe_allow_html=True)


def predict_rows(rows: list) -> np.ndarray:
    df = pd.DataFrame(rows)[feature_columns]
    return np.asarray(model.predict(df), dtype=float)


def cast(col: str, value: float):
    return int(round(value)) if numeric_spec(col)[3] else float(value)


def derived_metrics(v: dict, price: float) -> list:
    """Engineering ratios that can be computed from the available specs."""
    out = []
    rng, bat = v.get("range_miles"), v.get("battery_capacity_kwh")
    hp, kg = v.get("horsepower"), v.get("weight_kg")
    if rng and bat:
        out.append(("Efficiency", f"{rng / bat:.2f} mi/kWh", "Range divided by battery capacity"))
    if rng:
        out.append(("Price per mile of range", f"${price / rng:,.0f}", "Estimated price divided by range"))
    if bat:
        out.append(("Price per kWh", f"${price / bat:,.0f}", "Estimated price divided by battery size"))
    if hp and kg:
        out.append(("Power-to-weight", f"{hp / (kg / 1000):,.0f} hp/tonne", "Horsepower per tonne of weight"))
    return out


def money(x: float) -> str:
    return f"${x:,.0f}"


def signed_money(x: float) -> str:
    return f"{'+' if x >= 0 else '-'}${abs(x):,.0f}"


def baseline_values() -> dict:
    """The vehicle configured in the estimator tab (typical values until submitted)."""
    return {c: st.session_state.get(f"est_{c}", default_value(c)) for c in feature_columns}


def show_table(df: pd.DataFrame, index: bool = False) -> None:
    """Render a table as styled HTML so it matches the app in any Streamlit theme."""
    html_table = df.to_html(index=index, border=0, classes="tbl", escape=True, na_rep="")
    st.markdown(f'<div class="tbl-wrap">{html_table}</div>', unsafe_allow_html=True)


def show_error(message: str, err: Exception) -> None:
    st.error(message)
    with st.expander("Technical details"):
        st.code(f"{type(err).__name__}: {err}")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="app-header">
    <h1>EV Price Intelligence</h1>
    <p>Estimate EV prices from specifications, compare vehicles side by side,
    explore how each spec moves the price, and review how well the model performs.</p>
</div>
""",
    unsafe_allow_html=True,
)


# Leakage guard: a feature whose name contains "price" is probably derived from the target.
suspect = [c for c in feature_columns if "price" in c.lower()]
if suspect:
    st.warning("Possible data leakage: the model uses " + ", ".join(f"`{c}`" for c in suspect) +
               ", which looks derived from the price. Retrain without it for reliable estimates.")

if "history" not in st.session_state:
    st.session_state.history = []


# ---------------------------------------------------------------------------
# TAB 1: Price estimator
# ---------------------------------------------------------------------------
def tab_estimator() -> None:
    title("Vehicle specifications",
          "Defaults are the typical (median) values from the training data.")

    input_values = {}

    identity = [c for c in SECTIONS.get("Vehicle identity", []) if c in feature_columns]
    if identity:
        st.markdown('<div class="group-title">Vehicle identity</div>', unsafe_allow_html=True)
        left, right = st.columns(2, gap="large")
        for i, col in enumerate(identity):
            with (left if i % 2 == 0 else right):
                input_values[col] = render_input(col, key=f"est_{col}", chosen=input_values)
        if HIER_COLS:
            st.caption("Model and trim options update to match the selected brand.")
        elif {"brand", "model"} <= set(feature_columns):
            st.caption("This model file has no brand-to-model links. Re-run train_model.py to enable them.")

    with st.form("ev_prediction_form"):
        for section, cols in SECTIONS.items():
            if section == "Vehicle identity":
                continue
            active = [c for c in cols if c in feature_columns]
            if not active:
                continue
            st.markdown(f'<div class="group-title">{html.escape(section)}</div>', unsafe_allow_html=True)
            left, right = st.columns(2, gap="large")
            for i, col in enumerate(active):
                with (left if i % 2 == 0 else right):
                    input_values[col] = render_input(col, key=f"est_{col}")
            st.write("")
        submitted = st.form_submit_button("Estimate price", use_container_width=True)

    if submitted:
        try:
            input_df = pd.DataFrame([[input_values[c] for c in feature_columns]], columns=feature_columns)
            prediction = float(model.predict(input_df)[0])
        except Exception as err:  # noqa: BLE001
            show_error("The estimate could not be calculated. Check that the inputs match the training data.", err)
            return

        if prediction <= 0:
            st.error("The model returned a non-positive price. Try different inputs or retrain the model.")
            return

        mae = metrics.get("mae")
        band_html = ""
        if isinstance(mae, (int, float)) and mae > 0:
            band_html = (f'<div class="band">Typical error range: '
                         f'{money(max(prediction - mae, 0))} to {money(prediction + mae)}</div>')

        title("Estimate")
        result_html = (
            '<div class="result">'
            '<div class="lbl">Estimated price (USD)</div>'
            f'<div class="price">{money(prediction)}</div>'
            f'{band_html}'
            '<div class="note">A model-generated estimate based on the specifications entered. '
            'It is not a dealer quote or a guaranteed market price.</div>'
            '</div>'
        )
        st.markdown(result_html, unsafe_allow_html=True)

        kpis = derived_metrics(input_values, prediction)
        if kpis:
            for c, (k, val, hint) in zip(st.columns(len(kpis), gap="medium"), kpis):
                c.markdown(kpi(k, val, hint), unsafe_allow_html=True)

        bars = []
        for col in feature_columns:
            info, val = numeric_ranges.get(col), input_values.get(col)
            if col in categorical_options or not info or not isinstance(val, (int, float)):
                continue
            lo, hi = float(info.get("min", 0)), float(info.get("max", 0))
            if hi <= lo:
                continue
            pct = min(max((float(val) - lo) / (hi - lo), 0), 1) * 100
            bars.append(
                f'<div class="bar-row"><div class="bar-head"><span>{html.escape(label_for(col))}</span>'
                f'<span>{val:,.4g}</span></div>'
                f'<div class="track"><div class="fill" style="width:{pct:.0f}%"></div></div>'
                f'<div class="scale"><span>{lo:,.4g}</span><span>{hi:,.4g}</span></div></div>')
        if bars:
            title("How this vehicle compares with the training data",
                  "Each bar shows where an input sits between the lowest and highest values the model has seen. "
                  "Inputs near the edges are less reliable.")
            half = (len(bars) + 1) // 2
            b1, b2 = st.columns(2, gap="large")
            b1.markdown('<div class="panel">' + "".join(bars[:half]) + "</div>", unsafe_allow_html=True)
            b2.markdown('<div class="panel">' + "".join(bars[half:]) + "</div>", unsafe_allow_html=True)

        st.session_state.history.append({
            "Scenario": len(st.session_state.history) + 1,
            **{label_for(c): input_values[c] for c in feature_columns},
            "Estimated price (USD)": round(prediction),
        })

        report = input_df.copy()
        report["estimated_price_usd"] = round(prediction, 2)
        report["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.write("")
        st.download_button("Download this estimate (CSV)",
                           data=report.to_csv(index=False).encode("utf-8"),
                           file_name="ev_price_estimate.csv", mime="text/csv",
                           use_container_width=True)

    if st.session_state.history:
        title("Scenario history",
              "Every estimate from this session is kept here. Use the Compare vehicles tab for live side-by-side analysis.")
        hist = pd.DataFrame(st.session_state.history)
        show_table(hist)
        c_a, c_b = st.columns(2)
        c_a.download_button("Download all scenarios (CSV)", data=hist.to_csv(index=False).encode("utf-8"),
                            file_name="ev_price_scenarios.csv", mime="text/csv", use_container_width=True)
        if c_b.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()


# ---------------------------------------------------------------------------
# TAB 2: Compare vehicles (live, no submit button)
# ---------------------------------------------------------------------------
def tab_compare() -> None:
    title("Compare vehicles",
          "Edit up to three configurations and the comparison updates instantly. "
          "Vehicle B starts above and Vehicle C below the typical values so differences show straight away.")

    n = st.radio("Number of vehicles", [2, 3], horizontal=True, key="cmp_n")
    names = ["Vehicle A", "Vehicle B", "Vehicle C"][:n]
    biases = [0.0, 0.4, -0.4][:n]

    results = st.container()  # filled after the inputs below are read
    st.write("")

    vehicles = []
    for i, col_area in enumerate(st.columns(n, gap="medium")):
        with col_area:
            with st.expander(f"{names[i]} specifications", expanded=(i == 0)):
                vals = {}
                ordered = HIER_COLS + [c for c in feature_columns if c not in HIER_COLS]
                for col in ordered:
                    vals[col] = render_input(col, key=f"cmp{i}_{col}", bias=biases[i], chosen=vals)
                vehicles.append(vals)

    try:
        prices = predict_rows(vehicles)
    except Exception as err:  # noqa: BLE001
        with results:
            show_error("The comparison could not be calculated.", err)
        return

    with results:
        # Price cards with change versus Vehicle A
        for i, c in enumerate(st.columns(n, gap="medium")):
            hint = "Reference vehicle" if i == 0 else (
                f"{signed_money(prices[i] - prices[0])} ({(prices[i] / prices[0] - 1) * 100:+.1f}%) vs Vehicle A"
                if prices[0] else "")
            c.markdown(kpi(names[i], money(prices[i]), hint, size=30), unsafe_allow_html=True)
        st.write("")

        left, right = st.columns(2, gap="large")

        with left, st.container(border=True):
            fig = go.Figure(go.Bar(x=names, y=prices, marker_color=SERIES[:n],
                                   text=[money(p) for p in prices], textposition="outside",
                                   hovertemplate="%{x}: $%{y:,.0f}<extra></extra>"))
            fig.update_layout(title="Estimated price", showlegend=False,
                              yaxis=dict(tickprefix="$", tickformat=",.0f", rangemode="tozero"))
            st.plotly_chart(style_fig(fig, 380), theme=None)

        with right, st.container(border=True):
            radar_cols = NUM_FEATURES[:8]
            if len(radar_cols) >= 3:
                fig = go.Figure()
                for i, veh in enumerate(vehicles):
                    r = []
                    for col in radar_cols:
                        lo, hi, _, _ = numeric_spec(col)
                        r.append(min(max((float(veh[col]) - lo) / (hi - lo), 0), 1))
                    fig.add_trace(go.Scatterpolar(
                        r=r + r[:1], theta=[label_for(c) for c in radar_cols] + [label_for(radar_cols[0])],
                        name=names[i], fill="toself", opacity=0.45, line=dict(color=SERIES[i])))
                fig.update_layout(title="Specification profile (0 = lowest, 1 = highest seen)",
                                  polar=dict(radialaxis=dict(range=[0, 1], gridcolor=LINE)))
                st.plotly_chart(style_fig(fig, 380), theme=None)
            else:
                st.info("At least three numeric features are needed for the profile chart.")

        # Specification table
        title("Specification comparison")
        only_diff = st.toggle("Show only differences", value=True, key="cmp_diff")
        table = pd.DataFrame({names[i]: {label_for(c): str(vehicles[i][c]) for c in feature_columns}
                              for i in range(n)})
        if only_diff:
            table = table[table.nunique(axis=1) > 1]
        show_table(table if len(table) else pd.DataFrame({"Result": ["No differences"]}), index=len(table) > 0)

        # Derived metrics table
        rows = {}
        for i in range(n):
            for label, value, _ in derived_metrics(vehicles[i], prices[i]):
                rows.setdefault(label, {})[names[i]] = value
        if rows:
            title("Engineering ratios")
            show_table(pd.DataFrame(rows).T, index=True)


# ---------------------------------------------------------------------------
# TAB 3: Price explorer (what-if charts)
# ---------------------------------------------------------------------------
def tab_explorer() -> None:
    title("Price explorer",
          "What-if charts. Each one changes one or two specifications and holds everything else at the vehicle "
          "set in the Price estimator tab (typical values until you change them).")

    base = baseline_values()
    try:
        current = float(predict_rows([base])[0])
    except Exception as err:  # noqa: BLE001
        show_error("The explorer could not score the baseline vehicle.", err)
        return

    st.markdown(kpi("Baseline vehicle estimate", money(current),
                    "Charts below show how this price moves as specifications change"),
                unsafe_allow_html=True)
    st.write("")

    # --- 1. Sensitivity curve ---
    with st.container(border=True):
        st.markdown('<div class="group-title">Price sensitivity</div>', unsafe_allow_html=True)
        if not NUM_FEATURES:
            st.info("No numeric features with known ranges are available.")
        else:
            feat = st.selectbox("Specification to vary", NUM_FEATURES, format_func=label_for, key="ex_feat")
            lo, hi, _, is_int = numeric_spec(feat)
            xs = np.linspace(lo, hi, 40)
            if is_int:
                xs = np.unique(np.round(xs))
            ys = predict_rows([{**base, feat: cast(feat, x)} for x in xs])

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Estimated price",
                                     line=dict(color=ACCENT, width=3),
                                     hovertemplate=f"{label_for(feat)}: %{{x:,.4g}}<br>Price: $%{{y:,.0f}}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[base[feat]], y=[current], mode="markers", name="Baseline vehicle",
                                     marker=dict(color=INK, size=12)))
            fig.update_layout(xaxis_title=label_for(feat), yaxis=dict(tickprefix="$", tickformat=",.0f", title="Estimated price"))
            st.plotly_chart(style_fig(fig, 380), theme=None)
            st.caption((f"Across the observed range of {label_for(feat).lower()}, the estimate moves by "
                        f"{signed_money(ys[-1] - ys[0])} (from {money(ys[0])} to {money(ys[-1])}).").replace("$", "\\$"))

    # --- 2. Two-way heatmap ---
    with st.container(border=True):
        st.markdown('<div class="group-title">Two-feature price map</div>', unsafe_allow_html=True)
        if len(NUM_FEATURES) < 2:
            st.info("Two numeric features are needed for the price map.")
        else:
            c1, c2 = st.columns(2)
            fx = c1.selectbox("Horizontal axis", NUM_FEATURES, format_func=label_for, key="ex_hx")
            fy = c2.selectbox("Vertical axis", NUM_FEATURES, index=1, format_func=label_for, key="ex_hy")
            if fx == fy:
                st.info("Choose two different specifications.")
            else:
                g = 25
                gx = np.linspace(numeric_spec(fx)[0], numeric_spec(fx)[1], g)
                gy = np.linspace(numeric_spec(fy)[0], numeric_spec(fy)[1], g)
                rows = [{**base, fx: cast(fx, x), fy: cast(fy, y)} for y in gy for x in gx]
                z = predict_rows(rows).reshape(g, g)
                fig = go.Figure(go.Heatmap(
                    x=gx, y=gy, z=z, colorscale=[[0, "#e2f4f0"], [1, INK]],
                    colorbar=dict(title="USD", tickprefix="$"),
                    hovertemplate=(f"{label_for(fx)}: %{{x:,.4g}}<br>{label_for(fy)}: %{{y:,.4g}}"
                                   "<br>Price: $%{z:,.0f}<extra></extra>")))
                fig.add_trace(go.Scatter(x=[base[fx]], y=[base[fy]], mode="markers", name="Baseline vehicle",
                                         marker=dict(color="#d9822b", size=13, line=dict(color="#fff", width=2))))
                fig.update_layout(xaxis_title=label_for(fx), yaxis_title=label_for(fy))
                st.plotly_chart(style_fig(fig, 460), theme=None)

    # --- 3. Category impact ---
    with st.container(border=True):
        st.markdown('<div class="group-title">Price by category</div>', unsafe_allow_html=True)
        cat_cols = [c for c in feature_columns if categorical_options.get(c)]
        if not cat_cols:
            st.info("No categorical features are available.")
        else:
            cat = st.selectbox("Category", cat_cols, format_func=label_for, key="ex_cat")
            options = options_for(cat, base)[:40]
            rows = []
            for o in options:
                row = {**base, cat: o}
                if cat in HIER_COLS:  # a new brand/model needs valid children
                    for child in HIER_COLS[HIER_COLS.index(cat) + 1:]:
                        valid = options_for(child, row)
                        if row[child] not in valid:
                            row[child] = valid[0]
                rows.append(row)
            ys = predict_rows(rows)
            order = np.argsort(ys)
            labels = [str(options[i]) for i in order]
            colors = [INK if options[i] == base[cat] else "#b8c4d2" for i in order]
            fig = go.Figure(go.Bar(x=ys[order], y=labels, orientation="h", marker_color=colors,
                                   hovertemplate="%{y}: $%{x:,.0f}<extra></extra>"))
            fig.update_layout(xaxis=dict(tickprefix="$", tickformat=",.0f", title="Estimated price"),
                              yaxis=dict(type="category"), showlegend=False)
            st.plotly_chart(style_fig(fig, max(320, 26 * len(options) + 80)), theme=None)
            st.caption("Dark bar = current baseline selection. Other specifications are held fixed, and when the brand or model "
                       "changes, its model and trim switch to that brand's first listing.")


# ---------------------------------------------------------------------------
# TAB 4: Model performance dashboard
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Scoring test data...")
def run_predictions(_model, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(_model.predict(X), dtype=float)


def evaluate(actual, predicted) -> dict:
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    err = p - a
    ss_tot = np.sum((a - a.mean()) ** 2)
    nz = a != 0
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
        "mape": float(np.mean(np.abs(err[nz] / a[nz])) * 100) if nz.any() else float("nan"),
        "n": int(len(a)),
    }


def feature_importance():
    """Aggregate model importances back to the original input features, if available."""
    stored = bundle.get("feature_importance")
    if isinstance(stored, dict) and stored:
        return pd.Series(stored, dtype=float).sort_values(ascending=False)
    try:
        steps = getattr(model, "steps", None)
        est = steps[-1][1] if steps else model
        if hasattr(est, "feature_importances_"):
            vals = np.asarray(est.feature_importances_, float)
        elif hasattr(est, "coef_"):
            vals = np.abs(np.ravel(est.coef_)).astype(float)
        else:
            return None
        names = list(model[:-1].get_feature_names_out()) if steps else list(feature_columns)
        if len(names) != len(vals):
            return None
    except Exception:  # noqa: BLE001
        return None

    agg = {c: 0.0 for c in feature_columns}
    for name, val in zip(names, vals):
        base_name = name.split("__", 1)[-1]
        hits = [c for c in feature_columns if base_name == c or base_name.startswith(c + "_")]
        if hits:
            agg[max(hits, key=len)] += float(val)
    s = pd.Series(agg)
    s = s[s > 0]
    return (s / s.sum()).sort_values(ascending=False) if len(s) else None


def render_dashboard(df: pd.DataFrame) -> None:
    m = evaluate(df["actual"], df["predicted"])
    df = df.assign(error=df["predicted"] - df["actual"])
    df["abs_error"] = df["error"].abs()
    df["pct_error"] = np.where(df["actual"] != 0, df["abs_error"] / df["actual"].abs() * 100, np.nan)

    cards = [("Mean absolute error", money(m["mae"]), "Average miss in dollars"),
             ("RMSE", money(m["rmse"]), "Penalises large misses"),
             ("R²", f"{m['r2']:.3f}", "Share of price variation explained"),
             ("Average % error", f"{m['mape']:.1f}%", "Typical miss relative to price"),
             ("Records scored", f"{m['n']:,}", "")]
    for c, (k, v, h) in zip(st.columns(len(cards), gap="medium"), cards):
        c.markdown(kpi(k, v, h, size=22), unsafe_allow_html=True)
    st.write("")

    left, right = st.columns(2, gap="large")
    with left, st.container(border=True):
        lo, hi = float(min(df.actual.min(), df.predicted.min())), float(max(df.actual.max(), df.predicted.max()))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
                                 line=dict(color=MUTED, dash="dash")))
        fig.add_trace(go.Scatter(x=df.actual, y=df.predicted, mode="markers", name="Vehicles",
                                 marker=dict(color=ACCENT, size=7, opacity=0.65),
                                 hovertemplate="Actual: $%{x:,.0f}<br>Predicted: $%{y:,.0f}<extra></extra>"))
        fig.update_layout(title="Actual vs predicted price", xaxis=dict(title="Actual", tickprefix="$", tickformat=",.0f"),
                          yaxis=dict(title="Predicted", tickprefix="$", tickformat=",.0f"))
        st.plotly_chart(style_fig(fig, 380), theme=None)

    with right, st.container(border=True):
        fig = go.Figure(go.Histogram(x=df.error, nbinsx=30, marker_color=INK,
                                     hovertemplate="Error: $%{x:,.0f}<br>Count: %{y}<extra></extra>"))
        fig.add_vline(x=0, line_dash="dash", line_color=ACCENT)
        fig.update_layout(title="Prediction errors (predicted minus actual)", showlegend=False,
                          xaxis=dict(title="Error", tickprefix="$", tickformat=",.0f"), yaxis_title="Vehicles")
        st.plotly_chart(style_fig(fig, 380), theme=None)

    left, right = st.columns(2, gap="large")
    with left, st.container(border=True):
        if df.actual.nunique() >= 3:
            band = pd.qcut(df.actual, q=min(5, df.actual.nunique()), duplicates="drop")
            grp = df.groupby(band, observed=True)["pct_error"].mean()
            labels = [f"${iv.left:,.0f} to ${iv.right:,.0f}" for iv in grp.index]
            fig = go.Figure(go.Bar(x=labels, y=grp.values, marker_color=ACCENT,
                                   hovertemplate="%{x}<br>Average error: %{y:.1f}%<extra></extra>"))
            fig.update_layout(title="Average % error by price band", showlegend=False, yaxis=dict(ticksuffix="%"),
                              xaxis_title="Actual price band")
            st.plotly_chart(style_fig(fig, 360), theme=None)
        else:
            st.info("Not enough distinct prices for a price-band chart.")

    with right, st.container(border=True):
        groups = [c for c in categorical_options if c in df.columns and 1 < df[c].nunique() <= 60]
        if groups:
            g = st.selectbox("Break down error by", groups, format_func=label_for, key="perf_group")
            agg = df.groupby(g).agg(mae=("abs_error", "mean"), n=("abs_error", "size")).nlargest(15, "n")
            agg = agg.sort_values("mae")
            fig = go.Figure(go.Bar(x=agg["mae"], y=agg.index.astype(str), orientation="h", marker_color=INK,
                                   customdata=agg["n"],
                                   hovertemplate="%{y}<br>MAE: $%{x:,.0f}<br>Vehicles: %{customdata}<extra></extra>"))
            fig.update_layout(title=f"Mean absolute error by {label_for(g).lower()} (15 most common)",
                              showlegend=False, xaxis=dict(tickprefix="$", tickformat=",.0f"), yaxis=dict(type="category"))
            st.plotly_chart(style_fig(fig, 360), theme=None)
        else:
            st.info("Upload test data that includes the categorical columns to see errors by group.")

    id_cols = [c for c in ("brand", "model", "year", "variant") if c in df.columns]
    worst = df.nlargest(10, "abs_error")[id_cols + ["actual", "predicted", "error"]].rename(
        columns={"actual": "Actual", "predicted": "Predicted", "error": "Error"})
    title("Largest prediction errors", "Review these to see where the model struggles.")
    show_table(worst.round(0))


def tab_performance() -> None:
    # ----- Overview -----
    title("Model overview")
    trained = datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).strftime("%d %b %Y")
    cards = [("Algorithm", estimator_name(model)), ("Input features", str(len(feature_columns))),
             ("Model file updated", trained)]
    if isinstance(metrics.get("r2"), (int, float)):
        cards.append(("R² (held-out)", f"{metrics['r2']:.3f}"))
    if isinstance(metrics.get("mae"), (int, float)):
        cards.append(("Mean absolute error", money(metrics["mae"])))
    if isinstance(metrics.get("n_train"), (int, float)):
        cards.append(("Training records", f"{int(metrics['n_train']):,}"))
    for c, (k, v) in zip(st.columns(len(cards), gap="medium"), cards):
        c.markdown(kpi(k, v, size=22), unsafe_allow_html=True)

    # ----- Evaluation data -----
    title("Accuracy on test data",
          "Upload a CSV of vehicles the model has not seen, with the actual price column, to score it live.")

    eval_df = None
    source = ""
    up = st.file_uploader("Test data (CSV)", type="csv", key="perf_csv")
    if up is not None:
        try:
            raw = pd.read_csv(up)
            guess = next((i for i, c in enumerate(raw.columns) if "price" in c.lower()), 0)
            target = st.selectbox("Column holding the actual price", list(raw.columns), index=guess, key="perf_target")
            missing = [c for c in feature_columns if c not in raw.columns]
            if missing:
                st.error("The file is missing these model features: " + ", ".join(missing))
            else:
                data = raw.copy()
                data[target] = pd.to_numeric(data[target], errors="coerce")
                data = data.dropna(subset=feature_columns + [target]).reset_index(drop=True)
                if data.empty:
                    st.error("No complete rows were found after removing missing values.")
                else:
                    data["actual"] = data[target]
                    data["predicted"] = run_predictions(model, data[feature_columns])
                    eval_df, source = data, f"Scored {len(data):,} rows from the uploaded file."
        except Exception as err:  # noqa: BLE001
            show_error("The test file could not be scored. Check column names and category values.", err)
    else:
        stored = bundle.get("evaluation")
        if isinstance(stored, dict) and "y_true" in stored and "y_pred" in stored \
                and len(stored["y_true"]) == len(stored["y_pred"]) > 0:
            eval_df = pd.DataFrame({"actual": np.asarray(stored["y_true"], float),
                                    "predicted": np.asarray(stored["y_pred"], float)})
            source = "Showing the held-out test results stored in the model file."

    if eval_df is not None:
        st.caption(source)
        render_dashboard(eval_df)
    else:
        st.info("No test data yet. Upload a CSV above, or store `evaluation` (y_true, y_pred) in the model bundle "
                "so this dashboard is populated on first load.")

    # ----- Feature importance -----
    imp = feature_importance()
    if imp is not None:
        title("What drives the price", "Share of the model's decisions attributed to each input feature.")
        top = imp.head(15).iloc[::-1]
        with st.container(border=True):
            fig = go.Figure(go.Bar(x=top.values * 100, y=[label_for(c) for c in top.index], orientation="h",
                                   marker_color=ACCENT, hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
            fig.update_layout(showlegend=False, xaxis=dict(ticksuffix="%"), yaxis=dict(type="category"))
            st.plotly_chart(style_fig(fig, max(320, 28 * len(top) + 80)), theme=None)
        st.caption("Importance shows what the model relies on, not what causes a price to change.")

    with st.expander("Method and limitations"):
        st.markdown(
            "**How it works.** A supervised regression model learned the relationship between vehicle "
            "specifications and prices from a historical EV dataset. Inputs pass through the same preprocessing "
            "used in training, and the model returns a price in US dollars.\n\n"
            "**How to read the results.** Treat estimates as benchmarks for comparison, not quotes. Accuracy drops "
            "for unusual configurations, so check how inputs sit within the training range.\n\n"
            "**Limitations.** The model does not account for incentives, regional taxes, option packages, new "
            "technology, or market changes after the data was collected.")
        st.write("Features used: " + ", ".join(label_for(c) for c in feature_columns))


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
tab_a, tab_b, tab_c, tab_d = st.tabs(
    ["Price estimator", "Compare vehicles", "Price explorer", "Model performance"])
with tab_a:
    tab_estimator()
with tab_b:
    tab_compare()
with tab_c:
    tab_explorer()
with tab_d:
    tab_performance()

st.markdown("---")
st.markdown(
    '<div style="color:#5b6b7e;font-size:12.5px;text-align:center;">'
    "EV Price Intelligence. Built with Python, scikit-learn, Plotly and Streamlit."
    "</div>",
    unsafe_allow_html=True,
)


# In[ ]:




