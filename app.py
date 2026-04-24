"""
============================================================
  STREAMLIT APP — Tax Appeal Lead Scoring System
  5 Tabs: Prediction | Visualizations | Code Output | Description | Chatbot

  Model: generate_lead_scoring_model.py (v2.2)
  Columns: Owner_City, Owner_ZipCode, Property_Type,
           num_ExemptionCode, Properties_Count,
           MaxTrestlescore, total_market_value, client_status
  Removed: Owner_State, CountyName, num_ocaluc, SalesLeadID
============================================================
  Run: streamlit run app.py
============================================================
"""

import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import requests

# ──────────────────────────────────────────────────────────────
#  PAGE CONFIG  — must be FIRST st call
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tax Appeal Lead Scorer",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────
#  CUSTOM CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: #f8fafc; }

    .hero-banner {
        background: linear-gradient(135deg,#1e3a8a 0%,#3b82f6 50%,#0ea5e9 100%);
        padding:2rem 2.5rem; border-radius:16px; margin-bottom:1.5rem;
        color:white; box-shadow:0 4px 24px rgba(59,130,246,.3);
    }
    .hero-banner h1 { font-size:2rem; font-weight:700; margin:0; }
    .hero-banner p  { font-size:1rem; opacity:.85; margin:.4rem 0 0; }

    .metric-card {
        background:white; border-radius:14px; padding:1.4rem 1.6rem;
        box-shadow:0 2px 12px rgba(0,0,0,.07); border-left:5px solid #3b82f6; height:100%;
    }
    .metric-card.green  { border-left-color:#22c55e; }
    .metric-card.orange { border-left-color:#f97316; }
    .metric-card.purple { border-left-color:#8b5cf6; }
    .metric-card.red    { border-left-color:#ef4444; }
    .metric-card h3   { font-size:.8rem; text-transform:uppercase; letter-spacing:.08em; color:#6b7280; margin:0 0 .4rem; }
    .metric-card .value { font-size:2rem; font-weight:700; color:#111827; }
    .metric-card .sub   { font-size:.78rem; color:#6b7280; margin-top:.2rem; }

    .verdict-convert {
        background:linear-gradient(90deg,#d1fae5,#a7f3d0);
        border:2px solid #22c55e; border-radius:14px;
        padding:1.5rem 2rem; text-align:center; margin:1rem 0;
    }
    .verdict-no-convert {
        background:linear-gradient(90deg,#fee2e2,#fecaca);
        border:2px solid #ef4444; border-radius:14px;
        padding:1.5rem 2rem; text-align:center; margin:1rem 0;
    }
    .verdict-convert h2    { color:#065f46; margin:0; font-size:1.6rem; }
    .verdict-no-convert h2 { color:#991b1b; margin:0; font-size:1.6rem; }
    .verdict-convert p, .verdict-no-convert p { margin:.4rem 0 0; color:#374151; }

    .section-title {
        font-size:1rem; font-weight:600; color:#374151;
        border-bottom:2px solid #e5e7eb; padding-bottom:.4rem; margin:1.2rem 0 .8rem;
    }
    .code-area {
        background:#0f172a; color:#e2e8f0; border-radius:12px; padding:1.4rem;
        font-family:'Courier New',monospace; font-size:.82rem;
        line-height:1.6; overflow-x:auto; white-space:pre-wrap;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap:8px; background:white; border-radius:12px; padding:6px;
        box-shadow:0 2px 8px rgba(0,0,0,.06); margin-bottom:1rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius:8px; padding:8px 20px; font-weight:500; font-size:.9rem;
    }
    .stTabs [aria-selected="true"] { background:#3b82f6 !important; color:white !important; }

    .chat-bubble-user {
        background:#3b82f6; color:white;
        border-radius:18px 18px 4px 18px;
        padding:.75rem 1rem; margin:.4rem 0 .4rem 3rem;
        font-size:.9rem; line-height:1.5;
    }
    .chat-bubble-assistant {
        background:white; color:#111827;
        border-radius:18px 18px 18px 4px;
        padding:.75rem 1rem; margin:.4rem 3rem .4rem 0;
        font-size:.9rem; line-height:1.5;
        box-shadow:0 2px 8px rgba(0,0,0,.08);
    }
    .chat-label-user      { text-align:right; font-size:.72rem; color:#6b7280; margin-right:.3rem; }
    .chat-label-assistant { font-size:.72rem; color:#6b7280; margin-left:.3rem; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
#  SESSION STATE — initialise BEFORE any widget
# ──────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


# ──────────────────────────────────────────────────────────────
#  LOAD MODEL
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path="lead_scoring_models.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

MODEL_LOADED = False
bundle = None
conv_model = None
fit_meta = None

try:
    bundle     = load_model()
    conv_model = bundle["conv_model"]
    fit_meta   = bundle["fit_meta"]
    MODEL_LOADED = True
except Exception as e:
    pass


# ──────────────────────────────────────────────────────────────
#  INFERENCE HELPERS
#  Mirrors engineer_features() in generate_lead_scoring_model.py
#  Uses only: Owner_City, Owner_ZipCode, Property_Type,
#             num_ExemptionCode, Properties_Count,
#             MaxTrestlescore, total_market_value
# ──────────────────────────────────────────────────────────────
def build_inference_row(inputs: dict, fm: dict) -> pd.DataFrame:
    """
    Manually reproduce every feature created by engineer_features()
    so that the column order & names exactly match fit_meta["feature_cols"].
    """
    # ── Raw inputs (normalised) ──
    city    = str(inputs["Owner_City"]).strip().upper()
    zipcode = str(inputs["Owner_ZipCode"]).strip().zfill(5)   # always 5-char string
    ptype   = str(inputs["Property_Type"]).strip().upper()
    n_exemp = float(inputs["num_ExemptionCode"])
    n_prop  = float(inputs["Properties_Count"])
    trestle = float(inputs["MaxTrestlescore"])
    mktval  = float(inputs["total_market_value"])

    row = {}

    # ── Raw numeric features ──
    row["num_ExemptionCode"]  = n_exemp
    row["Properties_Count"]   = n_prop
    row["MaxTrestlescore"]    = trestle
    row["total_market_value"] = mktval

    # ── 1. Log transform ──
    row["log_market_value"] = np.log1p(max(mktval, 0))
    lv = row["log_market_value"]

    # ── 2. Quantile binary flags ──
    row["high_trestle"]         = int(trestle > fm["q75_trestle"])
    row["med_trestle"]          = int(fm["q50_trestle"] < trestle <= fm["q75_trestle"])
    row["low_trestle"]          = int(trestle < fm["q25_trestle"])
    row["high_value"]           = int(mktval  > fm["q75_value"])
    row["low_value"]            = int(mktval  < fm["q25_value"])
    row["multi_property"]       = int(n_prop  > 1)
    row["large_portfolio"]      = int(n_prop  >= 5)
    row["very_large_portfolio"] = int(n_prop  >= 10)
    row["has_exemption"]        = int(n_exemp > 0)
    row["multi_exemption"]      = int(n_exemp > 1)

    # ── 3. Interaction features ──
    row["value_x_trestle"]     = lv * trestle
    row["portfolio_x_value"]   = n_prop * lv
    row["exemption_x_value"]   = row["has_exemption"] * lv
    row["trestle_x_portfolio"] = trestle * n_prop

    # ── 4. City frequency encoding ──
    row["owner_city_frequency"] = fm.get("city_freq", {}).get(city, 0)

    # ── 5. ZipCode frequency encoding ──
    row["zip_frequency"] = fm.get("zip_freq", {}).get(zipcode, 0)

    # ── 6. One-hot: Property_Type_enc ──
    top_ptypes = fm.get("top_prop_types", [])
    pc = ptype if ptype in top_ptypes else "OTHER"
    all_ptypes = top_ptypes + (["OTHER"] if "OTHER" not in top_ptypes else [])
    for pt in all_ptypes:
        row[f"Property_Type_enc_{pt}"] = int(pc == pt)

    # ── 7. One-hot: City_enc ──
    top_cities = fm.get("top_cities", [])
    cc = city if city in top_cities else "OTHER"
    all_cities = top_cities + (["OTHER"] if "OTHER" not in top_cities else [])
    for ct in all_cities:
        row[f"City_enc_{ct}"] = int(cc == ct)

    # ── Align to exact feature_cols from training ──
    feature_cols = fm["feature_cols"]
    aligned = {col: row.get(col, 0) for col in feature_cols}
    return pd.DataFrame([aligned])


def predict_lead(inputs: dict) -> dict:
    """
    Run inference. Returns conversion probability + segment.
    Revenue estimate uses market value input directly (no value_model).
    """
    X         = build_inference_row(inputs, fit_meta)
    conv_prob = float(conv_model.predict_proba(X)[0, 1])
    mktval    = float(inputs["total_market_value"])

    # Expected revenue: P(convert) × market_value × 2% (typical savings rate) × 30% fee
    savings_rate = 0.02
    fee_rate     = 0.30
    exp_revenue  = conv_prob * mktval * savings_rate * fee_rate

    # Priority modifiers
    mod = 1.0
    if mktval                          > fit_meta.get("q75_value",   0): mod += 0.20
    if inputs["MaxTrestlescore"]        > fit_meta.get("q75_trestle", 0): mod += 0.15
    if inputs["Properties_Count"]       > 1:                              mod += 0.10
    if inputs["num_ExemptionCode"]      > 0:                              mod += 0.10
    lead_score = exp_revenue * mod

    # Segment
    value_median = fit_meta.get("value_median", mktval)
    high_p = conv_prob >= 0.50
    high_v = mktval    >= value_median
    if   high_p and high_v:     segment = "🔥 High value, high prob"
    elif high_v and not high_p: segment = "💡 High value, low prob"
    elif high_p and not high_v: segment = "⚡ Low value, high prob"
    else:                       segment = "Low value, low prob"

    return dict(
        conv_probability = conv_prob,
        expected_revenue = exp_revenue,
        lead_score       = lead_score,
        segment          = segment,
        will_convert     = conv_prob >= fit_meta.get("optimal_threshold", 0.50),
    )


# ──────────────────────────────────────────────────────────────
#  OLLAMA HELPERS
# ──────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

SYSTEM_PROMPT = """You are an expert assistant for a Tax Appeal Lead Scoring System.

WHAT THIS APP DOES:
- Scores real estate property owners as leads for tax appeal services
- Predicts whether a lead will CONVERT (sign up for tax appeal representation)
- Estimates expected business revenue using market value input
- Assigns each lead to a segment with a recommended sales action

HOW THE MODEL WORKS:
- Conversion Model: XGBoost + CalibratedClassifierCV → P(convert) from 0–1
- Expected Revenue = P(convert) × Market Value × 2% savings × 30% fee
- Lead Score = Expected Revenue × Priority Modifiers

PRIORITY MODIFIERS:
- +20% if market value > 75th percentile of training data
- +15% if Trestle Score > 75th percentile
- +10% if owner holds more than 1 property
- +10% if owner has at least 1 exemption code

INPUT FEATURES (only these are used — no state or county):
- Owner City / ZipCode: geographic identity, frequency-encoded
- Property Type: RESIDENTIAL, COMMERCIAL, INDUSTRIAL, BPP, etc.
- Total Market Value ($): county-assessed value
- Properties Count: portfolio size
- Max Trestle Score (0–100): proprietary appeal-likelihood score
- Exemption Code Count: number of tax exemptions on the property

LEAD SEGMENTS:
1. High value + high prob (>=50%, >=median) — CALL IMMEDIATELY
2. High value + low prob  (<50%,  >=median) — NURTURE
3. Low value  + high prob (>=50%, <median)  — AUTOMATE
4. Low value  + low prob  (<50%,  <median)  — DEPRIORITIZE

FEE MODEL: 30% contingency on recovered tax savings (assumed ~2% of market value).
Answer clearly and relate everything to tax appeal lead generation."""


def build_ollama_prompt(user_msg: str, history: list, lead_ctx: str) -> str:
    hist = "".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}\n"
        for m in history
    )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"CURRENT LEAD:\n{lead_ctx}\n\n"
        f"CONVERSATION:\n{hist}"
        f"User: {user_msg}\nAssistant:"
    )


def stream_ollama(prompt: str, placeholder) -> str:
    full = ""
    try:
        with requests.post(
            OLLAMA_URL,
            json={
                "model":   OLLAMA_MODEL,
                "prompt":  prompt,
                "stream":  True,
                "options": {"temperature": 0.7, "num_predict": 512},
            },
            stream=True,
            timeout=120,
        ) as resp:
            if resp.status_code != 200:
                placeholder.error(
                    f"Ollama HTTP {resp.status_code}. "
                    "Make sure Ollama is running: `ollama serve`"
                )
                return ""
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    data  = json.loads(raw_line.decode("utf-8"))
                    full += data.get("response", "")
                    placeholder.markdown(
                        f'<div class="chat-bubble-assistant">{full}▌</div>',
                        unsafe_allow_html=True,
                    )
                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
        placeholder.markdown(
            f'<div class="chat-bubble-assistant">{full}</div>',
            unsafe_allow_html=True,
        )
        return full
    except requests.exceptions.ConnectionError:
        placeholder.error(
            "❌ Cannot reach Ollama. Make sure it is running:\n"
            "```\nollama serve\nollama pull mistral\n```"
        )
    except requests.exceptions.Timeout:
        placeholder.error("❌ Ollama timed out. Try again or restart Ollama.")
    except Exception as exc:
        placeholder.error(f"❌ Unexpected error: {exc}")
    return ""


# ══════════════════════════════════════════════════════════════
#  HERO BANNER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
  <h1>🏠 Tax Appeal Lead Scoring System</h1>
  <p>AI-powered lead qualification · XGBoost Conversion Model · Calibrated Probability Output</p>
</div>
""", unsafe_allow_html=True)

if not MODEL_LOADED:
    st.error(
        "⚠️ Model file not found (`lead_scoring_models.pkl`). "
        "Run `python generate_lead_scoring_model.py` first to generate it. "
        "The Chatbot tab still works without the model."
    )

# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Lead Prediction",
    "📊 Visualizations",
    "🖥️ Code Output",
    "📝 Description",
    "🤖 Chatbot",
])


# ══════════════════════════════════════════════════════════════
#  TAB 1 — LEAD PREDICTION
#  Columns: Owner_City, Owner_ZipCode, Property_Type,
#           num_ExemptionCode, Properties_Count,
#           MaxTrestlescore, total_market_value
# ══════════════════════════════════════════════════════════════
with tab1:
    if not MODEL_LOADED:
        st.warning("Model not loaded — prediction unavailable.")
    else:
        col_input, col_result = st.columns([1, 1], gap="large")

        with col_input:
            st.markdown('<div class="section-title">🔎 Property & Owner Details</div>',
                        unsafe_allow_html=True)

            # ── Row 1: City + ZipCode ──
            c1, c2 = st.columns(2)
            with c1:
                owner_city = st.text_input(
                    "Owner City", value="HOUSTON", key="t1_city"
                )
            with c2:
                # ZipCode is a string — no number_input; use text_input
                owner_zip = st.text_input(
                    "Owner ZipCode", value="77001",
                    max_chars=10, key="t1_zip",
                    help="5-digit ZIP code (stored as text, e.g. 07001)"
                )

            # ── Row 2: Property Type ──
            property_type = st.selectbox(
                "Property Type",
                ["RESIDENTIAL", "COMMERCIAL", "INDUSTRIAL", "BPP",
                 "MULTI-FAMILY", "VACANT LAND", "AGRICULTURAL", "MIXED USE", "OTHER"],
                index=0, key="t1_ptype"
            )

            st.markdown('<div class="section-title">📊 Property Metrics</div>',
                        unsafe_allow_html=True)

            # ── Row 3: Market Value + Properties Count ──
            c3, c4 = st.columns(2)
            with c3:
                total_market_value = st.number_input(
                    "Total Market Value ($)", min_value=0.0,
                    value=450_000.0, step=10_000.0, format="%.0f", key="t1_mktval"
                )
            with c4:
                properties_count = st.number_input(
                    "Properties Count", min_value=1.0,
                    value=1.0, step=1.0, key="t1_propcount"
                )

            # ── Row 4: Trestle Score + Exemption Count ──
            c5, c6 = st.columns(2)
            with c5:
                max_trestle_score = st.slider(
                    "Max Trestle Score", 0.0, 100.0, 65.0, 0.5, key="t1_trestle"
                )
            with c6:
                num_exemption = st.number_input(
                    "Exemption Code Count", min_value=0, max_value=20,
                    value=1, step=1, key="t1_exemp"
                )

            predict_btn = st.button(
                "🚀 Score This Lead",
                use_container_width=True, type="primary", key="t1_predict"
            )

        with col_result:
            st.markdown('<div class="section-title">📈 Prediction Results</div>',
                        unsafe_allow_html=True)

            if predict_btn or "last_result" in st.session_state:
                if predict_btn:
                    inputs = {
                        "Owner_City":         owner_city,
                        "Owner_ZipCode":      owner_zip,          # string
                        "Property_Type":      property_type,
                        "num_ExemptionCode":  num_exemption,
                        "Properties_Count":   properties_count,
                        "MaxTrestlescore":    max_trestle_score,
                        "total_market_value": total_market_value,
                    }
                    result = predict_lead(inputs)
                    st.session_state["last_result"] = result
                    st.session_state["last_inputs"]  = inputs
                else:
                    result = st.session_state["last_result"]

                # ── Verdict banner ──
                if result["will_convert"]:
                    st.markdown(f"""
                    <div class="verdict-convert">
                      <h2>✅ LIKELY TO CONVERT</h2>
                      <p>Conversion probability: <strong>{result['conv_probability']:.1%}</strong></p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="verdict-no-convert">
                      <h2>❌ UNLIKELY TO CONVERT</h2>
                      <p>Conversion probability: <strong>{result['conv_probability']:.1%}</strong></p>
                    </div>""", unsafe_allow_html=True)

                # ── Metric cards ──
                m1, m2 = st.columns(2)
                with m1:
                    color = "green" if result["will_convert"] else "red"
                    st.markdown(f"""
                    <div class="metric-card {color}">
                      <h3>Conversion Probability</h3>
                      <div class="value">{result['conv_probability']:.1%}</div>
                      <div class="sub">XGBoost · calibrated</div>
                    </div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="metric-card orange">
                      <h3>Expected Revenue (30% fee)</h3>
                      <div class="value">${result['expected_revenue']:,.0f}</div>
                      <div class="sub">P(convert) × value × 2% × 30%</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                m3, m4 = st.columns(2)
                with m3:
                    st.markdown(f"""
                    <div class="metric-card">
                      <h3>Lead Score</h3>
                      <div class="value">${result['lead_score']:,.0f}</div>
                      <div class="sub">Revenue × priority modifiers</div>
                    </div>""", unsafe_allow_html=True)
                with m4:
                    threshold_pct = fit_meta.get("optimal_threshold", 0.50) * 100
                    st.markdown(f"""
                    <div class="metric-card purple">
                      <h3>Decision Threshold</h3>
                      <div class="value">{threshold_pct:.0f}%</div>
                      <div class="sub">Optimal threshold from training</div>
                    </div>""", unsafe_allow_html=True)

                # ── Segment & Action ──
                st.markdown("<br>", unsafe_allow_html=True)
                seg = result["segment"]
                strategies = {
                    "🔥 High value, high prob": "📞 CALL IMMEDIATELY — assign top rep",
                    "💡 High value, low prob":  "📧 NURTURE — education emails & follow-ups",
                    "⚡ Low value, high prob":  "🤖 AUTOMATE — drip email/SMS campaign",
                    "Low value, low prob":      "📁 DEPRIORITIZE — minimal effort",
                }
                st.markdown(f"""
                <div style="background:white;border-radius:14px;padding:1.2rem 1.4rem;
                            box-shadow:0 2px 10px rgba(0,0,0,.06);margin-top:.5rem;">
                  <div class="section-title" style="margin-top:0;">Segment & Action</div>
                  <div style="font-size:1.1rem;font-weight:600;margin-bottom:.5rem;">{seg}</div>
                  <div style="color:#374151;font-size:.9rem;">{strategies.get(seg,'')}</div>
                </div>""", unsafe_allow_html=True)

                # ── Gauge chart ──
                prob_pct = result["conv_probability"] * 100
                opt_thr  = fit_meta.get("optimal_threshold", 0.50) * 100
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number+delta", value=prob_pct,
                    title={"text": "Conversion Probability", "font": {"size": 14}},
                    delta={"reference": opt_thr, "valueformat": ".1f"},
                    number={"suffix": "%", "font": {"size": 28}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar":  {"color": "#22c55e" if prob_pct >= opt_thr else "#ef4444"},
                        "steps": [
                            {"range": [0,   opt_thr*0.6], "color": "#fee2e2"},
                            {"range": [opt_thr*0.6, opt_thr], "color": "#fef3c7"},
                            {"range": [opt_thr, 100],  "color": "#d1fae5"},
                        ],
                        "threshold": {
                            "line": {"color": "#1e3a8a", "width": 3},
                            "thickness": 0.75, "value": opt_thr,
                        },
                    }
                ))
                fig_g.update_layout(
                    height=220, margin=dict(t=30, b=10, l=20, r=20),
                    paper_bgcolor="white"
                )
                st.plotly_chart(fig_g, use_container_width=True)

            else:
                st.info("👈 Fill in the lead details and click **Score This Lead**.")


# ══════════════════════════════════════════════════════════════
#  TAB 2 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════
with tab2:
    if not MODEL_LOADED:
        st.warning("Model not loaded — visualizations unavailable.")
    else:
        st.markdown('<div class="section-title">📊 Model & Lead Analytics</div>',
                    unsafe_allow_html=True)
        train_probs = fit_meta.get("train_conv_probs", [])
        opt_thr     = fit_meta.get("optimal_threshold", 0.50)

        c1, c2 = st.columns(2)
        with c1:
            # ── Conv probability distribution ──
            fig1 = go.Figure()
            fig1.add_trace(go.Histogram(
                x=train_probs, nbinsx=30,
                marker_color="#3b82f6", opacity=0.8, name="Training data"
            ))
            fig1.add_vline(x=opt_thr, line_color="red", line_dash="dash",
                           annotation_text=f"Threshold {opt_thr:.0%}")
            if "last_result" in st.session_state:
                cp = st.session_state["last_result"]["conv_probability"]
                fig1.add_vline(x=cp, line_color="#f97316", line_width=2.5,
                               annotation_text=f"Your lead: {cp:.1%}")
            fig1.update_layout(
                title="Conv Probability Distribution (Training)",
                xaxis_title="Probability", yaxis_title="Count",
                height=320, paper_bgcolor="white",
                plot_bgcolor="#f8fafc", margin=dict(t=40, b=20, l=20, r=20)
            )
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            # ── Lead score waterfall ──
            if "last_result" in st.session_state:
                r   = st.session_state["last_result"]
                inp = st.session_state["last_inputs"]
                base_rev = r["expected_revenue"]
                q75_val  = fit_meta.get("q75_value",   0)
                q75_tr   = fit_meta.get("q75_trestle", 0)
                mb = {
                    "Base Revenue":      base_rev,
                    "High Value +20%":   base_rev * 0.20 if inp["total_market_value"] > q75_val else 0,
                    "High Trestle +15%": base_rev * 0.15 if inp["MaxTrestlescore"]    > q75_tr  else 0,
                    "Multi-Prop +10%":   base_rev * 0.10 if inp["Properties_Count"]  > 1       else 0,
                    "Exemption +10%":    base_rev * 0.10 if inp["num_ExemptionCode"] > 0       else 0,
                }
                fig2 = go.Figure(go.Waterfall(
                    orientation="v",
                    measure=["absolute"] + ["relative"] * (len(mb) - 1),
                    x=list(mb.keys()), y=list(mb.values()),
                    connector={"line": {"color": "#94a3b8"}},
                    increasing={"marker": {"color": "#22c55e"}},
                    decreasing={"marker": {"color": "#ef4444"}},
                ))
                fig2.update_layout(
                    title="Lead Score Breakdown",
                    yaxis_title="Revenue ($)", height=320,
                    paper_bgcolor="white", plot_bgcolor="#f8fafc",
                    margin=dict(t=40, b=20, l=20, r=20)
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Score a lead in Tab 1 to see the breakdown chart.")

        c3, c4 = st.columns(2)
        with c3:
            # ── Radar chart (no ocaluc axis) ──
            cats = [
                "Conv Probability", "Market Value",
                "Trestle Score", "Portfolio Size", "Exemptions",
            ]
            if "last_result" in st.session_state:
                inp  = st.session_state["last_inputs"]
                r    = st.session_state["last_result"]
                vals = [
                    r["conv_probability"],
                    min(inp["total_market_value"] / 2_000_000, 1.0),
                    inp["MaxTrestlescore"] / 100,
                    min(inp["Properties_Count"] / 10, 1.0),
                    min(inp["num_ExemptionCode"] / 5,  1.0),
                ]
            else:
                vals = [0.5] * len(cats)

            fig3 = go.Figure()
            fig3.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill="toself", fillcolor="rgba(59,130,246,.2)",
                line=dict(color="#3b82f6", width=2)
            ))
            fig3.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Lead Profile Radar", height=320,
                paper_bgcolor="white", margin=dict(t=50, b=20, l=20, r=20)
            )
            st.plotly_chart(fig3, use_container_width=True)

        with c4:
            # ── Segment matrix ──
            value_median_k = fit_meta.get("value_median", 475_000) / 1_000
            seg_df = pd.DataFrame({
                "CP":  [0.75, 0.30, 0.70, 0.20],
                "Val": [value_median_k * 1.5, value_median_k * 1.5,
                        value_median_k * 0.5, value_median_k * 0.5],
                "Lbl": ["Call Now", "Nurture", "Automate", "Deprioritize"],
                "Sz":  [80, 60, 40, 20],
                "Col": ["#22c55e", "#f59f00", "#3b82f6", "#9ca3af"],
            })
            fig4 = go.Figure()
            for _, row_s in seg_df.iterrows():
                fig4.add_trace(go.Scatter(
                    x=[row_s.CP], y=[row_s.Val], mode="markers+text",
                    marker=dict(size=row_s.Sz, color=row_s.Col, opacity=0.75,
                                line=dict(color="white", width=2)),
                    text=[row_s.Lbl], textposition="top center",
                    name=row_s.Lbl, showlegend=True
                ))
            if "last_result" in st.session_state:
                r   = st.session_state["last_result"]
                inp = st.session_state["last_inputs"]
                fig4.add_trace(go.Scatter(
                    x=[r["conv_probability"]],
                    y=[inp["total_market_value"] / 1_000],
                    mode="markers+text",
                    marker=dict(size=20, color="#f97316", symbol="star",
                                line=dict(color="white", width=2)),
                    text=["Your Lead"], textposition="middle right", name="Your Lead"
                ))
            fig4.add_vline(x=opt_thr, line_dash="dash", line_color="#6b7280", opacity=0.5)
            fig4.add_hline(y=value_median_k, line_dash="dash", line_color="#6b7280",
                           opacity=0.5, annotation_text="Median value")
            fig4.update_layout(
                title="Segment Matrix",
                xaxis_title="Conversion Probability",
                yaxis_title="Market Value ($K)",
                height=320, paper_bgcolor="white",
                plot_bgcolor="#f8fafc", margin=dict(t=40, b=20, l=20, r=20)
            )
            st.plotly_chart(fig4, use_container_width=True)

        # ── Feature importance ──
        st.markdown('<div class="section-title">🧠 Top Predictive Features</div>',
                    unsafe_allow_html=True)
        try:
            est = conv_model.calibrated_classifiers_[0].estimator
            feat_df = pd.DataFrame({
                "Feature":    fit_meta["feature_cols"],
                "Importance": est.feature_importances_,
            }).sort_values("Importance", ascending=False).head(15)
            fig5 = px.bar(
                feat_df, x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale="Blues",
                title="Top 15 Features — Conversion Model"
            )
            fig5.update_layout(
                height=400, yaxis=dict(autorange="reversed"),
                paper_bgcolor="white", plot_bgcolor="#f8fafc",
                margin=dict(t=40, b=20, l=20, r=20),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig5, use_container_width=True)
        except Exception:
            st.info("Feature importance chart not available for this model type.")


# ══════════════════════════════════════════════════════════════
#  TAB 3 — CODE OUTPUT
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">🖥️ Raw Model Output & Logs</div>',
                unsafe_allow_html=True)

    if "last_result" in st.session_state and MODEL_LOADED:
        r   = st.session_state["last_result"]
        inp = st.session_state["last_inputs"]

        lines = ["=" * 60, "  LEAD SCORING SYSTEM — PREDICTION OUTPUT", "=" * 60, ""]
        lines += ["-- INPUT FEATURES --"]
        for k, v in inp.items():
            lines.append(f"  {k:<30} {v}")
        lines += ["", "-- MODEL OUTPUTS --"]
        lines.append(f"  {'Conversion Probability':<30} {r['conv_probability']:.4f}  ({r['conv_probability']:.1%})")
        lines.append(f"  {'Expected Revenue (30%)':<30} ${r['expected_revenue']:,.2f}")
        lines.append(f"  {'Lead Score':<30} ${r['lead_score']:,.2f}")
        lines.append(f"  {'Segment':<30} {r['segment']}")
        lines.append(f"  {'Will Convert':<30} {'YES' if r['will_convert'] else 'NO'}")
        lines += ["", "-- BUSINESS LOGIC MODIFIERS --"]
        q75_val = fit_meta.get("q75_value",   0)
        q75_tr  = fit_meta.get("q75_trestle", 0)
        base_rev = r["expected_revenue"]
        lines.append(f"  Base expected revenue: ${base_rev:,.2f}")
        if inp["total_market_value"] > q75_val:
            lines.append(f"  + High value (+20%):   ${base_rev * 0.20:,.2f}")
        if inp["MaxTrestlescore"] > q75_tr:
            lines.append(f"  + High trestle (+15%): ${base_rev * 0.15:,.2f}")
        if inp["Properties_Count"] > 1:
            lines.append(f"  + Multi-prop (+10%):   ${base_rev * 0.10:,.2f}")
        if inp["num_ExemptionCode"] > 0:
            lines.append(f"  + Exemptions (+10%):   ${base_rev * 0.10:,.2f}")
        lines.append(f"  Final lead score:      ${r['lead_score']:,.2f}")
        lines += ["", "-- MODEL INFO --"]
        lines.append(f"  Conversion model:  XGBoost + CalibratedClassifierCV")
        lines.append(f"  Features used:     {len(fit_meta['feature_cols'])}")
        lines.append(f"  Test AUC-ROC:      {fit_meta.get('test_auc', 'N/A')}")
        lines.append(f"  CV AUC (5-fold):   {fit_meta.get('cv_auc_mean', 'N/A'):.4f}"
                     f" ± {fit_meta.get('cv_auc_std', 'N/A'):.4f}")
        lines.append(f"  Optimal threshold: {fit_meta.get('optimal_threshold', 0.50):.2f}")
        lines.append("=" * 60)

        code_text = "\n".join(lines)
        st.markdown(f'<div class="code-area">{code_text}</div>', unsafe_allow_html=True)
        st.download_button(
            "⬇️ Download Output as .txt", data=code_text,
            file_name="lead_score_output.txt", mime="text/plain",
            use_container_width=True, key="t3_dl"
        )

        with st.expander("📦 Raw JSON Output"):
            st.json({
                "input":  {k: str(v) for k, v in inp.items()},
                "output": {k: round(v, 4) if isinstance(v, float) else v
                           for k, v in r.items()},
            })

        with st.expander("🔢 Full Feature Vector"):
            X_row = build_inference_row(inp, fit_meta)
            feat_str = "\n".join(
                f"  {k:<50} {v}"
                for k, v in X_row.iloc[0].to_dict().items()
            )
            st.markdown(f'<div class="code-area">{feat_str}</div>', unsafe_allow_html=True)
    else:
        st.info("Score a lead in the Lead Prediction tab first.")
        sample = """  Conversion Probability         0.7823  (78.2%)
  Expected Revenue (30%)         $1,985
  Lead Score                     $2,580
  Segment                        🔥 High value, high prob
  Will Convert                   YES"""
        st.markdown(f'<div class="code-area">{sample}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  TAB 4 — DESCRIPTION
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">📝 Project Description</div>',
                unsafe_allow_html=True)
 
    sections = [
        ("📋", "Project Description",
         """This project delivers an AI-powered tax appeal lead scoring system designed to help property tax appeal firms identify and prioritise property owners most likely to engage their services.
The system ingests property-level data — including market value, trestle score, portfolio size, exemption codes, and geographic identifiers — and outputs a calibrated probability of client conversion for each lead.
A full end-to-end pipeline covers data ingestion from Excel, preprocessing and feature engineering, model training with imbalance handling, probability calibration, and an interactive Streamlit web interface for real-time scoring.
Each lead is assigned to one of four action segments — Call Immediately, Nurture, Automate, or Deprioritise — enabling sales teams to focus effort on the highest-value opportunities without manual triage.
The business impact is quantified through an expected revenue estimate (conversion probability × market value × assumed savings rate × 30% contingency fee), giving firms a dollar-value priority score for every prospect in their pipeline."""),
 
        ("🔧", "Tools Used",
         """Python served as the core development language, with Pandas and NumPy handling all data wrangling, feature engineering, quantile computation, frequency encoding, and log transformation of skewed financial columns.
XGBoost (XGBClassifier) was chosen as the primary model for its strong performance on tabular data with mixed feature types; it was trained with histogram-based tree construction, early stopping on a validation set, and tuned regularisation parameters to reduce overfitting.
Scikit-learn provided CalibratedClassifierCV (Platt scaling) for post-hoc probability calibration, StratifiedKFold cross-validation for robust AUC estimation, and evaluation utilities including ROC-AUC, F1, precision-recall, and confusion matrix displays.
Imbalanced-learn's SMOTE was applied when the class imbalance ratio exceeded 1.5:1, synthetically oversampling the minority (client) class in training so the model does not learn to default-predict the majority class; scale_pos_weight served as the fallback when SMOTE was unavailable.
Streamlit powered the interactive web application, with Plotly providing gauge charts, waterfall breakdowns, radar profiles, segment scatter matrices, and distribution histograms; the trained model bundle was serialised with Python's Pickle for portable, dependency-free deployment."""),
 
        ("🏆", "Project Outcome",
         """The trained XGBoost model achieved a strong ROC-AUC score on the held-out test set, with an additional five-fold stratified cross-validation confirming that the result generalises across data splits and is not an artefact of a favourable random partition.
Probability calibration via Platt scaling ensured that model output scores are genuine probability estimates — a lead scored at 70% truly converts approximately 70% of the time — making the outputs directly usable for business prioritisation without additional calibration effort.
The Streamlit application allows non-technical users to score individual leads in real time, entering seven property attributes and instantly receiving a conversion probability, expected revenue figure, lead score, segment label, and recommended sales action.
Feature importance analysis revealed that MaxTrestlescore, log-transformed market value, and their interaction term were the dominant predictors, confirming domain intuitions that properties with high appeal potential and high assessed value are the most attractive leads.
The system replaces ad-hoc, experience-based lead qualification with a reproducible, data-driven process, enabling firms to rank a full prospect database objectively and allocate sales resources to leads where the expected return on outreach is highest."""),
 
        ("📈", "Project Improvements",
         """Hyperparameter optimisation using Bayesian search (Optuna or Hyperopt) over the full XGBoost parameter space — including max_depth, learning_rate, subsample ratios, and regularisation penalties — would likely yield a meaningful AUC improvement over the current manually chosen configuration.
Replacing SMOTE with more sophisticated imbalance strategies such as ADASYN, Borderline-SMOTE, or class-conditional cost-sensitive learning could improve recall on the minority (client) class without introducing synthetic samples that may not reflect realistic property owner profiles.
Adding a second supervised model — such as a GradientBoostingRegressor trained on actual contract values — to replace the heuristic revenue formula (market value × 2% × 30%) would make the lead score a true expected-value estimate grounded in historical deal data rather than assumptions.
Incorporating additional data sources such as historical county appraisal records, prior appeal outcomes, property age and construction type, and owner tenure length would enrich the feature space and provide the model with signals that are known to correlate strongly with appeal success and client willingness to engage.
Implementing a periodic model retraining pipeline — triggered either on a schedule or when prediction drift is detected — would prevent score degradation as market conditions, appraisal practices, and client behaviour patterns evolve over time."""),
 
        ("💡", "Future Use Cases",
         """The scoring pipeline could be extended into a full CRM integration — automatically importing raw lead records nightly, scoring every prospect, and pushing ranked lists back into Salesforce or HubSpot so sales teams begin each day with a pre-prioritised call queue without any manual intervention.
A multi-county or multi-state deployment would allow the model to operate as a national lead intelligence platform, with county-level sub-models or geographic embeddings capturing regional appraisal culture, appeal success rates, and local market dynamics that a single aggregate model cannot represent.
The same modelling architecture could be adapted for churn prediction — identifying existing clients at risk of not renewing their tax appeal representation — by relabelling the target as "renewed vs. did not renew" and retraining on historical client lifecycle data.
The conversion probability score could power a personalised outreach engine where the content, channel, and timing of client communications (email, direct mail, or phone) are dynamically selected based on segment, property type, and portfolio size to maximise engagement rates at scale.
In the longer term, the system could evolve into a property intelligence platform combining lead scoring, appeal outcome prediction, comparable property analysis, and automated hearing preparation — creating an end-to-end AI-assisted workflow from initial prospect identification through to settled savings."""),
    ]
 
    for icon, title, body in sections:
        bullets = "".join(
            f'<li style="margin-bottom:.6rem;color:#374151;font-size:.92rem;line-height:1.6">{line.strip()}</li>'
            for line in body.strip().split("\n") if line.strip()
        )
        st.markdown(f"""
<div style="background:white;border-radius:14px;padding:1.6rem 2rem;
                    box-shadow:0 2px 10px rgba(0,0,0,.06);border-left:4px solid #6366f1;
                    margin-bottom:1.2rem;">
<div style="display:flex;align-items:center;gap:.6rem;margin-bottom:1rem;">
<span style="font-size:1.5rem;">{icon}</span>
<span style="font-size:1.1rem;font-weight:700;color:#1e1b4b;">{title}</span>
</div>
<ul style="margin:0;padding-left:1.2rem;">{bullets}</ul>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  TAB 5 — CHATBOT (Ollama / Mistral)
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown(
        '<div class="section-title">🤖 Lead Scoring Assistant — Powered by Ollama / Mistral</div>',
        unsafe_allow_html=True,
    )

    # ── Build lead context ──
    if "last_result" in st.session_state and MODEL_LOADED:
        r   = st.session_state["last_result"]
        inp = st.session_state["last_inputs"]
        lead_ctx = (
            f"Owner City: {inp['Owner_City']}\n"
            f"Owner ZipCode: {inp['Owner_ZipCode']}\n"
            f"Property Type: {inp['Property_Type']}\n"
            f"Market Value Input: ${inp['total_market_value']:,.0f}\n"
            f"Trestle Score: {inp['MaxTrestlescore']}\n"
            f"Properties Count: {inp['Properties_Count']}\n"
            f"Exemption Codes: {inp['num_ExemptionCode']}\n"
            f"Conversion Probability: {r['conv_probability']:.1%}\n"
            f"Expected Revenue: ${r['expected_revenue']:,.0f}\n"
            f"Lead Score: ${r['lead_score']:,.0f}\n"
            f"Segment: {r['segment']}\n"
            f"Will Convert: {'YES' if r['will_convert'] else 'NO'}"
        )
        st.success(
            f"Lead context loaded — {inp['Owner_City']} {inp['Owner_ZipCode']} | {r['segment']}"
        )
    else:
        lead_ctx = "No lead scored yet. Answer general questions about the lead scoring system."
        st.info("Score a lead in Tab 1 first to get context-aware answers.")

    st.markdown("---")

    # ── Chat history ──
    if not st.session_state["chat_history"]:
        st.markdown("""
        <div style="text-align:center;padding:2rem;color:#9ca3af;">
          <div style="font-size:2.5rem;margin-bottom:.5rem;">💬</div>
          <div>Ask about the scoring system, your lead results, or which inputs matter most.</div>
        </div>""", unsafe_allow_html=True)
    else:
        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-label-user">You</div>'
                    f'<div class="chat-bubble-user">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-label-assistant">Assistant</div>'
                    f'<div class="chat-bubble-assistant">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)

    user_input = st.text_area(
        "Message", value="", height=80,
        placeholder="e.g. Why is this lead's conversion probability low? What should I do next?",
        key="t5_input", label_visibility="collapsed",
    )

    col_send, col_clear = st.columns([4, 1])
    with col_send:
        send_btn  = st.button("📨 Send",  use_container_width=True, type="primary", key="t5_send")
    with col_clear:
        clear_btn = st.button("🗑️ Clear", use_container_width=True, key="t5_clear")

    if clear_btn:
        st.session_state["chat_history"] = []
        st.rerun()

    if send_btn:
        if not user_input.strip():
            st.warning("Please type a message before sending.")
        else:
            user_msg         = user_input.strip()
            history_snapshot = list(st.session_state["chat_history"])
            st.session_state["chat_history"].append({"role": "user", "content": user_msg})

            full_prompt = build_ollama_prompt(user_msg, history_snapshot, lead_ctx)

            with st.spinner("Thinking…"):
                resp_ph       = st.empty()
                full_response = stream_ollama(full_prompt, resp_ph)

            if full_response:
                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": full_response}
                )

            st.rerun()
