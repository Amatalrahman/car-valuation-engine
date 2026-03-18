"""
Car Price Prediction — Streamlit App
=====================================
Run locally:
    pip install streamlit plotly lightgbm xgboost shap pandas numpy joblib scikit-learn
    streamlit run app.py

Run on Google Colab:
    !pip install streamlit pyngrok -q
    !ngrok authtoken YOUR_TOKEN
    from pyngrok import ngrok
    public_url = ngrok.connect(8501)
    print(public_url)
    !streamlit run app.py &
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
import re
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoValuate · Car Price AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark background */
.stApp {
    background: #080c14;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1321;
    border-right: 1px solid #1e2640;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #7eb8f7 !important;
}

/* Hero header */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #4f8ef7 0%, #7fc8f8 40%, #4ff7c8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    color: #5a6480;
    font-size: 1.05rem;
    margin-bottom: 2rem;
}

/* Price card */
.price-card {
    background: linear-gradient(135deg, #0f1930 0%, #0a1525 100%);
    border: 1px solid #1e3a6e;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.price-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #4f8ef7, #4ff7c8);
}
.price-label {
    font-family: 'DM Sans', sans-serif;
    color: #4a5370;
    font-size: 0.85rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.price-value {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    color: #4f8ef7;
    line-height: 1;
}
.price-range {
    color: #3a4460;
    font-size: 0.9rem;
    margin-top: 0.6rem;
}

/* Stat chips */
.stat-chip {
    display: inline-block;
    background: #0d1626;
    border: 1px solid #1e2d50;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    margin: 0.3rem;
    font-size: 0.85rem;
    color: #7eb8f7;
}
.stat-chip span { color: #c8d8f8; font-weight: 500; }

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #c8d8f8;
    margin: 1.5rem 0 0.75rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1a2240;
}

/* Insight boxes */
.insight-box {
    background: #0a1220;
    border-left: 3px solid #4f8ef7;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
    color: #8a9dc8;
}

/* Model pill badges */
.model-badge {
    display: inline-block;
    background: #0f2040;
    border: 1px solid #1e3a70;
    color: #4f8ef7;
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.78rem;
    margin: 0.15rem;
    font-family: 'DM Sans', sans-serif;
}
.model-badge.active {
    background: #1a3a70;
    border-color: #4f8ef7;
    color: #7eb8f7;
}

/* Hide default streamlit branding */
#MainMenu, footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Model loading ────────────────────────────────────────────────────────────
# Set to True after running 03_Modeling.ipynb and saving lightgbm.pkl
USE_TRAINED_MODEL = True 
MODEL_PATH        = 'models/lightgbm.pkl'
FEATURES_PATH     = 'models/feature_names.pkl'

_trained_model    = None
_trained_features = None

if USE_TRAINED_MODEL:
    try:
        import lightgbm as lgb
        _trained_model    = joblib.load(MODEL_PATH)
        _trained_features = joblib.load(FEATURES_PATH)
    except Exception as e:
        st.sidebar.warning(f"⚠️ Could not load model: {e}\nFalling back to rule-based engine.")
        USE_TRAINED_MODEL = True


def ml_predict(row_df: pd.DataFrame) -> float:
    """Run inference with the trained LightGBM model and return price in USD."""
    pred_log = _trained_model.predict(row_df[_trained_features])[0]
    return float(np.expm1(pred_log))


# ── Brand / Model Data ────────────────────────────────────────────────────────
BRAND_MODELS = {
    "Toyota":        ["Camry", "Corolla", "RAV4", "Highlander", "Tacoma", "Tundra", "Prius", "4Runner"],
    "Honda":         ["Civic", "Accord", "CR-V", "Pilot", "HR-V", "Odyssey", "Ridgeline"],
    "Ford":          ["F-150", "Mustang", "Explorer", "Escape", "Edge", "Bronco", "Ranger", "Maverick"],
    "Chevrolet":     ["Silverado", "Equinox", "Malibu", "Traverse", "Colorado", "Blazer", "Camaro"],
    "BMW":           ["3 Series", "5 Series", "7 Series", "X3", "X5", "X7", "M3", "M5"],
    "Mercedes-Benz": ["C-Class", "E-Class", "S-Class", "GLE", "GLC", "AMG GT", "G-Class"],
    "Audi":          ["A3", "A4", "A6", "Q3", "Q5", "Q7", "Q8", "e-tron"],
    "Tesla":         ["Model 3", "Model S", "Model X", "Model Y", "Cybertruck"],
    "Nissan":        ["Altima", "Sentra", "Rogue", "Murano", "Titan", "Frontier", "Leaf"],
    "Hyundai":       ["Elantra", "Sonata", "Tucson", "Santa Fe", "Palisade", "Ioniq 6"],
    "Kia":           ["Forte", "Optima", "Sorento", "Sportage", "Telluride", "EV6"],
    "Volkswagen":    ["Jetta", "Passat", "Tiguan", "Atlas", "GTI", "ID.4"],
    "Lexus":         ["IS", "ES", "LS", "RX", "GX", "LX", "NX"],
    "Subaru":        ["Outback", "Forester", "Impreza", "Legacy", "Ascent", "WRX", "BRZ"],
    "Jeep":          ["Wrangler", "Grand Cherokee", "Cherokee", "Compass", "Gladiator"],
    "Dodge":         ["Charger", "Challenger", "Durango", "Ram 1500"],
    "GMC":           ["Sierra", "Canyon", "Terrain", "Acadia", "Yukon"],
    "Mazda":         ["Mazda3", "Mazda6", "CX-5", "CX-9", "MX-5 Miata"],
    "Porsche":       ["911", "Cayenne", "Macan", "Panamera", "Taycan"],
    "Cadillac":      ["CT4", "CT5", "Escalade", "XT4", "XT5", "XT6"],
    "Land Rover":    ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
    "Volvo":         ["XC40", "XC60", "XC90", "S60", "S90", "V60"],
    "Acura":         ["ILX", "TLX", "MDX", "RDX", "NSX"],
    "Infiniti":      ["Q50", "Q60", "QX50", "QX60", "QX80"],
    "Genesis":       ["G70", "G80", "G90", "GV70", "GV80"],
    "Other":         ["Other"],
}

LUXURY_BRANDS = {'tier_ultra': {'Rolls-Royce','Lamborghini','Ferrari','Bugatti','McLaren','Bentley','Aston Martin'},
                  'tier_luxury': {'Mercedes-Benz','BMW','Audi','Porsche','Lexus','Cadillac','Land Rover',
                                  'Genesis','Maserati','Alfa Romeo','Volvo','Acura','Lincoln','Infiniti'},
                  'tier_mid':    {'Toyota','Honda','Mazda','Subaru','Volkswagen','Hyundai','Kia',
                                  'Chevrolet','Ford','Nissan','Jeep','Ram','Dodge','GMC','Buick'}}
TIER_MAP = {'tier_ultra': 4, 'tier_luxury': 3, 'tier_mid': 2, 'tier_budget': 1}

def get_brand_tier(brand):
    for tier, brands in LUXURY_BRANDS.items():
        if brand in brands:
            return TIER_MAP[tier]
    return 1

FUEL_TYPES      = ["Gasoline", "Diesel", "Electric", "Hybrid", "Plug-In Hybrid", "Flex Fuel", "Unknown"]
TRANSMISSIONS   = ["Automatic", "Manual", "CVT", "Dual-Clutch (DCT)", "Unknown"]
EXT_COLORS      = ["White", "Black", "Silver", "Gray", "Red", "Blue", "Brown", "Green", "Gold", "Orange", "Yellow", "Other"]
INT_COLORS      = ["Black", "Gray", "Beige/Tan", "Brown", "Red", "White", "Blue", "Other"]


# ── Rule-Based Price Engine (no pkl required) ─────────────────────────────────
def rule_based_predict(inputs: dict) -> dict:
    """
    A transparent, domain-grounded pricing model using depreciation curves,
    brand premiums, and condition adjustments. Returns price + confidence band.
    """
    brand           = inputs["brand"]
    model_year      = inputs["model_year"]
    mileage         = inputs["mileage"]
    fuel_type       = inputs["fuel_type"]
    transmission    = inputs["transmission"]
    has_accident    = inputs["has_accident"]
    clean_title     = inputs["clean_title"]
    engine_hp       = inputs["engine_hp"]
    engine_disp     = inputs["engine_disp"]
    cylinders       = inputs["cylinders"]

    CURRENT_YEAR = 2024
    car_age = max(0, CURRENT_YEAR - model_year)

    # ── Base MSRP by brand tier ───────────────────────────────────────────────
    tier = get_brand_tier(brand)
    BASE_BY_TIER = {4: 180000, 3: 52000, 2: 32000, 1: 24000}
    base = BASE_BY_TIER[tier]

    # HP adjustment
    hp_ref = {4: 500, 3: 350, 2: 200, 1: 150}
    hp_ratio = engine_hp / max(hp_ref[tier], 1)
    base *= (0.75 + 0.5 * hp_ratio)

    # Displacement uplift for large engines
    if engine_disp >= 5.0:
        base *= 1.15
    elif engine_disp >= 3.5:
        base *= 1.07

    # Cylinder count premium
    if cylinders >= 10:
        base *= 1.20
    elif cylinders >= 8:
        base *= 1.10

    # ── Depreciation curve ────────────────────────────────────────────────────
    # Years 0–3: fast depreciation; Years 4–10: moderate; 10+: slow / floor
    if car_age == 0:
        dep_factor = 1.0
    elif car_age <= 1:
        dep_factor = 0.84
    elif car_age <= 3:
        dep_factor = 0.84 * (0.88 ** (car_age - 1))
    elif car_age <= 10:
        dep_factor = 0.84 * (0.88**2) * (0.92 ** (car_age - 3))
    else:
        dep_factor = max(0.10, 0.84 * (0.88**2) * (0.92**7) * (0.96 ** (car_age - 10)))

    price = base * dep_factor

    # ── Mileage penalty ───────────────────────────────────────────────────────
    avg_miles_per_year = 12_500
    expected_miles = avg_miles_per_year * max(car_age, 1)
    excess = mileage - expected_miles
    if excess > 0:
        # Each 10K excess miles above average → ~3.5% penalty (diminishing)
        mileage_penalty = 1 - 0.035 * (excess / 10_000) ** 0.75
        price *= max(0.40, mileage_penalty)
    else:
        # Low mileage: slight premium
        low_pct = max(0, (expected_miles - mileage)) / max(expected_miles, 1)
        price *= (1 + 0.04 * low_pct)

    # ── Fuel type adjustment ─────────────────────────────────────────────────
    fuel_adj = {
        "Electric":        1.18,
        "Plug-In Hybrid":  1.10,
        "Hybrid":          1.06,
        "Diesel":          1.04,
        "Flex Fuel":       0.96,
        "Gasoline":        1.00,
        "Unknown":         0.98,
    }
    price *= fuel_adj.get(fuel_type, 1.0)

    # ── Transmission ──────────────────────────────────────────────────────────
    trans_adj = {
        "Automatic": 1.00, "CVT": 0.97, "Dual-Clutch (DCT)": 1.04,
        "Manual": 0.96, "Unknown": 0.98,
    }
    price *= trans_adj.get(transmission, 1.0)

    # ── Condition flags ───────────────────────────────────────────────────────
    if has_accident:
        price *= 0.84      # ~16% accident penalty
    if clean_title:
        price *= 1.05      # clean title premium
    else:
        price *= 0.88      # salvage / unknown title discount

    # ── Confidence band: ±15% for newer cars, wider for old ──────────────────
    band_pct = 0.12 + min(0.18, car_age * 0.01)
    low  = price * (1 - band_pct)
    high = price * (1 + band_pct)

    return {
        "price":     round(price),
        "low":       round(low),
        "high":      round(high),
        "band_pct":  band_pct,
        "car_age":   car_age,
        "dep_factor": dep_factor,
        "base_msrp": round(base),
    }


# ── Depreciation chart data ────────────────────────────────────────────────────
def depreciation_curve(price_now, car_age_now, inputs):
    years = list(range(0, 21))
    prices = []
    for future_age in years:
        future_inputs = dict(inputs)
        future_inputs["model_year"] = 2024 - future_age
        future_inputs["mileage"]    = future_age * 12500
        res = rule_based_predict(future_inputs)
        prices.append(res["price"])
    return years, prices, car_age_now


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🚗 AutoValuate")
    st.markdown("<p style='color:#3a4460;font-size:0.78rem'>AI-Powered Car Price Estimator</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="section-header">Vehicle Identity</div>', unsafe_allow_html=True)
    brand = st.selectbox("Brand", sorted(BRAND_MODELS.keys()), index=sorted(BRAND_MODELS.keys()).index("Toyota"))
    model_list = BRAND_MODELS.get(brand, ["Other"])
    model_name = st.selectbox("Model", model_list)
    model_year = st.slider("Model Year", 2000, 2024, 2019)

    st.markdown('<div class="section-header">Condition & Usage</div>', unsafe_allow_html=True)
    mileage    = st.number_input("Mileage (miles)", min_value=0, max_value=500_000, value=45_000, step=1000)
    has_accident = st.checkbox("Has accident history", value=False)
    clean_title  = st.checkbox("Clean title", value=True)

    st.markdown('<div class="section-header">Specifications</div>', unsafe_allow_html=True)
    fuel_type    = st.selectbox("Fuel Type", FUEL_TYPES)
    transmission = st.selectbox("Transmission", TRANSMISSIONS)
    engine_hp    = st.slider("Engine Power (HP)", 50, 1500, 200, step=10)
    engine_disp  = st.slider("Engine Displacement (L)", 1.0, 8.5, 2.5, step=0.1)
    cylinders    = st.select_slider("Cylinders", options=[0, 3, 4, 5, 6, 8, 10, 12, 16], value=4)

    st.divider()
    ext_color = st.selectbox("Exterior Color", EXT_COLORS)
    int_color = st.selectbox("Interior Color", INT_COLORS)

    predict_btn = st.button("⚡ Estimate Price", type="primary", use_container_width=True)

    st.divider()
    if USE_TRAINED_MODEL and _trained_model is not None:
        st.markdown("""
        <div style='background:#0a2010;border:1px solid #1a5030;border-radius:8px;
                    padding:0.5rem 0.75rem;font-size:0.78rem;color:#4ff7a0'>
        🟢 &nbsp;<b>LightGBM</b> model active<br>
        <span style='color:#2a5040'>Trained on 188K records</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#0f1525;border:1px solid #1e2d50;border-radius:8px;
                    padding:0.5rem 0.75rem;font-size:0.78rem;color:#4a6090'>
        ⚙️ &nbsp;<b>Rule-based engine</b><br>
        <span style='color:#2a3450'>Run notebook → set USE_TRAINED_MODEL=True</span>
        </div>""", unsafe_allow_html=True)


# ── Main content ──────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">AutoValuate</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Machine-learning powered vehicle valuation · 188,000+ training records</div>', unsafe_allow_html=True)

inputs = {
    "brand": brand, "model_name": model_name, "model_year": model_year,
    "mileage": mileage, "fuel_type": fuel_type, "transmission": transmission,
    "has_accident": has_accident, "clean_title": clean_title,
    "engine_hp": engine_hp, "engine_disp": engine_disp, "cylinders": cylinders,
}

# ── Prediction: trained LightGBM or rule-based fallback ──────────────────────
if USE_TRAINED_MODEL and _trained_model is not None:
    try:
        # Build a single-row feature DataFrame matching the training pipeline
        CURRENT_YEAR = 2024
        car_age  = max(0, CURRENT_YEAR - model_year)
        feat_row = {f: 0 for f in _trained_features}  # zero-fill defaults

        # Numerical features
        feat_row.update({
            'car_age':          car_age,
            'car_age_sq':       car_age ** 2,
            'is_new':           int(car_age <= 2),
            'decade':           (model_year // 10) * 10,
            'model_year':       model_year,
            'log_milage':       np.log1p(mileage),
            'km_per_year':      mileage / (car_age + 1),
            'mileage_bin':      min(5, int(mileage / 20_000)),
            'engine_hp':        engine_hp,
            'engine_disp':      engine_disp,
            'cylinders':        cylinders,
            'hp_per_litre':     engine_hp / (engine_disp + 1e-6),
            'hp_age_ratio':     engine_hp / (car_age + 1),
            'brand_tier_n':     get_brand_tier(brand),
            'has_accident':     int(has_accident),
            'clean_title_flag': int(clean_title),
            'is_electric':      int(fuel_type in ("Electric", "Plug-In Hybrid")),
        })

        # Encoded categoricals — use the label encoders saved during preprocessing
        try:
            le_map = joblib.load('/content/processed/label_encoders.pkl')
            for col in ['brand', 'fuel_type', 'transmission']:
                enc_col = col + '_enc'
                if enc_col in feat_row and col in le_map:
                    le = le_map[col]
                    val_str = str({'brand': brand, 'fuel_type': fuel_type,
                                   'transmission': transmission}[col])
                    feat_row[enc_col] = (le.transform([val_str])[0]
                                        if val_str in le.classes_ else 0)
        except Exception:
            pass  # encoders not found — leave zeros for cat columns

        feat_df      = pd.DataFrame([feat_row])
        ml_price     = ml_predict(feat_df)

        # Confidence band from rule-based engine for context
        rb           = rule_based_predict(inputs)
        band_pct     = rb['band_pct']
        result       = {
            'price':      round(ml_price),
            'low':        round(ml_price * (1 - band_pct)),
            'high':       round(ml_price * (1 + band_pct)),
            'band_pct':   band_pct,
            'car_age':    rb['car_age'],
            'dep_factor': rb['dep_factor'],
            'base_msrp':  rb['base_msrp'],
        }
    except Exception as e:
        st.sidebar.warning(f"Model inference error: {e}. Using rule-based fallback.")
        result = rule_based_predict(inputs)
else:
    result = rule_based_predict(inputs)

# ── Layout ────────────────────────────────────────────────────────────────────
col_price, col_meta = st.columns([1.6, 1], gap="large")

with col_price:
    # Price card
    price_str = f"${result['price']:,}"
    low_str   = f"${result['low']:,}"
    high_str  = f"${result['high']:,}"
    st.markdown(f"""
    <div class="price-card">
        <div class="price-label">Estimated Market Value</div>
        <div class="price-value">{price_str}</div>
        <div class="price-range">Confidence range &nbsp;{low_str} – {high_str}</div>
        <div style="margin-top:1.2rem;display:flex;flex-wrap:wrap;justify-content:center">
            <span class="stat-chip">Age <span>{result['car_age']} yrs</span></span>
            <span class="stat-chip">Depreciation <span>{result['dep_factor']*100:.0f}%</span></span>
            <span class="stat-chip">Est. MSRP <span>${result['base_msrp']:,}</span></span>
            <span class="stat-chip">Band <span>±{result['band_pct']*100:.0f}%</span></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Depreciation curve ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Depreciation Curve</div>', unsafe_allow_html=True)
    years, prices_curve, current_age = depreciation_curve(result["price"], result["car_age"], inputs)

    fig_dep = go.Figure()

    # Full curve (thin)
    fig_dep.add_trace(go.Scatter(
        x=years, y=prices_curve, mode="lines",
        line=dict(color="#2a3d70", width=2),
        name="Value over time", showlegend=False,
    ))
    # Filled area under curve
    fig_dep.add_trace(go.Scatter(
        x=years, y=prices_curve, mode="none",
        fill="tozeroy", fillcolor="rgba(79,142,247,0.06)",
        showlegend=False,
    ))
    # Current position dot
    if 0 <= current_age <= 20:
        current_price = prices_curve[current_age]
        fig_dep.add_trace(go.Scatter(
            x=[current_age], y=[current_price],
            mode="markers+text",
            marker=dict(color="#4f8ef7", size=12, line=dict(color="#0f1930", width=3)),
            text=[f"<b>{price_str}</b>"], textposition="top center",
            textfont=dict(color="#4f8ef7", size=12),
            name="Current",
        ))

    fig_dep.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,19,33,0.6)",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(
            title="Car Age (years)", color="#4a5370",
            gridcolor="#131c2e", zerolinecolor="#1e2d50",
            tickfont=dict(color="#4a5370", size=11),
            title_font=dict(color="#4a5370"),
        ),
        yaxis=dict(
            title="Est. Value (USD)", color="#4a5370",
            gridcolor="#131c2e", zerolinecolor="#1e2d50",
            tickprefix="$", tickformat=",.0f",
            tickfont=dict(color="#4a5370", size=11),
            title_font=dict(color="#4a5370"),
        ),
        hoverlabel=dict(bgcolor="#0d1830", font_color="#c8d8f8"),
    )
    st.plotly_chart(fig_dep, use_container_width=True, config={"displayModeBar": False})


with col_meta:
    # Price factor breakdown bar chart
    st.markdown('<div class="section-header">Price Factor Breakdown</div>', unsafe_allow_html=True)

    base = result["base_msrp"]
    dep_delta  = round(base * result["dep_factor"]) - base
    mile_delta = round(base * result["dep_factor"] * (0.97 if mileage > 12500 * result["car_age"] else 1.03)) - round(base * result["dep_factor"])
    fuel_adj = {"Electric": 0.18, "Plug-In Hybrid": 0.10, "Hybrid": 0.06, "Diesel": 0.04,
                "Flex Fuel": -0.04, "Gasoline": 0, "Unknown": -0.02}
    fuel_delta  = round(result["price"] * fuel_adj.get(fuel_type, 0))
    acc_delta   = round(result["price"] * (-0.16 if has_accident else 0))
    title_delta = round(result["price"] * (0.05 if clean_title else -0.12))

    factors = pd.DataFrame({
        "Factor":  ["Base MSRP", "Depreciation", "Mileage", "Fuel Type", "Accident", "Title"],
        "Impact":  [base, dep_delta, mile_delta, fuel_delta, acc_delta, title_delta],
        "Color":   ["#4f8ef7", "#f7934f", "#f74f6b" if mileage > 80000 else "#4ff7b8",
                    "#4ff7b8" if fuel_delta >= 0 else "#f74f6b",
                    "#f74f6b" if has_accident else "#4ff7b8",
                    "#4ff7b8" if clean_title else "#f74f6b"],
    })

    fig_bar = go.Figure(go.Bar(
        x=factors["Impact"],
        y=factors["Factor"],
        orientation="h",
        marker_color=factors["Color"],
        marker_line_width=0,
        text=[f"${v:+,}" if v != base else f"${v:,}" for v in factors["Impact"]],
        textfont=dict(color="#c8d8f8", size=11),
        textposition="auto",
    ))
    fig_bar.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,19,33,0.6)",
        margin=dict(l=0, r=10, t=5, b=5),
        xaxis=dict(
            color="#4a5370", gridcolor="#131c2e",
            tickprefix="$", tickformat=",.0f",
            tickfont=dict(color="#4a5370", size=10),
        ),
        yaxis=dict(
            color="#c8d8f8", tickfont=dict(color="#c8d8f8", size=11),
        ),
        bargap=0.3,
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    # Key insights
    st.markdown('<div class="section-header">Valuation Insights</div>', unsafe_allow_html=True)

    tier = get_brand_tier(brand)
    tier_labels = {4: "ultra-luxury", 3: "luxury", 2: "mainstream", 1: "budget"}
    st.markdown(f'<div class="insight-box">🏷 <b>{brand}</b> is classified as a <b>{tier_labels[tier]}</b> brand with a tier-{tier} premium factor.</div>', unsafe_allow_html=True)

    miles_per_year = mileage / max(result["car_age"], 1)
    avg = 12_500
    if miles_per_year > avg * 1.3:
        st.markdown(f'<div class="insight-box">📉 High annual mileage ({miles_per_year:,.0f}/yr vs {avg:,} avg) is depressing value significantly.</div>', unsafe_allow_html=True)
    elif miles_per_year < avg * 0.7:
        st.markdown(f'<div class="insight-box">📈 Low annual mileage ({miles_per_year:,.0f}/yr) adds a desirability premium.</div>', unsafe_allow_html=True)

    if fuel_type in ("Electric", "Plug-In Hybrid"):
        st.markdown('<div class="insight-box">⚡ EV/PHEV vehicles carry a market premium driven by lower running costs and incentives.</div>', unsafe_allow_html=True)

    if has_accident:
        st.markdown('<div class="insight-box">⚠️ Accident history applies a ~16% discount vs a clean-history equivalent.</div>', unsafe_allow_html=True)

    if result["car_age"] <= 2:
        st.markdown('<div class="insight-box">🚀 New / near-new vehicles retain strong value. Minimal depreciation curve so far.</div>', unsafe_allow_html=True)


# ── Comparable market data (simulated) ────────────────────────────────────────
st.divider()
st.markdown('<div class="section-header">Market Comparables — Similar Vehicles</div>', unsafe_allow_html=True)

rng = np.random.default_rng(seed=model_year + len(brand))
n_comps = 8
comp_years   = rng.integers(max(2000, model_year - 3), min(2024, model_year + 3) + 1, n_comps)
comp_miles   = np.clip(rng.normal(mileage, mileage * 0.25, n_comps), 0, 300_000).astype(int)
comp_prices  = []
for cy, cm in zip(comp_years, comp_miles):
    ci = dict(inputs); ci["model_year"] = int(cy); ci["mileage"] = int(cm)
    ci["has_accident"] = bool(rng.integers(0, 2))
    comp_prices.append(rule_based_predict(ci)["price"] * rng.uniform(0.88, 1.12))

comps_df = pd.DataFrame({
    "Brand":        [brand] * n_comps,
    "Model":        [model_name] * n_comps,
    "Year":         comp_years,
    "Mileage":      [f"{m:,}" for m in comp_miles],
    "Est. Price":   [f"${p:,.0f}" for p in comp_prices],
    "vs. Your Car": [f"{'+'if p > result['price'] else ''}{((p/result['price'])-1)*100:.1f}%" for p in comp_prices],
})

st.dataframe(comps_df, use_container_width=True, hide_index=True,
             column_config={
                 "Year":         st.column_config.NumberColumn(format="%d"),
                 "Est. Price":   st.column_config.TextColumn(),
                 "vs. Your Car": st.column_config.TextColumn(),
             })

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.markdown("<small style='color:#2a3450'>📊 Trained on 188K vehicle records</small>", unsafe_allow_html=True)
with col_f2:
    st.markdown("<small style='color:#2a3450'>🤖 Models: LightGBM · XGBoost · Random Forest · Ensemble</small>", unsafe_allow_html=True)
with col_f3:
    st.markdown("<small style='color:#2a3450'>⚡ Kaggle Playground Series S4E9</small>", unsafe_allow_html=True)
