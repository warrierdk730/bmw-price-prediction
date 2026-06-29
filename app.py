import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go
import plotly.express as px

# ── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BMW Price Estimator",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #f5f7fa; }
    .price-card {
        background: linear-gradient(135deg, #1a3c5e 0%, #2d6a9f 100%);
        border-radius: 16px; padding: 32px 36px; text-align: center;
        color: white; margin: 16px 0; box-shadow: 0 6px 24px rgba(26,60,94,0.25);
    }
    .price-label { font-size: 15px; font-weight: 500; letter-spacing: 1.5px;
        text-transform: uppercase; opacity: 0.85; margin-bottom: 8px; }
    .price-value { font-size: 52px; font-weight: 800; letter-spacing: -1px; margin: 4px 0; }
    .price-range { font-size: 16px; opacity: 0.80; margin-top: 6px; }
    [data-testid="metric-container"] {
        background: white; border-radius: 12px; border: 1px solid #e0e6ed;
        padding: 14px 18px; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    [data-testid="metric-container"] label {
        color: #5a7a96 !important; font-size: 12px !important;
        font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.5px;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #1a3c5e !important; font-size: 22px !important; font-weight: 700 !important;
    }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0f2035 0%, #1a3c5e 100%); }
    [data-testid="stSidebar"] * { color: #e8f0f8 !important; }
    .section-header {
        background: linear-gradient(90deg, #1a3c5e, #2d6a9f); color: white;
        padding: 9px 18px; border-radius: 8px; font-size: 15px; font-weight: 600;
        margin: 20px 0 14px 0;
    }
    .insight-box {
        background: #eaf2fb; border-left: 4px solid #2d6a9f;
        border-radius: 0 8px 8px 0; padding: 12px 16px;
        font-size: 14px; color: #1a3c5e; margin: 10px 0; line-height: 1.6;
    }
    .warn-box {
        background: #fff8e6; border-left: 4px solid #e6a817;
        border-radius: 0 8px 8px 0; padding: 12px 16px;
        font-size: 13px; color: #7a5500; margin: 10px 0;
    }
    .good-box {
        background: #eafaf1; border-left: 4px solid #1e8449;
        border-radius: 0 8px 8px 0; padding: 12px 16px;
        font-size: 13px; color: #145a32; margin: 10px 0;
    }
    .main-title { font-size: 30px; font-weight: 800; color: #1a3c5e; margin-bottom: 2px; }
    .main-sub { font-size: 14px; color: #5a7a96; margin-bottom: 16px; }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL & DATA ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('bmw_model.pkl', 'rb') as f:
        mdl = pickle.load(f)
    with open('bmw_features.json') as f:
        feats = json.load(f)
    return mdl, feats

@st.cache_data
def load_data():
    df = pd.read_csv('bmw.csv')
    df['model'] = df['model'].str.strip()
    df = df[df['engineSize'] > 0].copy()
    df = df[df['price'] <= df['price'].quantile(0.99)].copy()
    df = df[df['mpg'] <= 200].copy()
    return df

try:
    model, FEATURE_COLS = load_model()
    df_data = load_data()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.markdown("Make sure **bmw_model.pkl**, **bmw_features.json**, and **bmw.csv** are in the same folder as app.py.")
    st.stop()

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
ALL_MODELS        = sorted(df_data['model'].unique())
ALL_FUELS         = sorted(df_data['fuelType'].unique())
ALL_TRANSMISSIONS = sorted(df_data['transmission'].unique())
ALL_ENGINES       = sorted(df_data['engineSize'].unique())
TAX_VALUES        = sorted(df_data['tax'].unique())

# ── PREDICTION FUNCTION ───────────────────────────────────────────────────────
def predict_price(model_name, year, mileage, engine_size,
                  transmission, fuel_type, tax, mpg):
    car_age     = max(2024 - year, 1)
    mil_per_yr  = mileage / car_age
    eng_age_int = engine_size * car_age

    row = {col: 0 for col in FEATURE_COLS}
    row['year']                   = year
    row['mileage']                = mileage
    row['tax']                    = tax
    row['mpg']                    = mpg
    row['engineSize']             = engine_size
    row['car_age']                = car_age
    row['mileage_per_year']       = mil_per_yr
    row['engine_age_interaction'] = eng_age_int

    for key in [f'model_{model_name}',
                f'transmission_{transmission}',
                f'fuelType_{fuel_type}']:
        if key in row:
            row[key] = 1

    X          = pd.DataFrame([row])[FEATURE_COLS]
    pred       = model.predict(X)[0]
    tree_preds = np.array([t.predict(X)[0] for t in model.estimators_])
    low        = max(np.percentile(tree_preds, 10), 500)
    high       = np.percentile(tree_preds, 90)

    return round(pred), round(low), round(high), tree_preds

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 Car Specifications")
    st.markdown("---")

    selected_model = st.selectbox("BMW Model", ALL_MODELS,
                                  index=ALL_MODELS.index('3 Series'))
    year = st.slider("Registration Year", 1996, 2020, 2017)
    mileage = st.slider("Mileage (miles)", 0, 200000, 20000, step=1000)
    engine_size = st.selectbox("Engine Size (litres)", ALL_ENGINES,
                               index=list(ALL_ENGINES).index(2.0))
    transmission = st.selectbox("Transmission", ALL_TRANSMISSIONS,
                                index=ALL_TRANSMISSIONS.index('Semi-Auto'))
    fuel_type = st.selectbox("Fuel Type", ALL_FUELS,
                             index=ALL_FUELS.index('Diesel'))
    tax = st.selectbox("Annual Road Tax (£)", TAX_VALUES,
                       index=TAX_VALUES.index(145))
    mpg = st.slider("Fuel Economy (MPG)", 10.0, 200.0, 53.3, step=0.5)

    st.markdown("---")
    predict_btn = st.button("🔍 Estimate Price", use_container_width=True, type="primary")
    st.markdown("---")
    st.markdown("""
    <small>
    <b>Model info</b><br>
    Algorithm: Random Forest (200 trees)<br>
    R² Score: 0.9585<br>
    Avg Error: ±£1,463<br>
    Training data: 10,673 UK listings<br>
    Dataset: BMW Used Cars — Kaggle
    </small>
    """, unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🚗 BMW Used Car Price Estimator</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="main-sub">Predict the fair market value of any used BMW · '
    'Random Forest model (R² = 0.9585) · 10,781 UK listings · 1996–2020</div>',
    unsafe_allow_html=True)

# ── PREDICTION VIEW ───────────────────────────────────────────────────────────
if predict_btn:
    pred, low, high, tree_preds = predict_price(
        selected_model, year, mileage, engine_size,
        transmission, fuel_type, tax, mpg
    )
    car_age = max(2024 - year, 1)

    # Main price card
    st.markdown(f"""
    <div class="price-card">
        <div class="price-label">Estimated Market Price — {year} BMW {selected_model}</div>
        <div class="price-value">£{pred:,}</div>
        <div class="price-range">Likely range &nbsp;·&nbsp; £{low:,} – £{high:,}</div>
    </div>""", unsafe_allow_html=True)

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📅 Car Age",        f"{car_age} yrs")
    c2.metric("📍 Miles/Year",     f"{mileage // car_age:,}")
    c3.metric("📊 Price Range",    f"£{high - low:,}")
    c4.metric("⚙️  Eng × Age",    f"{engine_size * car_age:.1f}")
    c5.metric("🔋 Fuel",          fuel_type)

    st.markdown("---")
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<div class="section-header">📊 Prediction Distribution (200 Trees)</div>',
                    unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=tree_preds, nbinsx=35,
                                   marker_color='#2d6a9f', opacity=0.85))
        fig.add_vline(x=pred, line_dash='dash', line_color='#c0392b', line_width=2.5,
                      annotation_text=f'Estimate: £{pred:,}',
                      annotation_position='top right',
                      annotation_font_color='#c0392b')
        fig.add_vrect(x0=low, x1=high, fillcolor='rgba(45,106,159,0.12)',
                      line_width=0, annotation_text='80% range',
                      annotation_position='top left',
                      annotation_font_size=11)
        fig.update_layout(xaxis_title='Predicted Price (£)', yaxis_title='Trees',
                          xaxis_tickprefix='£', xaxis_tickformat=',.0f',
                          template='plotly_white', height=300,
                          margin=dict(t=20, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""<div class="insight-box">
            Each of the 200 Random Forest trees votes independently.
            A tight cluster = high confidence. A wide spread = more uncertainty
            (common for rare spec combinations).
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">🔎 Similar Cars in Dataset</div>',
                    unsafe_allow_html=True)
        similar = df_data[
            (df_data['model'] == selected_model) &
            (df_data['year'].between(year - 2, year + 2)) &
            (df_data['transmission'] == transmission) &
            (df_data['fuelType'] == fuel_type)
        ]

        if len(similar) >= 3:
            sim_med = similar['price'].median()
            m1, m2 = st.columns(2)
            m1.metric("Dataset Median", f"£{sim_med:,.0f}")
            m2.metric("Listings Found", f"{len(similar)}")
            m1.metric("Lowest",         f"£{similar['price'].min():,.0f}")
            m2.metric("Highest",        f"£{similar['price'].max():,.0f}")

            diff     = pred - sim_med
            diff_pct = diff / sim_med * 100
            if abs(diff_pct) <= 5:
                box, msg = "insight-box", f"✅ Your estimate (£{pred:,}) is <b>in line</b> with similar market listings."
            elif diff_pct > 5:
                box, msg = "warn-box", f"⬆️ Estimate is <b>£{abs(diff):,.0f} above</b> median ({diff_pct:+.1f}%). Could reflect premium spec or recent model."
            else:
                box, msg = "good-box", f"⬇️ Estimate is <b>£{abs(diff):,.0f} below</b> median ({diff_pct:+.1f}%). This may be a great value find."
            st.markdown(f'<div class="{box}">{msg}</div>', unsafe_allow_html=True)

            # Mini chart — similar cars price distribution
            fig2 = px.histogram(similar, x='price', nbinsx=20,
                                color_discrete_sequence=['#4a90d9'],
                                template='plotly_white')
            fig2.add_vline(x=pred, line_dash='dash', line_color='#c0392b',
                           line_width=2, annotation_text='Your estimate',
                           annotation_font_color='#c0392b')
            fig2.update_layout(
                xaxis_tickprefix='£', xaxis_tickformat=',.0f',
                xaxis_title='Price (£)', yaxis_title='Count',
                height=200, margin=dict(t=10, b=10, l=10, r=10),
                showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.markdown(f"""<div class="warn-box">
                Only {len(similar)} exact matches found in dataset.
                Estimate is based on broader model patterns.
            </div>""", unsafe_allow_html=True)

    # Price drivers gauges
    st.markdown('<div class="section-header">💡 Price Driver Scores</div>',
                unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)

    gauges = [
        (g1, "Year Score",   int((year - 1996) / 24 * 100),
         f"Year {year} → newer = higher price"),
        (g2, "Mileage Score", max(0, int((1 - mileage / 200000) * 100)),
         f"{mileage:,} mi → lower mileage = higher price"),
        (g3, "Engine Score",  int((engine_size / 6.6) * 100),
         f"{engine_size}L → larger engine = higher price"),
    ]

    for col, title, val, caption_txt in gauges:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val,
            title={'text': title, 'font': {'size': 14}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#1a3c5e'},
                'steps': [
                    {'range': [0,  33], 'color': '#fde8e8'},
                    {'range': [33, 66], 'color': '#fff8e6'},
                    {'range': [66,100], 'color': '#e8f5e9'},
                ],
            },
            number={'suffix': '/100', 'font': {'size': 20}}
        ))
        fig_g.update_layout(height=210, margin=dict(t=30, b=5, l=20, r=20))
        col.plotly_chart(fig_g, use_container_width=True)
        col.caption(caption_txt)

    # Depreciation curve — show where this car sits
    st.markdown('<div class="section-header">📉 Where Does This Car Sit on the Depreciation Curve?</div>',
                unsafe_allow_html=True)

    model_df    = df_data[df_data['model'] == selected_model]
    yr_trend    = model_df.groupby('year')['price'].mean().reset_index()
    fig_dep     = go.Figure()
    fig_dep.add_trace(go.Scatter(
        x=yr_trend['year'], y=yr_trend['price'],
        mode='lines+markers', name=f'BMW {selected_model} avg price',
        line=dict(color='#2d6a9f', width=2.5),
        marker=dict(size=6),
        fill='tozeroy', fillcolor='rgba(45,106,159,0.10)'
    ))
    fig_dep.add_trace(go.Scatter(
        x=[year], y=[pred],
        mode='markers', name='Your car',
        marker=dict(size=16, color='#c0392b', symbol='star',
                    line=dict(width=2, color='white'))
    ))
    fig_dep.update_layout(
        xaxis_title='Registration Year',
        yaxis_title='Average Price (£)',
        yaxis_tickprefix='£', yaxis_tickformat=',.0f',
        template='plotly_white', height=320,
        legend=dict(orientation='h', y=1.08),
        margin=dict(t=20, b=10, l=10, r=10),
        hovermode='x unified'
    )
    st.plotly_chart(fig_dep, use_container_width=True)

# ── DEFAULT MARKET OVERVIEW ───────────────────────────────────────────────────
else:
    st.markdown("""<div class="insight-box" style="font-size:15px; padding:16px 20px;">
        👈 <b>Configure your BMW specs in the sidebar</b> and click
        <b>Estimate Price</b> to get an instant market valuation.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">📊 Market Overview — BMW Used Car Dataset</div>',
                unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Listings",  f"{len(df_data):,}")
    k2.metric("BMW Models",      f"{df_data['model'].nunique()}")
    k3.metric("Median Price",    f"£{df_data['price'].median():,.0f}")
    k4.metric("Lowest Price",    f"£{df_data['price'].min():,}")
    k5.metric("Highest Price",   f"£{df_data['price'].max():,}")

    col1, col2 = st.columns(2)

    with col1:
        model_avg = df_data.groupby('model')['price'].mean().sort_values(ascending=True)
        fig = px.bar(x=model_avg.values, y=model_avg.index,
                     orientation='h', title='Average Price by BMW Model',
                     color=model_avg.values,
                     color_continuous_scale=['#a8d0ed', '#2d6a9f', '#1a3c5e'],
                     template='plotly_white')
        fig.update_layout(xaxis_tickprefix='£', xaxis_tickformat=',.0f',
                          coloraxis_showscale=False, height=500,
                          title_font_size=14,
                          margin=dict(t=40, b=10, l=10, r=10),
                          xaxis_title='Average Price (£)', yaxis_title='')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.histogram(df_data, x='price', nbins=50,
                            title='Price Distribution (All Models)',
                            color_discrete_sequence=['#2d6a9f'],
                            template='plotly_white')
        fig2.add_vline(x=df_data['price'].median(), line_dash='dash',
                       line_color='#c0392b', line_width=2,
                       annotation_text=f"Median: £{df_data['price'].median():,.0f}",
                       annotation_position='top right')
        fig2.update_layout(xaxis_tickprefix='£', xaxis_tickformat=',.0f',
                           height=240, title_font_size=14,
                           margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.box(df_data, x='transmission', y='price',
                      title='Price by Transmission',
                      color='transmission',
                      color_discrete_sequence=['#1a3c5e','#2d6a9f','#4a90d9'],
                      template='plotly_white')
        fig3.update_layout(yaxis_tickprefix='£', yaxis_tickformat=',.0f',
                           showlegend=False, height=240, title_font_size=14,
                           margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig3, use_container_width=True)

    # Depreciation curve
    st.markdown('<div class="section-header">📈 BMW Price Depreciation Curve (All Models)</div>',
                unsafe_allow_html=True)
    yr_all = df_data.groupby('year')['price'].mean().reset_index()
    fig4   = go.Figure()
    fig4.add_trace(go.Scatter(
        x=yr_all['year'], y=yr_all['price'],
        mode='lines+markers',
        line=dict(color='#1a3c5e', width=3),
        marker=dict(size=7, color='#2d6a9f'),
        fill='tozeroy', fillcolor='rgba(45,106,159,0.10)',
        hovertemplate='<b>%{x}</b><br>Avg Price: £%{y:,.0f}<extra></extra>'
    ))
    fig4.update_layout(
        xaxis_title='Registration Year', yaxis_title='Average Price (£)',
        yaxis_tickprefix='£', yaxis_tickformat=',.0f',
        template='plotly_white', height=300,
        margin=dict(t=10, b=10, l=10, r=10), hovermode='x unified'
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("""<div class="insight-box">
        BMW prices rise sharply for cars registered after 2015. The steepest depreciation
        happens in the first 3–5 years — a 2–3 year old BMW often offers the best value for money.
    </div>""", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#8a9bb0;font-size:12px;padding:8px 0;">
BMW Price Estimator &nbsp;·&nbsp; Built by <b>Dileep Kumar Warrier</b> &nbsp;·&nbsp;
Random Forest (R² = 0.9585, MAE = £1,463) &nbsp;·&nbsp; Dataset: Kaggle &nbsp;·&nbsp;
<a href="https://github.com/warrierdk730/bmw-price-prediction"
   target="_blank" style="color:#2d6a9f;">GitHub</a>
</div>
""", unsafe_allow_html=True)
