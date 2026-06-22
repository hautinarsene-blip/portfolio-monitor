"""
streamlit_app.py
================
Private Markets Portfolio Monitor — Interactive Dashboard
----------------------------------------------------------
Run:  streamlit run streamlit_app.py

Dependencies: streamlit, pandas, numpy, scipy, openpyxl, plotly
Install:      pip install streamlit pandas numpy scipy openpyxl plotly
"""

import warnings
from pathlib import Path
from datetime import date, datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Private Markets Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── colour palette ─────────────────────────────────────────────────────────────
NAVY    = "#1F3864"
TEAL    = "#2E86AB"
SILVER  = "#A8DADC"
CREAM   = "#F8F9FA"
GREEN   = "#2D6A4F"
AMBER   = "#E9C46A"
CORAL   = "#E76F51"

# ── custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        border-left: 4px solid #1F3864;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .metric-label { font-size: 0.72rem; color: #6B7280; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #1F3864; margin-top: 2px; }
    .metric-sub   { font-size: 0.78rem; color: #9CA3AF; margin-top: 2px; }
    .section-title { font-size: 1rem; font-weight: 700; color: #1F3864;
                     border-bottom: 2px solid #1F3864; padding-bottom: 4px;
                     margin-bottom: 1rem; }
    div[data-testid="stSidebar"] { background-color: #F0F4F8; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS ENGINE  (self-contained — no external import needed)
# ══════════════════════════════════════════════════════════════════════════════

def _xirr(dates, cashflows, tol=1e-8):
    if len(cashflows) < 2:
        return None
    d0  = min(dates)
    t   = np.array([(d - d0).days / 365.0 for d in dates])
    cf  = np.array(cashflows, dtype=float)
    npv = lambda r: np.sum(cf / (1 + r) ** t)
    try:
        if npv(-0.999) * npv(100) > 0:
            return None
        from scipy.optimize import brentq
        return brentq(npv, -0.999, 100, xtol=tol)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_data(file_bytes, filename):
    import io
    xl  = pd.ExcelFile(io.BytesIO(file_bytes))

    def _read(sheet, date_cols):
        df = xl.parse(sheet, parse_dates=date_cols)
        df.columns = (df.columns.str.strip().str.lower()
                        .str.replace(r"[\s\(\)\$]", "_", regex=True)
                        .str.replace(r"_+", "_", regex=True)
                        .str.strip("_"))
        return df

    return {
        "funds": _read("Funds",        []),
        "calls": _read("Capital_Calls", ["Call Date"]),
        "dists": _read("Distributions", ["Distribution Date"]),
        "nav":   _read("NAV",           ["NAV Date"]),
    }


def compute_metrics(data, as_of):
    funds, calls, dists, nav_df = (
        data["funds"], data["calls"], data["dists"], data["nav"]
    )
    calls["call_date"]         = pd.to_datetime(calls["call_date"]).dt.date
    dists["distribution_date"] = pd.to_datetime(dists["distribution_date"]).dt.date
    nav_df["nav_date"]         = pd.to_datetime(nav_df["nav_date"]).dt.date

    rows = []
    for _, fund in funds.iterrows():
        name = fund["fund_name"]
        fc = calls[(calls["fund_name"] == name) & (calls["call_date"] <= as_of)]
        fd = dists[(dists["fund_name"] == name) & (dists["distribution_date"] <= as_of)]
        fn = nav_df[(nav_df["fund_name"] == name) & (nav_df["nav_date"] <= as_of)]

        called   = fc["amount"].sum()  if not fc.empty else 0.0
        distrib  = fd["amount"].sum()  if not fd.empty else 0.0
        nav_val  = fn.sort_values("nav_date").iloc[-1]["nav"] if not fn.empty else 0.0

        cf_d, cf_v = [], []
        for _, r in fc.iterrows():
            cf_d.append(r["call_date"]); cf_v.append(-r["amount"])
        for _, r in fd.iterrows():
            cf_d.append(r["distribution_date"]); cf_v.append(r["amount"])
        if nav_val > 0:
            cf_d.append(as_of); cf_v.append(nav_val)

        irr  = _xirr(cf_d, cf_v) if called > 0 else None
        dpi  = distrib / called  if called > 0 else 0.0
        rvpi = nav_val / called  if called > 0 else 0.0

        rows.append({
            "Fund":              name,
            "Vintage":           int(fund["vintage_year"]),
            "Strategy":          fund.get("strategy", ""),
            "Geography":         fund.get("geography", ""),
            "Committed ($M)":    fund["committed_capital"] / 1e6,
            "Called ($M)":       called / 1e6,
            "Distributions ($M)":distrib / 1e6,
            "NAV ($M)":          nav_val / 1e6,
            "Net IRR":           irr,
            "MOIC":              (distrib + nav_val) / called if called else 0,
            "DPI":               dpi,
            "RVPI":              rvpi,
            "TVPI":              dpi + rvpi,
        })
    return pd.DataFrame(rows)


def liquidity_model(fund_perf, as_of, horizon, call_rate, dist_rate, nav_growth):
    rows = []
    for _, fund in fund_perf.iterrows():
        unfunded = max(fund["Committed ($M)"] - fund["Called ($M)"], 0)
        nav      = fund["NAV ($M)"]
        for yr in range(1, horizon + 1):
            calls_est = unfunded * call_rate
            unfunded  = max(unfunded - calls_est, 0)
            dists_est = nav * dist_rate
            nav       = max(nav * (1 + nav_growth) - dists_est, 0)
            rows.append({
                "Fund":            fund["Fund"],
                "Vintage":         fund["Vintage"],
                "Year":            as_of.year + yr,
                "Est. Calls ($M)": round(calls_est, 2),
                "Est. Dists ($M)": round(dists_est, 2),
                "Net CF ($M)":     round(dists_est - calls_est, 2),
                "Proj. NAV ($M)":  round(nav, 2),
            })
    df = pd.DataFrame(rows)
    agg = (df.groupby("Year", as_index=False)
             .agg(**{
                 "Total Calls ($M)":  ("Est. Calls ($M)", "sum"),
                 "Total Dists ($M)":  ("Est. Dists ($M)", "sum"),
                 "Net CF ($M)":       ("Net CF ($M)",     "sum"),
                 "Proj. NAV ($M)":    ("Proj. NAV ($M)",  "sum"),
             }))
    agg["Cumulative CF ($M)"] = agg["Net CF ($M)"].cumsum()
    return df, agg


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f"<div style='color:{NAVY};font-size:1.1rem;font-weight:700;"
                f"padding-bottom:8px;border-bottom:2px solid {NAVY};"
                f"margin-bottom:1rem'>⚙️ Settings</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Excel file", type=["xlsx"],
                                help="Needs sheets: Funds, Capital_Calls, Distributions, NAV")

    # use bundled sample if nothing uploaded
    SAMPLE_PATH = Path(__file__).parent / "fund_data_input.xlsx"
    use_sample  = not uploaded

    st.markdown("---")
    st.markdown("**Valuation date**")
    as_of_dt = st.date_input("As-of date", value=date(2023, 12, 31),
                              min_value=date(2015, 1, 1), max_value=date(2030, 12, 31))
    as_of = as_of_dt

    st.markdown("---")
    st.markdown("**Liquidity model assumptions**")
    call_rate   = st.slider("Annual call rate (% of unfunded)", 5, 50, 20) / 100
    dist_rate   = st.slider("Annual distribution rate (% of NAV)", 5, 50, 25) / 100
    nav_growth  = st.slider("Annual NAV growth rate (%)", 0, 30, 12) / 100
    horizon     = st.slider("Projection horizon (years)", 3, 10, 5)

    st.markdown("---")
    if use_sample:
        st.info("Using built-in sample dataset (5 funds)")
    else:
        st.success(f"Loaded: {uploaded.name}")


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOAD
# ══════════════════════════════════════════════════════════════════════════════

try:
    if uploaded:
        data = load_data(uploaded.read(), uploaded.name)
    elif SAMPLE_PATH.exists():
        data = load_data(SAMPLE_PATH.read_bytes(), "fund_data_input.xlsx")
    else:
        st.error("No data source found. Upload an Excel file or place "
                 "fund_data_input.xlsx in the same folder as this script.")
        st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

fund_perf = compute_metrics(data, as_of)
fund_liq_detail, fund_liq_agg = liquidity_model(
    fund_perf, as_of, horizon, call_rate, dist_rate, nav_growth
)

total_called  = fund_perf["Called ($M)"].sum()
total_dists   = fund_perf["Distributions ($M)"].sum()
total_nav     = fund_perf["NAV ($M)"].sum()
port_dpi      = total_dists / total_called if total_called else 0
port_tvpi     = (total_dists + total_nav) / total_called if total_called else 0
weights       = fund_perf["Called ($M)"] / total_called
hhi           = (weights ** 2).sum()


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    f"<div style='background:{NAVY};padding:1rem 1.5rem;border-radius:8px;"
    f"margin-bottom:1.5rem'>"
    f"<span style='color:white;font-size:1.4rem;font-weight:700'>"
    f"Private Markets Portfolio Monitor</span>"
    f"<span style='color:{SILVER};font-size:0.85rem;margin-left:1rem'>"
    f"as of {as_of.strftime('%d %b %Y')} &nbsp;|&nbsp; "
    f"{len(fund_perf)} funds &nbsp;|&nbsp; "
    f"€{total_called:.0f}M called capital</span>"
    f"</div>",
    unsafe_allow_html=True
)


# ══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════════════════

k1, k2, k3, k4, k5 = st.columns(5)

def kpi(col, label, value, sub=""):
    col.markdown(
        f"<div class='metric-card'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div>"
        f"<div class='metric-sub'>{sub}</div>"
        f"</div>", unsafe_allow_html=True
    )

kpi(k1, "Portfolio TVPI",  f"{port_tvpi:.2f}x",
    f"DPI {port_dpi:.2f}x + RVPI {(port_tvpi-port_dpi):.2f}x")
kpi(k2, "Total NAV",       f"€{total_nav:.0f}M",
    f"Distributions: €{total_dists:.0f}M")
kpi(k3, "Called Capital",  f"€{total_called:.0f}M",
    f"Committed: €{fund_perf['Committed ($M)'].sum():.0f}M")
kpi(k4, "Unfunded",
    f"€{fund_perf['Committed ($M)'].sum() - total_called:.0f}M",
    "Remaining commitment")
kpi(k5, "HHI Concentration", f"{hhi:.3f}",
    f"Eff. {(1/hhi):.1f} funds equiv.")

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Fund Performance", "🗺️ Portfolio Exposure",
    "💧 Liquidity Projection", "📥 Export"
])


# ── Tab 1: Fund Performance ───────────────────────────────────────────────────
with tab1:
    st.markdown("<div class='section-title'>Fund-Level Metrics</div>",
                unsafe_allow_html=True)

    display = fund_perf.copy()
    display["Net IRR"] = display["Net IRR"].apply(
        lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
    display["MOIC"]  = display["MOIC"].apply(lambda x: f"{x:.2f}x")
    display["DPI"]   = display["DPI"].apply(lambda x: f"{x:.2f}x")
    display["RVPI"]  = display["RVPI"].apply(lambda x: f"{x:.2f}x")
    display["TVPI"]  = display["TVPI"].apply(lambda x: f"{x:.2f}x")
    for col in ["Committed ($M)", "Called ($M)", "Distributions ($M)", "NAV ($M)"]:
        display[col] = display[col].apply(lambda x: f"€{x:.1f}M")

    st.dataframe(display.set_index("Fund"), use_container_width=True, height=220)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='section-title'>TVPI vs Net IRR by Fund</div>",
                    unsafe_allow_html=True)
        scatter = fund_perf.copy()
        scatter["IRR_num"] = scatter["Net IRR"].fillna(0)
        fig = px.scatter(
            scatter, x="IRR_num", y="TVPI", size="Called ($M)",
            color="Strategy", hover_name="Fund",
            color_discrete_sequence=[NAVY, TEAL, AMBER, CORAL, GREEN],
            labels={"IRR_num": "Net IRR", "TVPI": "TVPI (x)"},
            size_max=40,
        )
        fig.update_xaxes(tickformat=".0%")
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Arial", size=11),
            margin=dict(t=10, b=40, l=40, r=10), height=280,
            legend=dict(orientation="h", y=-0.25)
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("<div class='section-title'>DPI / RVPI Split by Fund</div>",
                    unsafe_allow_html=True)
        bar_df = fund_perf[["Fund", "DPI", "RVPI"]].copy()
        short_names = [f.split()[0] for f in bar_df["Fund"]]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="DPI (realised)",  x=short_names, y=bar_df["DPI"],
                              marker_color=TEAL))
        fig2.add_trace(go.Bar(name="RVPI (unrealised)", x=short_names, y=bar_df["RVPI"],
                              marker_color=SILVER))
        fig2.update_layout(
            barmode="stack",
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Arial", size=11),
            margin=dict(t=10, b=40, l=40, r=10), height=280,
            yaxis_title="Multiple (x)",
            legend=dict(orientation="h", y=-0.25),
        )
        fig2.add_hline(y=1.0, line_dash="dash", line_color=CORAL,
                       annotation_text="1.0x", annotation_position="right")
        st.plotly_chart(fig2, use_container_width=True)


# ── Tab 2: Portfolio Exposure ──────────────────────────────────────────────────
with tab2:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='section-title'>NAV Exposure by Fund</div>",
                    unsafe_allow_html=True)
        pie_df = fund_perf[fund_perf["NAV ($M)"] > 0].copy()
        fig3 = px.pie(pie_df, values="NAV ($M)", names="Fund",
                      color_discrete_sequence=[NAVY, TEAL, SILVER, AMBER, CORAL])
        fig3.update_traces(textposition="inside", textinfo="percent+label",
                           hole=0.35)
        fig3.update_layout(
            showlegend=False, margin=dict(t=10, b=10, l=10, r=10),
            height=300, paper_bgcolor="white",
        )
        st.plotly_chart(fig3, use_container_width=True)

    with c2:
        st.markdown("<div class='section-title'>Called Capital by Vintage Year</div>",
                    unsafe_allow_html=True)
        vint = fund_perf.groupby("Vintage", as_index=False).agg(
            Called=("Called ($M)", "sum"),
            NAV=("NAV ($M)", "sum"),
        )
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(x=vint["Vintage"].astype(str), y=vint["Called"],
                              name="Called", marker_color=NAVY))
        fig4.add_trace(go.Bar(x=vint["Vintage"].astype(str), y=vint["NAV"],
                              name="NAV", marker_color=TEAL))
        fig4.update_layout(
            barmode="group", plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Arial", size=11),
            margin=dict(t=10, b=40, l=40, r=10), height=300,
            yaxis_title="€M", legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("<div class='section-title'>Exposure by Strategy</div>",
                unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    strat_called = fund_perf.groupby("Strategy")["Called ($M)"].sum().reset_index()
    strat_nav    = fund_perf.groupby("Strategy")["NAV ($M)"].sum().reset_index()
    geo_nav      = fund_perf.groupby("Geography")["NAV ($M)"].sum().reset_index()

    with c3:
        fig5 = px.bar(strat_called, x="Strategy", y="Called ($M)",
                      color="Strategy",
                      color_discrete_sequence=[NAVY, TEAL, AMBER, CORAL, GREEN],
                      labels={"Called ($M)": "Called Capital (€M)"})
        fig5.update_layout(showlegend=False, plot_bgcolor="white",
                           paper_bgcolor="white", height=260,
                           margin=dict(t=10, b=40, l=40, r=10),
                           font=dict(family="Arial", size=11))
        st.plotly_chart(fig5, use_container_width=True)

    with c4:
        fig6 = px.bar(geo_nav, x="Geography", y="NAV ($M)",
                      color="Geography",
                      color_discrete_sequence=[NAVY, TEAL, AMBER, CORAL, GREEN],
                      labels={"NAV ($M)": "NAV (€M)"})
        fig6.update_layout(showlegend=False, plot_bgcolor="white",
                           paper_bgcolor="white", height=260,
                           margin=dict(t=10, b=40, l=40, r=10),
                           font=dict(family="Arial", size=11))
        st.plotly_chart(fig6, use_container_width=True)


# ── Tab 3: Liquidity Projection ───────────────────────────────────────────────
with tab3:
    st.markdown(
        f"<div class='section-title'>Portfolio Liquidity Projection "
        f"({as_of.year + 1}–{as_of.year + horizon})</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Assumptions: call rate {call_rate:.0%} of unfunded  ·  "
        f"distribution rate {dist_rate:.0%} of NAV  ·  "
        f"NAV growth {nav_growth:.0%} p.a."
    )

    c1, c2 = st.columns([3, 2])

    with c1:
        fig7 = go.Figure()
        fig7.add_trace(go.Bar(
            x=fund_liq_agg["Year"].astype(str),
            y=fund_liq_agg["Total Dists ($M)"],
            name="Estimated Distributions", marker_color=GREEN,
        ))
        fig7.add_trace(go.Bar(
            x=fund_liq_agg["Year"].astype(str),
            y=-fund_liq_agg["Total Calls ($M)"],
            name="Estimated Capital Calls", marker_color=CORAL,
        ))
        fig7.add_trace(go.Scatter(
            x=fund_liq_agg["Year"].astype(str),
            y=fund_liq_agg["Cumulative CF ($M)"],
            name="Cumulative Net CF", mode="lines+markers",
            line=dict(color=NAVY, width=2.5, dash="dot"),
            yaxis="y2",
        ))
        fig7.update_layout(
            barmode="relative",
            yaxis=dict(title="Annual CF (€M)", gridcolor="#E5E7EB"),
            yaxis2=dict(title="Cumulative (€M)", overlaying="y", side="right",
                        showgrid=False),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Arial", size=11),
            margin=dict(t=10, b=40, l=50, r=50), height=360,
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig7, use_container_width=True)

    with c2:
        st.markdown("<div class='section-title'>Annual Summary</div>",
                    unsafe_allow_html=True)
        disp_liq = fund_liq_agg.copy()
        for col in ["Total Calls ($M)", "Total Dists ($M)",
                    "Net CF ($M)", "Cumulative CF ($M)", "Proj. NAV ($M)"]:
            if col in disp_liq.columns:
                disp_liq[col] = disp_liq[col].apply(lambda x: f"€{x:.1f}M")
        st.dataframe(disp_liq.set_index("Year"), use_container_width=True,
                     height=320)

    st.markdown("<div class='section-title'>Fund-Level Breakdown</div>",
                unsafe_allow_html=True)
    pivot = fund_liq_detail.pivot_table(
        index="Fund", columns="Year",
        values="Net CF ($M)", aggfunc="sum"
    ).round(1)
    st.dataframe(pivot.style.background_gradient(
        cmap="RdYlGn", axis=None
    ), use_container_width=True)


# ── Tab 4: Export ─────────────────────────────────────────────────────────────
with tab4:
    st.markdown("<div class='section-title'>Export Power BI-Ready Tables</div>",
                unsafe_allow_html=True)
    st.caption("All tables are formatted for direct connection to Power BI via Get Data > Text/CSV.")

    exports = {
        "fund_performance_summary.csv": fund_perf,
        "liquidity_projection.csv":     fund_liq_agg,
        "fund_liquidity_detail.csv":    fund_liq_detail,
    }
    for fname, df in exports.items():
        csv = df.to_csv(index=False).encode("utf-8")
        col1, col2 = st.columns([3, 1])
        col1.markdown(f"**{fname}** — {len(df)} rows, {len(df.columns)} columns")
        col2.download_button(f"Download", csv, file_name=fname,
                             mime="text/csv", key=fname)
        st.markdown("---")
