import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import plotly.graph_objects as go
from fredapi import Fred
import warnings
from datetime import datetime
import numpy as np

# Désactivation des warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION & DESIGN SYSTEM
# ==============================================================================
st.set_page_config(page_title="OptiWealth Terminal Pro", page_icon="🦅", layout="wide")

# API KEYS
FRED_API_KEY = 'e16626c91fa2b1af27704a783939bf72'
NEWS_API_KEY = '4a00780a63064eae8faf0b7cbcfc5507'

# Palette Couleurs
C_BG = "#0e1117"
C_CARD = "#161b22"
C_BLUE = "#2962ff"
C_RED = "#ff4b4b"
C_GREEN = "#00ff88"
C_TEXT = "#e0e0e0"
C_BORDER = "#2d3342"

st.markdown(f"""
    <style>
        .stApp {{ background-color: {C_BG}; color: {C_TEXT}; }}
        h1, h2, h3, h4 {{ font-family: 'Roboto', sans-serif; color: white; margin: 0; }}
        
        /* Layout Compact */
        .block-container {{ padding-top: 1rem; padding-bottom: 2rem; }}
        
        /* TABLEAUX / MATRICE */
        .stDataFrame td {{
            text-align: center !important;
            vertical-align: middle !important;
            font-weight: bold;
            font-size: 14px;
            border-bottom: 1px solid #1e2130;
        }}
        .stDataFrame th {{
            text-align: center !important;
            background-color: {C_CARD};
        }}
        
        /* Onglets */
        .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
        .stTabs [data-baseweb="tab"] {{
            height: 40px; background-color: {C_CARD}; border: 1px solid #333;
            border-radius: 4px 4px 0 0; color: #888; padding: 4px 16px;
        }}
        .stTabs [aria-selected="true"] {{ background-color: {C_BLUE}; color: white; }}
        
        /* News Item */
        .news-item {{
            background-color: {C_CARD}; border-left: 3px solid {C_BLUE};
            padding: 10px; margin-bottom: 8px; border-radius: 4px;
            transition: transform 0.1s;
        }}
        .news-item:hover {{ transform: translateX(5px); }}
        .news-link {{ color: {C_BLUE}; text-decoration: none; font-size: 12px; font-weight: bold; }}
        
        /* Métriques Bannière */
        div[data-testid="metric-container"] {{
            background-color: {C_CARD};
            border: 1px solid {C_BORDER};
            padding: 10px;
            border-radius: 6px;
        }}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DATA ENGINE
# ==============================================================================

MACRO_ZONES_DISPLAY = {
    "USA 🇺🇸": {"Rate": "FEDFUNDS", "CPI": "CPIAUCSL", "Unemp": "UNRATE", "GDP": "A191RL1Q225SBEA", "M2": "M2SL"},
    "Eurozone 🇪🇺": {"Rate": "ECBDFR", "CPI": "CP0000EZ19M086NEST", "Unemp": "LRHUTTTTEZM156S", "GDP": "CLVMEURSCAB1GQEA19", "M2": "MABMM301EZM189S"},
    "Japan 🇯🇵": {"Rate": "IRSTCI01JPM156N", "CPI": "JPNCPIALLMINMEI", "Unemp": "LRHUTTTTJPM156S", "GDP": "JPNNGDP", "M2": "MABMM201JPM189S"},
    "UK 🇬🇧": {"Rate": "IUDSOIA", "CPI": "CPALTT01GBM659N", "Unemp": "LRHUTTTTGBM156S", "GDP": "UKNGDP", "M2": "MABMM401GBM189S"}
}

YIELD_CODES = {
    "USA 🇺🇸": {"10Y": "DGS10", "3M": "DGS3MO"},
    "Eurozone 🇪🇺": {"10Y": "IRLTLT01EZM156N", "3M": "IR3TIB01EZM156N"},
    "UK 🇬🇧": {"10Y": "IRLTLT01GBM156N", "3M": "IUDSOIA"}, 
    "Japan 🇯🇵": {"10Y": "IRLTLT01JPM156N", "3M": "IRSTCI01JPM156N"}
}

ECONOMIES_SCORING = {
    "USD": {"Rate": "FEDFUNDS", "CPI": "CPIAUCSL", "Unemp": "UNRATE", "GDP": "A191RL1Q225SBEA"},
    "EUR": {"Rate": "ECBDFR", "CPI": "CP0000EZ19M086NEST", "Unemp": "LRHUTTTTEZM156S", "GDP": "CLVMEURSCAB1GQEA19"},
    "JPY": {"Rate": "IRSTCI01JPM156N", "CPI": "JPNCPIALLMINMEI", "Unemp": "LRHUTTTTJPM156S", "GDP": "JPNRGDPEXP"},
    "GBP": {"Rate": "IUDSOIA", "CPI": "CPALTT01GBM659N", "Unemp": "LRHUTTTTGBM156S", "GDP": "UKNGDP"},
    "CAD": {"Rate": "IRSTCI01CAM156N", "CPI": "CPALTT01CAM659N", "Unemp": "LRHUTTTTCAM156S", "GDP": "NGDP_CA"},
    "AUD": {"Rate": "IRSTCI01AUM156N", "CPI": "CPALTT01AUM659N", "Unemp": "LRHUTTTTAUM156S", "GDP": "NGDP_AU"},
    "CHF": {"Rate": "IRSTCI01CHM156N", "CPI": "CPALTT01CHM659N", "Unemp": "LRHUTTTTCHM156S", "GDP": "NGDP_CH"},
    "NZD": {"Rate": "IRSTCI01NZM156N", "CPI": "CPALTT01NZM659N", "Unemp": "LRHUTTTTNZM156S", "GDP": "NGDP_NZ"},
}

ASSET_MAP = {
    "EURUSD=X": {"name": "EUR/USD", "base": "EUR", "quote": "USD", "cftc": "EURO FX"},
    "GBPUSD=X": {"name": "GBP/USD", "base": "GBP", "quote": "USD", "cftc": "BRITISH POUND STERLING"},
    "USDJPY=X": {"name": "USD/JPY", "base": "USD", "quote": "JPY", "cftc": "JAPANESE YEN", "inv_cftc": True},
    "USDCAD=X": {"name": "USD/CAD", "base": "USD", "quote": "CAD", "cftc": "CANADIAN DOLLAR", "inv_cftc": True},
    "AUDUSD=X": {"name": "AUD/USD", "base": "AUD", "quote": "USD", "cftc": "AUSTRALIAN DOLLAR"},
    "NZDUSD=X": {"name": "NZD/USD", "base": "NZD", "quote": "USD", "cftc": "NEW ZEALAND DOLLAR"},
    "USDCHF=X": {"name": "USD/CHF", "base": "USD", "quote": "CHF", "cftc": "SWISS FRANC", "inv_cftc": True},
    "AUDJPY=X": {"name": "AUD/JPY", "base": "AUD", "quote": "JPY", "cftc": None},
    "EURJPY=X": {"name": "EUR/JPY", "base": "EUR", "quote": "JPY", "cftc": None},
    "^GSPC":    {"name": "S&P 500", "base": "USD", "quote": "USD", "cftc": "E-MINI S&P 500"},
    "GC=F":     {"name": "GOLD",    "base": "USD", "quote": "USD", "cftc": "GOLD"},
}

@st.cache_data(ttl=3600)
def get_banner_data(zone_key):
    try:
        fred = Fred(api_key=FRED_API_KEY)
        codes = MACRO_ZONES_DISPLAY[zone_key]
        def safe_get(code, change_type="last"):
            try:
                s = fred.get_series(code)
                if change_type == "last": return s.iloc[-1]
                elif change_type == "yoy": return s.pct_change(12).iloc[-1] * 100
            except: return 0.0
        return {
            "Rate": safe_get(codes["Rate"]), "CPI": safe_get(codes["CPI"], "yoy"),
            "Unemp": safe_get(codes["Unemp"]), "GDP": safe_get(codes["GDP"], "yoy"), "M2": safe_get(codes["M2"], "yoy")
        }
    except: return None

@st.cache_data(ttl=3600)
def get_macro_db():
    fred = Fred(api_key=FRED_API_KEY)
    db = {}
    for curr, codes in ECONOMIES_SCORING.items():
        d = {}
        for k, code in codes.items():
            try:
                s = fred.get_series(code)
                val = s.iloc[-1]
                if k == 'GDP' or k == 'CPI':
                    if val > 50: d[k] = s.pct_change(4 if k=='GDP' else 12).iloc[-1] * 100
                    else: d[k] = val
                else: d[k] = val
            except: d[k] = 0.0
        db[curr] = d
    return db


@st.cache_data(ttl=3600)
def get_yield_curves():
    fred = Fred(api_key=FRED_API_KEY)
    history = {}
    for zone, codes in YIELD_CODES.items():
        try:
            long = fred.get_series(codes['10Y']).iloc[-252:]
            short = fred.get_series(codes['3M']).iloc[-252:]

            # FORCE ALIGNMENT ON COMMON DATES (INNER JOIN)
            df = pd.concat([long.rename("10Y"), short.rename("3M")], axis=1, join="inner")

            # Clean & sort
            df = df.dropna().sort_index()

            if not df.empty:
                history[zone] = df
        except:
            pass
    return history

# New function to fetch DXY from FRED
@st.cache_data(ttl=3600)
def get_dxy_fred():
    try:
        fred = Fred(api_key=FRED_API_KEY)
        dxy = fred.get_series("DTWEXM")  # Traditional DXY (Major Currencies Index)
        return dxy
    except:
        return None

@st.cache_data(ttl=3600)
def get_cot_data():
    url = "https://www.cftc.gov/dea/newcot/FinFutWk.txt"
    headers = {"User-Agent": "Mozilla/5.0"}
    data = {}
    try:
        r = requests.get(url, headers=headers, verify=False, timeout=10)
        if r.status_code == 200:
            for line in r.text.split('\n'):
                row = line.split(',')
                if len(row) > 10:
                    name = row[0].strip().replace('"', '')
                    try:
                        longs, shorts = float(row[7].strip()), float(row[8].strip())
                        data[name] = {"Net": longs - shorts, "Total": longs+shorts, "Longs": longs, "Shorts": shorts}
                    except: continue
    except: pass
    return data

@st.cache_data(ttl=600)
def get_market_prices():
    tickers = list(ASSET_MAP.keys())
    try: return yf.download(tickers, period="1y", interval="1d", group_by='ticker', progress=False)
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_seasonality_data(tickers):
    """CORRECTIF SAISONNALITÉ : Nettoyage et Ré-indexation 12 mois"""
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, period="10y", interval="1d", progress=False)

            # Nettoyage MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                try: df = df.xs(t, axis=1, level=0)
                except: df.columns = df.columns.droplevel(1)

            if not df.empty and 'Close' in df.columns:
                monthly_close = df['Close'].resample("M").last()
                monthly_ret = monthly_close.pct_change() * 100
                monthly_ret = monthly_ret.dropna()
                monthly_ret.index = monthly_ret.index.to_period("M")
                monthly_ret = monthly_ret.groupby(monthly_ret.index.month).mean()
                monthly_ret = monthly_ret.reindex(range(1, 13), fill_value=0)
                data[t] = monthly_ret
        except: continue
    return data

@st.cache_data(ttl=1800)
def get_news():
    url = "https://newsapi.org/v2/top-headlines"
    params = {'apiKey': NEWS_API_KEY, 'category': 'business', 'language': 'en', 'pageSize': 6}
    try: return requests.get(url, params=params).json().get('articles', [])
    except: return []

# ==============================================================================
# 3. LOGIC
# ==============================================================================

def calculate_matrix_row(ticker, info, macro, cot, market):
    # MACRO
    b, q = macro.get(info['base'], {}), macro.get(info['quote'], {})
    
    def calc_diff(key, threshold_bull, threshold_bear, inverse=False):
        d = b.get(key, 0) - q.get(key, 0)
        if inverse: return 1 if d < threshold_bull else (-1 if d > threshold_bear else 0)
        return 1 if d > threshold_bull else (-1 if d < threshold_bear else 0)

    s_rate = calc_diff('Rate', 0.25, -0.25)
    s_gdp = calc_diff('GDP', 0.2, -0.2)
    s_unemp = calc_diff('Unemp', -0.2, 0.2, inverse=True)
    s_cpi = calc_diff('CPI', 0.5, -0.5)

    # TECH
    trend, rsi, price = 0, 50, 0
    try:
        df = market[ticker]
        if isinstance(df.columns, pd.MultiIndex): df = df.xs(ticker, axis=1, level=0)
        if not df.empty:
            c = df['Close']
            price = c.iloc[-1]
            delta = c.diff()
            rs = (delta.where(delta>0,0)).rolling(14).mean() / (-delta.where(delta<0,0)).rolling(14).mean()
            rsi = 100 - (100/(1+rs)).iloc[-1]
            trend = 1 if price > c.rolling(200).mean().iloc[-1] else -1
    except: pass

    # COT
    cot_s, net_pos, pct_long = 0, 0, 50
    if info.get('cftc'):
        for k, v in cot.items():
            if info['cftc'] in k:
                net_pos = v['Net']
                longs, total = v['Longs'], v['Total']
                pct_long = (longs / total * 100) if total > 0 else 50
                if info.get('inv_cftc'): 
                    net_pos = -net_pos
                    pct_long = 100 - pct_long
                if net_pos > 5000: cot_s = 1
                elif net_pos < -5000: cot_s = -1
                break

    total = s_rate + s_gdp + s_unemp + s_cpi + trend + cot_s
    bias = "STRONG BUY" if total>=3 else ("BUY" if total>=1 else ("STRONG SELL" if total<=-3 else ("SELL" if total<=-1 else "NEUTRAL")))

    return {
        "Asset": info['name'], "Bias": bias, "Total": total, "Rate": s_rate, "GDP": s_gdp,
        "Unemp": s_unemp, "CPI": s_cpi, "COT": cot_s, "Trend": trend, 
        "Net COT": net_pos, "Sentiment": pct_long, "Price": price, "RSI": rsi
    }

# ==============================================================================
# 4. CHARTING
# ==============================================================================

def plot_yield_comparison(zone_data, zone_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=zone_data.index, y=zone_data['10Y'], name="10Y (Long)", line=dict(color=C_BLUE)))
    fig.add_trace(go.Scatter(x=zone_data.index, y=zone_data['3M'], name="3M (Short)", line=dict(color=C_RED)))
    fig.update_layout(
        title=f"{zone_name} Yield Curve", template="plotly_dark", height=250, 
        paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=20,r=20,t=30,b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_dxy_base100(market):
    fig = go.Figure()
    global dxy_fred_global
    start_date = "2020-01-01"
    dxy = dxy_fred_global
    if dxy is None:
        st.warning("DXY data not available from FRED.")
    else:
        dxy = dxy.dropna()
        dxy = dxy[dxy.index >= start_date]
        if not dxy.empty:
            dxy_base = (dxy / dxy.iloc[0]) * 100
            fig.add_trace(go.Scatter(
                x=dxy_base.index,
                y=dxy_base.values,
                name="USD Index (FRED)",
                line=dict(color="white", width=3)
            ))

    colors = {"EURUSD=X": C_BLUE, "GBPUSD=X": C_GREEN, "AUDUSD=X": "orange"}
    names = {"EURUSD=X": "EUR", "GBPUSD=X": "GBP", "AUDUSD=X": "AUD"}
    for t, col in colors.items():
        try:
            s = market[t]["Close"].dropna()
            s = s[s.index >= start_date]
            if not s.empty:
                s_base = (s / s.iloc[0]) * 100
                fig.add_trace(go.Scatter(x=s_base.index, y=s_base.values, name=names[t], line=dict(color=col)))
        except: pass
    try:
        jpy = (1 / market["USDJPY=X"]["Close"]).dropna()
        jpy = jpy[jpy.index >= start_date]
        if not jpy.empty:
            jpy_base = (jpy / jpy.iloc[0]) * 100
            fig.add_trace(go.Scatter(x=jpy_base.index, y=jpy_base.values, name="JPY", line=dict(color=C_RED)))
    except: pass
    fig.update_layout(
        title="Force Relative Devises (Base 100)",
        template="plotly_dark",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=True,
            gridcolor="#333",
            tickformat="%Y-%m-%d",
            tickangle=-45,
            dtick="M1"
        )
    )
    return fig

def plot_seasonality(ticker1, ticker2, seas_db):
    fig = go.Figure()
    months = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]
    
    if ticker1 in seas_db:
        s1 = seas_db[ticker1]
        fig.add_trace(go.Scatter(
            x=months,
            y=s1.values,
            name=ASSET_MAP.get(ticker1, {'name': ticker1})['name'],
            mode="lines+markers",
            line=dict(color=C_BLUE, width=3)
        ))
        
    # Chart Ticker 2 (Ligne)
    if ticker2 and ticker2 in seas_db:
        s2 = seas_db[ticker2]
        fig.add_trace(go.Scatter(x=months, y=s2.values, name=ASSET_MAP.get(ticker2, {'name':ticker2})['name'], line=dict(color=C_GREEN, width=3)))

    # Marqueur mois actuel
    current_month_idx = datetime.now().month - 1
    fig.add_shape(type="line", x0=current_month_idx, x1=current_month_idx, y0=-5, y1=5, line=dict(color="white", width=1, dash="dot"))
    
    # Dynamic Y-axis scaling
    y_values = []
    if ticker1 in seas_db:
        y_values.extend(seas_db[ticker1].values.tolist())
    if ticker2 and ticker2 in seas_db:
        y_values.extend(seas_db[ticker2].values.tolist())

    if y_values:
        y_min = min(y_values)
        y_max = max(y_values)
        fig.update_yaxes(range=[y_min * 1.1, y_max * 1.1])

    fig.update_layout(title="Saisonnalité Moyenne (10 Ans)", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=450)
    return fig

# ==============================================================================
# 5. MAIN
# ==============================================================================

def main():
    c1, c2 = st.columns([3, 1])
    with c1: st.title("🦅 OPTIWEALTH TERMINAL v13")
    with c2: zone = st.selectbox("Zone Économique", list(MACRO_ZONES_DISPLAY.keys()))
    
    banner = get_banner_data(zone)
    if banner:
        cols = st.columns(5)
        cols[0].metric("Taux Directeur", f"{banner['Rate']:.2f}%")
        cols[1].metric("Inflation (CPI)", f"{banner['CPI']:.1f}%")
        cols[2].metric("Chômage", f"{banner['Unemp']:.1f}%")
        cols[3].metric("GDP (YoY)", f"{banner['GDP']:.1f}%")
        cols[4].metric("M2 Supply", f"{banner['M2']:.1f}%")

    st.markdown("---")

    with st.spinner("Synchronisation..."):
        macro_db = get_macro_db()
        cot_db = get_cot_data()
        market_db = get_market_prices()
        dxy_fred = get_dxy_fred()
        yields_db = get_yield_curves()
        seas_db = get_seasonality_data(list(ASSET_MAP.keys()))

    if market_db is None or market_db.empty:
        st.error("Erreur flux de données.")
        return

    rows = []
    for t, i in ASSET_MAP.items():
        rows.append(calculate_matrix_row(t, i, macro_db, cot_db, market_db))
    df = pd.DataFrame(rows)

    tab_matrix, tab_money, tab_seas, tab_scan = st.tabs(["🌍 MATRICE MACRO", "💸 MONEY FLOW", "📅 SAISONNALITÉ", "🚀 SCANNER"])

    # TAB 1: MATRICE (RESTORE PROGRESS BAR)
    with tab_matrix:
        st.subheader("Matrice de Différentiel")
        
        def color_map(val):
            if isinstance(val, (int, float)):
                if val > 0: return f'color: {C_BLUE}; font-weight:bold;'
                elif val < 0: return f'color: {C_RED}; font-weight:bold;'
                else: return f'color: #555;'
            return ''

        st.dataframe(
            df[['Asset', 'Bias', 'Total', 'Rate', 'GDP', 'Unemp', 'CPI', 'COT', 'Trend']]
            .style.applymap(color_map, subset=['Rate', 'GDP', 'Unemp', 'CPI', 'COT', 'Trend']),
            use_container_width=True, height=500,
            column_config={
                "Total": st.column_config.ProgressColumn(
                    "Score Global", 
                    help="Score de -6 à +6", 
                    format="%d", 
                    min_value=-6, 
                    max_value=6
                )
            }
        )
        
        st.markdown("---")
        st.subheader("📰 Flash News")
        news = get_news()
        if news:
            cols = st.columns(3)
            for i, n in enumerate(news[:6]):
                with cols[i%3]:
                    st.markdown(f"""<div class="news-item"><div style="font-weight:bold;">{n['title'][:60]}...</div><div style="font-size:11px; color:#888;">{n['source']['name']}</div><a href="{n['url']}" class="news-link" target="_blank">Lire</a></div>""", unsafe_allow_html=True)

    # TAB 2: MONEY FLOW (INDENTATION CORRIGÉE)
    with tab_money:
        global dxy_fred_global
        dxy_fred_global = dxy_fred

        st.subheader("1. Force Relative")
        st.plotly_chart(plot_dxy_base100(market_db), use_container_width=True)
        st.subheader("2. Yield Curve Control (10Y vs 3M)")
        
        yc1, yc2 = st.columns(2)
        yc3, yc4 = st.columns(2)
        
        if "USA 🇺🇸" in yields_db: 
            with yc1: st.plotly_chart(plot_yield_comparison(yields_db["USA 🇺🇸"], "USA"), use_container_width=True)
        if "Eurozone 🇪🇺" in yields_db: 
            with yc2: st.plotly_chart(plot_yield_comparison(yields_db["Eurozone 🇪🇺"], "Europe"), use_container_width=True)
        if "UK 🇬🇧" in yields_db: 
            with yc3: st.plotly_chart(plot_yield_comparison(yields_db["UK 🇬🇧"], "UK"), use_container_width=True)
        if "Japan 🇯🇵" in yields_db: 
            with yc4: st.plotly_chart(plot_yield_comparison(yields_db["Japan 🇯🇵"], "Japan"), use_container_width=True)

    # TAB 3: SEASONALITY (BUG FIXÉ)
    with tab_seas:
        st.subheader("📅 Analyse Saisonnière (10 Ans)")
        c_sel1, c_sel2 = st.columns(2)
        with c_sel1: asset1 = st.selectbox("Actif A", list(ASSET_MAP.keys()))
        with c_sel2: asset2 = st.selectbox("Actif B (Comparaison)", ["Aucun"] + list(ASSET_MAP.keys()))
        comp = None if asset2 == "Aucun" else asset2
        st.plotly_chart(plot_seasonality(asset1, comp, seas_db), use_container_width=True)

        # Infos importantes sous le graphique
        def compute_stats(ticker):
            s = seas_db.get(ticker)
            if s is None:
                return None
            mean_val = s.mean()
            std_val = s.std()
            best_month = s.idxmax()
            worst_month = s.idxmin()
            return mean_val, std_val, best_month, worst_month

        stats1 = compute_stats(asset1)
        stats2 = compute_stats(comp) if comp else None

        months_map = {
            1: "Jan", 2: "Fév", 3: "Mar", 4: "Avr", 5: "Mai", 6: "Juin",
            7: "Juil", 8: "Août", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Déc"
        }

        if stats1:
            mean1, std1, best1, worst1 = stats1
            st.markdown("## 🔎 Informations Clés (10 Ans)")
            colA, colB = st.columns(2)

            with colA:
                st.markdown(f"""
<div style="background-color:#161b22; padding:15px; border-radius:8px; border-left:4px solid {C_BLUE};">
<h3 style="margin:0; color:white;"> {ASSET_MAP.get(asset1, {'name': asset1})['name']}</h3>
<p style="font-size:16px; margin-top:10px;">
<b>Rendement moyen :</b> <span style="font-size:20px; font-weight:700; color:{C_GREEN};">{mean1:.2f}%</span><br>
<b>Écart-type :</b> <span style="font-size:20px; font-weight:700;">{std1:.2f}</span><br>
<b>Meilleur mois :</b> <span style="font-weight:700; color:{C_GREEN};">{months_map.get(best1)}</span> ({seas_db[asset1][best1]:.2f}%)<br>
<b>Pire mois :</b> <span style="font-weight:700; color:{C_RED};">{months_map.get(worst1)}</span> ({seas_db[asset1][worst1]:.2f}%)
</p>
</div>
""", unsafe_allow_html=True)

            if stats2:
                mean2, std2, best2, worst2 = stats2
                with colB:
                    st.markdown(f"""
<div style="background-color:#161b22; padding:15px; border-radius:8px; border-left:4px solid {C_BLUE};">
<h3 style="margin:0; color:white;"> {ASSET_MAP.get(comp, {'name': comp})['name']}</h3>
<p style="font-size:16px; margin-top:10px;">
<b>Rendement moyen :</b> <span style="font-size:20px; font-weight:700; color:{C_GREEN};">{mean2:.2f}%</span><br>
<b>Écart-type :</b> <span style="font-size:20px; font-weight:700;">{std2:.2f}</span><br>
<b>Meilleur mois :</b> <span style="font-weight:700; color:{C_GREEN};">{months_map.get(best2)}</span> ({seas_db[comp][best2]:.2f}%)<br>
<b>Pire mois :</b> <span style="font-weight:700; color:{C_RED};">{months_map.get(worst2)}</span> ({seas_db[comp][worst2]:.2f}%)
</p>
</div>
""", unsafe_allow_html=True)

    # TAB 4: SCANNER
    with tab_scan:
        st.dataframe(df[['Asset', 'Bias', 'Total', 'Price', 'RSI', 'Net COT']], use_container_width=True)

if __name__ == "__main__":
    main()