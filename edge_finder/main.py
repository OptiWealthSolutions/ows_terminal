import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import plotly.graph_objects as go
from fredapi import Fred
import warnings
from datetime import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
    "EURCHF=X": {"name": "EUR/CHF", "base": "EUR", "quote": "CHF", "cftc": None},
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
            df = pd.DataFrame({"10Y": long, "3M": short}).dropna()
            if not df.empty: history[zone] = df
        except: pass
    return history

@st.cache_data(ttl=3600)
def get_cot_data():
    url = "https://www.cftc.gov/dea/newcot/FinFutWk.txt"
    headers = {"User-Agent": "Mozilla/5.0"}
    data = {}

    try:
        r = requests.get(url, headers=headers, verify=False, timeout=10)
        if r.status_code == 200:
            import io
            df = pd.read_csv(io.StringIO(r.text), dtype=str)

            # Normalisation des noms de colonnes (certaines contiennent des espaces)
            df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

            # Colonnes CFTC généralement présentes
            name_col = "market_and_exchange_names"
            long_col = "noncommercial_long"
            short_col = "noncommercial_short"

            if name_col in df.columns and long_col in df.columns and short_col in df.columns:
                for _, row in df.iterrows():
                    try:
                        name = str(row[name_col]).strip()
                        longs = float(row[long_col])
                        shorts = float(row[short_col])
                        data[name] = {
                            "Net": longs - shorts,
                            "Total": longs + shorts,
                            "Longs": longs,
                            "Shorts": shorts
                        }
                    except:
                        continue
    except:
        pass

    return data

@st.cache_data(ttl=600)
def get_market_prices():
    # AJOUT DE ^NYICDX POUR LE DXY
    tickers = list(ASSET_MAP.keys()) + ["^NYICDX", "DX=F", "^TNX", "^IRX"]
    try: return yf.download(tickers, period="1y", interval="1d", group_by='ticker', progress=False)
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_seasonality_data(tickers):
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, period="10y", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                try: df = df.xs(t, axis=1, level=0) 
                except: df.columns = df.columns.droplevel(1)
            
            if not df.empty and 'Close' in df.columns:
                monthly_close = df['Close'].resample("M").last()
                monthly_ret = monthly_close.pct_change() * 100
                monthly_ret = monthly_ret.dropna()
                avg_ret = monthly_ret.groupby(monthly_ret.index.month).mean()
                avg_ret = avg_ret.reindex(range(1, 13), fill_value=0)
                data[t] = avg_ret
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
    trend = 0
    rsi = 50
    price = 0
    try:
        df = market[ticker]
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(ticker, axis=1, level=0)

        if not df.empty:
            c = df["Close"]
            price = c.iloc[-1]

            # SMA calculations
            sma20 = c.rolling(20).mean()
            sma50 = c.rolling(50).mean()
            sma200 = c.rolling(200).mean()

            # RSI
            delta = c.diff()
            rs = (delta.where(delta > 0, 0)).rolling(14).mean() / (-delta.where(delta < 0, 0)).rolling(14).mean()
            rsi = 100 - (100 / (1 + rs)).iloc[-1]

            # Trend logic
            trend_sma = 1 if price > sma200.iloc[-1] else -1

            # Slope of SMA20
            slope20 = sma20.diff().iloc[-1]
            trend_slope = 1 if slope20 > 0 else -1

            # ADX
            high = df["High"]
            low = df["Low"]
            close = df["Close"]

            plus_dm = (high.diff().where(high.diff() > low.diff(), 0)).fillna(0)
            minus_dm = (low.diff().where(low.diff() > high.diff(), 0)).fillna(0)
            tr = (high - low).combine((high - close.shift()).abs(), max).combine((low - close.shift()).abs(), max)

            atr = tr.rolling(14).mean()
            plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
            dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
            adx = dx.rolling(14).mean().iloc[-1]

            trend_adx = 1 if adx > 25 else 0

            # Final trend score
            trend = trend_sma + trend_slope + trend_adx
    except:
        pass

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
    
    # --- ALIGNEMENT TEMPOREL (Fix Bgs) ---
    # On prend l'index commun le plus récent (ex: 1 an glissant)
    # DXY PRIORITY: ^NYICDX
    dxy = None
    if "^NYICDX" in market and not market["^NYICDX"].empty:
        dxy = market["^NYICDX"]["Close"]
    elif "DX-Y.NYB" in market and not market["DX-Y.NYB"].empty:
        dxy = market["DX-Y.NYB"]["Close"]
    elif "DX=F" in market and not market["DX=F"].empty:
        dxy = market["DX=F"]["Close"]
    
    if dxy is not None:
        dxy = dxy.fillna(method='ffill').dropna()
        # BASE 100
        if not dxy.empty:
            base_val = dxy.iloc[0]
            if base_val > 0:
                dxy_b100 = (dxy / base_val) * 100
                fig.add_trace(go.Scatter(x=dxy.index, y=dxy_b100, name="USD Index (^NYICDX)", line=dict(color="white", width=3)))

    # DEVISES
    colors = {"EURUSD=X": C_BLUE, "GBPUSD=X": C_GREEN, "AUDUSD=X": "orange"}
    names = {"EURUSD=X": "EUR", "GBPUSD=X": "GBP", "AUDUSD=X": "AUD"}
    
    for t, col in colors.items():
        try:
            s = market[t]["Close"].dropna()
            # Alignement temporel simple: on prend tout l'historique dispo dans market (1y)
            if not s.empty:
                s_b100 = (s / s.iloc[0]) * 100
                fig.add_trace(go.Scatter(x=s.index, y=s_b100, name=names[t], line=dict(color=col)))
        except: pass
        
    try: 
        jpy_raw = market["USDJPY=X"]["Close"].dropna()
        if not jpy_raw.empty:
            jpy = 1 / jpy_raw # Inversion
            jpy_b100 = (jpy / jpy.iloc[0]) * 100
            fig.add_trace(go.Scatter(x=jpy.index, y=jpy_b100, name="JPY", line=dict(color=C_RED)))
    except: pass
    
    fig.update_layout(title="Force Relative Devises (Base 100)", template="plotly_dark", height=400, paper_bgcolor='rgba(0,0,0,0)')
    return fig

def plot_seasonality(ticker1, ticker2, seas_db):
    fig = go.Figure()
    months = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]
    
    if ticker1 in seas_db:
        s1 = seas_db[ticker1]
        fig.add_trace(go.Scatter(x=months, y=s1.values, name=ASSET_MAP.get(ticker1, {'name':ticker1})['name'], mode='lines+markers', line=dict(color=C_BLUE, width=3)))
        
    if ticker2 and ticker2 in seas_db:
        s2 = seas_db[ticker2]
        fig.add_trace(go.Scatter(x=months, y=s2.values, name=ASSET_MAP.get(ticker2, {'name':ticker2})['name'], line=dict(color=C_GREEN, width=3, dash='dot')))

    curr_idx = datetime.now().month - 1
    fig.add_shape(type="line", x0=curr_idx, x1=curr_idx, y0=-5, y1=5, line=dict(color="white", width=1, dash="dot"))
    fig.add_annotation(x=curr_idx, y=5, text="Actuel", showarrow=False, font=dict(color="white"))
    
    fig.update_layout(title="Saisonnalité Moyenne (10 Ans)", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=450)
    return fig

# ==============================================================================
# 5. MAIN
# ==============================================================================

def main():
    c1, c2 = st.columns([3, 1])
    with c1: st.title("🦅 OPTIWEALTH TERMINAL v16")
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

    st.markdown(
        """
        <style>
            .poly-container {
                display: flex;
                flex-wrap: nowrap;
                justify-content: space-between;
                gap: 10px;
            }
            .poly-item {
                flex: 1;
                max-width: 24%;
                background-color: #0e1117;
                border-radius: 6px;
                padding: 0;
            }
            .poly-item iframe {
                width: 100%;
                height: 180px;
                border: none;
                background-color: #0e1117;
            }
        </style>

        <div class="poly-container">
            <div class="poly-item">
                <iframe
                    title="polymarket-market-iframe"
                    src="https://embed.polymarket.com/market.html?market=fed-decreases-interest-rates-by-25-bps-after-december-2025-meeting&features=volume&theme=light"
                ></iframe>
            </div>
            <div class="poly-item">
                <iframe
                    title="polymarket-market-iframe"
                    src="https://embed.polymarket.com/market.html?market=will-the-ecb-announce-no-change-at-the-december-meeting&features=volume&theme=light"
                ></iframe>
            </div>
            <div class="poly-item">
                <iframe
                    title="polymarket-market-iframe"
                    src="https://embed.polymarket.com/market.html?market=bank-of-japan-increases-interest-rates-by-25-bps-after-december-2025-meeting&features=volume&theme=light"
                ></iframe>
            </div>
            <div class="poly-item">
                <iframe
                    title="polymarket-market-iframe"
                    src="https://embed.polymarket.com/market.html?market=will-the-bank-of-canada-announce-no-change-at-the-december-meeting&features=volume&theme=light"
                ></iframe>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.spinner("Synchronisation..."):
        macro_db = get_macro_db()
        cot_db = get_cot_data()
        market_db = get_market_prices()
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

    # TAB 1: MATRICE
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

    # TAB 2: MONEY FLOW
    with tab_money:
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

        st.markdown("---")
        st.subheader("3. Corrélation des Devises (1 an)")

        try:
            # Extraction des clôtures pour toutes les paires Forex disponibles
            fx_pairs = [t for t in ASSET_MAP.keys() if "=X" in t]
            closes = {}

            for t in fx_pairs:
                try:
                    df_t = market_db[t]
                    if isinstance(df_t.columns, pd.MultiIndex):
                        df_t = df_t.xs(t, axis=1, level=0)
                    closes[t] = df_t["Close"]
                except:
                    continue
            df_close = pd.DataFrame(closes).dropna()
            corr = df_close.corr()


            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Erreur corrélation : {e}")

    # TAB 3: SEASONALITY
    with tab_seas:
        st.subheader("📅 Analyse Saisonnière (10 Ans)")
        c_sel1, c_sel2 = st.columns(2)
        with c_sel1: asset1 = st.selectbox("Actif A", list(ASSET_MAP.keys()))
        with c_sel2: asset2 = st.selectbox("Actif B (Comparaison)", ["Aucun"] + list(ASSET_MAP.keys()))
        comp = None if asset2 == "Aucun" else asset2
        st.plotly_chart(plot_seasonality(asset1, comp, seas_db), use_container_width=True)

        # STATS
        def get_stats(t):
            if t not in seas_db: return None
            s = seas_db[t]
            return s.mean(), s.std(), s.idxmax(), s.idxmin()

        s1 = get_stats(asset1)
        s2 = get_stats(comp) if comp else None

        colA, colB = st.columns(2)

        if s1:
            mean, std, best, worst = s1
            months_map = {1:"Jan", 2:"Fév", 3:"Mar", 4:"Avr", 5:"Mai", 6:"Juin", 7:"Juil", 8:"Août", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Déc"}
            with colA:
                st.markdown(f"""
                <div style="background-color:#161b22; padding:15px; border-radius:5px; border-left:4px solid {C_BLUE};">
                    <h4 style="margin:0;">📊 {ASSET_MAP[asset1]['name']}</h4>
                    <p><b>Moyenne :</b> {mean:.2f}%<br>
                    <b>Volatilité :</b> {std:.2f}<br>
                    <b>Top Mois :</b> <span style="color:{C_GREEN}">{months_map[best]}</span><br>
                    <b>Flop Mois :</b> <span style="color:{C_RED}">{months_map[worst]}</span></p>
                </div>""", unsafe_allow_html=True)

        if s2:
            mean2, std2, best2, worst2 = s2
            months_map = {1:"Jan", 2:"Fév", 3:"Mar", 4:"Avr", 5:"Mai", 6:"Juin", 7:"Juil", 8:"Août", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Déc"}
            with colB:
                st.markdown(f"""
                <div style="background-color:#161b22; padding:15px; border-radius:5px; border-left:4px solid {C_GREEN};">
                    <h4 style="margin:0;">📊 {ASSET_MAP[comp]['name']}</h4>
                    <p><b>Moyenne :</b> {mean2:.2f}%<br>
                    <b>Volatilité :</b> {std2:.2f}<br>
                    <b>Top Mois :</b> <span style="color:{C_GREEN}">{months_map[best2]}</span><br>
                    <b>Flop Mois :</b> <span style="color:{C_RED}">{months_map[worst2]}</span></p>
                </div>""", unsafe_allow_html=True)

    # TAB 4: SCANNER
    with tab_scan:
        st.dataframe(df[['Asset', 'Bias', 'Total', 'Price', 'RSI', 'Net COT']], use_container_width=True)

if __name__ == "__main__":
    main()