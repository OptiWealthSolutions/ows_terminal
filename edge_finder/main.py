import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import plotly.graph_objects as go
from fredapi import Fred
import warnings
from datetime import datetime

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
        
        /* TABLEAUX / MATRICE - Design Amélioré */
        .stDataFrame td {{
            text-align: center !important;
            vertical-align: middle !important;
            font-weight: bold; /* Lisibilité accrue */
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
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DATA ENGINE
# ==============================================================================

# Codes FRED Taux (10Y vs 3M) - CORRIGÉ POUR EUROZONE
YIELD_CODES = {
    "USA 🇺🇸": {"10Y": "DGS10", "3M": "DGS3MO"},
    "Eurozone 🇪🇺": {"10Y": "IRLTLT01EZM156N", "3M": "IR3TIB01EZM156N"}, # Code 3M corrigé (Interbank Rate)
    "UK 🇬🇧": {"10Y": "IRLTLT01GBM156N", "3M": "IUDSOIA"}, 
    "Japan 🇯🇵": {"10Y": "IRLTLT01JPM156N", "3M": "IRSTCI01JPM156N"}
}

# Mapping Scoring
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
    """Récupère l'historique des taux pour les graphiques"""
    fred = Fred(api_key=FRED_API_KEY)
    history = {}
    for zone, codes in YIELD_CODES.items():
        try:
            long = fred.get_series(codes['10Y']).iloc[-252:]
            short = fred.get_series(codes['3M']).iloc[-252:]
            # Alignement des dates pour éviter les trous
            df = pd.DataFrame({"10Y": long, "3M": short}).dropna()
            if not df.empty:
                history[zone] = df
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
    tickers = list(ASSET_MAP.keys()) + ["DX-Y.NYB", "DX=F"]
    try:
        return yf.download(tickers, period="1y", interval="1d", group_by='ticker', progress=False)
    except: return pd.DataFrame()

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
    
    # Diff Calculations
    def calc_diff(key, threshold_bull, threshold_bear, inverse=False):
        d = b.get(key, 0) - q.get(key, 0)
        if inverse: 
            return 1 if d < threshold_bull else (-1 if d > threshold_bear else 0)
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
            # RSI
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
    # DXY Selection
    dxy = market["DX=F"]["Close"] if "DX=F" in market and not market["DX=F"].empty else (market["DX-Y.NYB"]["Close"] if "DX-Y.NYB" in market else None)
    
    if dxy is not None:
        dxy = dxy.fillna(method='ffill')
        fig.add_trace(go.Scatter(x=dxy.index, y=(dxy/dxy.iloc[0])*100, name="USD Index", line=dict(color="white", width=3)))

    colors = {"EURUSD=X": C_BLUE, "GBPUSD=X": C_GREEN, "AUDUSD=X": "orange"}
    names = {"EURUSD=X": "EUR", "GBPUSD=X": "GBP", "AUDUSD=X": "AUD"}
    for t, col in colors.items():
        try:
            s = market[t]["Close"]
            fig.add_trace(go.Scatter(x=s.index, y=(s/s.iloc[0])*100, name=names[t], line=dict(color=col)))
        except: pass
        
    try: # JPY Inverted
        jpy = 1 / market["USDJPY=X"]["Close"]
        fig.add_trace(go.Scatter(x=jpy.index, y=(jpy/jpy.iloc[0])*100, name="JPY", line=dict(color=C_RED)))
    except: pass
    fig.update_layout(title="Force Relative Devises (Base 100)", template="plotly_dark", height=400, paper_bgcolor='rgba(0,0,0,0)')
    return fig

# ==============================================================================
# 5. MAIN
# ==============================================================================

def main():
    c1, c2 = st.columns([3, 1])
    with c1: st.title("🦅 OPTIWEALTH TERMINAL v9")
    with c2: st.caption("Live Feed: CFTC • FRED • Yahoo • NewsAPI")

    with st.spinner("Synchronisation des données financières..."):
        macro_db = get_macro_db()
        cot_db = get_cot_data()
        market_db = get_market_prices()
        yields_db = get_yield_curves()

    if market_db is None or market_db.empty:
        st.error("Erreur critique: Flux de données inaccessible.")
        return

    # Calcul Matrice
    rows = []
    for t, i in ASSET_MAP.items():
        if "USD" in t or "JPY" in t or "EUR" in t or "GBP" in t:
             rows.append(calculate_matrix_row(t, i, macro_db, cot_db, market_db))
    df = pd.DataFrame(rows)

    # --- TABS ---
    tab_matrix, tab_money, tab_scan = st.tabs(["🌍 MATRICE MACRO & NEWS", "💸 MONEY FLOW (YIELDS)", "🚀 SCANNER TECHNIQUE"])

    # TAB 1: MATRICE
    with tab_matrix:
        st.subheader("Matrice de Différentiel Fondamental")
        
        def color_map(val):
            if isinstance(val, (int, float)):
                if val > 0:
                    return f'background-color: {C_BLUE}; color: white; border: 1px solid #1c212c'
                elif val < 0:
                    return f'color: white; border: 1px solid #1c212c'
                else:
                    return f'color: #555; border: 1px solid #1c212c'
            return ''

        st.dataframe(
            df[['Asset', 'Bias', 'Total', 'Rate', 'GDP', 'Unemp', 'CPI', 'COT', 'Trend']]
            .style.applymap(color_map, subset=['Total', 'Rate', 'GDP', 'Unemp', 'CPI', 'COT', 'Trend']),
            use_container_width=True, height=500,
            column_config={
                "Total": st.column_config.ProgressColumn("Score", min_value=-6, max_value=6, format="%d"),
                "Rate": st.column_config.NumberColumn("Rate", format="%d"),
                "GDP": st.column_config.NumberColumn("GDP", format="%d"),
                "Unemp": st.column_config.NumberColumn("Unemp", format="%d"),
                "CPI": st.column_config.NumberColumn("CPI", format="%d"),
                "COT": st.column_config.NumberColumn("COT", format="%d"),
                "Trend": st.column_config.NumberColumn("Trend", format="%d")
            }
        )
        
        # --- NOUVEAU : RAPPORT COT DÉTAILLÉ ---
        st.markdown("---")
        st.subheader("📊 Positionnement Institutionnel (COT Report)")
        st.caption("Positions nettes des Fonds à effet de levier (Hedge Funds). Barres bleues = Acheteurs dominants.")
        
        cot_display_df = df[['Asset', 'Net COT', 'Sentiment']].copy()
        cot_display_df['Net COT'] = cot_display_df['Net COT'].astype(int)
        
        st.dataframe(
            cot_display_df,
            column_config={
                "Asset": st.column_config.TextColumn("Actif", width="small"),
                "Net COT": st.column_config.NumberColumn("Contrats Nets (Vol)", format="%d"),
                "Sentiment": st.column_config.ProgressColumn(
                    "Long vs Short (%)",
                    help="Pourcentage de positions Longues vs Totales",
                    min_value=0, max_value=100, format="%.0f%%"
                ),
            },
            use_container_width=True,
            hide_index=True
        )

        # NEWS SECTION
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
        st.subheader("1. Force Relative (Base 100)")
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

    # TAB 3: SCANNER
    with tab_scan:
        st.dataframe(
            df[['Asset', 'Bias', 'Total', 'Price', 'RSI', 'Net COT']],
            column_config={
                "Total": st.column_config.ProgressColumn("Score", min_value=-6, max_value=6, format="%d"),
                "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),
                "Net COT": st.column_config.NumberColumn("Hedge Funds Net", format="%d"),
                "Price": st.column_config.NumberColumn("Prix", format="%.4f")
            },
            use_container_width=True
        )

if __name__ == "__main__":
    main()