import streamlit as st
import pandas as pd
import os
import re
import yfinance as yf
import plotly.express as px
from datetime import datetime, timezone
import concurrent.futures

st.set_page_config(page_title="Quant Terminal v2.9", layout="wide", page_icon="⚡")
DB_FILE = "master_database.csv"

# ==========================================
# 1. CACHED API & HELPER FUNCTIONS (MULTI-THREADED)
# ==========================================
@st.cache_data(show_spinner=False)
def fetch_sector_mapping(tickers):
    mapping = {}
    def get_sector(t):
        if "USD" in t or "BTC" in t or "ETH" in t: return t, "Cryptocurrency"
        try: return t, yf.Ticker(t).info.get('sector', 'Unknown / ETF')
        except: return t, "Unknown / ETF"
        
    # Launches 20 simultaneous connections for instant sector mapping
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        for t, sec in executor.map(get_sector, tickers):
            mapping[t] = sec
    return mapping

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_bulk_live_prices(tickers):
    if not tickers: return {}
    prices = {}
    try:
        # yfinance built-in multi-threading for mass downloading
        data = yf.download(list(tickers), period="1d", progress=False, threads=True)
        if len(tickers) == 1:
            t = list(tickers)[0]
            try: prices[t] = round(float(data['Close'].iloc[-1]), 2)
            except: prices[t] = None
        else:
            for t in tickers:
                try: prices[t] = round(float(data['Close'][t].iloc[-1]), 2)
                except: prices[t] = None
    except:
        pass
    return prices

@st.cache_data(ttl=86400, show_spinner=False)
def get_historical_price(ticker, date):
    try:
        start_str = date.strftime('%Y-%m-%d')
        end_str = (date + pd.Timedelta(days=3)).strftime('%Y-%m-%d')
        df = yf.download(ticker, start=start_str, end=end_str, progress=False)
        if not df.empty: return round(float(df['Close'].iloc[0]), 2)
    except: pass
    return None

def color_returns(val):
    if isinstance(val, str) and '%' in val:
        try:
            num = float(val.strip('%'))
            if num > 0: return 'color: #00CC96; font-weight: bold;'
            elif num < 0: return 'color: #EF553B; font-weight: bold;'
        except: pass
    return ''

def apply_color(df, col_names):
    style_meth = 'map' if hasattr(df.style, 'map') else 'applymap'
    return getattr(df.style, style_meth)(color_returns, subset=col_names)

# ==========================================
# 2. DATA INGESTION
# ==========================================
with st.sidebar:
    st.header("📥 Data Ingestion")
    uploaded_files = st.file_uploader("Upload CSVs", type=["csv"], accept_multiple_files=True)
    if st.button("Process & Save Database"):
        if uploaded_files:
            all_dfs = [pd.read_csv(f).rename(columns=lambda x: x.strip()) for f in uploaded_files]
            new_data = pd.concat(all_dfs, ignore_index=True)
            new_data['Time'] = pd.to_datetime(new_data['Time'], utc=True)
            
            if os.path.exists(DB_FILE):
                existing_db = pd.read_csv(DB_FILE)
                existing_db['Time'] = pd.to_datetime(existing_db['Time'], utc=True)
                combined_db = pd.concat([existing_db, new_data], ignore_index=True)
            else:
                combined_db = new_data
                
            combined_db.drop_duplicates(subset=['Time', 'Ticker', 'Description'], inplace=True)
            combined_db.sort_values('Time', ascending=False, inplace=True)
            combined_db.to_csv(DB_FILE, index=False)
            st.success(f"✅ DB Updated! ({len(combined_db)} total alerts)")

# ==========================================
# 3. MAIN DASHBOARD & PARSER
# ==========================================
st.title("⚡ Quantitative Signal Terminal")

if os.path.exists(DB_FILE):
    df = pd.read_csv(DB_FILE)
    df['Time'] = pd.to_datetime(df['Time'], utc=True)
    df.sort_values('Time', ascending=False, inplace=True)
    
    def parse_row(row):
        desc = str(row['Description']).upper()
        clean_ticker = str(row['Ticker']).split(':')[-1].split(',')[0].strip()
        name = str(row['Name']).upper()
        
        signal = "ℹ️ INFO"
        category = "Info"
        
        macro_keywords = ["LIQ", "MACRO", "RISK", "NET FUNDS", "LIQUIDITY"]
        is_macro = any(k in desc for k in macro_keywords) or any(k in name for k in macro_keywords)
        
        if is_macro:
            category = "Macro"
            if re.search(r'\b(GLD|XAUUSD|XAU|GOLD)\b', desc + name): clean_ticker = "GLD"
            elif re.search(r'\b(BTCUSD|BTC)\b', desc + name): clean_ticker = "BTC"
            elif re.search(r'\b(SPY|SPX)\b', desc + name): clean_ticker = "SPY"
                
            if "BUY" in desc or "LONG" in desc: signal = "🟢 LIQ BUY"
            elif "SELL" in desc or "SHORT" in desc: signal = "🔴 LIQ SELL"
            elif "HOLD" in desc: signal = "🟡 LIQ HOLD"
            else: signal = "ℹ️ LIQ INFO"
            
        elif "PRIME BUY" in desc or "BULLISH STACK" in desc: 
            signal = "⭐⭐ PRIME BUY" if "PRIME" in desc else "🟢 BULL STACK"
            category = "Bullish"
        elif "PRIME SELL" in desc or "BEARISH STACK" in desc:
            signal = "⭐⭐ PRIME SELL" if "PRIME" in desc else "🔴 BEAR STACK"
            category = "Bearish"
        elif "ORDER BUY" in desc: 
            signal = "🤖 STRAT BUY"
            category = "Strategy"
        elif "ORDER SELL" in desc: 
            signal = "🤖 STRAT SELL"
            category = "Strategy"
            
        price = None
        price_match = re.search(r"PRICE:\s*([\d\.]+)", desc, re.IGNORECASE)
        if price_match: price = float(price_match.group(1))
            
        return pd.Series([clean_ticker, row['Time'], signal, category, price, str(row['Name']), desc])
        
    parsed_df = df.apply(parse_row, axis=1)
    parsed_df.columns = ["Ticker", "Time", "Signal", "Category", "Price", "Name", "Raw_Desc"]
    parsed_df['Date_Str'] = parsed_df['Time'].dt.strftime('%b %d, %Y')
    parsed_df['Date_Only'] = parsed_df['Time'].dt.date
    
    unique_tickers = parsed_df['Ticker'].unique().tolist()
    
    # ⚡ THE SPEED BOTTLENECK FIX ⚡
    with st.spinner("🚀 Booting Async Data Engines (Fetching Live Data)..."):
        sector_map = fetch_sector_mapping(unique_tickers)
        live_prices_map = fetch_bulk_live_prices(unique_tickers)
        
    parsed_df['Sector'] = parsed_df['Ticker'].map(sector_map)

    # ==========================================
    # 4. MULTI-ASSET MACRO GAUGES
    # ==========================================
    st.markdown("### 🌐 Macro Liquidity Gauges")
    
    interval = st.radio("Analysis Interval (Applies to Breadth & Heatmap):", ["7 Days", "14 Days", "30 Days", "All-Time"], horizontal=True)
    if interval != "All-Time":
        days = int(interval.split()[0])
        cutoff = datetime.now(timezone.utc) - pd.Timedelta(days=days)
        filtered_df = parsed_df[parsed_df['Time'] >= cutoff]
    else:
        filtered_df = parsed_df

    def get_liquidity_state(ticker_regex):
        assets = parsed_df[(parsed_df['Ticker'].str.contains(ticker_regex, case=False, na=False, regex=True)) & 
                           (parsed_df['Category'] == 'Macro')].sort_values('Time', ascending=False)
        return assets.iloc[0] if not assets.empty else None

    spy_state = get_liquidity_state(r'^SPY|^SPX')
    gold_state = get_liquidity_state(r'^GLD|^XAU|^GOLD')
    btc_state = get_liquidity_state(r'^BTC')

    col_spy, col_gold, col_btc = st.columns(3)
    def render_gauge(col, title, state, emoji):
        with col:
            if state is not None:
                if "BUY" in state['Signal']:
                    st.success(f"{emoji} **{title}**: 🟢 RISK-ON\n\n*(via {state['Ticker']} on {state['Date_Str']})*")
                elif "SELL" in state['Signal']:
                    st.error(f"{emoji} **{title}**: 🔴 RISK-OFF\n\n*(via {state['Ticker']} on {state['Date_Str']})*")
                elif "HOLD" in state['Signal']:
                    st.warning(f"{emoji} **{title}**: 🟡 NEUTRAL\n\n*(via {state['Ticker']} on {state['Date_Str']})*")
                else:
                    st.info(f"{emoji} **{title}**: ℹ️ INFO ONLY\n\n*(via {state['Ticker']} on {state['Date_Str']})*")
            else:
                st.warning(f"{emoji} **{title}**: ⚖️ NEUTRAL\n\n*(No specific Liquidity alerts found)*")

    render_gauge(col_spy, "Equities (SPY)", spy_state, "🏢")
    render_gauge(col_gold, "Commodities (GLD/XAU)", gold_state, "🪙")
    render_gauge(col_btc, "Crypto (BTC)", btc_state, "₿")

    # ==========================================
    # 5. MARKET BREADTH & HEATMAP
    # ==========================================
    cvd_only = filtered_df[filtered_df['Category'].isin(['Bullish', 'Bearish'])]
    latest_cvd = cvd_only.sort_values('Time', ascending=False).drop_duplicates('Ticker').copy()
    
    col1, col2 = st.columns([1, 2])
    with col1:
        total = len(latest_cvd)
        if total > 0:
            bulls = len(latest_cvd[latest_cvd['Category'] == 'Bullish'])
            bears = len(latest_cvd[latest_cvd['Category'] == 'Bearish'])
            st.metric("🐂 Bullish Exposure", f"{(bulls/total)*100:.0f}%")
            st.metric("🐻 Bearish Exposure", f"{(bears/total)*100:.0f}%")
            st.metric(f"Active Assets Monitored", total)
            
    with col2:
        if total > 0:
            def get_sentiment_score(sig):
                if "PRIME BUY" in sig: return 2
                elif "BULL STACK" in sig: return 1
                elif "BEAR STACK" in sig: return -1
                elif "PRIME SELL" in sig: return -2
                return 0
                
            latest_cvd['Sentiment_Score'] = latest_cvd['Signal'].apply(get_sentiment_score)
            latest_cvd['Alert Price'] = latest_cvd['Price'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")
            latest_cvd['Days Active'] = (datetime.now(timezone.utc) - latest_cvd['Time']).dt.days

           custom_colors = ['#8B0000', '#EF553B', '#262730', '#00CC96', '#00FF00']

            fig = px.treemap(
                latest_cvd, path=[px.Constant("All Sectors"), 'Sector', 'Ticker'], 
                color='Sentiment_Score', color_continuous_scale=custom_colors, range_color=[-2, 2],
                hover_data={'Sentiment_Score': False, 'Signal': True, 'Alert Price': True, 'Date_Str': True, 'Days Active': True}
            )
            
            # --- THE TRADINGVIEW MODERNIZATION ---
            # 1. Force sharp, dark, thin borders
            fig.update_traces(marker=dict(line=dict(width=1, color='#0E1117')), root_color="#0E1117")
            
            # 2. Strip all padding and margins to make it edge-to-edge
            fig.update_layout(
                margin=dict(t=0, l=0, r=0, b=0), 
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)", 
                coloraxis_showscale=False,
                treemapcolorway=["#0E1117"] # Hides the top root box color
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    st.markdown("---")

    # ==========================================
    # 6. CONFLUENCE MATRIX
    # ==========================================
    st.subheader("🏆 Tier-1 Confluence Setups")
    confluence_list = []
    for (ticker, date), group in parsed_df.groupby(['Ticker', 'Date_Only']):
        sigs = group['Signal'].unique()
        has_strat_buy = any("STRAT BUY" in s for s in sigs)
        has_strat_sell = any("STRAT SELL" in s for s in sigs)
        has_bull = any(s in ["⭐⭐ PRIME BUY", "🟢 BULL STACK"] for s in sigs)
        has_bear = any(s in ["⭐⭐ PRIME SELL", "🔴 BEAR STACK"] for s in sigs)
        
        if (has_strat_buy and has_bull) or (has_strat_sell and has_bear):
            confluence_list.append({"Ticker": ticker, "Date": date, "Signals": ", ".join(sigs)})
            
    if confluence_list: st.dataframe(pd.DataFrame(confluence_list), use_container_width=True, hide_index=True)
    else: st.info("No pure directional Confluence setups found.")
    st.markdown("---")

    # ==========================================
    # 7. OPTIMAL TRADE MATRIX
    # ==========================================
    st.subheader("🎯 Optimal Trade Timing Matrix")
    
    with st.expander("ℹ️ How the Analytics Engine works"):
        st.markdown("""
        **Analytics Tags Explained:**
        * ⚡ **ACTIVE**: A standard active signal with no previous comparable data.
        * 🔥 **NEW ENTRY**: A new Bullish signal that successfully flipped a previous Bearish signal.
        * 🩸 **NEW SHORT**: A new Bearish signal that successfully flipped a previous Bullish signal.
        * ✅ **HIGHER LOW**: An algorithmic 'Buy the Dip' - Entry price is *higher* than the previous Bullish signal.
        * 🔪 **LOWER LOW**: 'Falling Knife' Warning - Entry price is *lower* than the previous Bullish signal.
        * ⚠️ **LOWER HIGH**: A warning that algorithms are shorting at a *lower* price than the previous short.
        """)
        
    analytics_data = []
    full_cvd = parsed_df[parsed_df['Category'].isin(['Bullish', 'Bearish'])].sort_values('Time', ascending=False)
    
    for ticker, group in full_cvd.groupby('Ticker'):
        if len(group) >= 1:
            latest = group.iloc[0]
            tag = "⚡ ACTIVE"
            prev_sig = "None"
            prev_date = "None"
            prev_price = "None"
            days_active = (datetime.now(timezone.utc) - latest['Time']).days
            
            if len(group) >= 2:
                prev = group.iloc[1]
                prev_sig = prev['Signal']
                prev_date = prev['Date_Str']
                if pd.notnull(latest['Price']) and pd.notnull(prev['Price']):
                    prev_price = f"${prev['Price']:.2f}"
                    if latest['Category'] == 'Bullish' and prev['Category'] == 'Bullish': tag = "✅ HIGHER LOW" if latest['Price'] > prev['Price'] else "🔪 LOWER LOW"
                    elif latest['Category'] == 'Bullish' and prev['Category'] == 'Bearish': tag = "🔥 NEW ENTRY"
                    elif latest['Category'] == 'Bearish' and prev['Category'] == 'Bearish': tag = "⚠️ LOWER HIGH"
                    elif latest['Category'] == 'Bearish' and prev['Category'] == 'Bullish': tag = "🩸 NEW SHORT"
            
            analytics_data.append({
                "Ticker": ticker, "Latest Signal": latest['Signal'], "Trigger Date": latest['Date_Str'],
                "Entry Price": f"${latest['Price']:.2f}" if pd.notnull(latest['Price']) else "N/A",
                "Prev Signal": prev_sig, "Prev Date": prev_date, "Prev Price": prev_price, 
                "Tag": tag, "Days Active": days_active, "Category": latest['Category']
            })

    matrix_df = pd.DataFrame(analytics_data)
    if not matrix_df.empty:
        tab1, tab2, tab3 = st.tabs(["[ View All Tickers ]", "[ 🟢 Bullish Only ]", "[ 🔴 Bearish Only ]"])
        with tab1: st.dataframe(matrix_df.drop(columns=['Category']), use_container_width=True, hide_index=True)
        with tab2: st.dataframe(matrix_df[matrix_df['Category'] == 'Bullish'].drop(columns=['Category']), use_container_width=True, hide_index=True)
        with tab3: st.dataframe(matrix_df[matrix_df['Category'] == 'Bearish'].drop(columns=['Category']), use_container_width=True, hide_index=True)
    st.markdown("---")

    # ==========================================
    # 8. ACTIVE STRATEGY TRACKER
    # ==========================================
    st.subheader("🤖 Strategy Execution Engine")
    strat_only = parsed_df[parsed_df['Category'] == 'Strategy'].sort_values('Time', ascending=False)
    latest_strats = strat_only.drop_duplicates('Ticker')
    
    active_data = []
    closed_data = []
    
    for _, row in latest_strats.iterrows():
        t = row['Ticker']
        buy_date = row['Time']
        
        derived_price = row['Price']
        if pd.isnull(derived_price):
            recent_cvd = parsed_df[(parsed_df['Ticker'] == t) & (parsed_df['Time'] <= buy_date)].head(1)
            if not recent_cvd.empty and pd.notnull(recent_cvd.iloc[0]['Price']):
                derived_price = recent_cvd.iloc[0]['Price']
            else:
                derived_price = get_historical_price(t, buy_date)
        
        if row['Signal'] == '🤖 STRAT BUY':
            # PULLED INSTANTLY FROM BULK MEMORY CACHE
            live_p = live_prices_map.get(t)
            ret_pct = f"{((live_p - derived_price)/derived_price)*100:.2f}%" if live_p and derived_price else "N/A"
            days_act = (datetime.now(timezone.utc) - buy_date).days
            active_data.append({
                "Ticker": t, "Trigger Date": row['Date_Str'], "Strategy": row['Name'],
                "Entry Price": f"${derived_price:.2f}" if derived_price else "N/A", 
                "Live Price": f"${live_p:.2f}" if live_p else "N/A",
                "Return %": ret_pct, "Days Active": days_act
            })
        else:
            closed_data.append({
                "Ticker": t, "Closed Date": row['Date_Str'], "Strategy": row['Name'],
                "Exit Price": f"${derived_price:.2f}" if derived_price else "N/A"
            })

    tab_act, tab_cls = st.tabs(["🟢 Active Strategies", "🔴 Closed Strategies"])
    with tab_act:
        if active_data: 
            st.dataframe(apply_color(pd.DataFrame(active_data), ['Return %']), use_container_width=True, hide_index=True)
        else: st.write("No active strategies.")
    with tab_cls:
        if closed_data: st.dataframe(pd.DataFrame(closed_data), use_container_width=True, hide_index=True)
        else: st.write("No closed strategies.")
    st.markdown("---")

    # ==========================================
    # 9. VOLATILITY SCANNER
    # ==========================================
    st.subheader("🌋 Advanced Volatility Scanner")
    vol_int = st.selectbox("Select Volatility Interval:", ["7 Days", "30 Days", "45 Days", "60 Days", "YTD"], index=1)
    
    if vol_int == "YTD": vol_cutoff = datetime(datetime.now(timezone.utc).year, 1, 1, tzinfo=timezone.utc)
    else: vol_cutoff = datetime.now(timezone.utc) - pd.Timedelta(days=int(vol_int.split()[0]))
        
    last_vol = parsed_df[parsed_df['Time'] >= vol_cutoff]
    vol_list = []
    for sym, group in last_vol.dropna(subset=['Price']).groupby('Ticker'):
        high, low = group['Price'].max(), group['Price'].min()
        if low > 0: vol_list.append({"Ticker": sym, "Range %": f"{((high-low)/low)*100:.1f}%", "High": f"${high:.2f}", "Low": f"${low:.2f}"})
    if vol_list: st.dataframe(pd.DataFrame(vol_list).sort_values("Range %", ascending=False).head(10), use_container_width=True, hide_index=True)
    st.markdown("---")
    
    # ==========================================
    # 10. DEEP-DIVE DRAWER
    # ==========================================
    st.subheader("📂 Ticker Deep-Dive")
    selected_ticker = st.selectbox("Search Ticker:", sorted(unique_tickers))
    if selected_ticker:
        ticker_hist = parsed_df[parsed_df['Ticker'] == selected_ticker].sort_values('Time', ascending=False)
        
        latest_sig = ticker_hist[ticker_hist['Category'].isin(['Bullish', 'Bearish'])].head(1)
        if not latest_sig.empty and pd.notnull(latest_sig.iloc[0]['Price']):
            alert_price = latest_sig.iloc[0]['Price']
            # PULLED INSTANTLY FROM BULK MEMORY CACHE
            live_price = live_prices_map.get(selected_ticker)
            if live_price:
                delta = ((live_price - alert_price) / alert_price) * 100
                color_css = "#00CC96" if delta > 0 else "#EF553B"
                st.markdown(f"**Live Price:** ${live_price:.2f} | **Alert Price:** ${alert_price:.2f} | **Unrealized Gap:** <span style='color:{color_css}; font-weight:bold;'>{delta:.2f}%</span>", unsafe_allow_html=True)

        st.dataframe(ticker_hist[['Time', 'Signal', 'Price', 'Name', 'Category', 'Raw_Desc']].assign(Time=lambda x: x['Time'].dt.strftime('%Y-%m-%d %H:%M')), use_container_width=True, hide_index=True)

else:
    st.info("👈 Upload your TradingView CSVs in the sidebar to begin.")
