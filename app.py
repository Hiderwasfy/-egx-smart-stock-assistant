# ============================================
# EGX NEXT GEN PLATFORM - SMART ENGINE
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from concurrent.futures import ThreadPoolExecutor
from tradingview_ta import TA_Handler, Interval

# ============================================
# Load Market data
# ============================================
def analyze_stock(symbol):
        try:

            handler = TA_Handler(
                symbol=symbol,
                screener="egypt",
                exchange="EGX",
                interval=Interval.INTERVAL_1_DAY
            )

            analysis = handler.get_analysis()

            ind = analysis.indicators

            current_price = round(ind.get("close", 0), 2)

            rsi = round(ind.get("RSI", 0), 2)

            macd = round(ind.get("MACD.macd", 0), 2)

            volume = int(ind.get("volume", 0))

            sector = stocks_df.loc[
                stocks_df["Symbol"] == symbol,
                "Sector"
            ].values[0]

            fair_value = calculate_fair_value(
                sector,
                current_price,
                rsi,
                macd,
                volume
            )

            # =========================
            # SIGNAL ENGINE
            # =========================
            if current_price > 0:

                upside = round(
                    (
                        (fair_value - current_price)
                        / current_price
                    ) * 100,
                    2
                )

            else:

                upside = 0

            if rsi < 30 and macd > 0 and upside > 25:
                signal = "STRONG BUY"

            elif upside > 20 and macd > 0:
                signal = "BUY"

            elif rsi > 75:
                signal = "OVERBOUGHT"

            elif (
                current_price > fair_value
                and rsi > 70
                and macd < 0
            ):

                signal = "STRONG SELL"

            elif current_price > fair_value:
                signal = "OVERVALUED"

            else:
                signal = "HOLD"

            # =========================
            # TREND ENGINE
            # =========================

            if macd > 0:
                trend = "UPTREND"
            else:
                trend = "DOWNTREND"
            # =========================
            # AI SCORE ENGINE
            # =========================

            score = 0

            # UPSIDE

            if upside > 40:
                score += 20

            elif upside > 25:
                score += 20

            elif upside > 15:
                score += 10


            # RSI

            if 45 <= rsi <= 60:
                score += 20

            elif rsi < 40:
                score += 10


            # MACD

            if macd > 0:
                score += 15


            # TREND

            if trend == "UPTREND":
                score += 10


            # VOLUME

            if volume > 1000000:
                score += 15

            elif volume > 500000:
                score += 10

            elif rsi > 70:
                score -= 20


            # SIGNAL BONUS

            if signal == "STRONG BUY":
                score += 25

            elif signal == "BUY":
                score += 15


            # LIMIT

            score = min(score, 100)
            # =========================
            # SAVE DATA
            # =========================

        return {
                    "Symbol": symbol,
                    "Sector": sector,
                    "Price": current_price,
                    "Fair Value": fair_value,
                    "RSI": rsi,
                    "MACD": macd,
                    "Volume": volume,
                    "Signal": signal,
                    "Trend": trend,
                    "AI Score": score
                }

        except:
            return None


# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="EGX NEXT GEN PLATFORM",
    layout="wide"
)

st.title("🚀 EGX NEXT GEN PLATFORM")

# ============================================
# LOAD STOCKS CSV
# ============================================

stocks_df = pd.read_csv("egx_stocks.csv")

all_stocks = stocks_df["Symbol"].tolist()

# ============================================
# FAIR VALUE ENGINE
# ============================================

def calculate_fair_value(
    sector,
    current_price,
    rsi,
    macd,
    volume
):

    sector = str(sector).upper()

    # =========================
    # BANKS
    # P/B + ROE
    # =========================

    if sector == "BANKS":

        pb_ratio = 1.4

        fair_value = current_price * pb_ratio

        return round(fair_value, 2)

    # =========================
    # REAL ESTATE
    # NAV + DCF
    # =========================

    elif sector == "REAL_ESTATE":

        growth_factor = 1.40

        return round(current_price * growth_factor, 2)

    # =========================
    # INDUSTRIAL
    # P/E + DCF
    # =========================

    elif sector == "INDUSTRIAL":

        pe_factor = 1.18

        return round(current_price * pe_factor, 2)

    # =========================
    # TECH
    # HIGH GROWTH
    # =========================

    elif sector == "TECH":

        growth_factor = 1.60

        return round(current_price * growth_factor, 2)

    # =========================
    # FERTILIZERS
    # CYCLICAL PROFITS
    # =========================

    elif sector == "FERTILIZERS":

        cycle_factor = 1.30

        return round(current_price * cycle_factor, 2)

    # =========================
    # CONSUMER
    # STABLE P/E
    # =========================

    elif sector == "CONSUMER":

        return round(current_price * 1.17, 2)

    # =========================
    # HEALTHCARE
    # GROWTH
    # =========================

    elif sector == "HEALTHCARE":

        return round(current_price * 1.12, 2)

    # =========================
    # MATERIALS
    # COMMODITIES
    # =========================

    elif sector == "MATERIALS":

        return round(current_price * 1.10, 2)

    # =========================
    # DEFAULT
    # =========================

    else:

        return round(current_price * 1.08, 2)

# ============================================
# Data
# ============================================


@st.cache_data(ttl=300)
def load_market_data(stock_list):

    with ThreadPoolExecutor(max_workers=20) as executor:

        results = list(
            executor.map(analyze_stock, stock_list)
        )

    market_data = [
        r for r in results
        if r is not None
    ]

    return pd.DataFrame(market_data)

# ============================================
# SIDEBAR
# ============================================

st.sidebar.title("📊 Market Controls")

selected_stocks = st.sidebar.multiselect(
    "Choose Stocks",
    stocks_df["Symbol"].tolist(),
    default=["COMI", "HRHO", "FWRY", "TMGH"]
)

run = st.sidebar.button("🚀 Run Market Scan")

dashboard_df = pd.DataFrame()
scanner_market_df = pd.DataFrame()

if run:
    dashboard_df = load_market_data(selected_stocks)
    scanner_market_df = load_market_data(all_stocks)

if st.sidebar.button("🔄 Refresh Cache"):
    st.cache_data.clear()



# =========================
# COLOR ENGINE
# =========================

def color_signal(val):

    if val == "BUY":
        return "background-color: #00cc66"

    elif val == "STRONG BUY":
        return "background-color: #00ff99"

    elif val == "SELL":
        return "background-color: #ff4d4d"

    elif val == "OVERBOUGHT":
        return "background-color: orange"

    return ""

# =========================
# STYLED DATAFRAME
# =========================
dashboard_df = dashboard_df.round({
    "Price": 2,
    "Fair Value": 2,
    "RSI": 2,
    "MACD": 2
})

styled_df = dashboard_df.style.map(
    color_signal,
    subset=["Signal"]
)

# ============================================
# LIVE DASHBOARD
# ============================================

st.subheader("📊 Live Market Dashboard")

dashboard_df = dashboard_df.copy()

st.dataframe(
    dashboard_df,
    use_container_width=True
)

# =========================
# SMART SCANNER
# =========================


scanner_df = scanner_market_df.copy()

# FILTERS

scanner_df = scanner_df[
    (scanner_df["AI Score"] >= 70) &
    (scanner_df["Trend"] == "UPTREND") &
    (scanner_df["RSI"] < 80) &
    (scanner_df["Volume"] > 300000)
]

# SORT

scanner_df = scanner_df.sort_values(
    by="AI Score",
    ascending=False
)

# STATS

total_scanned = len(scanner_market_df)
filtered_out = total_scanned - len(scanner_df)
error_count = 0


# ============================================
# SMART SCANNER TABLE
# ============================================

st.subheader("🧠 Smart Scanner")

st.dataframe(
    scanner_df,
    use_container_width=True
)

st.subheader("📊 Scanner Statistics")

col1, col2, col3 = st.columns(3)

col1.metric("Scanned", total_scanned)
col2.metric("Filtered", filtered_out)
col3.metric("Errors", error_count)
# ============================================
# AI SCORE CHART
# ============================================

if not scanner_df.empty:

    st.subheader("📈 AI Opportunity Scores")

    fig = px.bar(
        scanner_df.head(20),
        x="Symbol",
        y="AI Score",
        color="AI Score",
        text="AI Score"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
