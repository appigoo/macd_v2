import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import requests
from streamlit_autorefresh import st_autorefresh
import time

# ===== 技術指標計算 =====
def calculate_macd(df, fast=12, slow=26, signal=9):
    df = df.dropna(subset=['Close'])
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calculate_stochastic(df, k_period=14, d_period=3):
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    d = k.rolling(window=d_period).mean()
    return k, d

def calculate_obv(df):
    sign = np.sign(df['Close'].diff())
    return (sign * df['Volume']).fillna(0).cumsum()

def calculate_mfi(df, period=14):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    raw_money_flow = typical_price * df['Volume']
    pos_flow = raw_money_flow.where(typical_price.diff() > 0, 0).rolling(period).sum()
    neg_flow = raw_money_flow.where(typical_price.diff() < 0, 0).rolling(period).sum()
    ratio = pos_flow / neg_flow.replace(0, np.nan)
    return 100 - (100 / (1 + ratio))

def calculate_bb(df, period=20, std=2):
    sma = df['Close'].rolling(period).mean()
    stdv = df['Close'].rolling(period).std()
    upper = sma + std * stdv
    lower = sma - std * stdv
    return upper, sma, lower

# ===== 分歧偵測 =====
def detect_bullish_divergence(df, hist_col, lookback=5):
    lows = df['Low'].tail(lookback)
    hist = df[hist_col].tail(lookback)
    return (lows.iloc[-1] < lows.min()) and (hist.iloc[-1] > hist.min())

def detect_bearish_divergence(df, hist_col, lookback=5):
    highs = df['High'].tail(lookback)
    hist = df[hist_col].tail(lookback)
    return (highs.iloc[-1] > highs.max()) and (hist.iloc[-1] < hist.max())

# ===== Telegram 發送 =====
def send_telegram_notification(message):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {'chat_id': CHAT_ID, 'text': message, 'parse_mode': 'HTML'}
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200:
            st.toast("Telegram 通知已發送")
        else:
            st.warning(f"Telegram 通知失敗: {r.status_code}")
    except Exception as e:
        st.error(f"發送 Telegram 通知時出錯: {e}")

# ===== 數據獲取 =====
def get_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
        if df.empty and datetime.now().weekday() >= 5:
            df = yf.download(ticker, period='5d', interval='1d', auto_adjust=False, progress=False)
        if df.empty:
            return pd.DataFrame()
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna(subset=['Close'])
    except Exception as e:
        st.error(f"獲取數據失敗: {e}")
        return pd.DataFrame()

# ===== Streamlit 介面 =====
st.title('股票日內交易助手（正式穩定版）')

# Telegram 設定
telegram_ready = False
try:
    BOT_TOKEN = st.secrets["telegram"]["BOT_TOKEN"]
    CHAT_ID = st.secrets["telegram"]["CHAT_ID"]
    telegram_ready = True
except:
    st.warning("未設定 Telegram，請於 .streamlit/secrets.toml 配置 BOT_TOKEN 和 CHAT_ID")

# 側邊設定參數
with st.sidebar:
    st.subheader('參數設定')
    ticker = st.text_input('股票代碼', value='TSLA')
    period = st.selectbox('抓取時間', ['1d', '5d', '10d'], index=1)
    interval = st.selectbox('K線間隔', ['1m', '5m', '15m', '1d'], index=1)
    refresh_minutes = st.number_input('建議刷新間隔（分鐘）', value=5, min_value=1)
    enable_auto_refresh = st.checkbox('啟用自動刷新', value=False)
    auto_interval = st.selectbox('自動刷新間隔 (min)', [1, 2, 3, 5], index=0) if enable_auto_refresh else 0

    macd_fast = st.number_input('MACD Fast', value=12, min_value=1)
    macd_slow = st.number_input('MACD Slow', value=26, min_value=1)
    macd_signal = st.number_input('MACD Signal', value=9, min_value=1)
    rsi_period = st.number_input('RSI Period', value=14, min_value=1)
    stoch_k = st.number_input('Stoch K', value=14, min_value=1)
    stoch_d = st.number_input('Stoch D', value=3, min_value=1)
    mfi_period = st.number_input('MFI Period', value=14, min_value=1)
    bb_period = st.number_input('BB Period', value=20, min_value=1)
    bb_std = st.number_input('BB Std', value=2.0, min_value=0.1, step=0.1)
    if telegram_ready:
        enable_telegram_buy = st.checkbox('啟用買入 Telegram 通知', value=False)
        enable_telegram_sell = st.checkbox('啟用賣出 Telegram 通知', value=False)
    else:
        enable_telegram_buy = enable_telegram_sell = False

placeholder = st.empty()

# 自動刷新模組
if enable_auto_refresh and auto_interval > 0:
    st_autorefresh(interval=auto_interval * 60 * 1000, key="auto_refresh")

# ===== 主函數 =====
def refresh_data():
    df = get_data(ticker, period, interval)
    if df.empty:
        st.error("無法取得數據")
        return

    df = df.tail(500)
    df['MACD'], df['Signal'], df['Histogram'] = calculate_macd(df, macd_fast, macd_slow, macd_signal)
    df['RSI'] = calculate_rsi(df, rsi_period)
    df['%K'], df['%D'] = calculate_stochastic(df, stoch_k, stoch_d)
    df['OBV'] = calculate_obv(df)
    df['MFI'] = calculate_mfi(df, mfi_period)
    df['BBU'], df['BBM'], df['BBL'] = calculate_bb(df, bb_period, bb_std)
    df = df.dropna()

    latest = df.iloc[-1]
    hist_trend = df['Histogram'].tail(3).diff().dropna()
    hist_increasing = (hist_trend.gt(0).all()) and (df['Histogram'].iloc[-1] < 0)
    hist_decreasing = (hist_trend.lt(0).all()) and (df['Histogram'].iloc[-1] > 0)

    bull_div = detect_bullish_divergence(df, 'Histogram')
    bear_div = detect_bearish_divergence(df, 'Histogram')
    rsi_bull = (df['RSI'].iloc[-1] > 40 and df['RSI'].iloc[-2] < 30)
    rsi_bear = (df['RSI'].iloc[-1] < 60 and df['RSI'].iloc[-2] > 70)
    stoch_bull = (df['%K'].iloc[-1] > df['%D'].iloc[-1]) and (df['%K'].iloc[-2] < 20)
    stoch_bear = (df['%K'].iloc[-1] < df['%D'].iloc[-1]) and (df['%K'].iloc[-2] > 80)
    obv_up = df['OBV'].diff().iloc[-1] > 0
    obv_down = df['OBV'].diff().iloc[-1] < 0
    mfi_bull = (df['MFI'].iloc[-1] > 20 and df['MFI'].iloc[-2] < 20)
    mfi_bear = (df['MFI'].iloc[-1] < 80 and df['MFI'].iloc[-2] > 80)
    bb_bull = df['Close'].iloc[-1] < df['BBL'].iloc[-1]
    bb_bear = df['Close'].iloc[-1] > df['BBU'].iloc[-1]

    buy_flags = [hist_increasing, bull_div, rsi_bull, stoch_bull, obv_up, mfi_bull, bb_bull]
    sell_flags = [hist_decreasing, bear_div, rsi_bear, stoch_bear, obv_down, mfi_bear, bb_bear]
    buy_score, sell_score = sum(buy_flags), sum(sell_flags)

    buy_text = "無明顯買入信號"
    sell_text = "無明顯賣出信號"

    if buy_score >= 5:
        buy_text = "強烈買入信號"
        if enable_telegram_buy and telegram_ready:
            if "last_buy_time" not in st.session_state or time.time() - st.session_state.last_buy_time > 300:
                msg = f"<b>🚨 強烈買入信號</b>\n{ticker}\n時間: {datetime.now():%Y-%m-%d %H:%M:%S}\n收盤: {latest['Close']:.2f}\n信號: {buy_score}/7"
                send_telegram_notification(msg)
                st.session_state.last_buy_time = time.time()

    if sell_score >= 5:
        sell_text = "強烈賣出信號"
        if enable_telegram_sell and telegram_ready:
            if "last_sell_time" not in st.session_state or time.time() - st.session_state.last_sell_time > 300:
                msg = f"<b>⚠️ 強烈賣出信號</b>\n{ticker}\n時間: {datetime.now():%Y-%m-%d %H:%M:%S}\n收盤: {latest['Close']:.2f}\n信號: {sell_score}/7"
                send_telegram_notification(msg)
                st.session_state.last_sell_time = time.time()

    with placeholder:
        st.metric("最新收盤價", f"{latest['Close']:.2f}")
        st.write(f"RSI: {latest['RSI']:.2f} | MACD Histogram: {latest['Histogram']:.4f}")
        st.write(f"多頭分歧: {'是' if bull_div else '否'} | 熊頭分歧: {'是' if bear_div else '否'}")
        st.progress(buy_score / 7)
        st.progress(sell_score / 7)
        st.subheader("買入建議")
        st.write(buy_text)
        st.subheader("賣出建議")
        st.write(sell_text)
        st.line_chart(df['Close'].tail(50))
        st.line_chart(df['Histogram'].tail(50))
        st.bar_chart(df['Volume'].tail(50))

refresh_data()

st.sidebar.markdown("---")
if st.sidebar.button("立即更新"):
    st.rerun()
