import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import requests  # 用於 Telegram API 請求
import time  # 用於自動刷新時間檢查

# 計算 MACD
def calculate_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# 計算 RSI
def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 計算 Stochastic
def calculate_stochastic(df, k_period=14, d_period=3):
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    d = k.rolling(window=d_period).mean()
    return k, d

# 計算 OBV
def calculate_obv(df):
    sign = np.sign(df['Close'].diff())
    obv = (sign * df['Volume']).fillna(0).cumsum()
    return obv

# 計算 MFI
def calculate_mfi(df, period=14):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    raw_money_flow = typical_price * df['Volume']
    positive_flow = raw_money_flow.where(typical_price.diff() > 0, 0).rolling(window=period).sum()
    negative_flow = raw_money_flow.where(typical_price.diff() < 0, 0).rolling(window=period).sum()
    money_ratio = positive_flow / negative_flow
    mfi = 100 - (100 / (1 + money_ratio))
    return mfi

# 計算 Bollinger Bands
def calculate_bb(df, period=20, std=2):
    sma = df['Close'].rolling(window=period).mean()
    std_dev = df['Close'].rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

# 發送 Telegram 通知
def send_telegram_notification(message):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'  # 支援簡單格式化
        }
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            st.success("Telegram 通知已發送！")
        else:
            st.error(f"Telegram 通知失敗: {response.status_code}")
    except Exception as e:
        st.error(f"發送 Telegram 通知時出錯: {e}")

# 檢測多頭分歧
def detect_bullish_divergence(df, histogram):
    if len(df) < 3:
        return False
    recent_lows = pd.to_numeric(df['Low'].iloc[-3:], errors='coerce')
    hist_lows = pd.to_numeric(histogram.iloc[-3:], errors='coerce')
    diff_lows = recent_lows.diff().dropna()
    diff_hists = hist_lows.diff().dropna()
    # 確保數值比較
    lows_decreasing = all(pd.to_numeric(d, errors='coerce') <= 0 and pd.notna(pd.to_numeric(d, errors='coerce')) for d in diff_lows)
    hist_decreasing = all(pd.to_numeric(d, errors='coerce') <= 0 and pd.notna(pd.to_numeric(d, errors='coerce')) for d in diff_hists)
    # 多頭分歧判斷是價格創新低，但指標沒有創新低
    if lows_decreasing and not hist_decreasing:
        return True
    return False

# 檢測熊頭分歧
def detect_bearish_divergence(df, histogram):
    if len(df) < 3:
        return False
    recent_highs = pd.to_numeric(df['High'].iloc[-3:], errors='coerce')
    hist_highs = pd.to_numeric(histogram.iloc[-3:], errors='coerce')
    diff_highs = recent_highs.diff().dropna()
    diff_hists = hist_highs.diff().dropna()
    # 確保數值比較
    highs_increasing = all(pd.to_numeric(d, errors='coerce') >= 0 and pd.notna(pd.to_numeric(d, errors='coerce')) for d in diff_highs)
    hist_increasing = all(pd.to_numeric(d, errors='coerce') >= 0 and pd.notna(pd.to_numeric(d, errors='coerce')) for d in diff_hists)
    # 熊頭分歧判斷是價格創新高，但指標沒有創新高
    if highs_increasing and not hist_increasing:
        return True
    return False

# 獲取數據
def get_data(ticker, period, interval):
    try:
        # 嘗試使用 Ticker.history 以避免 download 的某些錯誤
        data = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
        if data.empty:
            # 後備：嘗試每日數據（適合周末）
            is_weekend = datetime.now().weekday() >= 5
            if is_weekend:
                data = yf.Ticker(ticker).history(period='5d', interval='1d', auto_adjust=False)
        if data.empty:
            return pd.DataFrame()
        
        # 確保 OHLCV 為數值型
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        data = data.dropna(subset=['Close'])  # 移除無效行
        
        return data
    except Exception as e:
        st.error(f"獲取數據失敗: {e}")
        # 後備每日數據
        try:
            data = yf.Ticker(ticker).history(period='5d', interval='1d', auto_adjust=False)
            if not data.empty:
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in data.columns:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                return data
        except:
            pass
        return pd.DataFrame()

# Streamlit app 主介面
st.title('股票日內交易助手')
st.write('基於 MACD、Histogram 變化、多頭分歧、RSI、Stochastic、OBV、MFI、BB 指標，自動更新。')

# Telegram 設定（整合用戶提供的 try 塊）
telegram_ready = False
try:
    # 假設 secrets.toml 已經設定
    BOT_TOKEN = st.secrets["telegram"]["BOT_TOKEN"]
    CHAT_ID = st.secrets["telegram"]["CHAT_ID"]
    telegram_ready = True
except:
    st.warning("Telegram 設定未完成，請在 .streamlit/secrets.toml 中添加 BOT_TOKEN 和 CHAT_ID。")

# 側邊欄輸入參數
with st.sidebar:
    st.subheader('自訂參數')
    ticker = st.text_input('股票代碼', value='TSLA')
    period = st.selectbox('數據天數', ['1d', '5d', '10d'],
