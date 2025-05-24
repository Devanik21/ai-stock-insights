# 📈 Stock Market Storyteller with Gemini AI (Streamlit + TA-Lib-Free Version)

import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import numpy as np
import google.generativeai as genai

# Page config
st.set_page_config(page_title="📈 Stock Market Storyteller", layout="wide")
st.title("📈 Stock Market Storyteller")
st.write("Narrate your favorite stocks with technical indicators & Gemini-powered summaries!")

# Sidebar for Gemini API key
gemini_api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

# Sidebar stock selection
st.sidebar.header("📦 Stock Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g. AAPL)", value="AAPL")
period = st.sidebar.selectbox("Select Data Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
interval = st.sidebar.selectbox("Select Interval", ["1d", "1h", "15m"])

# Fetch stock data
@st.cache_data
def load_data(ticker, period, interval):
    data = yf.download(ticker, period=period, interval=interval)
    data.dropna(inplace=True)
    return data

if ticker:
    data = load_data(ticker, period, interval)

    st.subheader(f"📊 Price Chart for {ticker}")
    st.line_chart(data['Close'])

    st.subheader("📉 Technical Indicators")
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    delta = data['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    st.line_chart(data[['Close', 'SMA_20', 'SMA_50']])
    st.line_chart(data[['RSI']])
    st.line_chart(data[['MACD', 'MACD_signal']])

    st.subheader("🧠 Gemini-Powered Summary")
    if gemini_api_key:
        latest = data.iloc[-1]
        summary_prompt = f"""
        The current stock price of {ticker} is {latest['Close']:.2f}.
        The 20-day SMA is {latest['SMA_20']:.2f} and 50-day SMA is {latest['SMA_50']:.2f}.
        RSI is {latest['RSI']:.2f}, MACD is {latest['MACD']:.2f}, and MACD signal is {latest['MACD_signal']:.2f}.

        Please provide a simple, friendly summary of what this might mean for investors.
        """
        try:
            response = model.generate_content(summary_prompt)
            st.success(response.text)
        except Exception as e:
            st.error("Gemini error: " + str(e))
    else:
        st.info("Enter your Gemini API key in the sidebar to get AI summaries.")

    st.subheader("📥 Download Processed Data")
    st.download_button("Download CSV", data.to_csv().encode(), file_name=f"{ticker}_indicators.csv")
