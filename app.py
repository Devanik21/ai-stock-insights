# 📈 Stock Market Storyteller with Gemini AI (Streamlit + TA-Lib-Free Version)

import streamlit as st
import yfinance as yf
import pandas as pd
import google.generativeai as genai
from typing import Optional, Dict, Any

# Page config
st.set_page_config(page_title="Stock Market Storyteller", page_icon="📈",layout="wide")
st.title("📈 Stock Market Storyteller")
st.write("Narrate your favorite stocks with technical indicators & Gemini-powered summaries!")

# --- Constants ---
SMA_SHORT_WINDOW = 20
SMA_LONG_WINDOW = 50
RSI_WINDOW = 14
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
GEMINI_MODEL_NAME = "gemini-2.0-flash" # Or "gemini-1.5-flash", "gemini-pro"

# Sidebar for Gemini API key
model: Optional[genai.GenerativeModel] = None
gemini_api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
if gemini_api_key:
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        st.sidebar.success("Gemini AI configured successfully!")
    except Exception as e:
        st.sidebar.error(f"Failed to configure Gemini: {e}")
        model = None
else:
    st.sidebar.info("Enter your Gemini API key to enable AI-powered summaries.")

# Sidebar stock selection
st.sidebar.header("📦 Stock Settings")

with st.sidebar.expander("💡 Help me find codes (Examples)"):
    st.markdown("""
    Here are some popular stock ticker symbols to get you started:

    **Tech & Software:**
    - AAPL (Apple Inc.)
    - MSFT (Microsoft Corp.)
    - GOOGL (Alphabet Inc. - Google)
    - AMZN (Amazon.com Inc.)
    - TSLA (Tesla, Inc.)
    - NVDA (NVIDIA Corp.)
    - META (Meta Platforms, Inc.)
    - CRM (Salesforce, Inc.)
    - INTC (Intel Corp.)
    - AMD (Advanced Micro Devices)
    - ORCL (Oracle Corp.)
    - ADBE (Adobe Inc.)
    - QCOM (QUALCOMM Inc.)
    - CSCO (Cisco Systems, Inc.)
    - IBM (IBM Corp.)

    **Finance & Payments:**
    - JPM (JPMorgan Chase & Co.)
    - V (Visa Inc.)
    - MA (Mastercard Inc.)
    - BAC (Bank of America Corp.)
    - GS (Goldman Sachs Group)
    - WFC (Wells Fargo & Co.)
    - BRK-B (Berkshire Hathaway Inc. Class B)
    - AXP (American Express Co.)
    - PYPL (PayPal Holdings, Inc.)

    **Consumer (Discretionary & Staples):**
    - WMT (Walmart Inc.)
    - COST (Costco Wholesale Corp.)
    - MCD (McDonald's Corp.)
    - SBUX (Starbucks Corp.)
    - NKE (NIKE, Inc.)
    - HD (The Home Depot, Inc.)
    - PG (Procter & Gamble Co.)
    - KO (The Coca-Cola Co.)
    - PEP (PepsiCo, Inc.)
    - TGT (Target Corp.)

    **Healthcare:**
    - JNJ (Johnson & Johnson)
    - PFE (Pfizer Inc.)
    - UNH (UnitedHealth Group Inc.)
    - LLY (Eli Lilly and Co.)
    - MRK (Merck & Co., Inc.)
    - ABBV (AbbVie Inc.)
    - MRNA (Moderna, Inc.)

    **Industrials & Energy:**
    - BA (The Boeing Company)
    - CAT (Caterpillar Inc.)
    - XOM (Exxon Mobil Corp.)
    - CVX (Chevron Corp.)
    - GE (General Electric Co.)
    - HON (Honeywell International Inc.)

    **Communication & Entertainment:**
    - NFLX (Netflix, Inc.)
    - DIS (The Walt Disney Co.)
    - CMCSA (Comcast Corp.)
    - T (AT&T Inc.)
    - VZ (Verizon Communications Inc.)

    **Logistics:**
    - UPS (United Parcel Service, Inc.)
    - FDX (FedEx Corp.)
    """)

ticker = st.sidebar.text_input("Enter Stock Ticker (e.g. AAPL)", value="AAPL")
period = st.sidebar.selectbox("Select Data Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
interval = st.sidebar.selectbox("Select Interval", ["1d", "1wk", "1mo", "1h", "15m"], index=0) # Added more intervals

# Fetch stock data
@st.cache_data
def load_data(ticker_symbol: str, data_period: str, data_interval: str) -> pd.DataFrame:
    """Fetches stock data from Yahoo Finance and flattens MultiIndex columns."""
    try:
        data = yf.download(ticker_symbol, period=data_period, interval=data_interval, progress=False)
        if data.empty:
            st.warning(f"No data found for ticker {ticker_symbol} with period {data_period} and interval {data_interval}.")
            return pd.DataFrame()

        # Flatten MultiIndex columns if present (yfinance can return multi-level for Single Ticker)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if 'Close' not in data.columns:
            st.error(f"'Close' column not found in data for {ticker_symbol}.")
            return pd.DataFrame()
            
        data.dropna(inplace=True) # Drop rows with any NaN values in essential OCHLV columns
        return data
    except Exception as e:
        st.error(f"Error loading data for {ticker_symbol}: {e}")
        return pd.DataFrame()

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and adds technical indicators to the DataFrame."""
    data_ti = df.copy()
    if 'Close' not in data_ti.columns:
        st.error("Cannot calculate indicators: 'Close' column missing.")
        return data_ti # Return original or empty if 'Close' is critical

    # Moving Averages
    data_ti[f'SMA_{SMA_SHORT_WINDOW}'] = data_ti['Close'].rolling(window=SMA_SHORT_WINDOW).mean()
    data_ti[f'SMA_{SMA_LONG_WINDOW}'] = data_ti['Close'].rolling(window=SMA_LONG_WINDOW).mean()
    
    # RSI calculation
    delta = data_ti['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=RSI_WINDOW, min_periods=1).mean() # Use min_periods=1 to get values earlier
    avg_loss = loss.rolling(window=RSI_WINDOW, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    data_ti['RSI'] = 100.0 - (100.0 / (1.0 + rs))
    data_ti['RSI'].fillna(50, inplace=True) # Fill initial NaNs with neutral 50, or handle as preferred

    # MACD calculation
    exp1 = data_ti['Close'].ewm(span=MACD_FAST_PERIOD, adjust=False).mean()
    exp2 = data_ti['Close'].ewm(span=MACD_SLOW_PERIOD, adjust=False).mean()
    data_ti['MACD'] = exp1 - exp2
    data_ti['MACD_signal'] = data_ti['MACD'].ewm(span=MACD_SIGNAL_PERIOD, adjust=False).mean()
    return data_ti

def generate_gemini_summary(ai_model: genai.GenerativeModel, stock_ticker: str, latest_data: pd.Series) -> str:
    """Generates a stock summary using the Gemini AI model."""
    prompt_data = {key: f"{value:.2f}" if isinstance(value, float) else str(value) for key, value in latest_data.items()}
    
    summary_prompt = f"""
    Analyze the following latest technical indicator data for {stock_ticker}:
    Close Price: {prompt_data.get('Close', 'N/A')}
    SMA {SMA_SHORT_WINDOW}-day: {prompt_data.get(f'SMA_{SMA_SHORT_WINDOW}', 'N/A')}
    SMA {SMA_LONG_WINDOW}-day: {prompt_data.get(f'SMA_{SMA_LONG_WINDOW}', 'N/A')}
    RSI ({RSI_WINDOW}-day): {prompt_data.get('RSI', 'N/A')}
    MACD: {prompt_data.get('MACD', 'N/A')}
    MACD Signal: {prompt_data.get('MACD_signal', 'N/A')}

    Provide a concise, easy-to-understand summary for an investor. What might these indicators suggest about the stock's current situation and potential short-term outlook? Focus on interpretation, not just restating values.
    """
    try:
        response = ai_model.generate_content(summary_prompt)
        return response.text
    except Exception as e:
        return f"Gemini error: An error occurred while generating the summary - {str(e)}"

if ticker:
    stock_data_raw = load_data(ticker, period, interval)

    if not stock_data_raw.empty and 'Close' in stock_data_raw.columns:
        st.subheader(f"📊 Price Chart for {ticker}")
        st.line_chart(stock_data_raw['Close'])

        data_with_indicators = calculate_technical_indicators(stock_data_raw)

        st.subheader("📉 Technical Indicators")
        # Plot indicators
        st.line_chart(data_with_indicators[['Close', f'SMA_{SMA_SHORT_WINDOW}', f'SMA_{SMA_LONG_WINDOW}']])
        st.line_chart(data_with_indicators[['RSI']])
        st.line_chart(data_with_indicators[['MACD', 'MACD_signal']])

        st.subheader("🧠 Gemini-Powered Summary")
        if model: # Check if model was successfully initialized
            if not data_with_indicators.empty:
                latest_indicators = data_with_indicators.iloc[-1]
                summary_text = generate_gemini_summary(model, ticker, latest_indicators)
                if "Gemini error:" in summary_text:
                    st.error(summary_text)
                else:
                    st.markdown(summary_text) # Use markdown for better formatting from Gemini
            else:
                st.warning("Not enough data to generate a summary after calculating indicators.")
        else:
            st.info("Enter a valid Gemini API key in the sidebar to get AI summaries.")

        # --- Placeholder for New Tools ---
        st.subheader("🛠️ Additional Analysis Tools")

        with st.expander("📊 Volume Analysis"):
            st.write("Detailed volume chart and analysis will be shown here.")
            # Placeholder: You could add a volume bar chart using data_with_indicators['Volume']
            if 'Volume' in data_with_indicators.columns:
                st.bar_chart(data_with_indicators['Volume'])
            else:
                st.info("Volume data not available for this selection.")

        with st.expander("📈 Volatility Insights (e.g., Bollinger Bands)"):
            st.write("Bollinger Bands and other volatility metrics will be displayed here.")
            st.info("Feature coming soon!")

        with st.expander("💰 Dividend Information"):
            st.write("Dividend history and yield will be shown here.")
            # Placeholder: stock_info = yf.Ticker(ticker).info; st.write(stock_info.get('dividendYield'))
            st.info("Feature coming soon! (May require additional data fetching)")

        with st.expander("🧾 Key Financial Ratios"):
            st.write("P/E Ratio, EPS, and other fundamental ratios will be displayed here.")
            # Placeholder: stock_info = yf.Ticker(ticker).info; st.write(stock_info.get('trailingPE'))
            st.info("Feature coming soon! (May require additional data fetching)")

        with st.expander("📰 Recent News Headlines"):
            st.write("Latest news articles related to the stock will be summarized or linked here.")
            st.info("Feature coming soon! (May require news API integration)")

        st.subheader("📥 Download Processed Data")
        csv_data = data_with_indicators.to_csv().encode('utf-8')
        st.download_button(
            label="Download CSV with Indicators",
            data=csv_data,
            file_name=f"{ticker}_data_with_indicators.csv",
            mime='text/csv',
        )
    else:
        st.info(f"Could not retrieve or process data for {ticker}. Please check the ticker symbol and selected period/interval.")
else:
    st.info("Enter a stock ticker in the sidebar to get started.")
