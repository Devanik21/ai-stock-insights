# 📈 Stock Market Storyteller with Gemini AI (Streamlit + TA-Lib-Free Version)

import streamlit as st
import yfinance as yf
import pandas as pd
import google.generativeai as genai
from datetime import datetime # Added for formatting news timestamps
from typing import Optional, Dict, Any

# Page config
st.set_page_config(page_title="📈 Stock Market Storyteller",  page_icon="📈",layout="wide")
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

@st.cache_data
def get_ticker_details(ticker_symbol: str) -> Dict[str, Any]:
    """Fetches detailed information (like fundamentals, dividends) and news for a stock ticker."""
    try:
        tick = yf.Ticker(ticker_symbol)
        info = tick.info
        news = tick.news
        return {"info": info, "news": news}
    except Exception as e:
        # Use st.sidebar.warning for non-critical errors
        st.sidebar.warning(f"Could not fetch some details for {ticker_symbol}: {e}")
        return {"info": {}, "news": []}

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

    # Bollinger Bands
    data_ti[f'BB_Middle_{SMA_SHORT_WINDOW}'] = data_ti['Close'].rolling(window=SMA_SHORT_WINDOW).mean()
    data_ti[f'BB_StdDev_{SMA_SHORT_WINDOW}'] = data_ti['Close'].rolling(window=SMA_SHORT_WINDOW).std()
    data_ti[f'BB_Upper_{SMA_SHORT_WINDOW}'] = data_ti[f'BB_Middle_{SMA_SHORT_WINDOW}'] + (data_ti[f'BB_StdDev_{SMA_SHORT_WINDOW}'] * 2)
    data_ti[f'BB_Lower_{SMA_SHORT_WINDOW}'] = data_ti[f'BB_Middle_{SMA_SHORT_WINDOW}'] - (data_ti[f'BB_StdDev_{SMA_SHORT_WINDOW}'] * 2)
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
    ticker_details: Dict[str, Any] = {"info": {}, "news": []} # Initialize

    if not stock_data_raw.empty: # Only fetch details if primary data is good
        # Fetch additional details only once
        ticker_details = get_ticker_details(ticker)

    if not stock_data_raw.empty and 'Close' in stock_data_raw.columns:
        st.subheader(f"📊 Price & Volume Chart for {ticker}")
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

        stock_info = ticker_details.get("info", {})
        stock_news = ticker_details.get("news", [])

        with st.expander("📊 Volume Analysis"):
            if 'Volume' in data_with_indicators.columns:
                st.write("Recent Trading Volume:")
                st.bar_chart(data_with_indicators['Volume'])
            else:
                st.info("Volume data not available for this selection.")

        with st.expander("📈 Volatility Insights (e.g., Bollinger Bands)"):
            st.write(f"**Bollinger Bands ({SMA_SHORT_WINDOW}-day, 2 Std Dev)**")
            bb_cols = ['Close', f'BB_Upper_{SMA_SHORT_WINDOW}', f'BB_Middle_{SMA_SHORT_WINDOW}', f'BB_Lower_{SMA_SHORT_WINDOW}']
            if all(col in data_with_indicators.columns for col in bb_cols):
                # Ensure columns exist before trying to plot
                st.line_chart(data_with_indicators[bb_cols])
                st.caption(f"The Bollinger Bands show the {SMA_SHORT_WINDOW}-day moving average (middle band) "
                           f"and two standard deviations above and below it (upper and lower bands). "
                           "They can help identify periods of high or low volatility and potential overbought/oversold conditions.")
            else:
                st.info("Bollinger Bands data could not be calculated or is not available for the selected period/interval.")

        with st.expander("💰 Dividend Information"):
            if stock_info:
                st.write(f"**{stock_info.get('shortName', ticker)} Dividend Details**")
                div_yield = stock_info.get('dividendYield')
                div_rate = stock_info.get('dividendRate')
                ex_div_date_ts = stock_info.get('exDividendDate')
                payout_ratio = stock_info.get('payoutRatio')

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Dividend Yield", f"{div_yield*100:.2f}%" if div_yield else "N/A")
                    if ex_div_date_ts:
                        st.metric("Ex-Dividend Date", datetime.fromtimestamp(ex_div_date_ts).strftime('%Y-%m-%d'))
                    else:
                        st.metric("Ex-Dividend Date", "N/A")
                with col2:
                    st.metric("Annual Dividend Rate", f"${div_rate:.2f}" if div_rate else "N/A")
                    st.metric("Payout Ratio", f"{payout_ratio*100:.2f}%" if payout_ratio else "N/A")

                if not any([div_yield, div_rate, ex_div_date_ts, payout_ratio]):
                    st.info(f"{stock_info.get('shortName', ticker)} may not pay dividends or data is unavailable.")
            else:
                st.info("Dividend information could not be fetched.")

        with st.expander("🧾 Key Financial Ratios"):
            if stock_info:
                st.write(f"**{stock_info.get('shortName', ticker)} Financial Ratios**")
                
                # Helper to format market cap
                def format_market_cap(cap):
                    if cap is None: return "N/A"
                    if cap >= 1e12: return f"${cap/1e12:.2f} T"
                    if cap >= 1e9: return f"${cap/1e9:.2f} B"
                    if cap >= 1e6: return f"${cap/1e6:.2f} M"
                    return f"${cap}"

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Market Cap", format_market_cap(stock_info.get('marketCap')))
                    st.metric("Trailing P/E", f"{stock_info.get('trailingPE'):.2f}" if stock_info.get('trailingPE') else "N/A")
                    st.metric("Forward P/E", f"{stock_info.get('forwardPE'):.2f}" if stock_info.get('forwardPE') else "N/A")
                with col2:
                    st.metric("Price to Sales (TTM)", f"{stock_info.get('priceToSalesTrailing12Months'):.2f}" if stock_info.get('priceToSalesTrailing12Months') else "N/A")
                    st.metric("Price to Book", f"{stock_info.get('priceToBook'):.2f}" if stock_info.get('priceToBook') else "N/A")
                    st.metric("Beta", f"{stock_info.get('beta'):.2f}" if stock_info.get('beta') else "N/A")
                with col3:
                    st.metric("Enterprise Value/Revenue", f"{stock_info.get('enterpriseToRevenue'):.2f}" if stock_info.get('enterpriseToRevenue') else "N/A")
                    st.metric("Enterprise Value/EBITDA", f"{stock_info.get('enterpriseToEbitda'):.2f}" if stock_info.get('enterpriseToEbitda') else "N/A")
                    st.metric("52 Week High", f"${stock_info.get('fiftyTwoWeekHigh'):.2f}" if stock_info.get('fiftyTwoWeekHigh') else "N/A")
                
                if not stock_info: # Fallback if stock_info was empty from the start
                    st.info("Key financial ratios could not be fetched.")
            else:
                st.info("Key financial ratios could not be fetched.")

        with st.expander("🏢 Company Profile"):
            if stock_info:
                company_name = stock_info.get('longName', stock_info.get('shortName', ticker))
                st.write(f"**About {company_name}**")
                business_summary = stock_info.get('longBusinessSummary')
                if business_summary:
                    st.markdown(business_summary)
                else:
                    st.info(f"A detailed business summary for {company_name} is not available.")
            else:
                st.info("Company profile information could not be fetched.")

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
