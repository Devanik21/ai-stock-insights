# 📈 Stock Market Storyteller with Gemini AI (Streamlit + TA-Lib-Free Version)

import streamlit as st
import yfinance as yf
import pandas as pd
import google.generativeai as genai
from datetime import datetime # Added for formatting news timestamps
from typing import Optional, Dict, Any
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures # For Polynomial Regression
from sklearn.pipeline import make_pipeline # For Polynomial Regression
import numpy as np # For numerical operations
 
# Page config
st.set_page_config(page_title="📈 Stock Market Storyteller",  page_icon="📈",layout="wide")
st.title("📈 Stock Market Storyteller 🚀")
st.write("Narrate your favorite stocks with technical indicators & Gemini-powered summaries! 🤖")

# --- Constants ---
SMA_SHORT_WINDOW = 20
SMA_LONG_WINDOW = 50
RSI_WINDOW = 14
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
GEMINI_MODEL_NAME = "gemini-2.0-flash" # Or "gemini-1.5-flash", "gemini-pro"

# --- Custom Theme CSS (Dark Theme) ---
custom_css = """
<style>
/* Base styles for dark theme */
body {
    color: #f0f0f0; /* Light text for dark backgrounds */
    background-color: #1a1a1a; /* Very dark background */
}

.stApp {
    background-color: #1a1a1a;
}

/* Text input and other input elements */
input, textarea, select, .stTextInput > div > div > input {
    color: #f0f0f0 !important;
    background-color: #333 !important; /* Slightly lighter dark for inputs */
    border-color: #555 !important; /* Darker border */
}

/* Streamlit elements like headers, subheaders, and text */
h1, h2, h3, h4, h5, h6, p, div, label {
    color: #f0f0f0 !important; /* Ensure text is light */
}

/* Sidebar styles */
.stSidebar {
    background-color: #222 !important; /* Darker sidebar */
    color: #f0f0f0 !important;
}

.stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6, .stSidebar p, .stSidebar div, .stSidebar label {
    color: #f0f0f0 !important;
}

/* Buttons */
.stButton>button {
    color: #f0f0f0 !important;
    background-color: #444 !important; /* Dark button background */
    border-color: #555 !important;
}

.stButton>button:hover {
    background-color: #666 !important; /* Slightly lighter on hover */
}

/* Expander styles */
.stExpander {
    border-color: #555 !important;
    background-color: #222 !important; /* Darker expander background */
}

.stExpander>label>p {
    color: #f0f0f0 !important;
}

/* Metric styles */
div[data-testid="stMetricValue"] {
    color: #f0f0f0 !important; /* Ensure metric values are visible */
}
div[data-testid="stMetricLabel"] {
    color: #bbb !important; /* Slightly dimmer label */
}
div[data-testid="stMetricDelta"] {
    color: #f0f0f0 !important; /* Ensure delta is visible */
}


/* Download button */
.stDownloadButton>button {
    color: #f0f0f0 !important;
    background-color: #444 !important;
}

/* Code blocks */
pre, code {
    background-color: #333 !important;
    color: #f0f0f0 !important;
}

/* Dataframes */
.dataframe {
    background-color: #333 !important;
    color: #f0f0f0 !important;
}
.dataframe th {
    background-color: #444 !important; /* Darker header */
    color: #f0f0f0 !important;
}
.dataframe td {
     color: #f0f0f0 !important;
}


/* Alerts (info, warning, error) - Streamlit handles these well, but ensure text is visible */
.stAlert {
    color: #f0f0f0 !important; /* Ensure text is light */
}
.stAlert > div > div > p {
     color: #f0f0f0 !important; /* Ensure text in paragraph is light */
}

/* Flashcard specific styles */
.flashcard-item {
    background: linear-gradient(145deg, #2c3e50, #1e2b37); /* Dark blue/grey gradient */
    border-radius: 10px;
    padding: 20px;
    margin-top: 10px; /* Add some space above the card within the expander */
    margin-bottom: 10px; /* Add some space below the card */
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.4);
    color: #e0e0e0; /* Light text color for definition */
}

.flashcard-item h3 {
    color: #66b2ff; /* A brighter, appealing blue for the term */
    margin-top: 0;
    margin-bottom: 12px;
    font-size: 1.5em; /* Make term slightly larger */
}

.flashcard-item p {
    color: #c5d5e5; /* Softer light color for definition text */
    font-size: 1em;
    line-height: 1.6;
}


/* Chart backgrounds - May need specific overrides depending on chart type */
/* This is harder to control with simple CSS, Streamlit's defaults might show light areas */
/* For simplicity, we rely on Streamlit's built-in dark theme handling for charts */


/* Add more specific overrides as needed */
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- Sidebar for Gemini API key ---
model: Optional[genai.GenerativeModel] = None
gemini_api_key = st.sidebar.text_input("🔑 Enter Gemini API Key", type="password")
if gemini_api_key:
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        st.sidebar.success("Gemini AI configured successfully! ✅")

        # --- AI Chat with Data Tool ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("🤖 AI Chat with Data")
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        user_prompt = st.sidebar.text_area("💬 Ask Gemini about your stock data:", key="ai_chat_input")
        if st.sidebar.button("Send 🚀", key="ai_chat_send"):
            if user_prompt and model:
                # Prepare a summary of the latest data for context
                if 'stock_data_raw' in locals() and not stock_data_raw.empty:
                    context = stock_data_raw.tail(10).to_csv()
                    chat_prompt = (
                        f"You are a helpful stock market AI assistant. 🏦 "
                        f"Here is the latest data for the selected stock (last 10 rows, CSV):\n{context}\n\n"
                        f"User question: {user_prompt}\n"
                        "Answer in a concise, clear way. If relevant, use the data provided."
                    )
                else:
                    chat_prompt = user_prompt
                try:
                    response = model.generate_content(chat_prompt)
                    st.session_state["chat_history"].append(("user", user_prompt))
                    st.session_state["chat_history"].append(("ai", response.text))
                except Exception as e:
                    st.session_state["chat_history"].append(("ai", f"Gemini error: {e}"))

        # Display chat history
        for role, msg in st.session_state["chat_history"]:
            if role == "user":
                st.sidebar.markdown(f"**You:** {msg}")
            else:
                st.sidebar.markdown(f"**Gemini:** {msg}")
        st.sidebar.markdown("---")

    except Exception as e:
        st.sidebar.error(f"Failed to configure Gemini: {e}")
        model = None
else:
    st.sidebar.info("🔑 Enter your Gemini API key to enable AI-powered summaries. 🤖")

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

ticker = st.sidebar.text_input("Enter Stock Ticker (e.g. AAPL) 🏷️", value="AAPL")
period = st.sidebar.selectbox("Select Data Period 📅", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
interval = st.sidebar.selectbox("Select Interval ⏱️", ["1d", "1wk", "1mo", "1h", "15m"], index=0) # Added more intervals

# --- Flashcard Feature ---
with st.expander("📚 Stock Market Flashcards"):
    flashcards = [
        {"term": "P/E Ratio", "definition": "Price per share divided by earnings per share. Used for valuing a company."},
        {"term": "RSI", "definition": "Relative Strength Index, a momentum indicator that measures the magnitude of recent price changes."},
        {"term": "MACD", "definition": "Moving Average Convergence Divergence, a trend-following momentum indicator."},
        {"term": "SMA", "definition": "Simple Moving Average, the unweighted mean of the previous data points."},
        {"term": "Bollinger Bands", "definition": "Volatility bands placed above and below a moving average."},
        {"term": "Market Capitalization (Market Cap)", "definition": "The total market value of a company's outstanding shares. Calculated by multiplying the current stock price by the total number of outstanding shares."},
        {"term": "Dividend Yield", "definition": "The annual dividend payment per share divided by the stock's current price, expressed as a percentage. It shows how much a company pays out in dividends relative to its stock price."},
        {"term": "Beta", "definition": "A measure of a stock's volatility in relation to the overall market (e.g., S&P 500). A beta greater than 1 indicates higher volatility than the market; less than 1 means lower volatility."},
        {"term": "Blue Chip Stocks", "definition": "Shares of large, well-established, and financially sound companies that have operated for many years and have dependable earnings, often paying dividends."},
        {"term": "Exchange Traded Fund (ETF)", "definition": "A type of investment fund and exchange-traded product, i.e., they are traded on stock exchanges. ETFs hold assets such as stocks, commodities, or bonds and generally operate with an arbitrage mechanism designed to keep it trading close to its net asset value."},
        {"term": "P/B Ratio (Price-to-Book Ratio)", "definition": "Compares a company's market capitalization to its book value. A lower P/B ratio could mean the stock is undervalued."},
        {"term": "Volume", "definition": "The number of shares or contracts traded in a security or an entire market during a given period. High volume can indicate strong interest in a security at its current price."},
    ]
    card_index = st.session_state.get('card_index', 0)
    card = flashcards[card_index % len(flashcards)]

    # Display the current (first) flashcard
    # The navigation buttons have been removed.
    # Using markdown to apply custom CSS class
    st.markdown(f"""
    <div class="flashcard-item">
        <h3>{card['term']}</h3>
        <p>{card['definition']}</p>
    </div>
    """, unsafe_allow_html=True)

# Fetch stock data
@st.cache_data
def load_data(ticker_symbol: str, data_period: str, data_interval: str) -> pd.DataFrame:
    """Fetches stock data from Yahoo Finance and flattens MultiIndex columns."""
    try:
        data = yf.download(ticker_symbol, period=data_period, interval=data_interval, progress=False)
        if data.empty:
            st.warning(f"No data found for ticker {ticker_symbol} with period {data_period} and interval {data_interval}. ⚠️")
            return pd.DataFrame()

        # Flatten MultiIndex columns if present (yfinance can return multi-level for Single Ticker)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if 'Close' not in data.columns:
            st.error(f"'Close' column not found in data for {ticker_symbol}. ❌")
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
        # news = tick.news # We removed the news section, so this can be commented out or removed
        recommendations = tick.recommendations
        major_holders = tick.major_holders
        institutional_holders = tick.institutional_holders
        calendar = tick.calendar
        return {
            "info": info,
            "recommendations": recommendations,
            "major_holders": major_holders,
            "institutional_holders": institutional_holders,
            "calendar": calendar
        }
    except Exception as e:
        st.sidebar.warning(f"Could not fetch some details for {ticker_symbol}: {e}")
        return {"info": {}, "recommendations": None, "major_holders": None, "institutional_holders": None, "calendar": None}

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and adds technical indicators to the DataFrame."""
    data_ti = df.copy()
    if 'Close' not in data_ti.columns:
        st.error("Cannot calculate indicators: 'Close' column missing. ❌")
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
    ticker_details: Dict[str, Any] = {} # Initialize

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
                st.warning("Not enough data to generate a summary after calculating indicators. ⚠️")
        else:
            st.info("🔑 Enter a valid Gemini API key in the sidebar to get AI summaries. 🤖")

        # --- Placeholder for New Tools ---
        st.subheader("🛠️ Additional Analysis Tools")

        stock_info = ticker_details.get("info", {})
        # stock_news = ticker_details.get("news", []) # Not used anymore
        recommendations_data = ticker_details.get("recommendations")
        major_holders_data = ticker_details.get("major_holders")
        institutional_holders_data = ticker_details.get("institutional_holders")
        calendar_data = ticker_details.get("calendar")

        # Helper function for formatting market cap
        def format_market_cap(cap):
            if cap is None: return "N/A"
            if cap >= 1e12: return f"${cap/1e12:.2f} T"
            if cap >= 1e9: return f"${cap/1e9:.2f} B"
            if cap >= 1e6: return f"${cap/1e6:.2f} M"
            return f"${cap}"

        # --- Consolidated Expander 1: Company Overview ---
        with st.expander("🏛️ Company Overview, Financials & ESG"):
            if stock_info:
                company_name = stock_info.get('longName', stock_info.get('shortName', ticker))
                st.subheader(f"About {company_name}")
                business_summary = stock_info.get('longBusinessSummary')
                if business_summary:
                    st.markdown(business_summary)
                else:
                    st.info(f"A detailed business summary for {company_name} is not available. ⚠️")
                st.markdown("---")

                st.subheader("Sector & Industry")
                st.write(f"**Sector:** {stock_info.get('sector', 'N/A')}")
                st.write(f"**Industry:** {stock_info.get('industry', 'N/A')}")
                st.markdown("---")

                st.subheader("Key Financial Ratios")
                col1_fin, col2_fin, col3_fin = st.columns(3)
                with col1_fin:
                    st.metric("Market Cap", format_market_cap(stock_info.get('marketCap')))
                    st.metric("Trailing P/E", f"{stock_info.get('trailingPE'):.2f}" if stock_info.get('trailingPE') else "N/A")
                    st.metric("Forward P/E", f"{stock_info.get('forwardPE'):.2f}" if stock_info.get('forwardPE') else "N/A")
                with col2_fin:
                    st.metric("Price to Sales (TTM)", f"{stock_info.get('priceToSalesTrailing12Months'):.2f}" if stock_info.get('priceToSalesTrailing12Months') else "N/A")
                    st.metric("Price to Book", f"{stock_info.get('priceToBook'):.2f}" if stock_info.get('priceToBook') else "N/A")
                    st.metric("Beta", f"{stock_info.get('beta'):.2f}" if stock_info.get('beta') else "N/A")
                with col3_fin:
                    st.metric("Enterprise Value/Revenue", f"{stock_info.get('enterpriseToRevenue'):.2f}" if stock_info.get('enterpriseToRevenue') else "N/A")
                    st.metric("Enterprise Value/EBITDA", f"{stock_info.get('enterpriseToEbitda'):.2f}" if stock_info.get('enterpriseToEbitda') else "N/A")
                    st.metric("52 Week High", f"${stock_info.get('fiftyTwoWeekHigh'):.2f}" if stock_info.get('fiftyTwoWeekHigh') else "N/A")
                st.markdown("---")

                st.subheader("ESG (Sustainability) Score")
                if 'esgScores' in stock_info and stock_info['esgScores']:
                    esg = stock_info['esgScores']
                    st.metric("ESG Total Score", esg.get('totalEsg', 'N/A'))
                    st.metric("Environment Score", esg.get('environmentScore', 'N/A'))
                    st.metric("Social Score", esg.get('socialScore', 'N/A'))
                    st.metric("Governance Score", esg.get('governanceScore', 'N/A'))
                else:
                    st.info("ESG score data not available. ⚠️")
            else:
                st.info("Company profile, sector, financial ratios, and ESG data could not be fetched as `stock_info` is unavailable. ⚠️")

        # --- Consolidated Expander 2: Ownership & Dividends ---
        with st.expander("🤝 Ownership, Dividends & Insider Activity"):
            if stock_info:
                st.subheader("Dividend Details")
                div_yield = stock_info.get('dividendYield')
                div_rate = stock_info.get('dividendRate')
                ex_div_date_ts = stock_info.get('exDividendDate')
                payout_ratio = stock_info.get('payoutRatio')

                col1_div, col2_div = st.columns(2)
                with col1_div:
                    st.metric("Dividend Yield", f"{div_yield*100:.2f}%" if div_yield else "N/A")
                    if ex_div_date_ts:
                        st.metric("Ex-Dividend Date", datetime.fromtimestamp(ex_div_date_ts).strftime('%Y-%m-%d'))
                    else:
                        st.metric("Ex-Dividend Date", "N/A")
                with col2_div:
                    st.metric("Annual Dividend Rate", f"${div_rate:.2f}" if div_rate else "N/A")
                    st.metric("Payout Ratio", f"{payout_ratio*100:.2f}%" if payout_ratio else "N/A")

                if not any([div_yield, div_rate, ex_div_date_ts, payout_ratio]):
                    st.info(f"{stock_info.get('shortName', ticker)} may not pay dividends or data is unavailable. ⚠️")
                st.markdown("---")

                st.subheader("Dividend Growth")
                if 'dividendRate' in stock_info and 'fiveYearAvgDividendYield' in stock_info:
                    st.info("5Y Dividend CAGR calculation requires historical dividend data, which is not directly available in .info. ⚠️")
                else:
                    st.info("Data for Dividend CAGR calculation not available. ⚠️")
                st.markdown("---")

                st.subheader("Interest & Ownership (Shares, Float, Short Interest)")
                col1_own, col2_own = st.columns(2)
                with col1_own:
                    st.metric("Shares Outstanding", f"{stock_info.get('sharesOutstanding', 0)/1e6:.2f}M" if stock_info.get('sharesOutstanding') else "N/A")
                    st.metric("Float", f"{stock_info.get('floatShares', 0)/1e6:.2f}M" if stock_info.get('floatShares') else "N/A")
                    st.metric("% Held by Insiders", f"{stock_info.get('heldPercentInsiders', 0)*100:.2f}%" if stock_info.get('heldPercentInsiders') is not None else "N/A")
                with col2_own:
                    st.metric("% Held by Institutions", f"{stock_info.get('heldPercentInstitutions', 0)*100:.2f}%" if stock_info.get('heldPercentInstitutions') is not None else "N/A")
                    st.metric("Short Ratio (days to cover)", f"{stock_info.get('shortRatio'):.2f}" if stock_info.get('shortRatio') else "N/A")
                    st.metric("Short % of Float", f"{stock_info.get('shortPercentOfFloat')*100:.2f}%" if stock_info.get('shortPercentOfFloat') is not None else "N/A")
                st.markdown("---")

                st.subheader("Insider Transactions")
                if 'insiderTransactions' in stock_info:
                    transactions = stock_info['insiderTransactions']
                    if transactions and isinstance(transactions, list):
                        df_insider = pd.DataFrame(transactions)
                        if not df_insider.empty:
                            cols_to_show = []
                            if 'filerName' in df_insider.columns: cols_to_show.append('filerName')
                            if 'transactionText' in df_insider.columns: cols_to_show.append('transactionText')
                            if 'shares' in df_insider.columns: cols_to_show.append('shares')
                            if 'value' in df_insider.columns: cols_to_show.append('value')
                            if 'startDate' in df_insider.columns:
                                try:
                                    df_insider['startDate'] = pd.to_datetime(df_insider['startDate']).dt.strftime('%Y-%m-%d')
                                    cols_to_show.append('startDate')
                                except Exception: pass
                            if cols_to_show: st.dataframe(df_insider[cols_to_show].head(10))
                            else: st.write(transactions)
                        else: st.info("No insider transactions data found in the expected list format. ⚠️")
                    elif transactions: st.write(transactions)
                    else: st.info("No insider transactions data available. ⚠️")
                else: st.info("Insider transactions data could not be fetched. ⚠️")
                st.markdown("---")

            st.subheader("Major Holders & Institutional Ownership")
            if major_holders_data is not None and not major_holders_data.empty:
                st.write(f"**Major Holders for {stock_info.get('shortName', ticker)}**")
                st.dataframe(major_holders_data)
            elif institutional_holders_data is not None and not institutional_holders_data.empty:
                st.write(f"**Top Institutional Holders for {stock_info.get('shortName', ticker)}**")
                st.dataframe(institutional_holders_data.head(10))
            else:
                st.info("Major & institutional holder data could not be fetched or is not available. ⚠️")
            if institutional_holders_data is not None and not institutional_holders_data.empty:
                 st.caption("Note: This shows top institutional holders. Historical changes data is not readily available via yfinance .info. ⚠️")

        # --- Consolidated Expander 3: Technical Signals & Volatility ---
        with st.expander("📈 Technical Signals & Volatility"):
            st.subheader("Volume Analysis")
            if 'Volume' in data_with_indicators.columns:
                st.write("Recent Trading Volume:")
                st.bar_chart(data_with_indicators['Volume'])
            else:
                st.info("Volume data not available for this selection. ⚠️")
            st.markdown("---")

            st.subheader(f"Volatility Insights (Bollinger Bands {SMA_SHORT_WINDOW}-day)")
            bb_cols = ['Close', f'BB_Upper_{SMA_SHORT_WINDOW}', f'BB_Middle_{SMA_SHORT_WINDOW}', f'BB_Lower_{SMA_SHORT_WINDOW}']
            if all(col in data_with_indicators.columns for col in bb_cols):
                st.line_chart(data_with_indicators[bb_cols])
                st.caption(f"The Bollinger Bands show the {SMA_SHORT_WINDOW}-day moving average (middle band) "
                           f"and two standard deviations above and below it (upper and lower bands). "
                           "They can help identify periods of high or low volatility and potential overbought/oversold conditions.")
            else:
                st.info("Bollinger Bands data could not be calculated or is not available for the selected period/interval. ⚠️")
            st.markdown("---")
            
            st.subheader("Moving Average Crossover Signal")
            if 'Close' in data_with_indicators.columns:
                short_ma = data_with_indicators[f'SMA_{SMA_SHORT_WINDOW}']
                long_ma = data_with_indicators[f'SMA_{SMA_LONG_WINDOW}']
                if not short_ma.isnull().all() and not long_ma.isnull().all():
                    signal = "Bullish 🐂" if short_ma.iloc[-1] > long_ma.iloc[-1] else "Bearish 🐻"
                    st.metric("MA Crossover Signal", signal)
                else:
                    st.info("Not enough data for MA crossover. ⚠️")
            else:
                st.info("MA crossover data not available. ⚠️")
            st.markdown("---")

            st.subheader("RSI Overbought/Oversold")
            if 'RSI' in data_with_indicators.columns:
                rsi = data_with_indicators['RSI'].iloc[-1]
                if rsi > 70: st.warning(f"RSI: {rsi:.2f} (Overbought) 🥵")
                elif rsi < 30: st.success(f"RSI: {rsi:.2f} (Oversold) 🥶")
                else: st.info(f"RSI: {rsi:.2f} (Neutral) 👍")
            else: st.info("RSI data not available. ⚠️")
            st.markdown("---")

            st.subheader("MACD Crossover Signal")
            if 'MACD' in data_with_indicators.columns and 'MACD_signal' in data_with_indicators.columns:
                macd = data_with_indicators['MACD'].iloc[-1]
                macd_signal = data_with_indicators['MACD_signal'].iloc[-1]
                signal = "Bullish 🐂" if macd > macd_signal else "Bearish 🐻"
                st.metric("MACD Signal", signal)
            else: st.info("MACD data not available. ⚠️")
            st.markdown("---")

            st.subheader("Average True Range (ATR)")
            if all(col in stock_data_raw.columns for col in ['High', 'Low', 'Close']):
                high, low, close = stock_data_raw['High'], stock_data_raw['Low'], stock_data_raw['Close']
                prev_close = close.shift(1)
                tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
                atr = tr.rolling(window=14).mean()
                st.line_chart(atr)
            else: st.info("ATR data not available. ⚠️")
            st.markdown("---")

            st.subheader("Price Channel (Highest/Lowest 20)")
            if 'Close' in stock_data_raw.columns:
                high20 = stock_data_raw['Close'].rolling(20).max()
                low20 = stock_data_raw['Close'].rolling(20).min()
                st.line_chart(pd.DataFrame({'High 20': high20, 'Low 20': low20}))
            else: st.info("Price channel data not available. ⚠️")
            st.markdown("---")

            st.subheader("Volume Spike Detector")
            if 'Volume' in stock_data_raw.columns:
                avg_vol = stock_data_raw['Volume'].rolling(20).mean()
                spikes = stock_data_raw['Volume'] > 2 * avg_vol
                st.write("Recent Volume Spikes (True = Spike):")
                st.dataframe(spikes.tail(10))
            else: st.info("Volume spike data not available. ⚠️")
            st.markdown("---")

            st.subheader(f"Price Above/Below {SMA_LONG_WINDOW}-SMA")
            if 'Close' in data_with_indicators.columns and f'SMA_{SMA_LONG_WINDOW}' in data_with_indicators.columns:
                above = data_with_indicators['Close'].iloc[-1] > data_with_indicators[f'SMA_{SMA_LONG_WINDOW}'].iloc[-1]
                st.metric(f"Price Above {SMA_LONG_WINDOW}-SMA", "Yes ✅" if above else "No ❌")
            else: st.info("SMA comparison data not available. ⚠️")
            st.markdown("---")

            st.subheader("Price Volatility Bands (10/30/60 Day Std Dev)")
            if 'Close' in stock_data_raw.columns:
                for window in [10, 30, 60]:
                    if len(stock_data_raw) >= window:
                        std = stock_data_raw['Close'].rolling(window).std().iloc[-1]
                        st.metric(f"{window}-Period Volatility (Std Dev)", f"{std:.2f}")
                    else: st.info(f"Not enough data for {window}-Period Volatility. ⚠️")
            else: st.info("Volatility bands data not available. ⚠️")

        # --- Consolidated Expander 4: Performance & Risk Analysis ---
        with st.expander("📊 Performance & Risk Analysis"):
            st.subheader("Historical Performance Summary")
            if stock_info:
                st.metric("52 Week Change", f"{stock_info.get('52WeekChange', 0)*100:.2f}%" if stock_info.get('52WeekChange') is not None else "N/A")
                st.metric("YTD Return", f"{stock_info.get('ytdReturn', 0)*100:.2f}%" if stock_info.get('ytdReturn') is not None else "N/A")
                st.metric("Previous Close", f"${stock_info.get('previousClose'):.2f}" if stock_info.get('previousClose') else "N/A")
            else: st.info("Historical performance summary data could not be fetched. ⚠️")
            st.markdown("---")

            # ... (Add other performance & risk tools here, following the pattern) ...
            # Example for Drawdown Analysis:
            st.subheader("Drawdown Analysis")
            if 'Close' in data_with_indicators.columns:
                max_close = data_with_indicators['Close'].cummax()
                drawdown = (data_with_indicators['Close'] - max_close) / max_close * 100
                st.line_chart(drawdown, height=150)
                st.caption("Shows the percentage drop from the highest close (drawdown). Useful for risk assessment.")
            else: st.info("Drawdown data not available. ⚠️")
            st.markdown("---")

            # (Continue for: Correlation with S&P 500, Price Seasonality, Simple Value Score, Momentum Score,
            # Price Volatility (Std Dev), Price Change (%), Cumulative Returns, Max Drawdown Value,
            # Sharpe Ratio, Rolling Volatility, Beta vs S&P 500, Price Gap, Price Momentum (3,6,12M),
            # Skewness & Kurtosis, Downside Deviation, Rolling Correlation, Price/Volume Correlation)

        # --- Consolidated Expander 5: Market Pulse & Company Intel ---
        with st.expander("📰 Market Pulse & Company Intel"):
            st.subheader("Analyst Recommendations")
            if recommendations_data is not None and not recommendations_data.empty:
                st.write(f"**Analyst Recommendations for {stock_info.get('shortName', ticker)} (Recent)**")
                # Display the most recent few recommendations
                recent_recs = recommendations_data.tail(5).sort_index(ascending=False)
                # st.dataframe(recent_recs[['Firm', 'To Grade', 'Action']]) # Show more details
                
                # Count recommendations for a summary
                if 'To Grade' in recent_recs.columns:
                    recommendation_counts = recent_recs['To Grade'].value_counts()
                    st.write("Recent Recommendation Summary:")
                    for grade, count in recommendation_counts.items():
                        st.markdown(f"- **{grade}**: {count}")
                else:
                    st.info("Detailed 'To Grade' column not available in recent recommendations. ⚠️")
                
                if 'recommendationKey' in stock_info:
                     st.metric("Overall Recommendation", stock_info['recommendationKey'].replace('_', ' ').title() if stock_info['recommendationKey'] else "N/A")

            elif 'recommendationKey' in stock_info: # Fallback to info if recommendations DataFrame is empty
                st.metric("Overall Recommendation", stock_info['recommendationKey'].replace('_', ' ').title() if stock_info['recommendationKey'] else "N/A")
                st.info("Summary based on overall recommendation key. Detailed recent recommendations table not available. ⚠️")
            else:
                st.info("Analyst recommendation data not available. ⚠️")
            st.markdown("---")
            st.subheader("Earnings Calendar")
            if calendar_data is not None and ((isinstance(calendar_data, pd.DataFrame) and not calendar_data.empty) or (isinstance(calendar_data, dict) and calendar_data)):
                st.write(f"**Earnings Calendar for {stock_info.get('shortName', ticker)}**")
                # yfinance calendar often returns a DataFrame with 'Earnings Date' and 'EPS Estimate' etc.
                # The structure can vary, so let's be a bit flexible
                if isinstance(calendar_data, pd.DataFrame):
                    st.dataframe(calendar_data)
                elif isinstance(calendar_data, dict) and 'Earnings Date' in calendar_data: # Sometimes it's a dict
                     earnings_df = pd.DataFrame(calendar_data['Earnings Date'], columns=['Earnings Date'])
                     if 'EPS Estimate' in calendar_data: earnings_df['EPS Estimate'] = calendar_data['EPS Estimate']
                     if 'Revenue Estimate' in calendar_data: earnings_df['Revenue Estimate'] = calendar_data['Revenue Estimate']
                     st.dataframe(earnings_df)
                else:
                     st.info("Earnings calendar data format is unexpected or limited. ⚠️")
            elif 'earningsTimestamp' in stock_info: # Fallback from .info
                earnings_date = datetime.fromtimestamp(stock_info['earningsTimestamp']).strftime('%Y-%m-%d') if stock_info.get('earningsTimestamp') else "N/A"
                st.metric("Next Earnings Date (approx.)", earnings_date)
            else:
                st.info("Earnings calendar data not available. ⚠️")
            st.markdown("---")

            st.subheader("Intraday Price Range")
            if not stock_data_raw.empty and 'High' in stock_data_raw.columns and 'Low' in stock_data_raw.columns:
                st.metric("Day High", f"${stock_data_raw['High'].iloc[-1]:.2f}")
                st.metric("Day Low", f"${stock_data_raw['Low'].iloc[-1]:.2f}")
            else:
                st.info("Intraday high/low data not available. ⚠️")

        with st.expander("📉 Drawdown Analysis"):
            # ... (This was already part of the "Performance & Risk Analysis" group in thought process,
            # so it should be moved there. For now, keeping it here as per the diff structure,
            # but ideally, it would be in the Performance/Risk section.)
            # For brevity, the rest of the "Market Pulse & Company Intel" tools are added below.
            st.markdown("---")

            st.subheader("Latest News Headlines")
            try:
                tick = yf.Ticker(ticker)
                news = tick.news
                if news:
                    st.write(f"**Latest News for {stock_info.get('shortName', ticker)}**")
                    for item in news[:10]:
                        st.markdown(f"- [{item['title']}]({item['link']})")
                        if 'providerPublishTime' in item:
                             try:
                                 publish_time = datetime.fromtimestamp(item['providerPublishTime']).strftime('%Y-%m-%d %H:%M')
                                 st.caption(f"Published: {publish_time}")
                             except Exception: pass
                        st.markdown("---")
                else: st.info("Could not fetch news headlines. 📰")
            except Exception as e: st.info(f"Error fetching news: {e} ⚠️")
            st.markdown("---")

            st.subheader("Financial Statements (Summary)")
            try:
                tick = yf.Ticker(ticker)
                st.write("**Income Statement (Last 4 Periods)**"); income_stmt = tick.income_stmt
                if income_stmt is not None and not income_stmt.empty: st.dataframe(income_stmt.T.tail(4))
                else: st.info("Income statement data not available. ⚠️")

                st.write("**Balance Sheet (Last 4 Periods)**"); balance_sheet = tick.balance_sheet
                if balance_sheet is not None and not balance_sheet.empty: st.dataframe(balance_sheet.T.tail(4))
                else: st.info("Balance sheet data not available. ⚠️")

                st.write("**Cash Flow Statement (Last 4 Periods)**"); cashflow = tick.cashflow
                if cashflow is not None and not cashflow.empty: st.dataframe(cashflow.T.tail(4))
                else: st.info("Cash flow statement data not available. ⚠️")
            except Exception as e: st.info(f"Error fetching financial statements: {e} ⚠️")
            st.markdown("---")

            st.subheader("Options Chain (Summary)")
            try:
                tick = yf.Ticker(ticker)
                options = tick.options
                if options:
                    st.write(f"**Options Chain for {stock_info.get('shortName', ticker)}**")
                    selected_date = st.selectbox("Select Expiration Date 📅", options, key="options_exp_date")
                    if selected_date:
                        opt_chain = tick.option_chain(selected_date)
                        st.write("**Calls**"); st.dataframe(opt_chain.calls.head())
                        st.write("**Puts**"); st.dataframe(opt_chain.puts.head())
                    else: st.info("No expiration dates available. ⚠️")
                else: st.info("Options data not available for this ticker. ⚠️")
            except Exception as e: st.info(f"Error fetching options data: {e} ⚠️")
            st.markdown("---")

            st.subheader("Institutional Ownership")
            if institutional_holders_data is not None and not institutional_holders_data.empty:
                st.write(f"**Top Institutional Holders for {stock_info.get('shortName', ticker)}**")
                st.dataframe(institutional_holders_data.head(10))
                st.caption("Note: This shows top institutional holders. Historical changes data is not readily available via yfinance .info. ⚠️")
            else: st.info("Institutional ownership data not available. ⚠️")

        # --- Consolidated Expander 6: Predictive & Experimental Tools ---
        with st.expander("🔮 Predictive & Experimental Tools"):
            st.subheader("Future Price Prediction (Simple)")
            if 'Close' in stock_data_raw.columns and len(stock_data_raw) >= 5: # Require at least 5 data points for a somewhat meaningful trend
                try:
                    model_type = st.radio(
                        "Select Prediction Model:",
                        ("Linear Regression", "Polynomial Regression"),
                        key="prediction_model_type",
                        horizontal=True
                    )

                    # 1. Prepare data for Linear Regression
                    df_regr = stock_data_raw[['Close']].copy()
                    # Create a numerical time index as the feature
                    df_regr['time_idx'] = np.arange(len(df_regr)) 
                    
                    X_regr = df_regr[['time_idx']] # Feature (must be 2D for sklearn)
                    y_regr = df_regr['Close']      # Target

                    # 2. Train selected model
                    model_name_display = ""
                    if model_type == "Linear Regression":
                        model_to_use = LinearRegression()
                        model_to_use.fit(X_regr, y_regr)
                        model_name_display = "Linear Regression"
                    elif model_type == "Polynomial Regression":
                        poly_degree = st.slider("Polynomial Degree:", min_value=2, max_value=5, value=3, key="poly_degree_slider")
                        model_to_use = make_pipeline(PolynomialFeatures(degree=poly_degree, include_bias=False), LinearRegression())
                        model_to_use.fit(X_regr, y_regr)
                        model_name_display = f"Polynomial Regression (Degree {poly_degree})"

                    # 3. Predict on historical data
                    historical_predictions = model_to_use.predict(X_regr)

                    # 4. Predict future prices
                    future_periods = 10 # Predict for the next 10 periods
                    last_time_idx = df_regr['time_idx'].iloc[-1]
                    # Create future time indices
                    future_time_idx_array = np.arange(last_time_idx + 1, last_time_idx + 1 + future_periods).reshape(-1, 1)
                    future_price_predictions = model_to_use.predict(future_time_idx_array)

                    # 5. Prepare data for plotting
                    # Determine future dates for the x-axis
                    last_date = stock_data_raw.index[-1]
                    # Try to infer frequency from the data's index
                    freq = pd.infer_freq(stock_data_raw.index)
                    if not freq: # Fallback if inference fails
                        freq_map = {'15m': '15T', '1h': 'H', '1d': 'D', '1wk': 'W', '1mo': 'MS'}
                        freq = freq_map.get(interval)
                        if not freq: # Ultimate fallback if interval is unusual
                            st.warning(f"Could not infer data frequency for interval '{interval}'. Using daily ('D') frequency for future dates. Predictions might be misaligned.")
                            freq = 'D' 
                    
                    future_idx_dates = pd.date_range(start=last_date, periods=future_periods + 1, freq=freq)[1:]

                    # Create a combined DataFrame for charting
                    combined_index = stock_data_raw.index.union(future_idx_dates)
                    chart_df = pd.DataFrame(index=combined_index)
                    chart_df['Actual Price'] = stock_data_raw['Close'] # Historical actuals
                    
                    # Create a continuous series for the regression line (historical fit + future forecast)
                    historical_trend_series = pd.Series(historical_predictions, index=stock_data_raw.index)
                    future_trend_series = pd.Series(future_price_predictions, index=future_idx_dates)
                    chart_df[f'{model_name_display} Trend'] = pd.concat([historical_trend_series, future_trend_series])

                    st.write(f"Predicting trend for the next {future_periods} periods using {model_name_display}.")
                    st.line_chart(chart_df[['Actual Price', f'{model_name_display} Trend']])
                    
                    st.markdown(f"""
                    **Disclaimer:** This is a basic trend extrapolation using **{model_name_display}** and is **not financial advice**. 
                    Actual prices are influenced by many complex factors not captured by this simple model. 
                    The '{model_name_display} Trend' line shows the historical fit and its extension into the future.
                    """)
                    st.markdown("""
                    **Note on Model Simplicity:**
                    The models used here (Linear and Polynomial Regression) provide basic trendlines. For more nuanced predictions, consider:
                    *   **Time Series Models (ARIMA, SARIMA):** For data with seasonality and autocorrelation.
                    *   **Machine Learning Models (e.g., Prophet, LSTMs):** For more complex patterns.
                    *   **Fundamental Analysis:** Which considers company performance, industry trends, and economic factors.
                    These models require more data, careful tuning, and a deeper understanding of time series analysis.
                    """)

                except Exception as e:
                    st.error(f"Error generating price prediction: {e}")
                    st.info("Could not generate prediction. Ensure sufficient data is available for the selected period and interval.")
            else:
                st.info("Not enough data to perform price prediction (requires at least 5 data points). Select a longer period or ensure data is available. ⚠️")
            st.markdown("---")

            st.subheader("Price/Volume Heatmap (Last 30 Periods)")
            if 'Close' in stock_data_raw.columns and 'Volume' in stock_data_raw.columns:
                heatmap_df = pd.DataFrame({
                    'Close': stock_data_raw['Close'].tail(30),
                    'Volume': stock_data_raw['Volume'].tail(30)
                })
                st.dataframe(heatmap_df.style.background_gradient(cmap='viridis'))
            else:
                st.info("Price/volume heatmap data not available. ⚠️")

        st.subheader("📥 Download Processed Data")
        csv_data = data_with_indicators.to_csv().encode('utf-8')
        st.download_button(
            label="Download CSV with Indicators",
            data=csv_data,
            file_name=f"{ticker}_data_with_indicators.csv",
            mime='text/csv',
        )
    else:
        st.info(f"Could not retrieve or process data for {ticker}. Please check the ticker symbol and selected period/interval. ⚠️")
else:
    st.info("Enter a stock ticker in the sidebar to get started. 🚀")
