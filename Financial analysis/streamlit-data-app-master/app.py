import datetime
import streamlit as st
import pandas_datareader as pdr
import cufflinks as cf

APP_NAME = "Stock App!"

# Page Configuration
st.set_page_config(
    page_title=APP_NAME,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add some markdown
st.sidebar.markdown("Made with love using [Streamlit](https://streamlit.io/).")
st.sidebar.markdown("# :chart_with_upwards_trend:")

# Add app title
st.sidebar.title(APP_NAME)

# List of tickers
TICKERS = ['FB', 'AMZN', 'AAPL', 'NFLX', 'GOOGL', 'MSFT']

# Select ticker
ticker = st.sidebar.selectbox('Select ticker', sorted(TICKERS), index=0)

# Set start and end point to fetch data
start_date = st.sidebar.date_input('Start date', datetime.datetime(2021, 1, 1))
end_date = st.sidebar.date_input('End date', datetime.datetime.now().date())

# Fetch the data for specified ticker e.g. AAPL from yahoo finance
df_ticker = pdr.DataReader(ticker, 'yahoo', start_date, end_date)

st.header(f'{ticker} Stock Price')

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(df_ticker)

    

    

    
    
    
    
    
    
# Interactive data visualizations using cufflinks
# Create candlestick chart 
qf = cf.QuantFig(df_ticker, legend='top', name=ticker)


# Technical Analysis Studies can be added on demand
# Add Relative Strength Indicator (RSI) study to QuantFigure.studies
qf.add_rsi(periods=20, color='java')

# Add Bollinger Bands (BOLL) study to QuantFigure.studies
qf.add_bollinger_bands(periods=20,boll_std=2,colors=['magenta','grey'],fill=True)

# Add 'volume' study to QuantFigure.studies
qf.add_volume()

fig = qf.iplot(asFigure=True, dimensions=(800, 600))

# Render plot using plotly_chart
st.plotly_chart(fig)