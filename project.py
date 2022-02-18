import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yfinance as yf
import time
from contextlib import contextmanager, redirect_stdout
from io import StringIO

#portfolio optimisation module
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.cla import CLA
from matplotlib.ticker import FuncFormatter
from pypfopt import discrete_allocation
import pypfopt.plotting as pplt

st.set_page_config(layout="wide")
st.title('ESG Portfolio Optimiser (S&P)')

st.markdown("""
This app retrieves the list of the **S&P 500** (from Wikipedia) and its corresponding **stock closing price** (year-to-date)! 
This app will allow you to remove industry codes from the universe to generate your ESG positive 
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn
* **Data source:** 
    * [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies).
    * [yfinance](https://github.com/ranaroussi/yfinance).
""")

st.sidebar.header('User Input Features')

# Web scraping of S&P 500 data
#
@st.cache
def load_data():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = pd.read_html(url, header = 0)
    df = html[0]
    return df

df = load_data()

esg_scores = pd.read_excel("esg_scores.xlsx")
combined_df = pd.merge(df,esg_scores,on='Symbol')

# Sidebar - Sector selection
sorted_sector_unique = sorted( combined_df['GICS Sub-Industry'].unique() )
selected_sector = st.sidebar.multiselect('Sector to remove', sorted_sector_unique, ['Tobacco','Casinos & Gaming','Aerospace & Defense'])

# Filtering data
combined_df_filtered = combined_df[ ~combined_df['GICS Sub-Industry'].isin(selected_sector) ]
combined_df_filtered = combined_df_filtered.drop(columns=['SEC filings','CIK','ticker_name'])

# Filter for esg companies
min_esg_score = st.sidebar.slider('Minimum ESG score', 1, 100, value=90)
combined_df = combined_df[combined_df['esg_score']>min_esg_score]
esg_positive_tickers = combined_df.Symbol

#Parameter: maximum weight of 1 asset
max_wt = st.sidebar.slider('Max weight (%)', 1, 100, value=90)

st.header('Universe')
st.write('Data Dimension: ' + str(combined_df.shape[0]) + ' rows and ' + str(combined_df.shape[1]) + ' columns.')
st.dataframe(combined_df)

# Download S&P500 data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="SP500.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(combined_df_filtered), unsafe_allow_html=True)

# Plot Closing Price of Query Symbol
def price_plot(symbol):
  df = pd.DataFrame(data[symbol].Close)
  df['Date'] = df.index
  plt.fill_between(df.Date, df.Close, color='skyblue', alpha=0.3)
  plt.plot(df.Date, df.Close, color='skyblue', alpha=0.8)
  plt.xticks(rotation=90)
  plt.title(symbol, fontweight='bold')
  plt.xlabel('Date', fontweight='bold')
  plt.ylabel('Closing Price', fontweight='bold')
  return st.pyplot()

# note cache causes some error if code is not ready
# @st.cache 
def load_price_data():
    data = yf.download(
            tickers = list(esg_positive_tickers),
            period = "ytd",
            interval = "1D",
            threads = True
        )
    return data

if st.button('Download Price data'):
    data = load_price_data()


#drop tickers that can't find data
cleaned_adj_close = data['Adj Close'].dropna(axis=1,how='all')

#Annualised return
mu = expected_returns.mean_historical_return(cleaned_adj_close)
#Sample var
Sigma = risk_models.sample_cov(cleaned_adj_close)

#Max Sharpe Ratio - Tangent to the EF
#ef = EfficientFrontier(mu, Sigma, weight_bounds=(-1,1)) #weight bounds do not allow shorting of stocks
ef = CLA(mu, Sigma)
#ef.add_constraint(lambda x : x <= max_wt/100)

fig, ax = plt.subplots()
risk_free_rate = 0.0065

ax = pplt.plot_efficient_frontier(ef, show_assets=True)

ret_tangent, std_tangent, _ = ef.portfolio_performance(verbose=True, risk_free_rate = risk_free_rate)
ef.max_sharpe()
ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
st.pyplot(fig)

st.write(ef.clean_weights())
