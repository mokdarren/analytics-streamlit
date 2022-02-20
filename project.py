import streamlit as st
import pandas as pd
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

#import from other files
from helpers import filedownload

def set_page_config():
    st.set_page_config(layout="wide")
    st.title('ESG Portfolio Optimiser (S&P)')

    st.markdown("""
    This app retrieves the list of the **S&P 500** (from Wikipedia) and its corresponding **stock closing price** (year-to-date)! 
    This app will allow you to remove industry codes from the universe to generate your ESG portfolio 
    * **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn, pypfopt
    * **Data source:** 
        * [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies).
        * [yfinance](https://github.com/ranaroussi/yfinance).
    """)

def set_sidebar(combined_df):
    # returns all user inputs from side bar
    st.sidebar.header('User Input Features')

    # Sidebar - Sector selection
    sorted_sector_unique = sorted( combined_df['GICS Sub-Industry'].unique() )
    selected_sector = st.sidebar.multiselect('Sector to remove', sorted_sector_unique, ['Tobacco','Casinos & Gaming','Aerospace & Defense'])

    #Parameter: maximum weight of 1 asset
    max_wt = st.sidebar.slider('Max weight (%)', 1, 100, value=20)
    #Parameter: minimum esg score
    min_esg_score = st.sidebar.slider('Minimum ESG score', 1, 100, value=80)

    return selected_sector, max_wt, min_esg_score
    
@st.cache
def load_snp_data():
    # Web scraping of S&P 500 data
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = pd.read_html(url, header = 0)
    df = html[0]
    return df

@st.cache
def load_esg_scores():
    esg_scores = pd.read_excel(ESG_SCORE_FILENAME)
    return esg_scores

def load_all_data():
    snp_data = load_snp_data()
    esg_scores = load_esg_scores()
    return pd.merge(snp_data,esg_scores,on='Symbol')

def clean_data(combined_df, selected_sector, min_esg_score):
    # Filtering data
    combined_df_filtered = combined_df[ ~combined_df['GICS Sub-Industry'].isin(selected_sector) ]
    combined_df_filtered = combined_df_filtered.drop(columns=['SEC filings','CIK','ticker_name'])
    
    # Removing tickers below ESG threshold set
    combined_df = combined_df[combined_df['esg_score']>min_esg_score]
    
    return combined_df

def display_filtered_universe(combined_df_filtered):
    esg_positive_tickers = combined_df_filtered.Symbol
    st.header('Universe')
    st.write('Data Dimension: ' + str(combined_df_filtered.shape[0]) + ' rows and ' + str(combined_df_filtered.shape[1]) + ' columns.')
    st.dataframe(combined_df_filtered)
    st.markdown(filedownload(combined_df_filtered), unsafe_allow_html=True)

# note cache causes some error if code is not ready
@st.cache 
def load_price_data(combined_df_filtered):
    data = yf.download(
            tickers = list(combined_df_filtered.Symbol),
            period = "ytd",
            interval = "1D",
            threads = True
        )
        #drop tickers that can't find data
    cleaned_adj_close = data['Adj Close'].dropna(axis=1,how='all')
    return cleaned_adj_close

def run_ef_model(cleaned_adj_close):    
    
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
    #ef.max_sharpe()
    ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    st.pyplot(fig)
    #st.write(ret_tangent, std_tangent)
    st.write(ef.clean_weights())


# Global variables
ESG_SCORE_FILENAME = "esg_scores.xlsx"

def main():
    # Main logic
    set_page_config()
    combined_df = load_all_data()
    selected_sector, max_wt , min_esg_score = set_sidebar(combined_df)
    combined_df_filtered = clean_data(combined_df, selected_sector, min_esg_score)
    display_filtered_universe(combined_df_filtered)
    if st.button('Download Price data'):
        cleaned_adj_close = load_price_data(combined_df_filtered)
    else:
        st.stop()

    run_ef_model(cleaned_adj_close)
    
main()


