import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yfinance as yf
import time

#portfolio optimisation module
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.cla import CLA
from matplotlib.ticker import FuncFormatter
from pypfopt import discrete_allocation
from pypfopt import black_litterman
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt import objective_functions
import pypfopt.plotting as pplt
from st_aggrid import AgGrid

#import from other files
from helpers import filedownload
from grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode, DataReturnMode 

def set_page_config():
    st.set_page_config(layout="wide")
    st.title('ESG Portfolio Optimiser (S&P)')
    
    st.markdown("""
    This app retrieves the list of the **S&P 500** (from Wikipedia) and its corresponding **stock closing price** (year-to-date)! 
    This app will allow you to remove industry codes from the universe to generate your ESG portfolio 
    * **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn, pypfopt, yfinance
    * **Data source:** 
        * [wikipedia (Tickers)](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
        * [yfinance (Price)](https://github.com/ranaroussi/yfinance)
        * [refinitiv (ESG scores)](https://www.refinitiv.com/en/products/refinitiv-workspace/download-workspace)
    """)

def set_sidebar(combined_df):
    # returns all user inputs from side bar
    st.sidebar.header('User Input Features')

    # Sidebar - Sector selection
    sorted_sector_unique = sorted( combined_df['GICS Sub-Industry'].unique() )
    selected_sector = st.sidebar.multiselect('Sector to remove', sorted_sector_unique, ['Tobacco','Casinos & Gaming','Aerospace & Defense'])

    # Model Selection 
    model = st.sidebar.selectbox("Model", ['Modern Portfolio Theory', 'Black-Litterman'])

    #Parameter: maximum weight of 1 asset
    max_wt = st.sidebar.slider('Max weight (%)', 0, 100, value=10, help="Maximum weight for each asset")/100

    #Parameter: minimum weight for 1 asset
    min_wt = st.sidebar.slider('Min weight (%)', -100, 0, value=0, help="Minimum weight for each asset")/100

    #Parameter: minimum esg score
    min_esg_score = st.sidebar.slider('Minimum ESG score (50-100)', 50, 100, value=80)
    
    #Parameter: objective function
    objective_fn = st.sidebar.selectbox("Objective Function", ['Max Sharpe', 'Min Vol'], help="Use the Critical Line Algorithm to solve for selected objective function")

    #Parameter: Risk free rate
    risk_free_rate = st.sidebar.number_input("Risk Free Rate (%)", min_value = 0.0, max_value=20.0, step=0.01, value=6.5)/100

    #Parameter: Period to download
    period = st.sidebar.selectbox("Data load period", ['ytd','1mo','3mo','6mo','1y','2y','5y','10y','max'])    

    return selected_sector, model, min_wt, max_wt, min_esg_score, objective_fn, risk_free_rate, period
    
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

def load_mcaps(combined_df_filtered):
    mcaps = pd.read_excel(MCAP_FILENAME)
    to_keep = list(combined_df_filtered.Symbol)
    df_tokeep = mcaps[mcaps["ticker"].isin(to_keep)]
    df_tokeep = df_tokeep.reset_index()

    mcaps = {}
    for index, row in df_tokeep.iterrows():
        mcaps[row['ticker']] = row['mcap']
    return mcaps

def load_all_data():
    snp_data = load_snp_data()
    esg_scores = load_esg_scores()
    return pd.merge(snp_data,esg_scores,on='Symbol').sort_values(by="Symbol").reset_index(drop=True)

def clean_data(combined_df, selected_sector, min_esg_score):
    # Filtering data
    combined_df_filtered = combined_df[ ~combined_df['GICS Sub-Industry'].isin(selected_sector) ]
    combined_df_filtered = combined_df_filtered.drop(columns=['SEC filings','CIK','ticker_name'])
    
    # Removing tickers below ESG threshold set
    combined_df_filtered = combined_df_filtered[combined_df_filtered['esg_score']>min_esg_score].dropna(axis=1,how='all') #NEW
    
    # Reset index for cleaner display
    combined_df_filtered = combined_df_filtered.reset_index(drop=True)

    return combined_df_filtered
    
def display_filtered_universe(combined_df_filtered):
    st.header('Universe')
    st.write('Data Dimension: ' + str(combined_df_filtered.shape[0]) + ' rows and ' + str(combined_df_filtered.shape[1]) + ' columns.')
    st.markdown(filedownload(combined_df_filtered, "SP500.csv","Download ticker universe as CSV"), unsafe_allow_html=True)

# note cache causes some error if code is not ready
@st.cache 
def load_price_data(combined_df_filtered, period):
    data = yf.download(
            tickers = list(combined_df_filtered.Symbol),
            period = period,
            interval = "1D",
            threads = True
        )
        #drop tickers that can't find data
    cleaned_adj_close = data['Adj Close'].dropna(axis=1,how='all')
    return cleaned_adj_close

def run_ef_model(cleaned_adj_close, weight_bounds):
    min_wt, max_wt = weight_bounds
    #Annualised return
    mu = expected_returns.mean_historical_return(cleaned_adj_close)
    #Sample var
    Sigma = risk_models.sample_cov(cleaned_adj_close)
    ef = CLA(mu, Sigma, weight_bounds=(min_wt,max_wt))

    return ef

def run_bl_model(cleaned_adj_close, mcaps, views_dict, risk_free_rate, weight_bounds, view_confidence):
    min_wt, max_wt = weight_bounds

    delta = black_litterman.market_implied_risk_aversion(cleaned_adj_close, risk_free_rate = risk_free_rate)
    cov_matrix = risk_models.sample_cov(cleaned_adj_close) # can explore other covariance methods
    prior = black_litterman.market_implied_prior_returns(mcaps, delta, cov_matrix)
    bl = BlackLittermanModel(cov_matrix, pi = prior, absolute_views=views_dict, omega="idzorek", view_confidences=view_confidence)

    rets = bl.bl_returns()
    cov = bl.bl_cov()
    ef = CLA(rets, cov, weight_bounds = (min_wt,max_wt))

    return ef

def results(ef, objective_fn, risk_free_rate):
    fig, ax = plt.subplots()
    ax = pplt.plot_efficient_frontier(ef, ef_param="risk", show_assets=True)
    if objective_fn == "Max Sharpe":
        asset_weights = ef.max_sharpe()
    elif objective_fn == "Min Vol":
        asset_weights = ef.min_volatility()
    ret_tangent, std_tangent, _ = ef.portfolio_performance(risk_free_rate = risk_free_rate)
    
    ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label=objective_fn)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.legend()
    clean_wts = {key:value for (key,value) in asset_weights.items() if value != 0}

    # Separate into 2 columns 
    col1, col2 = st.columns(2)
    col1.markdown(f"### Efficient Frontier: {objective_fn}")
    col1.pyplot(fig)

    col2.markdown("### Optimal portfolio weights:")
    fig2, ax2 = plt.subplots()
    pplt.plot_weights(clean_wts)
    col2.pyplot(fig2)
    col2.write(clean_wts)
    col2.markdown(f'Annualised Returns: {ret_tangent*100:.2f}%  \n Sigma: {std_tangent*100:.2f}%  \n Sharpe Ratio: {(ret_tangent-risk_free_rate)/std_tangent:.2f}')
    
    # For performance plotting
    return asset_weights

# simple function for performance plotting for now;
def plot_portfolio_performance(cleaned_adj_close, asset_weights, benchmark, period):
    returns = np.log(cleaned_adj_close).diff()
    for asset, weight in asset_weights.items():
        returns[asset] = returns[asset] * weight
    returns['Portfolio_Ret'] = returns.sum(axis=1, skipna=True)
    returns = returns.cumsum()
    
    benchmark_adj_close = yf.download(
                            tickers = benchmark,
                            period = period,
                            interval = "1D",
                            threads = True
                        )['Adj Close'].rename(benchmark)
    benchmark_returns = np.log(benchmark_adj_close).diff().cumsum()
    benchmark_returns.iloc[0] = 0

    returns = returns.join(benchmark_returns)

    st.markdown("### Portfolio results:")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.set_ylabel("Cumulative Return")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.plot(returns[['Portfolio_Ret', benchmark]])

    plt.xticks(rotation=45)
    plt.legend(['Optimal Portfolio', str(benchmark)])
    st.pyplot(fig)


def get_user_input_black_litterman(tickers):
    view = [0] * len(tickers)
    confidence = [0] * len(tickers)

    views_confidence_df = pd.DataFrame({"View":view,"Confidence":confidence})
    views_confidence_df.index = tickers    
    views_confidence_df.reset_index(inplace=True)
    views_confidence_df = views_confidence_df.rename(columns={'index': 'ticker'})
    st.markdown("`View` of each ticker represent your view of the specified ticker's expected return within the range [-100 to 100]%")
    st.markdown("`Confidence` of each ticker is the confidence in the expected returns specified in `View`, within the range [0 to 100]%")
    st.markdown("If you have no views, leave `view` and `confidence` as 0")
    views_confidence_df_return = AgGrid(views_confidence_df, editable=True, fit_columns_on_grid_load=True)['data']
    views_dict ={}
    for _,row in views_confidence_df_return.iterrows():
        views_dict[row['ticker']] = row['View']/100
    view_confidence = [conf/100 for conf in list(views_confidence_df_return['Confidence'])]

    #data checks for views and confience
    if ((views_confidence_df_return['View'] > 100) | (views_confidence_df_return['View'] < -100)).any():        
        raise st.error('One or more of your views is out of range [-100,100]')

    if ((views_confidence_df_return['Confidence'] > 100) | (views_confidence_df_return['Confidence'] < 0)).any():        
        raise st.error('One or more of your confidence is out of range [0,100]')

    return views_dict, view_confidence
     
# Global variables
ESG_SCORE_FILENAME = "esg_scores.xlsx"
MCAP_FILENAME = "mcap.xlsx"
BENCHMARK = "^GSPC" #S&P500 index

def main():
    # Main logic
    set_page_config()
    combined_df = load_all_data()
    selected_sector, model, min_wt, max_wt, min_esg_score, objective_fn, risk_free_rate, period = set_sidebar(combined_df)
    combined_df_filtered = clean_data(combined_df, selected_sector, min_esg_score)

    gb = GridOptionsBuilder.from_dataframe(combined_df_filtered)
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
    gb.configure_selection(selection_mode='multiple', use_checkbox=True, pre_selected_rows = list(range(len(combined_df_filtered))), )
    grid_options = gb.build()
    grid_response = AgGrid(combined_df_filtered, gridOptions=grid_options, data_return_mode="AS_INPUT", update_mode="SELECTION_CHANGED")

    grid_data = grid_response['data']
    selected = pd.DataFrame(grid_response['selected_rows'])

    print('-'*30)
    display_filtered_universe(selected)
    combined_df_filtered = selected
    
    if model == "Black-Litterman":
        st.markdown("## Black Litterman Views")
        tickers = list(combined_df_filtered.Symbol)
        views_dict, view_confidence = get_user_input_black_litterman(tickers)  

    if len(combined_df_filtered) > 100:
        st.markdown(f'Too many tickers in universe for model to solve. <span style="color:red">Current: {len(combined_df_filtered)} Maximum: 100 </span>', unsafe_allow_html=True)
    elif st.button(f'Load Price data for {len(combined_df_filtered)} tickers'):
        cleaned_adj_close = load_price_data(combined_df_filtered, period)
        st.markdown(filedownload(cleaned_adj_close, "adj_close.csv","Download price data as CSV", index=True), unsafe_allow_html=True)
    else:
        st.stop()

    if model == "Black-Litterman":
        mcaps = load_mcaps(combined_df_filtered)
        ef = run_bl_model(cleaned_adj_close, mcaps=mcaps, views_dict=views_dict, risk_free_rate=risk_free_rate, weight_bounds=(min_wt,max_wt), view_confidence=view_confidence)
        asset_weights = results(ef, objective_fn = objective_fn, risk_free_rate=risk_free_rate)
    else:
        ef = run_ef_model(cleaned_adj_close, weight_bounds=(min_wt,max_wt))
        asset_weights = results(ef, objective_fn = objective_fn, risk_free_rate=risk_free_rate)
    # plotting
    plot_portfolio_performance(cleaned_adj_close, asset_weights, BENCHMARK, period)
    
main()


