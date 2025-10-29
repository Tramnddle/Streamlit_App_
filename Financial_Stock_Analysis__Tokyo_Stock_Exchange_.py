import streamlit as st
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("streamlit").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gcsfs
from st_files_connection import FilesConnection
import plotly.graph_objects as go

import os

# -------------------------------------------------
# DATA LOADING
# -------------------------------------------------

# Create a GCS connection (must be defined in .streamlit/secrets.toml)
conn = st.connection('gcs', type=FilesConnection)

# Read data from GCS
df = conn.read("gs://tokyostockexchange/stock_prices.csv", input_format="csv", encoding="utf-8")
stock_list = conn.read("gs://tokyostockexchange/stock_list.csv", input_format="csv", encoding="utf-8")

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'], encoding="utf-8")

st.title('Tokyo Stock Exchange JPX (2017-01-04 to 2021-12-03)')

# Dropdown with (SecuritiesCode, Name)
Securities_List = st.selectbox(
    'Securities reference list:',
    list(stock_list[['SecuritiesCode', 'Name']].itertuples(index=False, name=None))
)

# Text area for user-selected securities
user_inputs = st.text_area(
    'Enter Stock Codes with comma as delimiter',
    '6752, 6753, 6503'  # default example
)

securities_codes = [c.strip() for c in user_inputs.split(',') if c.strip()]


# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------

def get_data_for_code(df, code_as_str):
    """
    Returns a copy of the subset of df for this SecuritiesCode,
    with Date moved to index.
    """
    data = df[df['SecuritiesCode'] == int(code_as_str)].copy()
    data.index = data.pop('Date')
    return data


# -------------------------------------------------
# OPEN PRICE PLOT
# -------------------------------------------------

st.subheader('Open Price')

fig_open, ax_open = plt.subplots(figsize=(16, 8))

for code in securities_codes:
    data = get_data_for_code(df, code)
    ax_open.plot(data.index, data['Open'], label=f'Securities Code: {code}')

ax_open.set_title('Open Price')
ax_open.set_xlabel('Date')
ax_open.set_ylabel('Price')
ax_open.legend()

st.pyplot(fig_open)
plt.close(fig_open)

# Optional: show underlying rows for the selected codes
st.write(df[df['SecuritiesCode'].isin([int(code) for code in securities_codes])])


# -------------------------------------------------
# VOLUME PLOT
# -------------------------------------------------

st.subheader('Volume')

fig_vol, ax_vol = plt.subplots(figsize=(16, 8))

for code in securities_codes:
    data = get_data_for_code(df, code)
    ax_vol.plot(data.index, data['Volume'], label=f'Securities Code: {code}')

ax_vol.set_title('Volume')
ax_vol.set_xlabel('Date')
ax_vol.set_ylabel('Volume')
ax_vol.legend()

st.pyplot(fig_vol)
plt.close(fig_vol)


# -------------------------------------------------
# TOTAL TRADED (Volume * Open)
# -------------------------------------------------

st.subheader('Total Traded')

fig_tt, ax_tt = plt.subplots(figsize=(16, 8))

for code in securities_codes:
    data = get_data_for_code(df, code)
    data['Total_Traded'] = data['Volume'] * data['Open']
    ax_tt.plot(data.index, data['Total_Traded'], label=f'Securities Code: {code}')

ax_tt.set_title('Total Traded (Volume × Open)')
ax_tt.set_xlabel('Date')
ax_tt.set_ylabel('Total Traded Value')
ax_tt.legend()

st.pyplot(fig_tt)
plt.close(fig_tt)


# -------------------------------------------------
# HIGHEST TRADED DAY
# -------------------------------------------------

st.subheader('Highest traded day')

highest_traded_days = []

for code in securities_codes:
    data = get_data_for_code(df, code)
    data['Total_Traded'] = data['Volume'] * data['Open']
    # idxmax on a Series gives index label where it's max
    max_day = data['Total_Traded'].idxmax()
    highest_traded_days.append((code, max_day))

for code, max_day in highest_traded_days:
    st.write(f'Highest traded value day of {code}: {max_day}')


# -------------------------------------------------
# MOVING AVERAGE (MA50 / MA200 / Close)
# -------------------------------------------------

st.subheader('Moving Average Price')

for code in securities_codes:
    data = get_data_for_code(df, code)
    data['MA_50'] = data['Close'].rolling(50).mean()
    data['MA_200'] = data['Close'].rolling(200).mean()

    fig_ma, ax_ma = plt.subplots(figsize=(16, 8))
    ax_ma.plot(data.index, data['Close'], label='Close', color='green')
    ax_ma.plot(data.index, data['MA_50'], label='MA_50', color='blue')
    ax_ma.plot(data.index, data['MA_200'], label='MA_200', color='orange')

    ax_ma.set_title(f'Securities code {code}')
    ax_ma.set_xlabel('Date')
    ax_ma.set_ylabel('Price')
    ax_ma.legend()

    st.pyplot(fig_ma)
    plt.close(fig_ma)


# -------------------------------------------------
# CORRELATION MATRIX (Close price vs Close price)
# -------------------------------------------------

st.subheader('Correlation')

# Build wide dataframe: each column = Close of one code
correlation_matrix = pd.DataFrame()

for code in securities_codes:
    data = get_data_for_code(df, code)
    correlation_matrix[f'Securities Code {code}'] = data['Close']

corr_matrix = correlation_matrix.corr()

st.write("Correlation Matrix:")
st.write(corr_matrix)


# -------------------------------------------------
# CORRELATION SCATTER PLOTS
# -------------------------------------------------

st.subheader('Correlation Scatter Plot')

for i in range(len(securities_codes)):
    for j in range(i + 1, len(securities_codes)):
        code_i = securities_codes[i]
        code_j = securities_codes[j]

        data_i = get_data_for_code(df, code_i)[['Close']].rename(columns={'Close': f'{code_i}'}).reset_index(drop=True)
        data_j = get_data_for_code(df, code_j)[['Close']].rename(columns={'Close': f'{code_j}'}).reset_index(drop=True)

        merged = pd.concat([data_i, data_j], axis=1).dropna()

        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
        ax_scatter.scatter(
            merged[f'{code_i}'],
            merged[f'{code_j}'],
            alpha=0.5
        )
        ax_scatter.set_title(f'Correlation between {code_i} and {code_j}')
        ax_scatter.set_xlabel(code_i)
        ax_scatter.set_ylabel(code_j)
        ax_scatter.grid(True)

        st.pyplot(fig_scatter)
        plt.close(fig_scatter)


# -------------------------------------------------
# CANDLESTICK (NOV 2021)
# -------------------------------------------------

st.subheader('Candlestick chart in Nov 2021')

def Candlestick(df_in, Title):
    df_local = df_in.copy()
    df_local['Color'] = [
        'green' if close > open_ else 'red'
        for close, open_ in zip(df_local['Close'], df_local['Open'])
    ]

    fig_candle = go.Figure(
        data=[
            go.Candlestick(
                x=df_local.index,
                open=df_local['Open'],
                high=df_local['High'],
                low=df_local['Low'],
                close=df_local['Close'],
                increasing_line_color='green',
                decreasing_line_color='red',
                increasing_fillcolor='green',
                decreasing_fillcolor='red',
                line=dict(width=1),
                whiskerwidth=0.2,
                opacity=0.7,
                hoverinfo="x+y+z+text",
                hovertext=df_local['Color']
            )
        ]
    )

    fig_candle.update_layout(
        title=Title,
        xaxis_title='Date',
        yaxis_title='Price'
    )

    return fig_candle


for code in securities_codes:
    sel = get_data_for_code(df, code)
    sel_nov = sel.loc['2021-11-01':'2021-12-03']
    st.plotly_chart(Candlestick(sel_nov, f'Candlestick Chart for Securities Code {code}'))


# -------------------------------------------------
# DAILY RETURN HISTOGRAMS
# -------------------------------------------------

st.subheader('Daily return')

data_coll = pd.DataFrame()

for code in securities_codes:
    data = df[df['SecuritiesCode'] == int(code)].copy()
    data['Return'] = data['Close'] / data['Close'].shift(1) - 1
    data_coll = pd.concat([data_coll, data], axis=0)

for code in data_coll['SecuritiesCode'].unique():
    sub = data_coll[data_coll['SecuritiesCode'] == int(code)].copy()

    fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
    ax_hist.hist(sub['Return'].dropna(), bins=50)
    ax_hist.set_title(f'Daily Returns for Securities Code {code}')
    ax_hist.set_xlabel('Return')
    ax_hist.set_ylabel('Frequency')
    ax_hist.grid(True)

    st.pyplot(fig_hist)
    plt.close(fig_hist)


# -------------------------------------------------
# CUMULATIVE RETURN
# -------------------------------------------------

st.subheader('Cumulative return')

fig_cum, ax_cum = plt.subplots(figsize=(16, 8))

for code in securities_codes:
    data_selected = get_data_for_code(df, code)
    data_selected['Return'] = data_selected['Close'] / data_selected['Close'].shift(1) - 1
    data_selected['Cumulative_Return'] = (1 + data_selected['Return']).cumprod()
    ax_cum.plot(
        data_selected.index,
        data_selected['Cumulative_Return'],
        label=f'Securities Code: {code}'
    )

ax_cum.set_title('Cumulative Return')
ax_cum.set_xlabel('Date')
ax_cum.set_ylabel('Cumulative Return')
ax_cum.legend()

st.pyplot(fig_cum)
plt.close(fig_cum)


# -------------------------------------------------
# PORTFOLIO DEFINITION (WEIGHTS INPUT)
# -------------------------------------------------

st.header("Portfolio")

st.subheader('Enter Securities Weights (note: total weights should sum to 1.0)')

securities_weights = {}
for code in securities_codes:
    weight = st.number_input(
        f'Weight for Securities Code {code}',
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        key=f"weight_{code}"
    )
    securities_weights[code] = weight


# -------------------------------------------------
# PORTFOLIO RETURN TABLE
# -------------------------------------------------

st.subheader('Portfolio Return')

Portfolio = pd.DataFrame()

for code, weight in securities_weights.items():
    data = get_data_for_code(df, code)
    data['Return'] = data['Close'] / data['Close'].shift(1) - 1
    data['Cumulative_Return'] = (1 + data['Return']).cumprod()
    data[f'weighted_ret_{code}'] = data['Cumulative_Return'] * weight

    Portfolio[f'weighted_ret_{code}'] = data[f'weighted_ret_{code}']

Portfolio['Portfolio_ret'] = Portfolio.sum(axis=1)

st.write(Portfolio)


# -------------------------------------------------
# SHARPE RATIO
# -------------------------------------------------

st.subheader('Sharpe Ratio')

Portfolio_SR = pd.DataFrame()

for code, weight in securities_weights.items():
    data = get_data_for_code(df, code)
    data['Return'] = data['Close'] / data['Close'].shift(1) - 1
    data['Cumulative_Return'] = (1 + data['Return']).cumprod()
    data[f'weighted_ret_{code}'] = data['Cumulative_Return'] * weight

    Portfolio_SR[f'weighted_ret_{code}'] = data[f'weighted_ret_{code}']

Portfolio_SR['Portfolio_ret'] = Portfolio_SR.sum(axis=1)

# Daily Return of the portfolio
Portfolio_SR['Daily Return'] = Portfolio_SR['Portfolio_ret'].pct_change(1).fillna(0)

# guard against div by zero
Portfolio_SR['Daily Return'] = np.where(
    Portfolio_SR['Portfolio_ret'].shift(1) == 0,
    0,
    Portfolio_SR['Daily Return']
)

portfolio_daily_ret_mean = Portfolio_SR['Daily Return'].mean()
portfolio_daily_ret_std = Portfolio_SR['Daily Return'].std()

if portfolio_daily_ret_std == 0:
    Sharpe_ratio = np.nan
else:
    Sharpe_ratio = portfolio_daily_ret_mean / portfolio_daily_ret_std

st.write(f'The Sharpe Ratio for the portfolio is: {Sharpe_ratio:.2f}')
st.write(Portfolio_SR)


# -------------------------------------------------
# PORTFOLIO OPTIMIZATION / EFFICIENT FRONTIER
# -------------------------------------------------

st.subheader('Portfolio Optimization')

# Build joint price table for log returns
Stocks_com = pd.DataFrame()
for code in securities_codes:
    data = get_data_for_code(df, code)
    Stocks_com[f'{code}_close'] = data['Close']

log_ret = np.log(Stocks_com / Stocks_com.shift(1))

num_ports = 15000  # number of random portfolios to simulate
all_weights = np.zeros((num_ports, len(Stocks_com.columns)))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)

for ind in range(num_ports):

    # random weights
    weights = np.random.random(len(Stocks_com.columns))
    weights = weights / np.sum(weights)

    all_weights[ind, :] = weights

    # expected return (annualized ~252 trading days)
    ret_arr[ind] = np.sum((log_ret.mean() * weights) * 252)

    # expected volatility
    vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))

    # Sharpe Ratio
    if vol_arr[ind] == 0:
        sharpe_arr[ind] = np.nan
    else:
        sharpe_arr[ind] = ret_arr[ind] / vol_arr[ind]

# Find optimal portfolio (max Sharpe)
Optimal_index_point = np.nanargmax(sharpe_arr)
Max_Portfolio_Sharpe_Ratio = sharpe_arr[Optimal_index_point]
Optimal_weight_distribution = all_weights[Optimal_index_point, :]
max_sr_ret = ret_arr[Optimal_index_point]
max_sr_vol = vol_arr[Optimal_index_point]

st.write(f'Optimal Sharpe Ratio: {Max_Portfolio_Sharpe_Ratio}')
st.write(f'Optimal weight distribution for securities code {securities_codes} is: {Optimal_weight_distribution}')
st.write(f'Optimal Portfolio Return is {max_sr_ret}')
st.write(f'Optimal Portfolio Volatility is {max_sr_vol}')


# Efficient Frontier Plot
st.subheader('Efficient Frontier - Optimal Curve')

fig_eff, ax_eff = plt.subplots(figsize=(12, 8))
scatter = ax_eff.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='plasma')
fig_eff.colorbar(scatter, label='Sharpe Ratio')

ax_eff.set_xlabel('Volatility')
ax_eff.set_ylabel('Return')
ax_eff.set_title('Efficient Frontier')

# Mark the optimal point
ax_eff.scatter(max_sr_vol, max_sr_ret, c='red', s=50, edgecolors='black')

st.pyplot(fig_eff)
plt.close(fig_eff)
