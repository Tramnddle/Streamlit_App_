import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gcsfs
from st_files_connection import FilesConnection
import plotly.graph_objects as go

# -------------------------------------------------
# HELPER: robust CSV loader with encoding fallback
# -------------------------------------------------

def load_csv_from_gcs(path, encodings):
    """
    Read a CSV file from GCS as bytes, then try to decode with multiple encodings.
    Returns a pandas.DataFrame.
    """
    fs = gcsfs.GCSFileSystem()  # assumes your env is already authenticated to GCP
    last_err = None

    # read raw bytes once
    with fs.open(path, "rb") as f:
        raw_bytes = f.read()

    for enc in encodings:
        try:
            text = raw_bytes.decode(enc)
            st.caption(f"Loaded {path} using encoding: {enc}")
            return pd.read_csv(io.StringIO(text))
        except Exception as e:
            last_err = e

    # if we got here: none of the encodings worked
    raise last_err


# -------------------------------------------------
# DATA LOADING (GCS via st.connection, no Snowflake)
# -------------------------------------------------
try:
    df = load_csv_from_gcs(
        "gs://tokyostockexchange/stock_prices.csv",
        encodings=["utf-8", "cp932", "shift_jis", "cp1252", "latin1"]
    )
except Exception as e:
    st.error(f"Failed to load stock_prices.csv with any known encoding: {e}")
    st.stop()

try:
    stock_list = load_csv_from_gcs(
        "gs://tokyostockexchange/stock_list.csv",
        encodings=["utf-8", "cp932", "shift_jis", "cp1252", "latin1"]
    )
except Exception as e:
    st.error(f"Failed to load stock_list.csv with any known encoding: {e}")
    st.stop()

# make sure Date column is datetime if present
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# make sure Date column is datetime
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

st.title('Tokyo Stock Exchange JPX (2017-01-04 to 2021-12-03)')

# dropdown for reference list
if 'SecuritiesCode' in stock_list.columns and 'Name' in stock_list.columns:
    Securities_List = st.selectbox(
        'Securities reference list:',
        list(stock_list[['SecuritiesCode', 'Name']].itertuples(index=False, name=None))
    )
else:
    st.warning("stock_list.csv does not contain expected columns SecuritiesCode / Name")

# user input for selected securities
user_inputs = st.text_area(
    'Enter Stock Codes with comma as delimiter',
    '6752, 6753, 6503'
)
securities_codes = [c.strip() for c in user_inputs.split(',') if c.strip()]

# helper to slice df for one code and set Date as index
def get_data_for_code(df_full, code_str):
    # We cast to int() because SecuritiesCode looks numeric in your data
    sub = df_full[df_full['SecuritiesCode'] == int(code_str)].copy()
    if 'Date' not in sub.columns:
        st.error("Column 'Date' missing from data.")
        st.stop()
    sub.index = pd.to_datetime(sub.pop('Date'), errors='coerce')
    return sub

# -------------------------------------------------
# OPEN PRICE
# -------------------------------------------------

st.subheader('Open Price')

fig_open, ax_open = plt.subplots(figsize=(16, 8))
for code in securities_codes:
    data = get_data_for_code(df, code)
    if 'Open' not in data.columns:
        continue
    ax_open.plot(data.index, data['Open'], label=f'Securities Code: {code}')

ax_open.set_title('Open Price')
ax_open.set_xlabel('Date')
ax_open.set_ylabel('Price')
ax_open.legend()

st.pyplot(fig_open)
plt.close(fig_open)

# Show raw rows for inspection
if 'SecuritiesCode' in df.columns:
    st.write(df[df['SecuritiesCode'].isin([int(code) for code in securities_codes])])

# -------------------------------------------------
# VOLUME
# -------------------------------------------------

st.subheader('Volume')

fig_vol, ax_vol = plt.subplots(figsize=(16, 8))
for code in securities_codes:
    data = get_data_for_code(df, code)
    if 'Volume' not in data.columns:
        continue
    ax_vol.plot(data.index, data['Volume'], label=f'Securities Code: {code}')

ax_vol.set_title('Volume')
ax_vol.set_xlabel('Date')
ax_vol.set_ylabel('Volume')
ax_vol.legend()

st.pyplot(fig_vol)
plt.close(fig_vol)

# -------------------------------------------------
# TOTAL TRADED
# -------------------------------------------------

st.subheader('Total Traded')

fig_tt, ax_tt = plt.subplots(figsize=(16, 8))
highest_traded_days = []

for code in securities_codes:
    data = get_data_for_code(df, code)
    if not {'Volume', 'Open'}.issubset(data.columns):
        continue

    data['Total_Traded'] = data['Volume'] * data['Open']
    ax_tt.plot(data.index, data['Total_Traded'], label=f'Securities Code: {code}')

    if data['Total_Traded'].notna().any():
        max_day = data['Total_Traded'].idxmax()
        highest_traded_days.append((code, max_day))

ax_tt.set_title('Total Traded (Volume � Open)')
ax_tt.set_xlabel('Date')
ax_tt.set_ylabel('Total Traded Value')
ax_tt.legend()

st.pyplot(fig_tt)
plt.close(fig_tt)

st.subheader('Highest traded day')
for code, max_day in highest_traded_days:
    st.write(f'Highest traded value day of {code}: {max_day}')

# -------------------------------------------------
# MOVING AVERAGE PRICE
# -------------------------------------------------

st.subheader('Moving Average Price')

for code in securities_codes:
    data = get_data_for_code(df, code)
    if 'Close' not in data.columns:
        continue

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
# CORRELATION MATRIX
# -------------------------------------------------

st.subheader('Correlation')

correlation_matrix = pd.DataFrame()

for code in securities_codes:
    data = get_data_for_code(df, code)
    if 'Close' not in data.columns:
        continue
    # align all series by position, not timestamp
    correlation_matrix[f'Securities Code {code}'] = data['Close'].reset_index(drop=True)

if not correlation_matrix.empty:
    corr_matrix = correlation_matrix.corr()
    st.write("Correlation Matrix:")
    st.write(corr_matrix)
else:
    st.write("Not enough 'Close' price data to compute correlation.")

# -------------------------------------------------
# CORRELATION SCATTER PLOTS (PAIRWISE)
# -------------------------------------------------

st.subheader('Correlation Scatter Plot')

for i in range(len(securities_codes)):
    for j in range(i + 1, len(securities_codes)):
        code_i = securities_codes[i]
        code_j = securities_codes[j]

        data_i = get_data_for_code(df, code_i)
        data_j = get_data_for_code(df, code_j)

        if 'Close' not in data_i.columns or 'Close' not in data_j.columns:
            continue

        merged = pd.concat(
            [
                data_i['Close'].reset_index(drop=True).rename(code_i),
                data_j['Close'].reset_index(drop=True).rename(code_j)
            ],
            axis=1
        ).dropna()

        if merged.empty:
            continue

        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
        ax_scatter.scatter(
            merged[code_i],
            merged[code_j],
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
    needed_cols = {'Open', 'High', 'Low', 'Close'}
    if not needed_cols.issubset(df_local.columns):
        return None

    # Optional: bullish/bearish coloring
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
    if not {'Open', 'High', 'Low', 'Close'}.issubset(sel.columns):
        continue

    # filter to Nov 2021 range
    sel_nov = sel.loc['2021-11-01':'2021-12-03']
    fig_candle = Candlestick(sel_nov, f'Candlestick Chart for Securities Code {code}')
    if fig_candle is not None:
        st.plotly_chart(fig_candle)

# -------------------------------------------------
# DAILY RETURN HISTOGRAMS
# -------------------------------------------------

st.subheader('Daily return')

# build a combined table of all daily returns
data_coll = pd.DataFrame()

for code in securities_codes:
    tmp = df[df['SecuritiesCode'] == int(code)].copy()
    if 'Close' not in tmp.columns:
        continue
    tmp['Return'] = tmp['Close'] / tmp['Close'].shift(1) - 1
    tmp['SecuritiesCode'] = tmp['SecuritiesCode'].astype(str)
    data_coll = pd.concat([data_coll, tmp], axis=0)

for code in data_coll['SecuritiesCode'].unique():
    sub = data_coll[data_coll['SecuritiesCode'] == code].copy()
    if 'Return' not in sub.columns:
        continue

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
    if 'Close' not in data_selected.columns:
        continue

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
# PORTFOLIO INPUT (WEIGHTS)
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
    data_code = get_data_for_code(df, code)
    if 'Close' not in data_code.columns:
        continue

    data_code['Return'] = data_code['Close'] / data_code['Close'].shift(1) - 1
    data_code['Cumulative_Return'] = (1 + data_code['Return']).cumprod()
    data_code[f'weighted_ret_{code}'] = data_code['Cumulative_Return'] * weight

    Portfolio[f'weighted_ret_{code}'] = data_code[f'weighted_ret_{code}']

if not Portfolio.empty:
    Portfolio['Portfolio_ret'] = Portfolio.sum(axis=1)
    st.write(Portfolio)
else:
    st.write("No valid return data yet (maybe all weights are 0 or missing Close values).")

# -------------------------------------------------
# SHARPE RATIO
# -------------------------------------------------

st.subheader('Sharpe Ratio')

Portfolio_SR = pd.DataFrame()

for code, weight in securities_weights.items():
    data_code = get_data_for_code(df, code)
    if 'Close' not in data_code.columns:
        continue

    data_code['Return'] = data_code['Close'] / data_code['Close'].shift(1) - 1
    data_code['Cumulative_Return'] = (1 + data_code['Return']).cumprod()
    data_code[f'weighted_ret_{code}'] = data_code['Cumulative_Return'] * weight

    Portfolio_SR[f'weighted_ret_{code}'] = data_code[f'weighted_ret_{code}']

if not Portfolio_SR.empty:
    Portfolio_SR['Portfolio_ret'] = Portfolio_SR.sum(axis=1)

    Portfolio_SR['Daily Return'] = Portfolio_SR['Portfolio_ret'].pct_change(1).fillna(0)
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
else:
    st.write("Not enough data to compute Sharpe Ratio.")

# -------------------------------------------------
# PORTFOLIO OPTIMIZATION / EFFICIENT FRONTIER
# -------------------------------------------------

st.subheader('Portfolio Optimization')

Stocks_com = pd.DataFrame()
for code in securities_codes:
    data_code = get_data_for_code(df, code)
    if 'Close' not in data_code.columns:
        continue
    # reset_index(drop=True) so column vectors line up
    Stocks_com[f'{code}_close'] = data_code['Close'].reset_index(drop=True)

if Stocks_com.shape[1] >= 2:
    log_ret = np.log(Stocks_com / Stocks_com.shift(1))

    num_ports = 15000
    all_weights = np.zeros((num_ports, len(Stocks_com.columns)))
    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)

    for ind in range(num_ports):
        # random weights
        weights = np.random.random(len(Stocks_com.columns))
        weights = weights / np.sum(weights)

        all_weights[ind, :] = weights

        # annualized expected return (252 trading days)
        ret_arr[ind] = np.sum((log_ret.mean() * weights) * 252)

        # annualized vol
        vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))

        if vol_arr[ind] == 0:
            sharpe_arr[ind] = np.nan
        else:
            sharpe_arr[ind] = ret_arr[ind] / vol_arr[ind]

    # pick best Sharpe
    Optimal_index_point = np.nanargmax(sharpe_arr)
    Max_Portfolio_Sharpe_Ratio = sharpe_arr[Optimal_index_point]
    Optimal_weight_distribution = all_weights[Optimal_index_point, :]
    max_sr_ret = ret_arr[Optimal_index_point]
    max_sr_vol = vol_arr[Optimal_index_point]

    st.write(f'Optimal Sharpe Ratio: {Max_Portfolio_Sharpe_Ratio}')
    st.write(f'Optimal weight distribution for securities code {securities_codes} is: {Optimal_weight_distribution}')
    st.write(f'Optimal Portfolio Return is {max_sr_ret}')
    st.write(f'Optimal Portfolio Volatility is {max_sr_vol}')

    fig_eff, ax_eff = plt.subplots(figsize=(12, 8))
    scatter = ax_eff.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='plasma')
    fig_eff.colorbar(scatter, label='Sharpe Ratio')

    ax_eff.set_xlabel('Volatility')
    ax_eff.set_ylabel('Return')
    ax_eff.set_title('Efficient Frontier')

    # mark optimal point
    ax_eff.scatter(max_sr_vol, max_sr_ret, c='red', s=50, edgecolors='black')

    st.pyplot(fig_eff)
    plt.close(fig_eff)

else:
    st.write("Need at least 2 securities with valid Close prices to build an efficient frontier.")
