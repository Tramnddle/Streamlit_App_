import streamlit as st
import numpy as np
import pandas as pd
import gcsfs, io
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from google.oauth2 import service_account

# ========================
# Auth & data helpers
# ========================
@st.cache_resource
def get_fs():
    cfg = dict(st.secrets["connections"]["gcs"])
    creds = service_account.Credentials.from_service_account_info(
        cfg,
        scopes=["https://www.googleapis.com/auth/devstorage.read_write"]
    )
    return gcsfs.GCSFileSystem(token=creds)

def load_csv_from_gcs(path, encodings=("utf-8", "cp932", "shift_jis", "cp1252", "latin1")):
    fs = get_fs()
    last_err = None
    with fs.open(path, "rb") as f:
        raw_bytes = f.read()
    for enc in encodings:
        try:
            return pd.read_csv(io.StringIO(raw_bytes.decode(enc)))
        except Exception as e:
            last_err = e
    raise last_err

def get_data_for_code(df_full, code_str):
    sub = df_full[df_full['SecuritiesCode'] == int(code_str)].copy()
    if 'Date' not in sub.columns:
        st.error("Column 'Date' missing from data.")
        st.stop()
    sub.index = pd.to_datetime(sub.pop('Date'), errors='coerce')
    return sub

# ========================
# Load data once
# ========================
st.title('Tokyo Stock Exchange JPX (2017-01-04 to 2021-12-03)')

try:
    df = load_csv_from_gcs("gs://tokyostockexchange/stock_prices.csv")
except Exception as e:
    st.error(f"Failed to load stock_prices.csv with any known encoding: {e}")
    st.stop()

try:
    stock_list = load_csv_from_gcs("gs://tokyostockexchange/stock_list.csv")
except Exception as e:
    st.error(f"Failed to load stock_list.csv with any known encoding: {e}")
    st.stop()

# Normalize dates (both, if present)
for _df in (df, stock_list):
    if 'Date' in _df.columns:
        _df['Date'] = pd.to_datetime(_df['Date'], errors='coerce')

# ============================================================
# STEP 1 — TICKER SELECTION (gated by form_submit_button)
# ============================================================
st.header("Step 1 — Select securities")

with st.form("select_codes_form", clear_on_submit=False):
    # Optional reference dropdown (informational)
    if {'SecuritiesCode', 'Name'}.issubset(stock_list.columns):
        st.caption("Reference list (SecuritiesCode, Name)")
        st.dataframe(stock_list[['SecuritiesCode', 'Name']].head(20))
    user_inputs = st.text_area('Enter Stock Codes (comma-separated)', '6752, 6753, 6503')
    submitted_codes = st.form_submit_button("Load series")

if submitted_codes:
    codes = [c.strip() for c in user_inputs.split(',') if c.strip()]
    st.session_state['securities_codes'] = codes

# Guard: stop until Step 1 is completed
if 'securities_codes' not in st.session_state or not st.session_state['securities_codes']:
    st.info("Enter the securities codes above and click **Load series** to continue.")
    st.stop()

securities_codes = st.session_state['securities_codes']

# ========================
# Plots & analytics (run only after Step 1)
# ========================

st.subheader('Open Price')
fig_open, ax_open = plt.subplots(figsize=(16, 8))
for code in securities_codes:
    data = get_data_for_code(df, code)
    if 'Open' in data.columns:
        ax_open.plot(data.index, data['Open'], label=f'{code}')
ax_open.set_title('Open Price'); ax_open.set_xlabel('Date'); ax_open.set_ylabel('Price'); ax_open.legend()
st.pyplot(fig_open); plt.close(fig_open)

st.subheader('Volume')
fig_vol, ax_vol = plt.subplots(figsize=(16, 8))
for code in securities_codes:
    data = get_data_for_code(df, code)
    if 'Volume' in data.columns:
        ax_vol.plot(data.index, data['Volume'], label=f'{code}')
ax_vol.set_title('Volume'); ax_vol.set_xlabel('Date'); ax_vol.set_ylabel('Volume'); ax_vol.legend()
st.pyplot(fig_vol); plt.close(fig_vol)

st.subheader('Total Traded')
fig_tt, ax_tt = plt.subplots(figsize=(16, 8))
highest_traded_days = []
for code in securities_codes:
    data = get_data_for_code(df, code)
    if {'Volume', 'Open'}.issubset(data.columns):
        data['Total_Traded'] = data['Volume'] * data['Open']
        ax_tt.plot(data.index, data['Total_Traded'], label=f'{code}')
        if data['Total_Traded'].notna().any():
            highest_traded_days.append((code, data['Total_Traded'].idxmax()))
ax_tt.set_title('Total Traded (Volume × Open)'); ax_tt.set_xlabel('Date'); ax_tt.set_ylabel('Total Traded'); ax_tt.legend()
st.pyplot(fig_tt); plt.close(fig_tt)
st.caption("Highest traded day:")
for code, max_day in highest_traded_days:
    st.write(f"{code}: {max_day.date()}")

st.subheader('Moving Average Price')
for code in securities_codes:
    data = get_data_for_code(df, code)
    if 'Close' in data.columns:
        data['MA_50'] = data['Close'].rolling(50).mean()
        data['MA_200'] = data['Close'].rolling(200).mean()
        fig_ma, ax_ma = plt.subplots(figsize=(16, 8))
        ax_ma.plot(data.index, data['Close'], label='Close')
        ax_ma.plot(data.index, data['MA_50'], label='MA_50')
        ax_ma.plot(data.index, data['MA_200'], label='MA_200')
        ax_ma.set_title(f'{code}'); ax_ma.set_xlabel('Date'); ax_ma.set_ylabel('Price'); ax_ma.legend()
        st.pyplot(fig_ma); plt.close(fig_ma)

st.subheader('Correlation')
corr_df = pd.DataFrame()
for code in securities_codes:
    d = get_data_for_code(df, code)
    if 'Close' in d.columns:
        corr_df[f'{code}'] = d['Close'].reset_index(drop=True)
if not corr_df.empty:
    st.write(corr_df.corr())
else:
    st.write("Not enough 'Close' data to compute correlation.")

st.subheader('Correlation Scatter Plot')
for i in range(len(securities_codes)):
    for j in range(i + 1, len(securities_codes)):
        ci, cj = securities_codes[i], securities_codes[j]
        di, dj = get_data_for_code(df, ci), get_data_for_code(df, cj)
        if 'Close' in di.columns and 'Close' in dj.columns:
            merged = pd.concat(
                [di['Close'].reset_index(drop=True).rename(ci),
                 dj['Close'].reset_index(drop=True).rename(cj)], axis=1
            ).dropna()
            if merged.empty: continue
            fig_sc, ax_sc = plt.subplots(figsize=(8, 6))
            ax_sc.scatter(merged[ci], merged[cj], alpha=0.5)
            ax_sc.set_title(f'{ci} vs {cj}'); ax_sc.set_xlabel(ci); ax_sc.set_ylabel(cj); ax_sc.grid(True)
            st.pyplot(fig_sc); plt.close(fig_sc)

st.subheader('Candlestick chart in Nov 2021')
def Candlestick(df_in, Title):
    need = {'Open','High','Low','Close'}
    if not need.issubset(df_in.columns): return None
    fig = go.Figure(data=[go.Candlestick(
        x=df_in.index, open=df_in['Open'], high=df_in['High'],
        low=df_in['Low'], close=df_in['Close'],
        increasing_line_color='green', decreasing_line_color='red',
        increasing_fillcolor='green', decreasing_fillcolor='red',
        line=dict(width=1), whiskerwidth=0.2, opacity=0.7
    )])
    fig.update_layout(title=Title, xaxis_title='Date', yaxis_title='Price')
    return fig

for code in securities_codes:
    sel = get_data_for_code(df, code)
    if {'Open','High','Low','Close'}.issubset(sel.columns):
        sel_nov = sel.loc['2021-11-01':'2021-12-03']
        fig_candle = Candlestick(sel_nov, f'{code}')
        if fig_candle: st.plotly_chart(fig_candle)

st.subheader('Daily return')
data_coll = pd.DataFrame()
for code in securities_codes:
    tmp = df[df['SecuritiesCode'] == int(code)].copy()
    if 'Close' in tmp.columns:
        tmp['Return'] = tmp['Close'] / tmp['Close'].shift(1) - 1
        tmp['SecuritiesCode'] = tmp['SecuritiesCode'].astype(str)
        data_coll = pd.concat([data_coll, tmp], axis=0)
for code in data_coll['SecuritiesCode'].unique():
    sub = data_coll[data_coll['SecuritiesCode'] == code]
    if 'Return' in sub.columns:
        fig_h, ax_h = plt.subplots(figsize=(8, 6))
        ax_h.hist(sub['Return'].dropna(), bins=50)
        ax_h.set_title(f'Returns {code}'); ax_h.set_xlabel('Return'); ax_h.set_ylabel('Freq'); ax_h.grid(True)
        st.pyplot(fig_h); plt.close(fig_h)

st.subheader('Cumulative return')
fig_cum, ax_cum = plt.subplots(figsize=(16, 8))
for code in securities_codes:
    ds = get_data_for_code(df, code)
    if 'Close' in ds.columns:
        ds['Return'] = ds['Close'] / ds['Close'].shift(1) - 1
        ds['Cumulative_Return'] = (1 + ds['Return']).cumprod()
        ax_cum.plot(ds.index, ds['Cumulative_Return'], label=f'{code}')
ax_cum.set_title('Cumulative Return'); ax_cum.set_xlabel('Date'); ax_cum.set_ylabel('Cumulative Return'); ax_cum.legend()
st.pyplot(fig_cum); plt.close(fig_cum)

# ============================================================
# STEP 2 — WEIGHTS (also gated by form)
# ============================================================
st.header("Step 2 — Portfolio weights")
with st.form("weights_form", clear_on_submit=False):
    st.caption("Weights should sum to 1.0")
    weights = {}
    for code in securities_codes:
        weights[code] = st.number_input(
            f'Weight for {code}', min_value=0.0, max_value=1.0, step=0.01, key=f"weight_{code}"
        )
    submitted_weights = st.form_submit_button("Calculate portfolio")

if submitted_weights:
    st.session_state['weights'] = weights

# Guard: stop here until weights are submitted
if 'weights' not in st.session_state:
    st.info("Enter weights and click **Calculate portfolio** to continue.")
    st.stop()

securities_weights = st.session_state['weights']
total_w = sum(securities_weights.values())
if not (0.999 <= total_w <= 1.001):
    st.error(f"Weights must sum to 1.0 (current sum = {total_w:.3f}). Adjust and resubmit.")
    st.stop()

# ========================
# Portfolio analytics (run only after Step 2)
# ========================
st.subheader('Portfolio Return')
Portfolio = pd.DataFrame()
for code, weight in securities_weights.items():
    data_code = get_data_for_code(df, code)
    if 'Close' not in data_code.columns: continue
    data_code['Return'] = data_code['Close'] / data_code['Close'].shift(1) - 1
    data_code['Cumulative_Return'] = (1 + data_code['Return']).cumprod()
    Portfolio[f'weighted_ret_{code}'] = data_code['Cumulative_Return'] * weight

if not Portfolio.empty:
    Portfolio['Portfolio_ret'] = Portfolio.sum(axis=1)
    st.dataframe(Portfolio)
else:
    st.write("No valid return data.")

st.subheader('Sharpe Ratio')
Portfolio_SR = Portfolio.copy()
if 'Portfolio_ret' in Portfolio_SR.columns:
    Portfolio_SR['Daily Return'] = Portfolio_SR['Portfolio_ret'].pct_change(1).fillna(0)
    Portfolio_SR['Daily Return'] = np.where(
        Portfolio_SR['Portfolio_ret'].shift(1) == 0, 0, Portfolio_SR['Daily Return']
    )
    std = Portfolio_SR['Daily Return'].std()
    sharpe = np.nan if std == 0 else Portfolio_SR['Daily Return'].mean() / std
    st.write(f'The Sharpe Ratio for the portfolio is: {sharpe:.2f}')
    st.dataframe(Portfolio_SR)
else:
    st.write("Not enough data to compute Sharpe Ratio.")

st.subheader('Portfolio Optimization (Efficient Frontier)')

# Build close matrix
Stocks_com = pd.DataFrame()
for code in securities_codes:
    data_code = get_data_for_code(df, code)
    if 'Close' in data_code.columns:
        Stocks_com[f'{code}_close'] = data_code['Close'].reset_index(drop=True)

if Stocks_com.shape[1] >= 2:
    log_ret = np.log(Stocks_com / Stocks_com.shift(1)).dropna()
    if log_ret.empty:
        st.warning("Not enough overlapping data to optimize.")
        st.stop()

    np.random.seed(42)  # reproducible samples
    n_assets = Stocks_com.shape[1]
    num_ports = 15000

    all_weights = np.zeros((num_ports, n_assets))
    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.full(num_ports, np.nan)

    mu = log_ret.mean() * 252
    cov = log_ret.cov() * 252

    for i in range(num_ports):
        w = np.random.random(n_assets)
        w /= w.sum()
        all_weights[i] = w

        port_ret = np.sum(mu.values * w)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov.values, w)))
        ret_arr[i] = port_ret
        vol_arr[i] = port_vol
        if port_vol > 0:
            sharpe_arr[i] = port_ret / port_vol

    if np.all(np.isnan(sharpe_arr)):
        st.warning("Could not compute a valid Sharpe ratio (all NaN).")
        st.stop()

    idx = np.nanargmax(sharpe_arr)
    opt_w = all_weights[idx].copy()

    # numerical hygiene: clip and renormalize
    opt_w = np.clip(opt_w, 0, 1)
    opt_w /= opt_w.sum()

    max_sr, max_ret, max_vol = sharpe_arr[idx], ret_arr[idx], vol_arr[idx]

    # Pretty output as a table
    tickers = [c for c in securities_codes]
    weights_df = pd.DataFrame({"Security": tickers, "Optimal Weight": opt_w})
    weights_df["Optimal Weight"] = (weights_df["Optimal Weight"] * 100).round(2)
    st.write(f"**Optimal Sharpe Ratio:** {max_sr:.4f}")
    st.write(f"**Optimal Portfolio Return:** {max_ret:.6f}")
    st.write(f"**Optimal Portfolio Volatility:** {max_vol:.6f}")
    st.dataframe(weights_df, use_container_width=True)

    # Efficient frontier plot
    import matplotlib.pyplot as plt
    fig_eff, ax_eff = plt.subplots(figsize=(12, 8))
    sc = ax_eff.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='plasma')
    fig_eff.colorbar(sc, label='Sharpe Ratio')
    ax_eff.set_xlabel('Volatility')
    ax_eff.set_ylabel('Return')
    ax_eff.set_title('Efficient Frontier')

    # mark optimum
    ax_eff.scatter(max_vol, max_ret, c='red', s=60, edgecolors='black', zorder=3)
    st.pyplot(fig_eff)
    plt.close(fig_eff)
else:
    st.info("Need at least 2 securities with valid Close prices to build an efficient frontier.")

