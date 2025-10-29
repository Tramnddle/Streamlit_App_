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
# DATA LOADING (robust encoding handling)
# -------------------------------------------------

conn = st.connection('gcs', type=FilesConnection)

def read_csv_from_gcs(path, encoding_guess="cp932"):
    # Read file from GCS with explicit encoding to avoid UTF-8 decode errors
    with conn.fs.open(path, "rb") as f:
        return pd.read_csv(f, encoding=encoding_guess)

# read both CSVs
df = read_csv_from_gcs("gs://tokyostockexchange/stock_prices.csv", encoding_guess="cp932")
stock_list = read_csv_from_gcs("gs://tokyostockexchange/stock_list.csv", encoding_guess="cp932")

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

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
# HELPER
# -------------------------------------------------

def get_data_for_code(df, code_as_str):
    """
    Returns a copy of subset of df for this SecuritiesCode,
    with Date moved to index.
    """
    data = df[df['SecuritiesCode'] == int(code_as_str)].copy()
    # move Date to index
    data.index = data.pop('Date')
    return data

# -------------------------------------------------
# OPEN PRICE
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

# show underlying rows
st.write(df[df['SecuritiesCode'].isin([int(code) for code in securities_codes])])

# -------------------------------------------------
# VOLUME
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
# TOTAL TRADED
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
