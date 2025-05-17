import subprocess

# Install required packages from requirements.txt
subprocess.call("pip install -r requirements.txt", shell=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import gcsfs
from st_files_connection import FilesConnection


import os

#secrets = st.secrets["connections_gcs"]
#secret_value = os.environ.get('connections_gcs')

# Create a GCS connection
conn = st.connection('gcs', type=FilesConnection)

# Read a file from the cloud storage
df = conn.read("gs://tokyostockexchange/stock_prices.csv", input_format="csv")

# Open the df file
st.dataframe(df)

# Read the CSV file with Dask
stock_list = conn.read("gs://tokyostockexchange/stock_list.csv", input_format="csv")
st.dataframe(stock_list)

# Create a dropdown menu
Securities_List = st.selectbox('Securities reference list: ', list(stock_list[['SecuritiesCode', 'Name']].itertuples(index=False, name=None)))


# Convert the Date column
df['Date'] = df['Date'].astype('M8[ns]')  # This is the datetime dtype in Dask

# Compute if necessary
#df = df.compute()

st.title('Tokyo Stock Exchange JPX 2017-01-04 to 2021-12-03')
user_inputs = st.text_area('Enter Stock Codes (comma-separated)', '6752, 6753, 6503')  # Example input

#st.set_option('deprecation.showPyplotGlobalUse', False)
print(st.config.get_option('deprecation.showPyplotGlobalUse'))

securities_codes = user_inputs.split(',')

plt.figure(figsize=(16, 8))

st.subheader('Open Price')
for code in securities_codes:
    data = df[df['SecuritiesCode'] == int(code)]
    data.index = data.pop('Date')
    data['Open'].plot(label=f'Securities Code: {code}')

plt.title('Open Price')
plt.legend()
st.pyplot()

# Optional: Display the dataframe
st.write(df[df['SecuritiesCode'].isin([int(code) for code in securities_codes])])

st.subheader('Volume')
for code in securities_codes:
    data = df[df['SecuritiesCode'] == int(code)]
    data.index = data.pop('Date')
    data['Volume'].plot(label=f'Securities Code: {code}')
   
plt.title('Volume')
plt.legend()
st.pyplot()

# Total traded
st.subheader('Total Traded')
for code in securities_codes:
    data = df[df['SecuritiesCode'] == int(code)]
    data.index = data.pop('Date')
    data['Total_Traded'] = data['Volume']*data['Open']
    data['Total_Traded'].plot(label=f'Securities Code: {code}')
    
plt.legend()
st.pyplot()

# Max traded days
st.subheader('Highest traded day')
highest_traded_days = []  # Initialize an empty list to store the highest traded days

for code in securities_codes:
    data = df[df['SecuritiesCode'] == int(code)]
    data.index = data.pop('Date')
    data['Total_Traded'] = data['Volume']*data['Open']
    Max_T = data.index[data['Total_Traded'].argmax()]
    highest_traded_days.append((code, Max_T))  # Store the code and highest traded day

# Display the highest traded days for all codes
for code, Max_T in highest_traded_days:
    st.write(f'Highest values traded day of {code}:' , Max_T)

# Moving Average
st.subheader('Moving Average Price')
for code in securities_codes:
    data = df[df['SecuritiesCode'] == int(code)]
    data.index = data.pop('Date')
    data['MA_50'] = data['Close'].rolling(50).mean()
    data['MA_200'] = data['Close'].rolling(200).mean()

# Create plot
    data['MA_50'].plot(title=f'Securities code {code}', figsize=(16,8))
    data['MA_200'].plot(title=f'Securities code {code}', figsize=(16,8))
    data['Close'].plot(title=f'Securities code {code}', figsize=(16,8))
    plt.legend()
    st.pyplot()
    

# Correlation
st.subheader('Correlation')

# Create an empty DataFrame to store the correlation matrix
correlation_matrix = pd.DataFrame()

for code in securities_codes:
    data = df[df['SecuritiesCode'] == int(code)]
    data.index = data.pop('Date')
    data_close = data[['Close']]
    
    # Concatenate the Close data of each code as a new column
    correlation_matrix[f'Securities Code {code}'] = data_close['Close']

# Compute the correlation matrix
corr_matrix = correlation_matrix.corr()

# Display the correlation matrix
st.write("Correlation Matrix:")
st.write(corr_matrix)

# Correlation Scatter Plot
st.subheader('Correlation Scatter Plot')

# Iterate through all pairs of securities codes
for i in range(len(securities_codes)):
    for j in range(i+1, len(securities_codes)):
        selected_data_i = df[df['SecuritiesCode'] == int(securities_codes[i])]
        selected_data_j = df[df['SecuritiesCode'] == int(securities_codes[j])]
        
        # Create a scatter plot for the correlation between securities codes i and j
        plt.figure(figsize=(8, 6))
        plt.scatter(selected_data_i['Close'], selected_data_j['Close'], alpha=0.5)
        plt.title(f'Correlation between {securities_codes[i]} and {securities_codes[j]}')
        plt.xlabel(securities_codes[i])
        plt.ylabel(securities_codes[j])
        plt.grid(True)
        st.pyplot()
        
st.subheader('Candle stick chart in November 2021')
import plotly.graph_objects as go

def Candlestick(df, Title):
    df['Color'] = ['green' if close > open else 'red' for close, open in zip(df['Close'], df['Open'])]
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                increasing_line_color='green',
                decreasing_line_color='red',
                increasing_fillcolor='green',
                decreasing_fillcolor='red',
                line=dict(width=1),
                whiskerwidth=0.2,
                opacity=0.7,
                hoverinfo="x+y+z+text",
                hovertext=df['Color'])])

    fig.update_layout(title=Title,
                      xaxis_title='Date',
                      yaxis_title='Price')

    return fig


# Call the Candlestick function for each securities code
for code in securities_codes:
    selected_data = df[df['SecuritiesCode'] == int(code)]
    selected_data.index = selected_data.pop('Date')
    selected_data = selected_data.loc['2021-11-01':'2021-12-03']

    # Use st.plotly_chart to display the Plotly figure in Streamlit
    st.plotly_chart(Candlestick(selected_data, f'Candlestick Chart for Securities Code {code}'))

# Daily return
st.subheader('Daily return')
data_coll = pd.DataFrame()

for code in securities_codes:
    data = df[df['SecuritiesCode'] == int(code)]
    data['Return'] = data['Close'] / data['Close'].shift(1) - 1
    data_coll = pd.concat([data_coll, data], axis=0)

# Loop over the unique codes in data_coll
for code in data_coll['SecuritiesCode'].unique():
    data = data_coll[data_coll['SecuritiesCode'] == code]
    plt.figure(figsize=(8, 6))
    data['Return'].hist(bins=50)
    plt.title(f'Daily Returns for Securities Code {code}')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    st.pyplot()

# Cumulative return
st.subheader('Cumulative return')
for code in securities_codes:
    data_selected = df[df['SecuritiesCode'] == int(code)]
    data_selected.index = data_selected.pop('Date')
    data_selected['Return'] = data_selected['Close'] / data_selected['Close'].shift(1) - 1
    data_selected['Cumulative_Return'] = (1 + data_selected['Return']).cumprod()
    data_selected['Cumulative_Return'].plot(label=f'Securities Code: {code}')

plt.legend()
st.pyplot()


# Create Portfolio
st.header("Portfolio")
securities_weights = {}

st.subheader('Enter Securities Weights (pls note that total portfolio weights add up to 1.0)')

for code in securities_codes:
    weight = st.number_input(f'Weight for Securities Code {code}', min_value=0.0, max_value=1.0, step=0.01)
    securities_weights[code] = weight  


# Portfolio Return
st.subheader('Portfolio Return')

# Create an empty DataFrame to store the portfolio returns
Portfolio = pd.DataFrame()

for code, weight in zip(securities_codes, securities_weights.values()):
    data = df[df['SecuritiesCode'] == int(code)]
    data.index = data.pop('Date')
    data['Return'] = data['Close'] / data['Close'].shift(1) - 1
    data['Cumulative_Return'] = (1 + data['Return']).cumprod()
    data[f'weighted_ret {code}'] = data['Cumulative_Return'] * weight

    # Add the weighted return to the Portfolio DataFrame
    Portfolio[f'weighted_ret {code}'] = data[f'weighted_ret {code}']

# Sum of Portfolio Returns
Portfolio['Portfolio_ret'] = Portfolio.sum(axis=1)

# Display the Portfolio DataFrame
st.write(Portfolio)


# Sharpe Ratio
st.subheader('Sharpe Ratio')
Portfolio = pd.DataFrame()

for code, weight in zip(securities_codes, securities_weights.values()):
    data = df[df['SecuritiesCode'] == int(code)]
    data.index = data.pop('Date')
    data['Return'] = data['Close'] / data['Close'].shift(1) - 1
    data['Cumulative_Return'] = (1 + data['Return']).cumprod()
    data[f'weighted_ret {code}'] = data['Cumulative_Return'] * weight

    # Add the weighted return to the Portfolio DataFrame
    Portfolio[f'weighted_ret {code}'] = data[f'weighted_ret {code}']

# Sum of Portfolio Returns
Portfolio['Portfolio_ret'] = Portfolio.sum(axis=1)
Portfolio['Daily Return']=Portfolio['Portfolio_ret'].pct_change(1).fillna(0)
#use np.where to handle the case where the previous value is 0
Portfolio['Daily Return'] = np.where(Portfolio['Portfolio_ret'].shift(1) == 0, 0, Portfolio['Daily Return']) 
Portfolio_daily_ret_mean = Portfolio['Daily Return'].mean()
Portfolio_daily_ret_std = Portfolio['Daily Return'].std()
Sharpe_ratio=Portfolio_daily_ret_mean / Portfolio_daily_ret_std

st.write(f'The Sharpe Ratio for the portfolio is: {Sharpe_ratio:.2f}')
st.write(Portfolio)

# Portfolio Optimization
st.subheader('Portfolio Optimization')
Stocks_com = pd.DataFrame()
for code in securities_codes:
    data = df[df['SecuritiesCode'] == int(code)]
    data.index = data.pop('Date')
    Stocks_com[f'Securities Code {code} close'] = data['Close']
    log_ret = np.log(Stocks_com/Stocks_com.shift(1))

num_ports = 15000 #Assume there are 15000 portfolio combinations of security allocation.

all_weights = np.zeros((num_ports,len(Stocks_com.columns)))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)

for ind in range(num_ports):

    # Create Random Weights
    weights = np.array(np.random.random(len(securities_codes)))

    # Rebalance Weights
    weights = weights / np.sum(weights)
    
    # Save Weights
    all_weights[ind,:] = weights

    # Expected Return
    ret_arr[ind] = np.sum((log_ret.mean() * weights) *252)

    # Expected Variance
    vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))

    # Sharpe Ratio
    sharpe_arr[ind] = ret_arr[ind]/vol_arr[ind]

    # Optimal weight distribution
    Max_Portfolio_Sharpe_Ratio = sharpe_arr.max()
    Optimal_index_point=sharpe_arr.argmax()
    Optimal_weight_distribution= all_weights[Optimal_index_point,:]
    max_sr_ret = ret_arr[Optimal_index_point]
    max_sr_vol = vol_arr[Optimal_index_point]

st.write(f'Optimal Sharpe Ratio: {Max_Portfolio_Sharpe_Ratio}')
st.write(f'Optimal weight distribution for securities code {securities_codes} is: {Optimal_weight_distribution}')
st.write(f'Optimal Portfolio Return is {max_sr_ret}')
st.write(f'Optimal Portfolio Volatility is {max_sr_vol}')

# Plotting the Curve    
st.subheader('Efficient Frontier - Optimal Curve')

# Create scatter plot
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='plasma')
fig.colorbar(scatter, label='Sharpe Ratio')
ax.set_xlabel('Volatility')
ax.set_ylabel('Return')

# Add red dot for max SR
ax.scatter(max_sr_vol, max_sr_ret, c='red', s=50, edgecolors='black')

# Display the plot in Streamlit app
st.pyplot(fig)
























