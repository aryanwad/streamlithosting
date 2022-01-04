# All imports
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import plotly.express as px
import re
import numpy as np
import datetime as dt
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pandas.plotting import lag_plot
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

# Streamlit title
st.title("Stock Market Analysis")

# Streamlit Sidebar
st.sidebar.title("Stocks")

st.sidebar.subheader("Stock Search")

# Pick variables such as stock ticker and what we want to train on (Close, High, Open)
stock_search_test = st.sidebar.text_input('Stock Search')
display_cols = st.sidebar.multiselect('Display Data', ["Close","Adj Close","Low","Open","High","Volume"],default = "High")
graph_type = st.sidebar.selectbox('Graph Type', ["Line","Bar","Vega Lite","AltAir","High","Volume"])

# Pick start and end data for datatset
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')

# Ticker string if multiple tickers
yf.pdr_override()
tickers = ""

tickers = stock_search_test.replace(","," ")

# Download data from yahoo finance
data = yf.download(tickers, start=start_date, end=end_date)

dataframe = data[display_cols]
dataframe.reset_index(level=0, inplace=True)

# Writting dataframe to webapp
st.write(dataframe)

# Plot stock data without ml or nlp
plt.plot(dataframe["Date"],dataframe["High"])
plt.ylabel("Dollars")
plt.xlabel("Date")
plt.draw()

st.pyplot(plt)

# Machine Learning starts
cf=dataframe

# ARIMA datatset code
train_ARIMA_data, test_ARIMA_data = cf[0:int(len(cf)*0.7)], cf[int(len(cf)*0.7):]

training_ARIMA_data = train_ARIMA_data['Close'].values
test_ARIMA_data = test_ARIMA_data['Close'].values

# LSTM dataset code
cf.index=cf['Date']

data=cf.sort_index(ascending=True,axis=0)
new_dataset=pd.DataFrame(index=range(0,len(cf)),columns=['Date',"High"])

for i in range(0,len(data)):
    new_dataset["Date"][i]=data['Date'][i]
    new_dataset[display_cols[0]][i]=data[display_cols[0]][i]

new_dataset["Date"].apply('str')
final_dataset=new_dataset.values

train_data=[]
valid_data=[]

# *********************** splitting the dataset
train_data_number = int((len(final_dataset)*0.8))
train_data=final_dataset[0:train_data_number,:]
valid_data=final_dataset[train_data_number:,:]

new_dataset.index=new_dataset.Date
new_dataset.drop("Date",axis=1,inplace=True)

# ********************** Feature Scaling
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(new_dataset)

x_train_data,y_train_data=[],[]

for i in range(60,len(train_data)):
    x_train_data.append(scaled_data[i-60:i,0])
    y_train_data.append(scaled_data[i,0])
    
x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data) #(len(data),60))

x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1)) #(len(data), 60, 1)

# ARIMA architecture
history = [x for x in training_ARIMA_data]
model_predictions = []
N_test_observations = len(test_ARIMA_data)
for time_point in range(N_test_observations):
    model = ARIMA(history, order=(4,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    model_predictions.append(yhat)
    true_test_value = test_ARIMA_data[time_point]
    history.append(true_test_value)
MSE_error = mean_squared_error(test_ARIMA_data, model_predictions

# LSTM architecture
lstm_model=Sequential()
lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))

inputs_data=new_dataset[len(new_dataset)-len(valid_data)-60:].values
inputs_data=inputs_data.reshape(-1,1)
inputs_data=scaler.transform(inputs_data)

lstm_model.compile(loss='mean_squared_error',optimizer='adam')
lstm_model.fit(x_train_data,y_train_data,epochs=25,batch_size=20,verbose=2)

# Testing
X_test=[]
for i in range(60,inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i,0])
X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_closing_price=lstm_model.predict(X_test)
predicted_closing_price=scaler.inverse_transform(predicted_closing_price)

train_data=new_dataset[:train_data_number]
valid_data=new_dataset[train_data_number:]

valid_data['Predictions']=predicted_closing_price

# Visuliazation LSTM
plt.plot(train_data["High"])
plt.plot(valid_data[['High',"Predictions"]])
st.pyplot(plt)

# Visualization ARIMA
test_set_range = df[int(len(df)*0.7):].index
plt.plot(test_set_range, model_predictions, color='blue', marker='o', linestyle='dashed',label='Predicted Price')
plt.plot(test_set_range, test_data, color='red', label='Actual Price')
plt.title('Prices Prediction')
plt.xlabel('Date')
plt.ylabel('Prices')
plt.xticks(np.arange(881,1259,50), df.Date[881:1259:50])
plt.legend()
st.pyplot(plt)

# Display LSTM prediction datatset
st.write(valid_data)
