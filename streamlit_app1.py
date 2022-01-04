%%writefile app.py

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

# download dataframe
st.title("Stock Market Analysis")

#sidebar
st.sidebar.title("Stocks")

st.sidebar.subheader("Stock Search")

stock_search_test = st.sidebar.text_input('Stock Search')
display_cols = st.sidebar.multiselect('Display Data', ["Close","Adj Close","Low","Open","High","Volume"],default = "High")
graph_type = st.sidebar.selectbox('Graph Type', ["Line","Bar","Vega Lite","AltAir","High","Volume"])

start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')

yf.pdr_override()
tickers = ""

tickers = stock_search_test.replace(","," ")

data = yf.download(tickers, start=start_date, end=end_date)

dataframe = data[display_cols]
dataframe.reset_index(level=0, inplace=True)

# dataframe.replace('^\b+', '', regex=True, inplace=True)
st.write(dataframe)

plt.plot(dataframe["Date"],dataframe["High"])
# plt.legend(dataframe["High"])
plt.ylabel("Dollars")
plt.xlabel("Date")
plt.draw()

st.pyplot(plt)

# # machine learning starts
cf=dataframe




# cf["Date"]=pd.to_datetime(cf.Date,format="%Y-%m-%d")
cf.index=cf['Date']

# st.write(dataframe)

# plt.figure(figsize=(16,8))
# plt.plot(cf["Close"],label='Close Price history')

data=cf.sort_index(ascending=True,axis=0)
new_dataset=pd.DataFrame(index=range(0,len(cf)),columns=['Date',"High"])

for i in range(0,len(data)):
    new_dataset["Date"][i]=data['Date'][i]
    new_dataset[display_cols[0]][i]=data[display_cols[0]][i]

new_dataset["Date"].apply('str')
final_dataset=new_dataset.values

train_data=[]
valid_data=[]

#splitting the dataset
train_data_number = int((len(final_dataset)*0.8))
train_data=final_dataset[0:train_data_number,:]
valid_data=final_dataset[train_data_number:,:]

new_dataset.index=new_dataset.Date
new_dataset.drop("Date",axis=1,inplace=True)

#Feature Scaling
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(new_dataset)

# new_dataset.reset_index(level=0, inplace=True)
# new_dataset["High"] = scaled_data

# st.write(new_dataset)

x_train_data,y_train_data=[],[]

for i in range(60,len(train_data)):
    x_train_data.append(scaled_data[i-60:i,0])
    y_train_data.append(scaled_data[i,0])
    
x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data) #(len(data),60))

x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1)) #(len(data), 60, 1)

# lstm architecture
lstm_model=Sequential()
lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))

inputs_data=new_dataset[len(new_dataset)-len(valid_data)-60:].values
inputs_data=inputs_data.reshape(-1,1)
inputs_data=scaler.transform(inputs_data)

lstm_model.compile(loss='mean_squared_error',optimizer='adam')
lstm_model.fit(x_train_data,y_train_data,epochs=25,batch_size=20,verbose=2)

#Testing
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
plt.plot(train_data["High"])
plt.plot(valid_data[['High',"Predictions"]])
st.pyplot(plt)

st.write(valid_data)
