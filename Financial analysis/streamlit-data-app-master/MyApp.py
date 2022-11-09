# importer les bibliothèques nécessaires

import datetime
import streamlit as st
import pandas_datareader as pdr
import cufflinks as cf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import requests, json
import seaborn as sb
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from itertools import cycle

APP_NAME = "Cryptocurrency Candlestick Dashboard"

# Page Configuration
st.set_page_config(
    page_title=APP_NAME,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add some markdown
#st.sidebar.markdown("Enjoy vizualisation!!!")


# Add app title
st.sidebar.title(APP_NAME)

# List of tickers
TICKERS = ['BTC-USD','ETH-USD','BNB-USD','XRP-USD','ADA-USD',
           'SOL-USD','MATIC-USD','DOT-USD','TRX-USD','UNI-USD',
           'ATOM-USD','LINK-USD','ETC-USD','BCH-USD','AAVE-USD']

# Select ticker
ticker = st.sidebar.selectbox('Select ticker', sorted(TICKERS), index=0)

# Set start and end point to fetch data
start_date = st.sidebar.date_input('Start date', datetime.datetime(2018, 1, 1))
end_date = st.sidebar.date_input('End date', datetime.datetime.now().date())

# Fetch the data for specified ticker e.g. AAPL from yahoo finance
df_ticker = pdr.DataReader(ticker, 'yahoo', start_date, end_date)

st.header(f'{ticker} Price evolution')

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(df_ticker)
    
    
# Interactive data visualizations using cufflinks
# Create candlestick chart 
qf = cf.QuantFig(df_ticker, legend='top', name=ticker,down_color='red',up_color='green')


# Technical Analysis Studies can be added on demand
# Add Relative Strength Indicator (RSI) study to QuantFigure.studies
qf.add_rsi(periods=20, color='black')
# Add MACD
qf.add_macd(color = ['red', 'black'])
# Add 'volume' study to QuantFigure.studies
#qf.add_volume()



qf.add_annotations([
                               dict(text = "RSI",
                                    font = dict(size = 15, color ='white'),
                                    showarrow=False,
                                    x = 0.5, y = 0.38,
                                    xref = 'paper', yref = "paper",
                                    align = "center",textangle=0),
                              
                               dict(text = "MACD",
                                    font = dict(size = 15, color ='white'),
                                    showarrow=False,
                                    x = 0.5, y = 0.14,
                                    xref = 'paper', yref = "paper",
                                    align = "center",textangle=0),
                               ])

fig = qf.iplot(asFigure=True, dimensions=(1100, 1000))

# Render plot using plotly_chart
st.plotly_chart(fig)

#########################################################Prediction 

st.header(f'{ticker} Price Prediction using Deep Learning')

closedf = df_ticker[['Close']]
closedf = closedf[closedf.index> '2020-09-13']
close_stock = closedf.copy()

scaler=MinMaxScaler(feature_range=(0,1))
closedf=scaler.fit_transform(np.array(closedf).reshape(-1,1))

training_size=int(len(closedf)*0.70)
test_size=len(closedf)-training_size
train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:]

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   #i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


time_step = 15
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

LSTM_model=Sequential()
LSTM_model.add(LSTM(100,input_shape=(None,1), return_sequences=True,activation="relu"))
LSTM_model.add(LSTM(100, return_sequences=True))
LSTM_model.add(LSTM(100, return_sequences=False))
LSTM_model.add(Dense(25))
LSTM_model.add(Dense(1))

LSTM_model.compile(loss="mean_squared_error",optimizer="adam")

history = LSTM_model.fit( X_train, y_train,
                         validation_data=(X_test,y_test),
                         epochs=100, batch_size=32,
                         verbose=True)




train_predict=LSTM_model.predict(X_train)
test_predict=LSTM_model.predict(X_test)
train_predict.shape, test_predict.shape

# Transform back to original form

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 



st.title("Long Short-Term Memory (LSTM)") 
st.text("R2 SCORE")
st.text(r2_score(original_ytest, test_predict))

# shift train predictions for plotting

look_back=time_step
trainPredictPlot = np.empty_like(closedf)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

# shift test predictions for plotting
testPredictPlot = np.empty_like(closedf)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict

names = cycle(['Original close price','Train predicted close price','Test predicted close price'])


plotdf = pd.DataFrame({'date': close_stock.index,
                       'original_close': close_stock['Close'],
                      'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                      'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})

fig = px.line(plotdf,x=plotdf['date'], y=[plotdf['original_close'],plotdf['train_predicted_close'],
                                          plotdf['test_predicted_close']],
              labels={'value':'Close price','date': 'Date'},width=1100, height=900)
fig.update_layout(title_text='Prediction avec LSTM',
                  plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
fig.for_each_trace(lambda t:  t.update(name = next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()

st.plotly_chart(fig)


































