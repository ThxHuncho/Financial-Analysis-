# importer les bibliothèques nécessaires

import datetime
import streamlit as st
import yfinance as yf
import cufflinks as cf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Flatten, Dense
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K

import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as py



APP_NAME = "Deep Learning for Portfolio Selection"


#---------------------------------------- page config

# Page Configuration
st.set_page_config(
    page_title=APP_NAME,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.image('https://www.neoma-bs.fr/events/emailings/2020/05/lancement-campagne-de-communication/images/neoma_logotype_rvb.png', width=200)


# Add app title
st.sidebar.title(APP_NAME)

# Apple Stock Price Visualisation WebApp
    
st.markdown('---')
# Sidebar Configuration
st.sidebar.markdown('In this app the user will be able to construct his own portfolio and optimize his portfolio with deep learning methode')
st.sidebar.markdown('This app was created by student of MSc in Finance and Big Data')



#---------------------------------------- List of tickers
TICKERS = [
           'MSFT','AAPL','GOOGL','ACN','IBM','JPM','BLK','WMT','AMZN', 'NKE',
           'NDAQ','EQOP','VTI','DBC','AGG'
]

#---------------------------------------- Function to load data
def filter_data(symbol_selections: list[str],start_date,end_date) -> pd.DataFrame:
    """
    Returns Dataframe with only accounts and symbols selected
    Args:symbol_selections (list[str]): list of symbols to include
    Returns:
        pd.DataFrame: data only for the given accounts and symbols
    """
    
    df = yf.download(symbol_selections, start=start_date, end=end_date)
    return df


#---------------------------------------- Select ticker
st.sidebar.subheader("Portfolio construction")
symbols = TICKERS
symbol_selections = st.sidebar.multiselect("Select your asset", options=symbols, default=symbols[0:3])

#---------------------------------------- Method

st.sidebar.subheader("Backtesting period")
start_date = st.sidebar.date_input('Start date', datetime.datetime(2010, 1, 1))
end_date = st.sidebar.date_input('End date', datetime.datetime.now().date())

#---------------------------------------- Data for Optimization 
df = filter_data(symbol_selections,start_date,end_date)
df = df['Close']
#---------------------------------------- Analysis 
#df normed
st.subheader("Normed return")
df_normed = df/df.iloc[0]

st.line_chart(df_normed)

#log return 
st.subheader("Log return")
log_ret = np.log(df/df.shift(1))
st.bar_chart(df)



#----------------------------------------Deep Learning Model for Portfolio Selection 

class Model:
    def __init__(self):
        self.data = None
        self.model = None
        
    def __build_model(self, input_shape, outputs):
        '''
        Builds and returns the Deep Neural Network that will compute the allocation ratios
        that optimize the Sharpe Ratio of the portfolio
        
        inputs: input_shape - tuple of the input shape, outputs - the number of assets
        returns: a Deep Neural Network model
        La fonction mathématique softmax peut être utilisée en 
        machine learning pour convertir un score en probabilité
        '''
        
        model = Sequential()
        
        model.add(LSTM(64, input_shape=input_shape))
        #model.add(LSTM(100, return_sequences=True))
        #model.add(LSTM(100, return_sequences=True))
        #model.add(LSTM(100, return_sequences=True))
        #model.add(LSTM(100, return_sequences=True))
        #model.add(LSTM(100, return_sequences=False))
        model.add(Flatten()) 
        model.add(Dense(outputs, activation='softmax')) 
        

        def sharpe_loss(_, y_pred):
            # make all time-series start at 1
            data = tf.divide(self.data, self.data[0])  
            
            # value of the portfolio after allocations applied
            portfolio_values = tf.reduce_sum(tf.multiply(data, y_pred), axis=1) 
            
            portfolio_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]  
            # % change formula

            sharpe = K.mean(portfolio_returns) / K.std(portfolio_returns)
            
            # since we want to maximize Sharpe, while gradient descent minimizes the loss, 
            #   we can negate Sharpe (the min of a negated function is its max)
            return -sharpe
        
        model.compile(loss=sharpe_loss, optimizer='adam')
        return model
    
    
    def get_allocations(self, data: pd.DataFrame):
        '''
        Computes and returns the allocation ratios that optimize the Sharpe over the given data
        
        input: data - DataFrame of historical closing prices of various assets
        
        return: the allocations ratios for each of the given assets
        '''
        
        # data with returns
        data_w_ret = np.concatenate([ data.values[1:], data.pct_change().values[1:] ], axis=1)
        
        data = data.iloc[1:]
        self.data = tf.cast(tf.constant(data), float)
        
        if self.model is None:
            self.model = self.__build_model(data_w_ret.shape, len(data.columns))
        
        fit_predict_data = data_w_ret[np.newaxis,:]        
        self.model.fit(fit_predict_data, np.zeros((1, len(data.columns))), epochs=100, shuffle=False)
        return self.model.predict(fit_predict_data)[0]
    
model = Model()

#---------------------------------------- 

def get_ret_vol_sr(weights):
    """
    Takes in weights, returns array or return,volatility, sharpe ratio
    """
    weights = np.array(weights)
    ret = np.sum(log_ret.mean() * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
    sr = ret/vol
    return np.array([ret,vol,sr])

def neg_sharpe(weights):
    return  get_ret_vol_sr(weights)[2] * -1

# Contraints
def check_sum(weights):
    '''
    Returns 0 if sum of weights is 1.0
    '''
    return np.sum(weights) - 1

#if Method == 'Modern Portfolio Theory':
# By convention of minimize function it should be a function that returns zero for conditions
cons = ({'type':'eq','fun': check_sum})

# 0-1 bounds for each weight
bounds = ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1),
          (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1),
          (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1),(0, 1))


# Initial Guess (equal distribution)
init_guess = []

for i in range(len(symbol_selections)):
    init_guess.append(1/len(symbol_selections))
    
# Sequential Least SQuares Programming (SLSQP).
opt_results = minimize(neg_sharpe,init_guess,method='SLSQP',bounds=bounds[0:len(symbol_selections)],constraints=cons)

st.subheader("Allocation")

#coef_M = pd.DataFrame(opt_results.x,index=symbol_selections)
coef=model.get_allocations(df)
#df_coef = pd.DataFrame(coef,index=symbol_selections)
#df_alloc=pd.concat(coef_M,df_coef)

d={'Coef_MKV':list(opt_results.x),'Coef_DLS':list(coef)}
df_coef = pd.DataFrame(data=d)
st.dataframe(df_coef)



st.subheader("Return - Volatility - Shape Ratio")

#summ_M = pd.DataFrame(get_ret_vol_sr(opt_results.x),index=['Return','Volatility','Shape Ratio'])
#df_summ = pd.DataFrame(get_ret_vol_sr(coef),index=['Return','Volatility','Shape Ratio'])
#df_res=pd.concat(summ_M,df_summ)

d={'MKV':list(get_ret_vol_sr(opt_results.x)),'DLS':list(get_ret_vol_sr(coef))}
df_rvs = pd.DataFrame(data=d)
st.dataframe(df_rvs)

#---------------------------------------- Backtesting_Mkv

st.subheader("Backtesting with 1000$")

allocation_M=list(opt_results.x)
alloc_M=df_normed*allocation_M
pos_M=alloc_M*1000
pos_t_M=pos_M.sum(axis=1)

    
#---------------------------------------- Backtesting_Deep L

allocation_D=list(coef)
alloc_D=df_normed*allocation_D
pos_D=alloc_D*1000
pos_t_D=pos_D.sum(axis=1)

Pos=pd.DataFrame({"MKV":pos_t_M,"DLS":pos_t_D})
st.line_chart(Pos)













