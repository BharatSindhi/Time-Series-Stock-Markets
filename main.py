import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
# import tensorflow as tf
# from tensorflow.keras.models import Sequential   # type: ignore
# from tensorflow.keras.layers import LSTM, Dense # type: ignore
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_errorṇ
import streamlit as st
from math import sqrt

data=pd.read_csv('D:\Zidio Developments\Time Series Stock Markets\stock_data.csv')
data['Date']=pd.to_datetime(data['Date'])
data.set_index('Date',inplace=True)

# 📊 Visualize Stock Price
st.title("📈 Stock Market Forecast Dashboard")
st.subheader("Historical Stock Price")
st.line_chart(data['Close'])
