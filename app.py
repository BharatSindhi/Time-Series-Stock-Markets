import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sn 
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Header 
st.title("Stock Market Forecasting App")
st.subheader("This app is created to forecast the selected company")

# Sidebar With Dates 
st.sidebar.title("Select the Parameters Below")
start_date=st.sidebar.date_input("Start Date",date(2020,1,1))
end_date=st.sidebar.date_input("End Date",date(2025,5,30))

# Company List 
company_list=["AAPL","GOOG","GOOGL","AMZN","META","TSLA","NVDA","AABA"]
company=st.sidebar.selectbox("Select the Company",company_list)

# Download Data using yfinance
data=yf.download(company,start=start_date,end=end_date)

# Read data and merge 
data.insert(0,"Date",data.index,True)
data.reset_index(drop=True,inplace=True)
st.write("Date from",start_date ,"to", end_date)
data.columns = ['_'.join(col).strip('_') for col in data.columns]
st.write(data)


st.header("Data visualization")

#  Data visualization with All Colunms 
stocks=px.line(data,x="Date",y=data.columns,title="closing price of the stock",width=1000,height=800)
st.plotly_chart(stocks)

# Select the columns which Forecast
columns=st.selectbox("select the columns to used forecation",data.columns[1:])
data=data[["Date",columns]]
st.write("Selected data")
st.write(data)





# Decomposition Better understanding 
st.header('Decomposition')
decompose = seasonal_decompose(data[columns], model='additive', period=252)
st.write(decompose.plot())

st.write('## Plotting the Decomposition in Plotly')
st.plotly_chart(px.line(x=data["Date"], y=decompose.trend, title='Trend', labels={"x": "Date", "y": "Price"}).update_traces(line_color="Red"))
st.plotly_chart(px.line(x=data["Date"], y=decompose.seasonal, title='Seasonality', labels={"x": "Date", "y": "Price"}).update_traces(line_color="Green"))
st.plotly_chart(px.line(x=data["Date"], y=decompose.resid, title='Residual', labels={"x": "Date", "y": "Price"}).update_traces(line_dash="dot"))


# Stationarity or NOT
st.header('Adata Stationarity Test')
data_diff = data[columns].diff().dropna()
result = adfuller(data_diff)
st.subheader("Adata Test on Differenced Data")
st.write(f"**Adata Statistic**: {result[0]}")
st.write(f"**p-value**: {result[1]}")
st.write("**Critical Values:**")
for key, value in result[4].items():
    st.write(f"   {key}: {value}")



# ARIMA
st.header("ARIMA Model")
p = st.slider('Select the value of p', 0, 5, 2)
d = st.slider('Select the value of d', 0, 5, 1)
q = st.slider('Select the value of q', 0, 5, 2)


model = ARIMA(data[columns], order=(p, d, q))
model_fit = model.fit()

st.header("ARIMA Uummary")
st.write(model_fit.summary())



st.header('ARIMA Forecasting')
forecast_period = st.number_input('Select the number of days to forecast', 1, 365, 30)
arima_prediction=model_fit.get_prediction(start=len(data),end=len(data)+forecast_period-1).predicted_mean


arima_prediction.index = pd.date_range(start=end_date, periods=len(arima_prediction), freq='D')
arima_prediction = pd.DataFrame(arima_prediction)
arima_prediction.insert(0, 'Date', arima_prediction.index)
arima_prediction.reset_index(drop=True, inplace=True)
st.write("## ARIMA Predictions")
st.write(arima_prediction)
st.write("## Actual Data")
st.write(data)
st.write("---")

fig_arima = go.Figure()
fig_arima.add_trace(go.Scatter(x=data["Date"], y=data[columns], name="Actual Data"))
fig_arima.add_trace(go.Scatter(x=arima_prediction["Date"], y=arima_prediction["predicted_mean"], name="ARIMA Predictions", mode="lines", line=dict(color="Red")))
fig_arima.update_layout(title_text="Actual Data vs ARIMA Predictions", xaxis_title="Date", yaxis_title="Price", width=1000, height=400)
st.plotly_chart(fig_arima)

# SARIMA
st.header("SARIMA Model")
p = st.slider('Select the value of p', 0, 5, 2, key="srm1")
d = st.slider('Select the value of d', 0, 5, 1, key="srm2")
q = st.slider('Select the value of q', 0, 5, 2, key="srm3")
seasonal_order=st.number_input('select the value of seasonal',0,24,12)


model=SARIMAX(data[columns],order=(p,d,q),seasonal_order=(p,d,q,seasonal_order))
model_fit=model.fit()


st.header("SARIMA Summary")
st.write(model_fit.summary())

st.header('SARIMA Forecasting')
forecast_period = st.number_input('Select the number of days to forecast', 1, 365, 10)
sarima_predictions = model_fit.get_prediction(start=len(data), end=len(data) + forecast_period - 1).predicted_mean

sarima_predictions.index = pd.date_range(start=end_date, periods=len(sarima_predictions), freq='D')
sarima_predictions = pd.DataFrame(sarima_predictions)
sarima_predictions.insert(0, 'Date', sarima_predictions.index)
sarima_predictions.reset_index(drop=True, inplace=True)
st.write("## SARIMA Predictions")
st.write(sarima_predictions)
st.write("## Actual Data")
st.write(data)
st.write("---")

fig_sarima = go.Figure()
fig_sarima.add_trace(go.Scatter(x=data["Date"], y=data[columns], name="Actual Data"))
fig_sarima.add_trace(go.Scatter(x=sarima_predictions["Date"], y=sarima_predictions["predicted_mean"], name="SARIMA Predictions", mode="lines", line=dict(color="Red")))
fig_sarima.update_layout(title_text="Actual Data vs SARIMA Predictions", xaxis_title="Date", yaxis_title="Price", width=1000, height=400)
st.plotly_chart(fig_sarima)


# LSTM
st.header('LSTM Model')
st.write('**Note**: LSTM Model is trained and predicted on the selected column of the data.')

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[columns].values.reshape(-1, 1))

train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size
train_data, test_data = scaled_data[:train_size, :], scaled_data[train_size:, :]

lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(1, 1)))
lstm_model.add(LSTM(50))
lstm_model.add(Dense(1))

lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(train_data.reshape(-1, 1, 1), train_data.reshape(-1, 1), epochs=10, batch_size=1, verbose=2)

lstm_predictions = lstm_model.predict(test_data.reshape(-1, 1, 1))
lstm_predictions = scaler.inverse_transform(lstm_predictions)

test_dates = pd.to_datetime(data["Date"].values[train_size:])

lstm_predictions_df = pd.DataFrame({"Date": test_dates, "Predicted_Price": lstm_predictions.flatten()})

st.write("## LSTM Predictions")
st.write(lstm_predictions_df)
st.write("---")

fig_lstm = go.Figure()
fig_lstm.add_trace(go.Scatter(x=data["Date"], y=data[columns], name="Actual Data"))
fig_lstm.add_trace(go.Scatter(x=lstm_predictions_df["Date"], y=lstm_predictions_df["Predicted_Price"], name="LSTM Predictions", mode="lines", line=dict(color="Red")))
fig_lstm.update_layout(title_text="Actual Data vs LSTM Predictions", xaxis_title="Date", yaxis_title="Price", width=1000, height=400)
st.plotly_chart(fig_lstm)


