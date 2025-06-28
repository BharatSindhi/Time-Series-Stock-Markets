import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from tensorflow.keras.models import Sequential   # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import streamlit as st
from math import sqrt
from sklearn.model_selection import train_test_split

data=pd.read_csv('D:\Zidio Developments\Time Series Stock Markets\stock_data.csv')
data['Date']=pd.to_datetime(data['Date'])
data.set_index('Date',inplace=True)

# 📊 Visualize Stock Price
st.title("📈 Stock Market Forecast Dashboard")
st.subheader("Historical Stock Price")
st.line_chart(data['Close'])


# ARIMA
model = ARIMA(data['Close'], order=(1, 1, 1))
model_fit = model.fit()

forecast_steps = 30
forecast_arima = model_fit.forecast(steps=forecast_steps)



# SARIMAX
model = SARIMAX(
    data['Close'],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)
model_fit = model.fit(disp=False)

forecast_steps = 30
forecast_sarima = model_fit.forecast(steps=forecast_steps)


# Dates
df_prophet = data[['Close']].reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
model_prophet = Prophet(daily_seasonality=True)
model_prophet.fit(df_prophet)
future = model_prophet.make_future_dataframe(periods=30)
forecast = model_prophet.predict(future)

# 📏 LSTM Forecast
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data[['Close']])

X, y = [], []
for i in range(60, len(data_scaled)):
    X.append(data_scaled[i - 60:i, 0])
    y.append(data_scaled[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model_lstm.add(LSTM(units=50))
model_lstm.add(Dense(1))

model_lstm.compile(loss='mean_squared_error', optimizer='adam')
model_lstm.fit(X, y, epochs=10, batch_size=32, verbose=0)

# 🔮 Predict Future with LSTM
pred_input = data_scaled[-60:]
pred_input = pred_input.reshape(1, 60, 1)

lstm_predictions = []

for _ in range(30):
    next_pred = model_lstm.predict(pred_input, verbose=0)[0][0]
    lstm_predictions.append(next_pred)
    pred_input = np.append(pred_input[:, 1:, :], [[[next_pred]]], axis=1)

lstm_predictions_actual = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1))

# 📅 Create date index for forecast
last_date = data.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)
lstm_forecast_df = pd.DataFrame({'Date': future_dates, 'LSTM_Predicted_Close': lstm_predictions_actual.flatten()})
lstm_forecast_df.set_index('Date', inplace=True)





st.subheader("ARIMA Forecast")
st.line_chart(forecast_arima)



st.subheader("SARIMA Forecast")
st.line_chart(forecast_sarima)



st.subheader("Prophet Forecast")
st.line_chart(forecast[['ds', 'yhat']].set_index('ds').tail(30))

st.subheader("LSTM Forecast")
st.line_chart(lstm_forecast_df)



st.success("✅ All forecasts generated successfully!")

