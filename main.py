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

# model = ARIMA(data['Close'], order=(1, 1, 1))
# model_fit = model.fit()

# forecast_steps = 30
# forecast_arima = model_fit.forecast(steps=forecast_steps)

# last_date = data.index[-1]
# forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
# forecast_arima = pd.Series(forecast_arima.values, index=forecast_index)

model_arima = ARIMA(data['Close'], order=(1, 1, 1))
result_arima = model_arima.fit()
forecast_arima = result_arima.forecast(steps=30)


# model = SARIMAX(
#     data['Close'],
#     order=(1, 1, 1),
#     seasonal_order=(1, 1, 1, 12),
#     enforce_stationarity=False,
#     enforce_invertibility=False
# )
# model_fit = model.fit(disp=False)

# forecast_steps = 30
# forecast_sarima = model_fit.forecast(steps=forecast_steps)

# last_date = data.index[-1]
# forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='B')  # 'B' = business days
# forecast_sarima = pd.Series(forecast_sarima.values, index=forecast_index)


# 📏 SARIMA Forecast
model_sarima = SARIMAX(data['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
result_sarima = model_sarima.fit()
forecast_sarima = result_sarima.forecast(steps=30)


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

# close_data = data[['Close']].values
# scaler = MinMaxScaler()
# scaled_data = scaler.fit_transform(close_data)

# def create_sequences(data, window_size):
#     X, y = [], []
#     for i in range(window_size, len(data)):
#         X.append(data[i-window_size:i, 0])
#         y.append(data[i, 0])
#     return np.array(X), np.array(y)

# window_size = 60
# X, y = create_sequences(scaled_data, window_size)

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# X_train = X_train.reshape((X_train.shape[0], window_size, 1))
# X_val = X_val.reshape((X_val.shape[0], window_size, 1))



# model = Sequential()
# model.add(LSTM(50, return_sequences=False, input_shape=(window_size, 1)))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mean_squared_error')

# model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)


# last_window = scaled_data[-window_size:]
# future_predictions = []

# for _ in range(30):
#     input_seq = last_window.reshape((1, window_size, 1))
#     next_pred = model.predict(input_seq, verbose=0)[0, 0]
    
#     future_predictions.append(next_pred)
    
#     last_window = np.append(last_window[1:], [[next_pred]], axis=0)

# future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))


# from datetime import timedelta

# last_date = data.index[-1]
# future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]


st.subheader("ARIMA Forecast")
st.line_chart(forecast_arima)



st.subheader("SARIMA Forecast")
st.line_chart(forecast_sarima)



st.subheader("Prophet Forecast")
st.line_chart(forecast[['ds', 'yhat']].set_index('ds').tail(30))

st.subheader("LSTM Forecast")
st.line_chart(lstm_forecast_df)



st.success("✅ All forecasts generated successfully!")

