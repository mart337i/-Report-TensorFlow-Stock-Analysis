import typing_extensions
import sys
import numpy as np
import pandas as pd
import hvplot.pandas
import tensorflow as tf

import matplotlib

# Set the random seed for reproducibility
from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(2)

# get more recent data as of 6/20/2020
import requests
import urllib.request
import csv
import json
from io import StringIO

# get fng data from api
get_fng_index = "https://api.alternative.me/fng/"
url = get_fng_index + "?limit=2000&format=csv&date_format=us"

#get data as csv from api
#urllib.request.urlretrieve(url, 'btc_fng.csv')

btc_fng_df = pd.read_csv('btc_fng.csv')
btc_fng_df

# Load the fear and greed sentiment data for Bitcoin from the most recent data as  of June 20, 2020
df = pd.read_csv('btc_fng.csv', index_col="date", infer_datetime_format=True, parse_dates=True)
df = df.drop(columns="fng_classification")
df.tail()

# Load the historical closing prices for Bitcoin from yahoo finance
df2 = pd.read_csv('btc_usd.csv', index_col="Date", infer_datetime_format=True, parse_dates=True)['Close']
df2 = df2.sort_index()
df2.head()

# Join the data into a single DataFrame
df = df.join(df2, how="inner")
df.tail()

df.dropna(inplace=True)
df.head()

df = df[::-1]
df.head()

# This function accepts the column number for the features (X) and the target (y)
# It chunks the data up with a rolling window of Xt-n to predict Xt
# It returns a numpy array of X any y
def window_data(df, window, feature_col_number, target_col_number):
    X = []
    y = []
    for i in range(len(df) - window - 1):
        features = df.iloc[i:(i + window), feature_col_number]
        target = df.iloc[(i + window), target_col_number]
        X.append(features)
        y.append(target)
    return np.array(X), np.array(y).reshape(-1, 1)

    # Predict Closing Prices using a 10 day window of fear and greed index values and a target of the 11th day closing price
# Try a window size anywhere from 1 to 10 and see how the model performance changes
window_size = 10

# Column index 1 is the `Close` column
feature_column = 0
target_column = 1
X, y = window_data(df, window_size, feature_column, target_column)

# Use 70% of the data for training and the remainder for testing

# x split
split = int(.7 * len(X))
X_train = X[:split - 1]
X_test = X[split:]

# y split             
y_train = y[:split - 1]
y_test = y[split:]

# Use MinMaxScaler to scale the data between 0 and 1
from sklearn.preprocessing import  MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
scaler.fit(y)
y_train = scaler.transform(y_train)
y_test = scaler.transform(y_test)

# Reshape the features for the model
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Build the LSTM model

model = Sequential() 
model.add(LSTM(
    units=30, return_sequences=True,
    input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=30, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=30))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")

# Summarize the model
model.summary()
# Train the model
# Use at least 10 epochs
model.fit(X_train, y_train, epochs=50, shuffle=False, batch_size=1, verbose=1)

# Evaluate the model
model.evaluate(X_test, y_test)

# Make some predictions
predicted = model.predict(X_test)

# Recover the original prices instead of the scaled version
predicted_prices = scaler.inverse_transform(predicted)
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Create a DataFrame of Real and Predicted values
btc_price_predictions = pd.DataFrame({
    "Real": real_prices.ravel(),
    "Predicted": predicted_prices.ravel()
})
btc_price_predictions.head()

# Plot the real vs predicted values as a line chart
# YOUR CODE HERE!
btc_price_predictions.head(100).hvplot()