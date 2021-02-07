import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt






print("Loading Data")
df = pd.read_pickle("./data/solar/solar_data_daily.pkl")

input_features_fixed = ['Year', 'Month', 'Day',
                  'Latitude', 'Longitude', 'Elevation']

input_features_timeseries = ['DHI',
                  'DNI',
                  'GHI',
                  'Clearsky DHI',
                  'Clearsky DNI',
                  'Clearsky GHI',
                  'Dew Point',
                  'Solar Zenith Angle',
                  'Surface Albedo',
                  'Wind Speed',
                  'Precipitable Water',
                  'Wind Direction',
                  'Relative Humidity',
                  'Temperature',
                  'Pressure',
                  'Global Horizontal UV Irradiance (280-400nm)',
                  'Global Horizontal UV Irradiance (295-385nm)',
                  ]

output_features_GHI = ['GHI+1', 
                   'GHI+2',
                   'GHI+3',
                   'GHI+4',
                   'GHI+5',
                   'GHI+6',
                   'GHI+7']

output_features_temperature = ['Temperature+1', 
                   'Temperature+2',
                   'Temperature+3',
                   'Temperature+4',
                   'Temperature+5',
                   'Temperature+6',
                   'Temperature+7']

output_features_wind = ['Wind Speed+1', 
                   'Wind Speed+2',
                   'Wind Speed+3',
                   'Wind Speed+4',
                   'Wind Speed+5',
                   'Wind Speed+6',
                   'Wind Speed+7']


x_fixed = df[input_features_fixed].values
x_timeseries = df[input_features_timeseries].values
x_timeseries = np.asarray(x_timeseries.tolist())

y_GHI = df[output_features_GHI].values
y_GHI = np.asarray(y_GHI.tolist())

y_temperature = df[output_features_temperature].values
y_temperature = np.asarray(y_temperature.tolist())

y_wind = df[output_features_wind].values
y_wind = np.asarray(y_wind.tolist())

# print(x_fixed.shape)
# print(x_timeseries.shape)
# print(y.shape)



x_train_fixed, x_test_fixed, \
x_train_timeseries, x_test_timeseries, \
y_train_GHI, y_test_GHI, \
y_train_wind, y_test_wind, \
y_train_temperature, y_test_temperature = train_test_split(
x_fixed, x_timeseries, y_GHI, y_temperature, y_wind, test_size=0.3, random_state=42)


# print(x_train_fixed.shape)
# print(x_test_fixed.shape)
# print(x_train_timeseries.shape)
# print(x_test_timeseries.shape)
# print(y_train.shape)
# print(y_test.shape)



model_GHI = keras.models.load_model("./models/model_GHI_20.h5")
y_pred_GHI = model_GHI.predict([x_test_fixed, x_test_timeseries])
print(y_test_GHI.shape)
print(y_pred_GHI.shape)

model_wind = keras.models.load_model("./models/model_wind_20.h5")
y_pred_wind = model_wind.predict([x_test_fixed, x_test_timeseries])
print(y_test_wind.shape)
print(y_pred_wind.shape)

model_temperature = keras.models.load_model("./models/model_temperature_20.h5")
y_pred_temperature = model_temperature.predict([x_test_fixed, x_test_timeseries])
print(y_test_temperature.shape)
print(y_pred_temperature.shape)





# fig, axs = plt.subplots(7,1)
# for i in range(7):
#     axs[i].plot(list(range(24)), y_test[500,i,:], c='k')
#     axs[i].plot(list(range(24)), y_pred[500,i,:], c='g')
#     axs[i].set_yticklabels([])
#     axs[i].set_xticklabels([])

# plt.tight_layout()
# plt.show()