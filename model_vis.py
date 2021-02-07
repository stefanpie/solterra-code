import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt






print("Loading Data")
df = pd.read_pickle("./data/solar/solar_data_daily.pkl")
station_locations_df = pd.read_csv("./data/solar/solar_data_station_locations.csv", header=0)


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


id_columns = ['year_month_day', "Location ID", "Latitude", "Longitude"]

x_fixed = df[input_features_fixed].values
x_timeseries = df[input_features_timeseries].values
x_timeseries = np.asarray(x_timeseries.tolist())

y_GHI = df[output_features_GHI].values
y_GHI = np.asarray(y_GHI.tolist())

y_temperature = df[output_features_temperature].values
y_temperature = np.asarray(y_temperature.tolist())

y_wind = df[output_features_wind].values
y_wind = np.asarray(y_wind.tolist())

id_data = df[id_columns].values
print(id_data)

# print(x_fixed.shape)
# print(x_timeseries.shape)
# print(y.shape)



# x_train_fixed, x_test_fixed, \
# x_train_timeseries, x_test_timeseries, \
# y_train_GHI, y_test_GHI, \
# y_train_wind, y_test_wind, \
# y_train_temperature, y_test_temperature, \
# id_data_train, id_data_test = train_test_split(x_fixed, x_timeseries, y_GHI, y_temperature, y_wind, id_data, test_size=1.0, random_state=42)

x_train_fixed, x_test_fixed = x_fixed, x_fixed
x_train_timeseries, x_test_timeseries = x_timeseries, x_timeseries
y_train_GHI, y_test_GHI = y_GHI, y_GHI
y_train_wind, y_test_wind = y_wind, y_wind
y_train_temperature, y_test_temperature = y_temperature, y_temperature
id_data_train, id_data_test = id_data, id_data


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

# model_wind = keras.models.load_model("./models/model_wind_20.h5")
# y_pred_wind = model_wind.predict([x_test_fixed, x_test_timeseries])
# print(y_test_wind.shape)
# print(y_pred_wind.shape)

# model_temperature = keras.models.load_model("./models/model_temperature_20.h5")
# y_pred_temperature = model_temperature.predict([x_test_fixed, x_test_timeseries])
# print(y_test_temperature.shape)
# print(y_pred_temperature.shape)


lats = np.sort(station_locations_df['Latitude'].unique())
lons = np.sort(station_locations_df['Longitude'].unique())

ymd = "2019_3_1"
hour = 13
forcast_day = 6

test = y_test_GHI[:,forcast_day,hour]
pred = y_pred_GHI[:,forcast_day,hour]


results_df = pd.DataFrame((test,pred))
results_df = results_df.T
print(results_df)
results_df.columns = ['GHI_test', 'GHI_pred']
print(results_df)
id_data_df = pd.DataFrame(id_data_test, columns=id_columns)
print(id_data_df)

vis_df = pd.concat([id_data_df, results_df], axis=1)
vis_df = vis_df.loc[(vis_df['year_month_day'] == ymd)]

fig, axs = plt.subplots(1,2)



val_pivot_df = vis_df.pivot(index='Latitude', columns='Longitude', values='GHI_test')
lons = val_pivot_df.columns.values
lats = val_pivot_df.index.values
xx, yy = np.meshgrid(lons,lats) 
data_values = val_pivot_df.values

axs[0].pcolormesh(xx, yy, data_values, shading='auto')
axs[0].scatter(-84.3880, 33.7490, marker='.', color='k',)
axs[0].annotate("Atlanta", (-84.3880+.01, 33.749+.01))
axs[0].scatter(-84.4277, 33.6407, marker='.', color='k',)
axs[0].annotate("Hartsfield-Jackson International Airport", (-84.4277+.01, 33.6407+.01))


val_pivot_df = vis_df.pivot(index='Latitude', columns='Longitude', values='GHI_pred')
lons = val_pivot_df.columns.values
lats = val_pivot_df.index.values
xx, yy = np.meshgrid(lons,lats) 
data_values = val_pivot_df.values

axs[1].pcolormesh(xx, yy, data_values, shading='auto')
axs[1].scatter(-84.3880, 33.7490, marker='.', color='k',)
axs[1].annotate("Atlanta", (-84.3880+.01, 33.749+.01))

plt.show()








# fig, axs = plt.subplots(7,1)
# for i in range(7):
#     axs[i].plot(list(range(24)), y_test[500,i,:], c='k')
#     axs[i].plot(list(range(24)), y_pred[500,i,:], c='g')
#     axs[i].set_yticklabels([])
#     axs[i].set_xticklabels([])

# plt.tight_layout()
# plt.show()