import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.metrics import MeanSquaredError, RootMeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split







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

output_features = ['Temperature+1', 
                   'Temperature+2',
                   'Temperature+3',
                   'Temperature+4',
                   'Temperature+5',
                   'Temperature+6',
                   'Temperature+7']


x_fixed = df[input_features_fixed].values
x_timeseries = df[input_features_timeseries].values
x_timeseries = np.asarray(x_timeseries.tolist())
y = df[output_features].values
y = np.asarray(y.tolist())
# y = y.reshape(-1,24)

# print(x_fixed.shape)
# print(x_timeseries.shape)
# print(y.shape)


x_train_fixed, x_test_fixed, x_train_timeseries, x_test_timeseries, y_train, y_test = train_test_split(x_fixed, x_timeseries, y, test_size=0.3, random_state=42)


print(x_train_fixed.shape)
print(x_test_fixed.shape)
print(x_train_timeseries.shape)
print(x_test_timeseries.shape)
print(y_train.shape)
print(y_test.shape)

print("Building Model")

input_fixed = layers.Input(shape=(6,))
input_timeseries = layers.Input(shape=(17,24))

x_timeseries = layers.Conv1D(filters=128, kernel_size=5, padding='same')(input_timeseries)
x_timeseries = layers.Dropout(0.2)(x_timeseries)
x_timeseries = layers.BatchNormalization()(x_timeseries)
x_timeseries = layers.ReLU()(x_timeseries)
x_timeseries = layers.MaxPooling1D(pool_size=2)(x_timeseries)
x_timeseries = layers.Conv1D(filters=256, kernel_size=3, padding='same')(x_timeseries)
x_timeseries = layers.Dropout(0.2)(x_timeseries)
x_timeseries = layers.BatchNormalization()(x_timeseries)
x_timeseries = layers.ReLU()(x_timeseries)
x_timeseries = layers.MaxPooling1D(pool_size=2)(x_timeseries)
x_timeseries = layers.Flatten()(x_timeseries)

x_fixed = layers.Dense(512)(input_fixed)
x_fixed = layers.Dropout(0.2)(x_fixed)
x_fixed = layers.BatchNormalization()(x_fixed)
x_fixed = layers.ReLU()(x_fixed)
x_fixed = layers.Dense(256)(x_fixed)
x_fixed = layers.Dropout(0.2)(x_fixed)
x_fixed = layers.BatchNormalization()(x_fixed)
x_fixed = layers.ReLU()(x_fixed)
x_fixed = layers.Dense(128)(x_fixed)
x_fixed = layers.Dropout(0.2)(x_fixed)
x_fixed = layers.BatchNormalization()(x_fixed)
x_fixed = layers.ReLU()(x_fixed)
x_fixed = layers.Flatten()(x_fixed)

x = layers.Concatenate()([x_timeseries, x_fixed])
x = layers.Dense(1024)(x)
x = layers.ReLU()(x)
x = layers.Dense(512)(x)
x = layers.ReLU()(x)
x = layers.Dense(7*24*8)(x)
x = layers.ReLU()(x)
x = layers.Reshape((7, 24*8))(x)
x = layers.Conv1D(filters=128, kernel_size=5, padding='same')(x)
x = layers.ReLU()(x)
x = layers.Conv1D(filters=24, kernel_size=3, padding='same')(x)
x = layers.ReLU()(x)


model = Model(inputs=[input_fixed, input_timeseries], outputs=x)
model.summary()
tf.keras.utils.plot_model(model, './model.png',show_shapes=True, dpi=300)

model.compile(loss='mae', optimizer='adam', metrics=[MeanSquaredError(), RootMeanSquaredError(), MeanAbsoluteError(), MeanAbsolutePercentageError()])
csv_logger = CSVLogger('./models/model_temperature_20.log')

print("Fitting Model")
model.fit([x_train_fixed, x_train_timeseries], y_train, validation_data=([x_test_fixed, x_test_timeseries], y_test), epochs=20, callbacks=[csv_logger])
y_pred = model.predict([x_test_fixed, x_test_timeseries])
model.save('./models/model_temperature_20.h5')
