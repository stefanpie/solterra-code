
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error


print("Loading Data")
df = pd.read_pickle("./data/solar/solar_data_daily.pkl")

input_features = ['Year', 'Month', 'Day',
                  'Latitude', 'Longitude', 'Elevation',
                  'DHI',
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

output_features = ['GHI+1']



def flat2gen(alist):
  for item in alist:
    if isinstance(item, list):
      for subitem in item: yield subitem
    else:
      yield item

x = df[input_features].values.tolist()
y = df[output_features].values.tolist()


x = [list(flat2gen(d)) for d in x]
y = [list(flat2gen(d)) for d in y]

x = np.array(x)
y = np.array(y)

print(x.shape)
print(y.shape)

scaler_x = StandardScaler()
x_scaled = scaler_x.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=42)

linear = LinearRegression()
svr = SVR(verbose=True, tol=0.001, kernel ='linear')
gb = GradientBoostingRegressor(n_estimators = 500, max_depth = 8, verbose = 1)

multi = MultiOutputRegressor(svr)

print("fitting")
multi.fit(x_train, y_train)
r2 = multi.score(x_test, y_test)
print(r2)
y_pred = multi.predict(x_test)
