import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

station_locations_df = pd.read_csv("./data/solar/solar_data_station_locations.csv", header=0)
solar_data_df = pd.read_csv("./data/solar/solar_data_combined.csv", header=0)

solar_data_df['year_month_day'] = solar_data_df['Year'].apply(str) + '_' + solar_data_df['Month'].apply(str) + "_" + solar_data_df['Day'].apply(str)
print(solar_data_df)

solar_data_2019_1_1_12_df = solar_data_df.loc[(solar_data_df['year_month_day'] == '2019_1_1') & (solar_data_df['Hour'] == 12)]
print(solar_data_2019_1_1_12_df)

lats = np.sort(station_locations_df['Latitude'].unique())
lons = np.sort(station_locations_df['Longitude'].unique())

val_pivot_df = solar_data_2019_1_1_12_df.pivot(index='Latitude', columns='Longitude', values='GHI')
lons = val_pivot_df.columns.values
lats = val_pivot_df.index.values
xx, yy = np.meshgrid(lons,lats) 
data_values = val_pivot_df.values


fig, ax = plt.subplots(1,1)
ax.pcolormesh(xx, yy, data_values, shading='auto')
ax.scatter(-84.3880, 33.7490, marker='.', color='k',)
ax.annotate("Atlanta", (-84.3880+.01, 33.749+.01))
plt.show()



solar_data_2019_1_1_960460_df = solar_data_df.loc[(solar_data_df['year_month_day'] == '2019_1_1') & (solar_data_df['Location ID'] == 960460)]
print(solar_data_2019_1_1_960460_df)

fig, ax = plt.subplots(1,1)
ax.plot(solar_data_2019_1_1_960460_df["Hour"].tolist(), solar_data_2019_1_1_960460_df["GHI"].tolist())
plt.show()


