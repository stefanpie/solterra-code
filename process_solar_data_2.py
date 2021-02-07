import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from tqdm import tqdm
from joblib import Parallel, delayed
import itertools



station_locations_df = pd.read_csv("./data/solar/solar_data_station_locations.csv", header=0)
solar_data_df = pd.read_csv("./data/solar/solar_data_combined.csv", header=0)

solar_data_df.sort_values(by=['Year', 'Month', "Day", "Hour", "Minute", "Location ID"])
solar_data_df['year_month_day'] = solar_data_df['Year'].apply(str) + '_' + solar_data_df['Month'].apply(str) + "_" + solar_data_df['Day'].apply(str)



# def flatten_to_daily(ymd, loc_id):
#     station_day_df = solar_data_df.loc[(solar_data_df['year_month_day'] == ymd) & (solar_data_df["Location ID"] == loc_id)]
#     data = {}
#     data['Year'] = station_day_df['Year'].tolist()[0]
#     data['Month'] = station_day_df['Month'].tolist()[0]
#     data['Day'] = station_day_df['Day'].tolist()[0]
#     data['year_month_day'] = station_day_df['year_month_day'].tolist()[0]
#     data['Location ID'] = station_day_df['Location ID'].tolist()[0]
#     data['Latitude'] = station_day_df['Latitude'].tolist()[0]
#     data['Longitude'] = station_day_df['Longitude'].tolist()[0]
#     data['Elevation'] = station_day_df['Elevation'].tolist()[0]
    
#     data['Hour'] = station_day_df['Hour'].to_numpy()
#     data['DHI'] = station_day_df['DHI'].to_numpy()
#     data['DNI'] = station_day_df['DNI'].to_numpy()
#     data['GHI'] = station_day_df['GHI'].to_numpy()
#     data['Clearsky DHI'] = station_day_df['Clearsky DHI'].to_numpy()
#     data['Clearsky DNI'] = station_day_df['Clearsky DNI'].to_numpy()
#     data['Clearsky GHI'] = station_day_df['Clearsky GHI'].to_numpy()
#     data['Cloud Type'] = station_day_df['Cloud Type'].to_numpy()
#     data['Dew Point'] = station_day_df['Dew Point'].to_numpy()
#     data['Solar Zenith Angle'] = station_day_df['Solar Zenith Angle'].to_numpy()
#     data['Surface Albedo'] = station_day_df['Surface Albedo'].to_numpy()
#     data['Wind Speed'] = station_day_df['Wind Speed'].to_numpy()
#     data['Precipitable Water'] = station_day_df['Precipitable Water'].to_numpy()
#     data['Wind Direction'] = station_day_df['Wind Direction'].to_numpy()
#     data['Relative Humidity'] = station_day_df['Relative Humidity'].to_numpy()
#     data['Temperature'] = station_day_df['Temperature'].to_numpy()
#     data['Pressure'] = station_day_df['Pressure'].to_numpy()
#     data['Global Horizontal UV Irradiance (280-400nm)'] = station_day_df['Global Horizontal UV Irradiance (280-400nm)'].to_numpy()
#     data['Global Horizontal UV Irradiance (295-385nm)'] = station_day_df['Global Horizontal UV Irradiance (295-385nm)'].to_numpy()
#     return data


# # solar_data_daily_df = pd.DataFrame()
# combos = itertools.product( list(solar_data_df['year_month_day'].unique()), list(solar_data_df["Location ID"].unique()))
# solar_data_daily_list = Parallel(n_jobs=-1)(delayed(flatten_to_daily)(i[0], i[1]) for i in combos)

# for ymd in tqdm(list(solar_data_df['year_month_day'].unique())):
#     for loc_id in tqdm(list(solar_data_df["Location ID"].unique())):
#         station_day_df = solar_data_df.loc[(solar_data_df['year_month_day'] == ymd) & (solar_data_df["Location ID"] == loc_id)]
#         data = {}
#         data['Year'] = station_day_df['Year'].tolist()[0]
#         data['Month'] = station_day_df['Month'].tolist()[0]
#         data['Day'] = station_day_df['Day'].tolist()[0]
#         data['year_month_day'] = station_day_df['year_month_day'].tolist()[0]
#         data['Location ID'] = station_day_df['Location ID'].tolist()[0]
#         data['Latitude'] = station_day_df['Latitude'].tolist()[0]
#         data['Longitude'] = station_day_df['Longitude'].tolist()[0]
#         data['Elevation'] = station_day_df['Elevation'].tolist()[0]
        
#         data['Hour'] = station_day_df['Hour'].to_numpy()
#         data['DHI'] = station_day_df['DHI'].to_numpy()
#         data['DNI'] = station_day_df['DNI'].to_numpy()
#         data['GHI'] = station_day_df['GHI'].to_numpy()
#         data['Clearsky DHI'] = station_day_df['Clearsky DHI'].to_numpy()
#         data['Clearsky DNI'] = station_day_df['Clearsky DNI'].to_numpy()
#         data['Clearsky GHI'] = station_day_df['Clearsky GHI'].to_numpy()
#         data['Cloud Type'] = station_day_df['Cloud Type'].to_numpy()
#         data['Dew Point'] = station_day_df['Dew Point'].to_numpy()
#         data['Solar Zenith Angle'] = station_day_df['Solar Zenith Angle'].to_numpy()
#         data['Surface Albedo'] = station_day_df['Surface Albedo'].to_numpy()
#         data['Wind Speed'] = station_day_df['Wind Speed'].to_numpy()
#         data['Precipitable Water'] = station_day_df['Precipitable Water'].to_numpy()
#         data['Wind Direction'] = station_day_df['Wind Direction'].to_numpy()
#         data['Relative Humidity'] = station_day_df['Relative Humidity'].to_numpy()
#         data['Temperature'] = station_day_df['Temperature'].to_numpy()
#         data['Pressure'] = station_day_df['Pressure'].to_numpy()
#         data['Global Horizontal UV Irradiance (280-400nm)'] = station_day_df['Global Horizontal UV Irradiance (280-400nm)'].to_numpy()
#         data['Global Horizontal UV Irradiance (295-385nm)'] = station_day_df['Global Horizontal UV Irradiance (295-385nm)'].to_numpy()
        
#         solar_data_daily_list.append(data)

# pprint(solar_data_daily_list)
# solar_data_daily_df = pd.DataFrame(solar_data_daily_list)
# print(solar_data_daily_df)


# solar_data_group_day_df['Hour'] = solar_data_group_day_df['Hour'].apply(lambda x: [x], convert_dtype=False)
# solar_data_group_day_df['DHI'] = solar_data_group_day_df['DHI'].apply(lambda x: np.asarray([x]), convert_dtype=False)
# solar_data_group_day_df['DNI'] = solar_data_group_day_df['DNI'].apply(lambda x: np.asarray([x]), convert_dtype=False)
# solar_data_group_day_df['GHI'] = solar_data_group_day_df['GHI'].apply(lambda x: np.asarray([x]), convert_dtype=False)
# solar_data_group_day_df['Clearsky DHI'] = solar_data_group_day_df['Clearsky DHI'].apply(lambda x: np.asarray([x]), convert_dtype=False)
# solar_data_group_day_df['Clearsky DNI'] = solar_data_group_day_df['Clearsky DNI'].apply(lambda x: np.asarray([x]), convert_dtype=False)
# solar_data_group_day_df['Clearsky GHI'] = solar_data_group_day_df['Clearsky GHI'].apply(lambda x: np.asarray([x]), convert_dtype=False)
# solar_data_group_day_df['Cloud Type'] = solar_data_group_day_df['Cloud Type'].apply(lambda x: np.asarray([x]), convert_dtype=False)
# solar_data_group_day_df['Dew Point'] = solar_data_group_day_df['Dew Point'].apply(lambda x: np.asarray([x]), convert_dtype=False)
# solar_data_group_day_df['Solar Zenith Angle'] = solar_data_group_day_df['Solar Zenith Angle'].apply(lambda x: np.asarray([x]), convert_dtype=False)
# solar_data_group_day_df['Surface Albedo'] = solar_data_group_day_df['Surface Albedo'].apply(lambda x: np.asarray([x]), convert_dtype=False)
# solar_data_group_day_df['Wind Speed'] = solar_data_group_day_df['Wind Speed'].apply(lambda x: np.asarray([x]), convert_dtype=False)
# solar_data_group_day_df['Precipitable Water'] = solar_data_group_day_df['Precipitable Water'].apply(lambda x: np.asarray([x]), convert_dtype=False)
# solar_data_group_day_df['Wind Direction'] = solar_data_group_day_df['Wind Direction'].apply(lambda x: np.asarray([x]), convert_dtype=False)
# solar_data_group_day_df['Relative Humidity'] = solar_data_group_day_df['Relative Humidity'].apply(lambda x: np.asarray([x]), convert_dtype=False)
# solar_data_group_day_df['Temperature'] = solar_data_group_day_df['Temperature'].apply(lambda x: np.asarray([x]), convert_dtype=False)
# solar_data_group_day_df['Pressure'] = solar_data_group_day_df['Pressure'].apply(lambda x: np.asarray([x]), convert_dtype=False)
# solar_data_group_day_df['Global Horizontal UV Irradiance (280-400nm)'] = solar_data_group_day_df['Global Horizontal UV Irradiance (280-400nm)'].apply(lambda x: np.asarray([x]), convert_dtype=False)
# solar_data_group_day_df['Global Horizontal UV Irradiance (295-385nm)'] = solar_data_group_day_df['Global Horizontal UV Irradiance (295-385nm)'].apply(lambda x: np.asarray([x]), convert_dtype=False)


#         data['Year'] = station_day_df['Year'].tolist()[0]
#         data['Month'] = station_day_df['Month'].tolist()[0]
#         data['Day'] = station_day_df['Day'].tolist()[0]
#         data['year_month_day'] = station_day_df['year_month_day'].tolist()[0]
#         data['Location ID'] = station_day_df['Location ID'].tolist()[0]
#         data['Latitude'] = station_day_df['Latitude'].tolist()[0]
#         data['Longitude'] = station_day_df['Longitude'].tolist()[0]
#         data['Elevation'] = station_day_df['Elevation'].tolist()[0]

flatten = lambda t: t.tolist()
print("doing agg")

solar_data_group_day_df = solar_data_df
solar_data_group_day_df = solar_data_group_day_df.groupby(['year_month_day', 'Location ID'], as_index=False)
agg_dict = {
    

    'Year': 'first',
    'Month': 'first',
    'Day': 'first',
    'year_month_day':  'first',
    'Location ID':  'first',
    'Latitude':  'first',
    'Longitude':  'first',
    'Elevation':  'first',

    'Hour':  flatten,
    'DHI': flatten,
    'DNI': flatten,
    'GHI': flatten,
    'Clearsky DHI': flatten,
    'Clearsky DNI': flatten,
    'Clearsky GHI': flatten,
    'Cloud Type': flatten,
    'Dew Point': flatten,
    'Solar Zenith Angle': flatten,
    'Surface Albedo': flatten,
    'Wind Speed': flatten,
    'Precipitable Water': flatten,
    'Wind Direction': flatten,
    'Relative Humidity': flatten,
    'Temperature': flatten,
    'Pressure': flatten,
    'Global Horizontal UV Irradiance (280-400nm)': flatten,
    'Global Horizontal UV Irradiance (295-385nm)': flatten,
}

solar_data_daily_df = solar_data_group_day_df.agg(agg_dict)
solar_data_daily_df = solar_data_daily_df.sort_values(by=['Year', 'Month', "Day", "Location ID"])
solar_data_daily_df['GHI+1'] = solar_data_daily_df.groupby('Location ID')['GHI'].shift(-1)
solar_data_daily_df['GHI+2'] = solar_data_daily_df.groupby('Location ID')['GHI'].shift(-2)
solar_data_daily_df['GHI+3'] = solar_data_daily_df.groupby('Location ID')['GHI'].shift(-3)
solar_data_daily_df['GHI+4'] = solar_data_daily_df.groupby('Location ID')['GHI'].shift(-4)
solar_data_daily_df['GHI+5'] = solar_data_daily_df.groupby('Location ID')['GHI'].shift(-5)
solar_data_daily_df['GHI+6'] = solar_data_daily_df.groupby('Location ID')['GHI'].shift(-6)
solar_data_daily_df['GHI+7'] = solar_data_daily_df.groupby('Location ID')['GHI'].shift(-7)
solar_data_daily_df = solar_data_daily_df[solar_data_daily_df['GHI+1'].notna()]
solar_data_daily_df = solar_data_daily_df[solar_data_daily_df['GHI+2'].notna()]
solar_data_daily_df = solar_data_daily_df[solar_data_daily_df['GHI+3'].notna()]
solar_data_daily_df = solar_data_daily_df[solar_data_daily_df['GHI+4'].notna()]
solar_data_daily_df = solar_data_daily_df[solar_data_daily_df['GHI+5'].notna()]
solar_data_daily_df = solar_data_daily_df[solar_data_daily_df['GHI+6'].notna()]
solar_data_daily_df = solar_data_daily_df[solar_data_daily_df['GHI+7'].notna()]

solar_data_daily_df['Wind Speed+1'] = solar_data_daily_df.groupby('Location ID')['Wind Speed'].shift(-1)
solar_data_daily_df['Wind Speed+2'] = solar_data_daily_df.groupby('Location ID')['Wind Speed'].shift(-2)
solar_data_daily_df['Wind Speed+3'] = solar_data_daily_df.groupby('Location ID')['Wind Speed'].shift(-3)
solar_data_daily_df['Wind Speed+4'] = solar_data_daily_df.groupby('Location ID')['Wind Speed'].shift(-4)
solar_data_daily_df['Wind Speed+5'] = solar_data_daily_df.groupby('Location ID')['Wind Speed'].shift(-5)
solar_data_daily_df['Wind Speed+6'] = solar_data_daily_df.groupby('Location ID')['Wind Speed'].shift(-6)
solar_data_daily_df['Wind Speed+7'] = solar_data_daily_df.groupby('Location ID')['Wind Speed'].shift(-7)
solar_data_daily_df = solar_data_daily_df[solar_data_daily_df['Wind Speed+1'].notna()]
solar_data_daily_df = solar_data_daily_df[solar_data_daily_df['Wind Speed+2'].notna()]
solar_data_daily_df = solar_data_daily_df[solar_data_daily_df['Wind Speed+3'].notna()]
solar_data_daily_df = solar_data_daily_df[solar_data_daily_df['Wind Speed+4'].notna()]
solar_data_daily_df = solar_data_daily_df[solar_data_daily_df['Wind Speed+5'].notna()]
solar_data_daily_df = solar_data_daily_df[solar_data_daily_df['Wind Speed+6'].notna()]
solar_data_daily_df = solar_data_daily_df[solar_data_daily_df['Wind Speed+7'].notna()]

solar_data_daily_df['Temperature+1'] = solar_data_daily_df.groupby('Location ID')['Temperature'].shift(-1)
solar_data_daily_df['Temperature+2'] = solar_data_daily_df.groupby('Location ID')['Temperature'].shift(-2)
solar_data_daily_df['Temperature+3'] = solar_data_daily_df.groupby('Location ID')['Temperature'].shift(-3)
solar_data_daily_df['Temperature+4'] = solar_data_daily_df.groupby('Location ID')['Temperature'].shift(-4)
solar_data_daily_df['Temperature+5'] = solar_data_daily_df.groupby('Location ID')['Temperature'].shift(-5)
solar_data_daily_df['Temperature+6'] = solar_data_daily_df.groupby('Location ID')['Temperature'].shift(-6)
solar_data_daily_df['Temperature+7'] = solar_data_daily_df.groupby('Location ID')['Temperature'].shift(-7)
solar_data_daily_df = solar_data_daily_df[solar_data_daily_df['Temperature+1'].notna()]
solar_data_daily_df = solar_data_daily_df[solar_data_daily_df['Temperature+2'].notna()]
solar_data_daily_df = solar_data_daily_df[solar_data_daily_df['Temperature+3'].notna()]
solar_data_daily_df = solar_data_daily_df[solar_data_daily_df['Temperature+4'].notna()]
solar_data_daily_df = solar_data_daily_df[solar_data_daily_df['Temperature+5'].notna()]
solar_data_daily_df = solar_data_daily_df[solar_data_daily_df['Temperature+6'].notna()]
solar_data_daily_df = solar_data_daily_df[solar_data_daily_df['Temperature+7'].notna()]


print(solar_data_daily_df)
print(solar_data_daily_df.loc[solar_data_daily_df["Location ID"] == 960460])
solar_data_daily_df.to_pickle("./data/solar/solar_data_daily.pkl")


# lats = np.sort(station_locations_df['Latitude'].unique())
# lons = np.sort(station_locations_df['Longitude'].unique())

# val_pivot_df = station_locations_df.pivot(index='Latitude', columns='Longitude', values='Location ID')
# lons = val_pivot_df.columns.values
# lats = val_pivot_df.index.values
# xx, yy = np.meshgrid(lons,lats) 
# data_values = val_pivot_df.values


# fig, ax = plt.subplots(1,1)
# ax.pcolormesh(xx, yy, data_values, shading='auto')
# ax.scatter(-84.3880, 33.7490, marker='.', color='k',)
# ax.annotate("Atlanta", (-84.3880+.01, 33.749+.01))