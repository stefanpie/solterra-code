import os
import glob
import pandas as pd
import csv
from tqdm import tqdm


solar_data_file_paths = glob.glob("./data/solar/raw/*.csv")
station_dfs = []

for fp in tqdm(solar_data_file_paths):
    station_info_df = pd.read_csv(fp, header=0, nrows=1)
    # print(station_info_df.columns)
    station_data_df = pd.read_csv(fp, header=0, skiprows=2)
    station_data_df = station_data_df.drop(columns=['Unnamed: 24'])
    station_df = station_data_df
    # print(station_info_df["Location ID"].loc[0])
    station_df["Location ID"] = station_info_df["Location ID"].loc[0]
    station_df["Latitude"] = station_info_df["Latitude"].loc[0]
    station_df["Longitude"] = station_info_df["Longitude"].loc[0]
    station_df["Time Zone"] = station_info_df["Time Zone"].loc[0]
    station_df["Elevation"] = station_info_df["Elevation"].loc[0]
    station_df["Local Time Zone"] = station_info_df["Local Time Zone"].loc[0]
    station_dfs.append(station_df)

all_data_df = pd.concat(station_dfs, ignore_index=True)
print(all_data_df)

all_data_df.to_csv("./data/solar/solar_data_combined.csv", index=False)

station_locations_df = all_data_df[["Location ID", "Latitude", "Longitude"]].groupby(by="Location ID", dropna=False).mean().reset_index()
station_locations_df.to_csv("./data/solar/solar_data_station_locations.csv", index=False)
