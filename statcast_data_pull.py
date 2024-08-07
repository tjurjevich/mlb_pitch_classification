"""
This script will scrape full season's worth of statcast data. This data will be saved out to a csv file in
the current directory, and will be used for eventual model training and testing.
"""
from pybaseball import statcast, cache
import warnings
import datetime
import os

current_directory = os.getcwd()

warnings.filterwarnings(action='ignore', category=FutureWarning)

cache.enable()

def statsPull():    
    return statcast(start_dt='2024-03-28', end_dt=datetime.datetime.today().strftime('%Y-%m-%d'), verbose=False)


if __name__ == "__main__":
    data = statsPull()
    data.to_csv(f'{current_directory}/statcast_data.csv', index=False)

