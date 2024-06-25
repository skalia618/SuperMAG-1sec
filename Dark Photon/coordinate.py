from loader import get_all_stations
import numpy as np
import os
from utils import *

def load_station_data(path = STATION_DATA_PATH):
    """
    Returns station data
    Each entry includes: year, station name, latitude, longitude, and IGRF declination for the year
    (Latitude and longitude do not change from year to year)
    """
    return np.genfromtxt(
        path,
        dtype=[('year', np.int32), ('IAGA', 'U4'), ('geolat', np.float64),
               ('geolon', np.float64), ('declination', np.float64)],
        names=True
    )

# Initialize once
STATION_DATA = load_station_data()

def get_station_coordinates(data = STATION_DATA, stations = get_all_stations()):
    """
    Returns dictionary of all station coordinates
    Keys are station names, and values are (latitude, longitude) pairs
    """

    coordinates = {}

    for station in stations:
        # Determine index of first entry corresponding to station
        # (Since latitude and longitude are the same for all years, any entry will do)
        ind = np.where(data['IAGA'] == station)[0][0]

        coordinates[station] = (data[ind]['geolat'], data[ind]['geolon'])
    
    return coordinates

def interpolate_angle(station, seconds, data = STATION_DATA, start_year = START_YEAR):
    """
    Interpolate the declination angle (in radians) for the given station and seconds.
    (seconds are measured beginning from start_year)
    """
    # Filter data for the specific station
    mask = (data['IAGA'] == station)
    filtered_data = data[mask]

    # Calculate seconds for each entry
    # Declination angles are given for first second of 180th day of each year
    year_seconds = np.array([year_start_sec(row['year'], start_year) + 179 * 24 * 60 * 60
                            for row in filtered_data])

    # Ensure entries are sorted
    # Sort by year_seconds to ensure monotonicity
    sort_indices = np.argsort(year_seconds)
    sorted_year_seconds = year_seconds[sort_indices]
    sorted_declinations = filtered_data['declination'][sort_indices]

    return np.interp(seconds, sorted_year_seconds, sorted_declinations) * np.pi / 180
