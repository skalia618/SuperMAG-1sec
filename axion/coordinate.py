import csv
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

def load_IGRFcoeffs(path = IGRF_PATH):
    """
    Loads g and h Gauss coefficients (in nT) at five-year intervals
    First relevant row of data file contains years
    Other rows contain coefficients for each year (for fixed l,m), plus derivative for current period
    g coefficients exist for l>=m>=0 (except l=0)
    h coefficients exist for l>=m>=1
    """
    file = open(IGRF_PATH)
    reader = csv.reader(file, delimiter = ',')

    # Initialize g and h so that g[0][0] and h[0][0] are empty
    g = [[[]]]
    h = [[[]]]

    for row in reader:
        # Load years
        if row[0] == 'g/h':
            IGRFyears = [int(x) for x in row[3:-1]]
            # Instead of last entry, add final year + 5
            IGRFyears.append(IGRFyears[-1] + 5)
            IGRFyears = np.array(IGRFyears)
            # Truncate list beginning at START_YEAR
            start_ind = np.searchsorted(IGRFyears, START_YEAR)
            IGRFyears = IGRFyears[start_ind:]

        # Load g values
        elif row[0] == 'g':
            l = int(row[1])
            # Only load if l <= LMAX
            if l > LMAX: continue
            m = int(row[2])
            # Create g[l] if it doesn't already exist
            if len(g) <= l:
                g.append([])
            g[l].append([float(x) for x in row[3:-1]])
            # Extrapolate coefficient to added year
            g[l][m].append(g[l][m][-1] + 5 * float(row[-1]))
            g[l][m] = np.array(g[l][m])[start_ind:]

        # Load h values
        elif row[0] == 'h':
            l = int(row[1])
            if l > LMAX: continue
            # Create h[l] so that h[l][0] is empty
            if len(h) <= l:
                h.append([[]])
            h[l].append([float(x) for x in row[3:-1]])
            h[l][m].append(h[l][m][-1] + 5 * float(row[-1]))
            h[l][m] = np.array(h[l][m])[start_ind:]

    return IGRFyears, g, h

def interpolate_angle(station, seconds, data = STATION_DATA, start_year = START_YEAR):
    """
    Interpolate the declination angle (in radians) for the given station and seconds
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
