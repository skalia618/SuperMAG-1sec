from glob import glob
import numpy as np
import os
import re
import struct
import sys
from utils import *

# The end of each component is marked with the following byte string
END_OF_FIELD_COMPONENT = bytes([0, 0, 0, 2])

# Size of header of every file in bytes
HEADER_SIZE = 315 * 4

# Size of buffer between fields
BUFFER_SIZE = 25 * 4

# Size of final buffer before end of file
FINAL_BUFFER_SIZE = 48 * 4
FINAL_BUFFER_SIZE_2018_2019 = 29 * 4 # for 2018 and 2019, the final buffer size is different

SIZE_OF_FLOAT = 4
SIZE_OF_DOUBLE = 8

# Cached arrays
CACHE = ('', 0, np.array([]), np.array([]))


def validate_buffer_size(buffer_size, year):
    """
    Validate the buffer size based on the year and expected size.
    Return True for single precision, False for double precision, or raise an exception if invalid.
    """

    leap_year = year % 4 == 0
    expected_size = 4 * (365 + leap_year) * 24 * 60 * 60

    if year != 2018 and year != 2019:
        # Check for single precision
        if buffer_size == HEADER_SIZE + 3 * expected_size + 2 * BUFFER_SIZE + FINAL_BUFFER_SIZE:
            return True

        # Check for double precision
        elif buffer_size == HEADER_SIZE + 3 * 2 * expected_size + 2 * BUFFER_SIZE + FINAL_BUFFER_SIZE:
            return False

        # Otherwise something is wrong
        else:
            raise ValueError(
                f"Invalid buffer size: size diff is {buffer_size - (HEADER_SIZE + 3 * expected_size + 2 * BUFFER_SIZE + FINAL_BUFFER_SIZE)}")
    else:
        # Check for single precision
        if buffer_size == HEADER_SIZE + 3 * expected_size + 2 * BUFFER_SIZE + FINAL_BUFFER_SIZE_2018_2019:
            return True

        # Check for double precision
        elif buffer_size == HEADER_SIZE + 3 * 2 * expected_size + 2 * BUFFER_SIZE + FINAL_BUFFER_SIZE_2018_2019:
            return False

        # Otherwise something is wrong
        else:
            raise ValueError(
                f"Invalid buffer size: size diff is {buffer_size - (HEADER_SIZE + 3 * expected_size + 2 * BUFFER_SIZE + FINAL_BUFFER_SIZE_2018_2019)}")

def station_year_file(station, year):
    """
    Returns path of datafile for given station and year
    """
    return os.path.join(RAW_DIR, str(year), f"{year}_{station}_1s_final.xdr")

def load_period(station, start_sec, end_sec_exclusive, cache = CACHE):
    """
    Loads period (start_sec, end_sec_exclusive) from given station

    As the data is stored by year, each year that the period overlaps
    with is loaded in sequentially

    The last year that is loaded will be cached so that the next
    iteration of load_period does not need to load it again
    """
    global CACHE

    # Find years that period overlaps with
    years, ranges = find_years(start_sec, end_sec_exclusive)

    # Initialize f32 array with SUPERMAG_NANs
    bn = np.zeros(end_sec_exclusive - start_sec, dtype=float) + SUPERMAG_NAN
    be = np.zeros(end_sec_exclusive - start_sec, dtype=float) + SUPERMAG_NAN

    written = 0  # index for multi-year periods
    write_to = 0
    for (year, (start_ind, end_ind)) in zip(years, ranges):
        if os.path.exists(station_year_file(station, year)):

            if CACHE[0] == station and CACHE[1] == year:
                # Load from cache if station-year is already cached
                bn_year = CACHE[2]
            else:
                if VERBOSE:
                    print(f'Loading data for {year}')
                    sys.stdout.flush()
                # Load first field if not cached
                # Open file and skip first header
                station_file_path = station_year_file(station, year)
                file = open(station_file_path, 'rb')
                file.seek(HEADER_SIZE, 1)
                is_single_precision = validate_buffer_size(
                    os.path.getsize(station_file_path), year)
                data_type = 'f' if is_single_precision else 'd'

                # NOTE: SUPERMAG data is stored in big-endian, indicated by >.
                array_elements = (365 + (year % 4 == 0)) * 24 * 60 * 60
                array_size = SIZE_OF_FLOAT * \
                    array_elements * (2 - is_single_precision)
                bn_year = np.array(struct.unpack(
                    '>' + data_type * array_elements, file.read(array_size))).astype(float)

            if CACHE[0] == station and CACHE[1] == year:
                # Load from cache if station-year is already cached
                be_year = CACHE[3]
            else:
                # Load second array if not cached
                array_elements = (365 + (year % 4 == 0)) * 24 * 60 * 60
                array_size = SIZE_OF_FLOAT * \
                    array_elements * (2 - is_single_precision)

                # Ensure we hit end-of-field
                end_of_field = file.read(4)
                assert end_of_field == END_OF_FIELD_COMPONENT, f"{bn_full} {end_of_field}"

                # Skip buffer (minus 4 bytes since it includes end-of-field), read data
                file.seek(BUFFER_SIZE - 4, 1)
                be_year = np.array(struct.unpack(
                    '>' + data_type * array_elements, file.read(array_size))).astype(float)

                # Update cache, but only after loading second array
                CACHE = (station, year, bn_year, be_year)

            # Get subarrays for this period
            length = end_ind - start_ind
            bn[write_to : write_to + length] = bn_year[start_ind : end_ind]
            be[write_to : write_to + length] = be_year[start_ind : end_ind]
            written += length

        # Index must be incremented whether or not there is data
        length = end_ind - start_ind
        write_to += length

    if not np.all(bn == SUPERMAG_NAN):
        return (bn, be)
    else:
        return None

def get_all_stations(raw_dir = RAW_DIR):
    """
    Returns list of all stations
    """

    # All files (all inner files (second *) across all years (first *))
    all_files = os.path.join(raw_dir, '*', '*')

    # Initialize set
    stations = set()

    # Seach for station name
    regex_pattern = r"(\d{4})_([A-Za-z0-9]+)_"

    for filename in glob(all_files):
        search = re.search(regex_pattern, os.path.basename(filename))
        if search is not None:
            stations.add(search.group(2))

    return list(stations)

def clean_fields(bn, be):
    """
    Given PRE-ROTATED(!!!) fields, finds all of the SUPERMAG_NAN
    values, sets them to zero, and returns an indicator array
    """

    # Indicator arrays
    bn_nan_indicator = (bn == SUPERMAG_NAN)
    be_nan_indicator = (be == SUPERMAG_NAN)

    # It is an unexpected result to have an absent value for only
    # one field component
    assert (bn_nan_indicator == be_nan_indicator).sum() == len(
        be_nan_indicator), f"got {bn_nan_indicator.sum()} vs {be_nan_indicator.sum()} nans"

    bn[bn_nan_indicator] = 0
    be[be_nan_indicator] = 0

    # Indicator is inverted so that it marks where there is data, rather than nans
    return np.invert(bn_nan_indicator)
