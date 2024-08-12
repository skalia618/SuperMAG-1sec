from matplotlib.ticker import LogFormatterSciNotation
import numpy as np

# Parameters/constants
START_YEAR = 2005
END_YEAR = 2020
THRESHOLD = 0.03
VDM = 1e-3 # velocity of DM (in c)

def secs_in_year(year):
    """
    Computes number of seconds in year (accounting for leap years)
    """
    days = 365 if year % 4 != 0 else 366
    return days * 24 * 60 * 60

def year_start_sec(year, start_year = START_YEAR):
    """
    Computes first second of year, beginning from start_year
    (If year < start_year, then will return negative seconds)
    """
    year_ = start_year
    secs = 0

    if year >= start_year:
        while year_ != year:
            secs += secs_in_year(year_)
            year_ += 1
    else:
        while year_ != year:
            secs -= secs_in_year(year_ - 1)
            year_ -= 1
    
    return secs

TOTAL_TIME = year_start_sec(END_YEAR + 1) # total length of dataset (in s)

def secs_in_month(year, month):
    """
    Computes number of seconds in month of specified year
    """
    assert month in range(12), "month must be integer less than 12"
    if month in [0, 2, 4, 6, 7, 9, 11]:
        days = 31
    elif month in [3, 5, 8, 10]:
        days = 30
    elif month == 1:
        days = 28 if year % 4 != 0 else 29
    return days * 24 * 60 * 60

def month_start_sec(year, month, start_year = START_YEAR):
    """
    Computes first second of month, beginning from start_year
    """
    month_ = 0
    secs = year_start_sec(year, start_year)
    while month_ != month:
        secs += secs_in_month(year, month_)
        month_ += 1
    return secs

def get_stationarity_chunks(stationarity_time):
    """
    Computes stationarity chunks
    For each chunk, returns [index, start, end_exclusive]
    For annual or monthly, are computed according to calendar years/months
    For daily, weekly, or integer stationarity_time, are computed as equal-length chunks
    (If TOTAL_TIME is not exact multiple of chunk_length, final chunk is extended to contain rest of dataset)
    """
    if stationarity_time == "monthly":
        chunk_list = [[12 * (year - START_YEAR) + month,
                       month_start_sec(year, month),
                       month_start_sec(year, month) + secs_in_month(year, month)]
                      for year in range(START_YEAR, END_YEAR + 1) for month in range(12)]
    elif stationarity_time == "annual":
        chunk_list = [[year - START_YEAR, year_start_sec(year), year_start_sec(year + 1)] for year in range(START_YEAR, END_YEAR + 1)]
    else:
        if stationarity_time == "daily":
            chunk_length = 24 * 60 * 60
        elif stationarity_time == "weekly":
            chunk_length = 7 * 24 * 60 * 60
        else:
            chunk_length = stationarity_time
        total_chunks = TOTAL_TIME // chunk_length
        chunk_list = [[k, k * chunk_length, (k + 1) * chunk_length] for k in range(total_chunks)]
        chunk_list[-1][2] = TOTAL_TIME
    return chunk_list

def maxprime(n):
    """
    Given a number n, calculate its largest prime factor
    """
    # Deal with base case
    if n == 1:
        return 1

    # Find upper_bound for checks
    upper_bound = int(np.sqrt(n))

    # Iterate through all numbers between 2 and the upper_bound
    for i in range(2, upper_bound + 1):
        if n % i == 0:
            # If n is divisible by i, recursively find the maximum prime of n / i
            return maxprime(n // i)

    # Because we are iterating up, this will return the largest prime factor
    return n

def coherence_times(total_time = TOTAL_TIME):
    """
    Calculates the coherence times used in the analysis, as given by Eq. (66) in 2108.08852
    """
    # Initialize return value
    times = []

    # Find max n
    max_n = int(round(-0.5 * np.log(VDM ** -2 / total_time) / np.log(1.0 + THRESHOLD)))

    for n in range(max_n + 1):
        # Find the raw coherence time
        raw_coherence_time = total_time / (1.0 + THRESHOLD) ** (2 * n)
        rounded = int(round(raw_coherence_time))

        # Find number in [rounded-10, .., rounded+10] with smallest max prime
        number = 0
        max_prime = np.iinfo(np.int64).max
        for candidate in range(rounded - 10, rounded + 11):
            x = maxprime(candidate)
            if x < max_prime:
                number = candidate
                max_prime = x

        times.append(number)

    # Values should be in descending order.
    return times

def frequencies_from_coherence_times(coherence_times):
    """
    Calculates relevant frequencies associated with each coherence time
    For each coherence time, returns a tuple containing (start, end, base_frequency), so that
    lowest relevant frequency is start * base_frequency and highest (inclusive) is end * base_frequency
    """
    # Calculate base frequencies, i.e. reciprocal of coherence times
    # Since coherence_times is in descending order, these will be in ascending order
    base_frequencies = [1.0 / x for x in coherence_times]

    # This constructs all frequency bins except for the highest one
    frequency_bins = []
    i_min = int(VDM ** -2 / (1.0 + THRESHOLD))
    for bin_index in range(len(base_frequencies) - 1):
        lower_bf, higher_bf = base_frequencies[bin_index], base_frequencies[bin_index + 1]

        # Find start and end multiples of frequency
        start = 0 if bin_index == 0 else i_min
        end = int(higher_bf * i_min / lower_bf)
        if end * lower_bf >= higher_bf * i_min:
            end -= 1

        # Append tuple containing lowest_multiple, highest_multiple and spacing to frequency_bins
        frequency_bins.append((start, end, lower_bf))

    # Add highest frequency bin
    last_coherence_time_in_seconds = coherence_times[-1]
    highest_frequency_end = last_coherence_time_in_seconds - 1
    highest_frequency_bf = 1.0 / last_coherence_time_in_seconds

    frequency_bins.append((i_min, highest_frequency_end, highest_frequency_bf))

    return frequency_bins

def div_ceil(a, b):
    """
    Shorthand for ceiling integer division (i.e., ceiling analog of //)
    """
    return -(a // -b)

def spectra_freqs(chunk_id, stationarity_time, window, downsample = None):
    """
    Returns the frequencies at which the spectra for chunk chunk_id are computed
    """
    if downsample == None: downsample = window // 10
    chunk_list = get_stationarity_chunks(stationarity_time)
    chunk = chunk_list[chunk_id]
    chunk_length = chunk[2] - chunk[1]
    freqs_length = div_ceil(chunk_length - 6 * window, downsample)
    return (3 * window + downsample * np.arange(freqs_length)) / chunk_length

def calculate_overlap(start1, end1, start2, end2):
    """
    Counts number of overlapping seconds between two intervals
    Note that both ends are exclusive
    Example:
    calculate_overlap(0, 4, 0, 4)
        = count(overlapping seconds: 0, 1, 2, 3)
        = 4
    """
    return max(0, (min(end1, end2)) - max(start1, start2))


def find_overlap_chunks(start, end, stationarity_time):
    """
    Computes overlap of period (start, end) with all stationarity chunks
    For each stationarity chunk that it overlaps with, returns (chunk index, overlap)
    """
    stationarity_chunks = []

    chunk_list = get_stationarity_chunks(stationarity_time)
    for chunk in chunk_list:
        chunk_ind, start_second, end_second = chunk

        # Record overlap if any
        overlap = calculate_overlap(start, end, start_second, end_second)
        if overlap > 0:
            stationarity_chunks.append((chunk_ind, overlap))

    return stationarity_chunks

class CustomTicker(LogFormatterSciNotation):
    """
    Custom ticker to render labels 0.1, 1, and 10 without scientific notation
    """
    def __call__(self, x, pos = None): 
        if x not in [0.1, 1, 10]:
            return LogFormatterSciNotation.__call__(self, x, pos = None) 
        else:
            return "{x:g}".format(x = x)
