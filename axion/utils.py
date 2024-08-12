import numpy as np
from params import *
from scipy import integrate
from scipy.special import factorial, lpmv

# Paths to external data
RAW_DIR = '../raw'
STATION_DATA_PATH = '../station_data.txt'
PERMUTATION_PATH = '../permutation.csv'
IGRF_PATH = '../IGRF13coeffs.csv'

# Constants
VDM = 1e-3 # velocity of DM (in c)
RHODM = 2.305e-42 # density of DM (in GeV^4)
R = 3.2323e22 # radius of Earth (in GeV^-1)
SUPERMAG_NAN = 999999. # value used by SUPERMAG to denote absent values

def get_proj_aux_dir():
    """
    Returns directory for projections/auxiliary timeseries
    """
    dirname = get_dirname(include_window = False, include_threshold = False, include_boundparams = False)
    return f'proj_aux/{dirname}'

def get_stationarity_chunk_dir(chunk_ind, window = WINDOW):
    """
    Returns spectra directory for chunk chunk_ind
    """
    dirname = get_dirname(include_threshold = False, include_boundparams = False, window = window)
    return f'spectra/{dirname}/chunk_{chunk_ind:04d}'

def get_analysis_vars_dir(subset = SUBSET):
    """
    Returns directory for analysis variables
    Includes option to get directory for subset analysis (used in candidate analysis)
    """
    dirname = get_dirname(include_boundparams = False, subset = subset)
    return f'analysis_vars/{dirname}'

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

def get_stationarity_chunks():
    """
    Computes stationarity chunks
    For each chunk, returns [index, start, end_exclusive]
    For annual or monthly, are computed according to calendar years/months
    For daily, weekly, or integer STATIONARITY_TIME, are computed as equal-length chunks
    (If TOTAL_TIME is not exact multiple of chunk_length, final chunk is extended to contain rest of dataset)
    """
    if STATIONARITY_TIME == "monthly":
        chunk_list = [[12 * (year - START_YEAR) + month,
                       month_start_sec(year, month),
                       month_start_sec(year, month) + secs_in_month(year, month)]
                      for year in range(START_YEAR, END_YEAR + 1) for month in range(12)]
    elif STATIONARITY_TIME == "annual":
        chunk_list = [[year - START_YEAR, year_start_sec(year), year_start_sec(year + 1)] for year in range(START_YEAR, END_YEAR + 1)]
    else:
        if STATIONARITY_TIME == "daily":
            chunk_length = 24 * 60 * 60
        elif STATIONARITY_TIME == "weekly":
            chunk_length = 7 * 24 * 60 * 60
        else:
            chunk_length = STATIONARITY_TIME
        total_chunks = TOTAL_TIME // chunk_length
        chunk_list = [[k, k * chunk_length, (k + 1) * chunk_length] for k in range(total_chunks)]
        chunk_list[-1][2] = TOTAL_TIME
    return chunk_list

CHUNK_LIST = get_stationarity_chunks()

def lat_lon_deg_to_theta_phi_radian(lat, lon):
    """
    Converts latitude and longitude (in degrees) to spherical coordinates (in radians)
    """
    theta = np.pi / 2 - lat * np.pi / 180
    phi = lon * np.pi / 180
    return theta, phi

def rotate_fields(bn, be, t):
    """
    Given input fields, applies rotation

    B_n' = B_n cos(t) - B_e sin(t)
    B_e' = B_e cos(t) + B_n sin(t)

    to the fields and return (B_n', B_e').
    """
    bnp = bn * np.cos(t) - be * np.sin(t)
    bep = be * np.cos(t) + bn * np.sin(t)
    return (bnp, bep)

def gh_to_C(g, h):
    """
    Converts from g and h Gauss coefficients to C [see Eq. (12) of 2112.09620]
    
    Note: C[l] consists of [m = 0 entry, positive m entries, negative m entries]
    (so that C[l][m] returns correct entry even for negative m)
    """
    C = [[]]
    for l in range(1, len(g)):
        C.append([])
        # m = 0 entry
        C[l].append(np.sqrt(np.pi / (2 * l + 1)) * g[l][0])
        # Positive m
        for m in range(1, l + 1):
            C[l].append((-1) ** m * np.sqrt(2 * np.pi / (2 * l + 1)) * (g[l][m] - 1j * h[l][m]))
        # Negative m are related to positive m
        for m in range(-l, 0):
            C[l].append((-1) ** m * np.conj(C[l][-m]))
    return C

def interpolate_C(C, seconds, IGRFyears):
    """
    Interpolate C coefficients to given seconds
    Given values should correspond to IGRFyears
    """
    # Find seconds corresponding to each IGRF year
    year_seconds = [year_start_sec(year) for year in IGRFyears]
    Cinterp = [[]]
    for l in range(1, len(C)):
        Cinterp.append([])
        for m in range(len(C[l])):
            # Interpolate given l,m mode to desired seconds
            Cinterp[l].append(np.interp(seconds, year_seconds, C[l][m]))
    return Cinterp

def Philm_theta(theta, phi, l, m):
    """
    Returns theta component of Phi vector spherical harmonic
    Defined as -dYlm/dphi / sin(theta), where Ylm is scalar spherical harmonic
    """
    prefactor = np.sqrt((2 * l + 1) / (4 * np.pi) * factorial(l - m) / factorial(l + m))
    exp = np.cos(m * phi) + 1j * np.sin(m * phi)
    return -prefactor * 1j * m * exp * lpmv(m, l, np.cos(theta)) / np.sin(theta)

def Philm_phi(theta, phi, l, m):
    """
    Returns phi component of Phi vector spherical harmonic
    Defined as dYlm/dtheta, where Ylm is scalar spherical harmonic
    """
    prefactor = np.sqrt((2 * l + 1) / (4 * np.pi) * factorial(l - m) / factorial(l + m))
    exp = np.cos(m * phi) + 1j * np.sin(m * phi)
    return prefactor * exp * (lpmv(m + 1, l, np.cos(theta)) + m * lpmv(m, l, np.cos(theta)) / np.tan(theta))

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

def find_years(start, end_exclusive, start_year = START_YEAR, end_year = END_YEAR):
    """
    Determine overlaps of range (start, end_exclusive) with each year
    In addition to returning list of the years that overlap,
    for each year, also returns range of overlapping indices within the year
    """
    years = []
    ranges = []
    for year in range(start_year, end_year + 1):
        # Find year's start and end
        year_start = year_start_sec(year, start_year)
        year_end_exclusive = year_start_sec(year + 1, start_year)
        if calculate_overlap(year_start, year_end_exclusive, start, end_exclusive) > 0:
            years.append(year)

            # Indices used
            start_within_year = max(0, start - year_start)
            end_within_year = min(year_end_exclusive,
                                  end_exclusive) - year_start
            ranges.append((start_within_year, end_within_year))

    return (years, ranges)

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

def find_overlap_chunks(coherence_time, coherence_chunk):
    """
    Computes overlap of a given coherence chunk with all stationarity chunks
    For each stationarity chunk that it overlaps with, returns (chunk index, overlap)
    """
    stationarity_chunks = []

    # First calculate start and end second for the coherence_time
    start_coh = coherence_time * coherence_chunk
    end_coh = min(start_coh + coherence_time, TOTAL_TIME)

    chunk_list = get_stationarity_chunks()
    for chunk in chunk_list:
        chunk_ind, start_second, end_second = chunk

        # Record overlap if any
        overlap = calculate_overlap(start_coh, end_coh, start_second, end_second)
        if overlap > 0:
            stationarity_chunks.append((chunk_ind, overlap))

    return stationarity_chunks

def spectra_freqs(chunk_id, window = WINDOW, downsample = DOWNSAMPLE):
    """
    Returns the frequencies at which the spectra for chunk chunk_id are computed
    """
    chunk = CHUNK_LIST[chunk_id]
    chunk_length = chunk[2] - chunk[1]
    freqs_length = div_ceil(chunk_length - 6 * window, downsample)
    return (3 * window + downsample * np.arange(freqs_length)) / chunk_length

def calculate_logpdf(g, sf, zf):
    """
    Calculates posterior log PDF, as in Eq. (C23) of 2112.09620
    The norm N is set to 1, as it will be fixed when computing the CDF
    """
    term_1 = np.log(np.sqrt(np.sum(4 * g ** 2 * sf ** 4 / (1 + g ** 2 * sf ** 2) ** 2)))

    term_2 = -np.sum(np.log(1 + g ** 2 * sf ** 2))

    term_3 = np.sum(-np.abs(zf) ** 2 / (1 + g ** 2 * sf ** 2))

    return term_1 + term_2 + term_3

def calculate_cdf(sf, zf):
    """
    Calculates posterior CDF, for given analysis variables
    sf and zf shapes: (number of coherence chunks)

    In order to integrate the PDF, it is important to identify an appropriate integration range.
    This is done by scanning over g_agamma to find the maximum of the PDF.
    The upper limit of integration is the g_agamma where log PDF reaches TAIL_START below its maximum.

    Returns two arrays: (grid of g_agamma's, CDF calculated along grid)
    If calculation fails because CDF has significant support above g_agamma = 10 ** (MAX_LOG10G - 1), returns None
    """
    # Calculate log PDF across epsilon grid and identify maximum
    grid_logpdf = np.array([calculate_logpdf(g, sf, zf) for g in SCAN_GRID])
    max_ind = np.argmax(grid_logpdf)
    # If maximum is above g_agamma = 10 ** (MAX_LOG10G - 1), exit
    if SCAN_GRID[max_ind] >= 10 ** (MAX_LOG10G - 1):
        return None

    # Identify where log PDF decreases to TAIL_START below its max
    max_logpdf = grid_logpdf[max_ind]
    tail = np.where(grid_logpdf[max_ind:] <= max_logpdf - TAIL_START)[0] + max_ind
    if len(tail) != 0:
        upper = SCAN_GRID[tail[0]]
    else:
        # If this doesn't occur within SCAN_GRID, exit
        return None

    # PDF (normalized to equal 1 at maximum)
    pdf = lambda g: np.exp(calculate_logpdf(g, sf, zf) - max_logpdf)

    # (Trapezoidal) integrate PDF from g_agamma = 0 up to start of tail
    int_grid = np.linspace(0, upper, NUM_G)
    grid_pdf = np.insert([pdf(g) for g in int_grid[1:]], 0, 0)
    cdf = integrate.cumtrapz(grid_pdf, x = int_grid, initial = 0)

    # Normalize so that total integral is 1
    cdf /= cdf[-1]

    return (int_grid, cdf)
