import numpy as np
import sys

# Years of data to use in analysis (END_YEAR is inclusive)
START_YEAR = 2005
END_YEAR = 2020

# Choice of data subset (-1 indicates full dataset)
SUBSET = -1

# Parameters for injected signal (if desired)
INJECT = False
G_AGAMMA = 1e-10 # in GeV^-1
FA = 0.12769081297860388 # in Hz

# Choice of stationarity time
# Preset options are 'daily', 'weekly', 'monthly', or 'annual'
# Alternatively can be a fixed integer, which yields equal-length stationarity chunks
STATIONARITY_TIME = 'weekly'

# Number of l modes of IGRF coefficients to sum over (for axion signal)
LMAX = 4

# Window size for smoothing
WINDOW = 15
DOWNSAMPLE = WINDOW // 10

# Percent level precision of coherence times
THRESHOLD = 0.03

# Range of frequencies to compute bound for
MIN_FREQ = 0.01
MAX_FREQ = 0.5

# Confidence level of bound (and candidate analysis)
CONFIDENCE = 0.95

# Parameters for grid to scan for maximum of PDF
MAX_LOG10G = -3.0
MIN_LOG10G = -13.0
NUM_G = 1000
SCAN_GRID = np.logspace(MIN_LOG10G, MAX_LOG10G, NUM_G)

# Threshold to set posterior integration cutoff
# Integration will be cutoff where logpdf = maximum - TAIL_START
TAIL_START = 10

# Whether or not to print progress updates
VERBOSE = True

def get_dirname(include_window = True, include_threshold = True, include_boundparams = True, subset = SUBSET):
    if STATIONARITY_TIME in ['daily', 'weekly', 'monthly', 'annual']:
        dirname = STATIONARITY_TIME
    else:
        dirname = f'stationarity{STATIONARITY_TIME}'
    if START_YEAR != 2005 or END_YEAR != 2020:
        dirname += f'_{START_YEAR}_{END_YEAR}'
    if subset != -1:
        dirname += f'_subset{subset}'
    if INJECT:
        dirname += '_injected'
    if LMAX != 4:
        dirname += f'_lmax{LMAX}'
    if include_window:
        dirname += f'_window{WINDOW}'
    if include_threshold and THRESHOLD != 0.03:
        dirname += f'_threshold{THRESHOLD}'
    if include_boundparams and CONFIDENCE != 0.95:
        dirname += f'_confidence{CONFIDENCE}'
    return dirname

def print_params(include_window = True, include_threshold = True, include_boundparams = True):
    print(f'Dataset duration: {START_YEAR} - {END_YEAR}')
    print('Stationarity time: ' + str(STATIONARITY_TIME).capitalize())
    if SUBSET == -1:
        print('Subset: Full')
    else:
        print(f'Subset: {SUBSET}')
    if INJECT:
        print(f'Injected signal: {FA:.4f} Hz, g_agamma = {G_AGAMMA:.1e}')
    else:
        print(f'Injected signal: False')
    print(f'Number of IGRF l modes: {LMAX}')
    if include_window:
        print(f'Smoothing window: {WINDOW}')
        if DOWNSAMPLE != WINDOW // 10:
            print(f'Downsampling factor: {DOWNSAMPLE}')
    if include_threshold:
        print(f'Coherence time precision: {THRESHOLD}')
    if include_boundparams:
        print(f'Bound range: ({MIN_FREQ}, {MAX_FREQ})')
        print(f'Confidence level: {CONFIDENCE}')
    print('')
    sys.stdout.flush()
