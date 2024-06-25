import multiprocessing as mp
import numpy as np
from pathlib import Path
import sys
from utils import *

# Number of cores for parallelization
CORES = 20

def calculate_bounds(coh):
    """
    Calculate bound for all frequencies with coherence time coh

    For each frequency, we compute the posterior CDF, and determine where it exceeds CONFIDENCE.

    If calculation fails for a frequency because CDF has
    significant support above epsilon = 1, bound is set to 1
    """

    # Load analysis variables for coherence time
    s = np.load(f'{analysis_vars_dir}/s_{coh}.npy')
    z = np.load(f'{analysis_vars_dir}/z_{coh}.npy')

    # Initiate bound array
    bounds = np.zeros(len(s))

    # Iterate over frequencies
    for (i, (sf, zf)) in enumerate(zip(s, z)):

        # Compute posterior CDF
        cdf_data = calculate_cdf(sf, zf)

        if cdf_data == None:
            # If CDF calculation fails, set bound = 1
            bounds[i] = 1
        else:
            # Otherwise, find first epsilon where CDF exceeds CONFIDENCE
            int_grid, cdf = cdf_data
            bound_ind = np.searchsorted(cdf, CONFIDENCE)
            bounds[i] = int_grid[bound_ind]

    if VERBOSE:
        print(f'Coherence time {coh} completed')
        sys.stdout.flush()
    
    return bounds

if __name__ == '__main__':
    if VERBOSE: print_params()

    analysis_vars_dir = get_analysis_vars_dir()

    # Calculate coherence times and their respective frequency bins
    coherence_times = coherence_times(TOTAL_TIME)
    freq_bins = frequencies_from_coherence_times(coherence_times)

    # Identify coherence times corresponding to MIN_FREQ and MAX_FREQ
    start_ind = np.where([freq_bin[1] * freq_bin[2] > MIN_FREQ for freq_bin in freq_bins])[0][0]
    end_ind = np.where([freq_bin[0] * freq_bin[2] < MAX_FREQ for freq_bin in freq_bins])[0][-1] # inclusive

    # Merge relevant frequencies into single array
    freqs = []
    for (lof, hif_inclusive, df) in freq_bins[start_ind:end_ind + 1]:
        coh_freqs = (lof + np.arange(hif_inclusive - lof + 1)) * df
        freqs.append(coh_freqs)
    freqs = np.concatenate(freqs)

    # Parallelize bound computation
    pool = mp.Pool(CORES)
    bounds = pool.map_async(calculate_bounds, coherence_times[start_ind:end_ind + 1]).get()
    pool.close()
    pool.join()

    # Merge bounds from different coherence times into single array
    bounds = np.concatenate(bounds)

    if VERBOSE:
        print('\nStoring results\n')
        sys.stdout.flush()

    # Store bounds
    Path('bounds').mkdir(exist_ok = True)
    filename = get_dirname()
    np.savez(f'bounds/{filename}', freqs = freqs, bounds = bounds)

    if VERBOSE: print('Done!')
