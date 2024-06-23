import numpy as np
import os
from params import *
from pathlib import Path
from scipy import stats
import sys
from utils import *

def calculate_spectra(ichunk, jchunk):
    """
    Computes the power between Xi and Xj within a stationarity chunk
    This is done by FFTing, and computing Xi * conj(Xj)
    The resulting PSD is then smoothed with a Gaussian filter
    The result can be downsampled to reduce the amount of data stored
    """

    # Count nans and set them to zero
    nans = np.sum(np.isnan(ichunk))
    if nans > 0:
        ichunk = np.nan_to_num(ichunk)
        jchunk = np.nan_to_num(jchunk)
    
    # FFT both chunks
    fft1 = np.fft.fft(ichunk)
    fft2 = np.fft.fft(jchunk)

    # Compute power
    # Power is normalized by number of valid data points (i.e., excluding nans)
    power = fft1 * np.conj(fft2) / (len(ichunk) - nans)

    # Gaussian filter (with standard deviation WINDOW)
    filter = stats.norm.pdf(np.arange(-3 * WINDOW, 3 * WINDOW + 1), loc = 0, scale = WINDOW)

    # Convolve with filter (only at downsampled frequencies)
    smoothed = []
    for n in range(0, len(power) - len(filter) + 1, DOWNSAMPLE):
        smoothed.append(np.sum(power[n:n + len(filter)] * filter))
    return np.array(smoothed)    

if __name__ == '__main__':
    if VERBOSE:
        print_params(include_threshold = False,
                     include_boundparams = False)

    # Iterate over stationarity chunks
    chunk_list = get_stationarity_chunks()
    for (chunk_ind, start_second, end_second) in chunk_list:
        if VERBOSE:
            print(f'Computing chunk {chunk_ind}')
            sys.stdout.flush()

        # Create output directory
        chunk_dir = get_stationarity_chunk_dir(chunk_ind)
        Path(chunk_dir).mkdir(parents = True, exist_ok = True)

        # Load chunks
        proj_aux_dir = get_proj_aux_dir()
        for i in range(5):
            ichunk = np.load(f'{proj_aux_dir}/X{i+1}.npy',
                             mmap_mode = 'r')[start_second:end_second]

            # Hermitian so only need i to 5
            for j in range(i, 5):
                jchunk = np.load(f'{proj_aux_dir}/X{j+1}.npy',
                                 mmap_mode = 'r')[start_second:end_second]

                spectra = calculate_spectra(ichunk, jchunk)

                # Save spectra
                np.save(f'{chunk_dir}/X{i+1}X{j+1}', spectra)

    if VERBOSE: print('\nDone!')
