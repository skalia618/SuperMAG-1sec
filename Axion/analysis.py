import numpy as np
from params import *
from pathlib import Path
import sys
from utils import *

# Start and end coherence times to be computed, if only a subset are required
# (Can both be set to None, if all are required)
COH_START = None
COH_END = None

if __name__ == '__main__':
    if VERBOSE:
        print_params(include_boundparams = False)
        print('\n')
        sys.stdout.flush()

    # Calculate coherence times and their respective frequency bins
    coherence_times = coherence_times(TOTAL_TIME)
    freq_bins = frequencies_from_coherence_times(coherence_times)

    # Create output directory
    analysis_vars_dir = get_analysis_vars_dir()
    Path(analysis_vars_dir).mkdir(parents = True, exist_ok = True)

    proj_aux_dir = get_proj_aux_dir()

    # Iterate over coherence times
    coh_zip = zip(coherence_times[COH_START : COH_END], freq_bins[COH_START : COH_END])
    for (coh, (lof, hif_inclusive, df)) in coh_zip:
        if VERBOSE:
            print(f'Computing coherence time {coh}\n')
            sys.stdout.flush()

        # Calculate number of coherence chunks
        total_chunks = div_ceil(TOTAL_TIME, coh)

        # Generate array of relevant frequencies
        num_frequencies = hif_inclusive - lof + 1
        coh_freqs = (lof + np.arange(num_frequencies)) * df

        # Initialize analysis variables (for each chunk and each frequency)
        s_chunks = np.zeros((total_chunks, num_frequencies))
        z_chunks = np.zeros((total_chunks, num_frequencies)) + 0j

        # For each coherence chunk, we compute the following:
        # 1) Data Vector
        # 2) Mean
        # 3) Covariance
        # 4) Analysis Variables
        for chunk in range(total_chunks):
            if VERBOSE:
                print(f'Beginning coherence chunk {chunk}')
                sys.stdout.flush()


            
            # Step 1: Data Vector
            data_vector = np.zeros((2, num_frequencies)) + 0j

            # Iterate over Xi series
            for i in range(2):
                series = np.load(f'{proj_aux_dir}/X{i+1}.npy', mmap_mode = 'r')

                # Start and end of this chunk in the series
                start = chunk * coh
                end = min(start + coh, len(series))

                # Take relevant subseries (and pad if chunk is not length of full coherence time)
                subseries = np.nan_to_num(series[start:end])
                if end - start != coh:
                    subseries = np.append(subseries, np.zeros(coh - len(subseries)))

                # FFT and zoom in on relevant frequency window
                subseries_fft = np.fft.fft(subseries)
                subseries_fft = subseries_fft[lof : hif_inclusive + 1]

                # Write to data_vector
                data_vector[i] = subseries_fft

            if VERBOSE:
                print('Data vector computed')
                sys.stdout.flush()



            # Step 2: Mean [see Eq. (C10) of 2112.09620]
            mu = np.zeros(2) + 0j
            
            # Load the appropriate chunks of the auxiliary series
            H1 = np.load(f'{proj_aux_dir}/H1.npy', mmap_mode = 'r')[start:end]
            H2 = np.load(f'{proj_aux_dir}/H2.npy', mmap_mode = 'r')[start:end]

            # Set nans in auxiliary series to zero
            nans = np.where(np.isnan(H1))[0]
            if len(nans) > 0:
                H1 = np.nan_to_num(H1)
                H2 = np.nan_to_num(H2)

            # Calculate mu
            mu[0] = np.sum(H1)
            mu[1] = np.sum(H2)
            mu *= 1j * R * np.sqrt(RHODM / 2)

            if VERBOSE:
                print('Mean computed')
                sys.stdout.flush()



            # Step 3: Covariance
            sigma = np.zeros((num_frequencies, 2, 2)) + 0j

            # Find which stationarity chunks this coherence chunk overlaps with
            overlapping_stationarity_chunks = find_overlap_chunks(coh, chunk)

            # Iterate over i,j elements of covariance matrix
            for i in range(2):
                for j in range(i, 2):

                    # Iterate over overlapping chunks
                    for (stationarity_chunk, overlap) in overlapping_stationarity_chunks:

                        # Load stationarity chunk
                        ijchunk_dir = get_stationarity_chunk_dir(stationarity_chunk)
                        ijchunk = np.load(f'{ijchunk_dir}/X{i+1}X{j+1}.npy')

                        # Find frequencies at which spectra were computed
                        spec_freqs = spectra_freqs(stationarity_chunk)

                        # For each entry, we interpolate the power to the appropriate frequencies
                        # Then add overlap * interpolated power to sigma
                        interpolated_power = np.interp(coh_freqs, spec_freqs, ijchunk)
                        if i == j:
                            sigma[:, i, i] += overlap * interpolated_power
                        else:
                            sigma[:, i, j] += overlap * interpolated_power
                            # Add conjugate tranpose entry, since sigma is Hermitian
                            sigma[:, j, i] += overlap * np.conj(interpolated_power)

            if VERBOSE:
                print('Variance computed')
                sys.stdout.flush()



            # Step 4: Analysis Variables

            # Calculate Ainv by Cholesky decomposing Sigma and inverting
            # Shape: (num_frequencies, 2, 2)
            ainv = np.linalg.inv(np.linalg.cholesky(sigma))

            # Calculate Y_k = Ainv_k * X_k
            # Shape: (num_frequencies, 2)
            y = np.einsum('fij, jf -> fi', ainv, data_vector)

            # Calculate nu_k = Ainv_k * mu_k
            # Shape: (num_frequencies, 2)
            nu = ainv @ mu

            # Compute s_k and z_k [see Eq. (C17)]
            # Shape for both: (num_frequencies)
            s = np.linalg.norm(nu, axis = 1)
            z = np.einsum('fi, fi -> f', np.conj(nu), y) / s

            # Write to s_chunks and z_chunks arrays
            s_chunks[chunk] = s
            z_chunks[chunk] = z

            if VERBOSE:
                print(f'Chunk {chunk} analysis complete\n')
                sys.stdout.flush()

        # Swap axes and store variables
        # Shape for both: (num_frequencies, total_chunks)
        s_chunks = np.ascontiguousarray(s_chunks.T)
        z_chunks = np.ascontiguousarray(z_chunks.T)

        if VERBOSE:
            print(f'Storing results for coherence time {coh}')
            sys.stdout.flush()

        np.save(f'{analysis_vars_dir}/s_{coh}', s_chunks)
        np.save(f'{analysis_vars_dir}/z_{coh}', z_chunks)

        if VERBOSE:
            print(f'Coherence time {coh} complete!\n\n\n')
            sys.stdout.flush()
        
    if VERBOSE: print('Done!')
