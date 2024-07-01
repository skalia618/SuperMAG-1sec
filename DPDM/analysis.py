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

        # Calculate approximate sidereal frequency
        approx_sidereal = approximate_sidereal(df)
        fdhat = approx_sidereal * df

        # Start and end of relevant frequency window (including sidereal padding)
        start_padded = lof - approx_sidereal
        end_padded_exclusive = hif_inclusive + approx_sidereal + 1
        num_frequencies = hif_inclusive - lof + 1
        coh_freqs = (lof + np.arange(num_frequencies)) * df

        # Initialize analysis variables (for each chunk and each frequency)
        s_chunks = np.zeros((total_chunks, num_frequencies, 3))
        z_chunks = np.zeros((total_chunks, num_frequencies, 3)) + 0j
        vh_chunks = np.zeros((total_chunks, num_frequencies, 3, 3)) + 0j

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
            data_vector = np.zeros((15, num_frequencies)) + 0j

            # Iterate over Xi series
            for i in range(5):
                series = np.load(f'{proj_aux_dir}/X{i+1}.npy', mmap_mode = 'r')

                # Start and end of this chunk in the series
                start = chunk * coh
                end = min(start + coh, len(series))

                # Take relevant subseries (and pad if chunk is not length of full coherence time)
                subseries = np.nan_to_num(series[start:end])
                if end - start != coh:
                    subseries = np.append(subseries, np.zeros(coh - len(subseries)))

                # FFT and zoom in on relevant frequency window
                # If padding goes below 0 Hz or above 1 Hz, then loop around
                subseries_fft = np.fft.fft(subseries)
                if start_padded > 0 and end_padded_exclusive < len(subseries_fft):
                    subseries_fft = subseries_fft[start_padded : end_padded_exclusive]
                elif start_padded < 0:
                    subseries_fft = np.concatenate((subseries_fft[:end_padded_exclusive],
                                                    subseries_fft[start_padded:]))
                else:
                    subseries_fft = np.concatenate((subseries_fft[start_padded:],
                                                    subseries_fft[:end_padded_exclusive - len(subseries_fft)]))
                    
                # Three sections of data vector correspond to: f - fdhat, f, f + fdhat
                lo = subseries_fft[:-2 * approx_sidereal]
                mid = subseries_fft[approx_sidereal : -approx_sidereal]
                hi = subseries_fft[2 * approx_sidereal:]

                # Write to data_vector
                data_vector[i] = lo
                data_vector[i + 5] = mid
                data_vector[i + 10] = hi

            if VERBOSE:
                print('Data vector computed')
                sys.stdout.flush()



            # Step 2: Mean [see Eqs. (36), (C1), and (C2) of 2108.08852]
            mux = np.zeros(15) + 0j
            muy = np.zeros(15) + 0j
            muz = np.zeros(15) + 0j

            # Construct DFT kernels, i.e. (cos - i * sin) at various frequencies
            cis_fh_f = np.exp(-1j * 2 * np.pi * (fdhat - FD) * np.arange(start, end)) # f = fdhat - fd
            cis_f = np.exp(-1j * 2 * np.pi * FD * np.arange(start, end)) # f = fd
            cis_f_fh = np.exp(-1j * 2 * np.pi * (FD - fdhat) * np.arange(start, end)) # f = fd - fdhat
            cis_fh = np.exp(-1j * 2 * np.pi * fdhat * np.arange(start, end)) # f = fdhat
            cis_mfh = np.exp(1j * 2 * np.pi * fdhat * np.arange(start, end)) # f = -fdhat
            
            # Load the appropriate chunks of the auxiliary series
            H1 = np.load(f'{proj_aux_dir}/H1.npy', mmap_mode = 'r')[start:end]
            H2 = np.load(f'{proj_aux_dir}/H2.npy', mmap_mode = 'r')[start:end]
            H3 = np.load(f'{proj_aux_dir}/H3.npy', mmap_mode = 'r')[start:end]
            H4 = np.load(f'{proj_aux_dir}/H4.npy', mmap_mode = 'r')[start:end]
            H5 = np.load(f'{proj_aux_dir}/H5.npy', mmap_mode = 'r')[start:end]
            H6 = np.load(f'{proj_aux_dir}/H6.npy', mmap_mode = 'r')[start:end]
            H7 = np.load(f'{proj_aux_dir}/H7.npy', mmap_mode = 'r')[start:end]

            # Set nans in auxiliary series and corresponding locations in kernels to zero
            nans = np.where(np.isnan(H1))[0]
            if len(nans) > 0:
                H1 = np.nan_to_num(H1)
                H2 = np.nan_to_num(H2)
                H3 = np.nan_to_num(H3)
                H4 = np.nan_to_num(H4)
                H5 = np.nan_to_num(H5)
                H6 = np.nan_to_num(H6)
                H7 = np.nan_to_num(H7)

                cis_fh_f[nans] = 0
                cis_f[nans] = 0
                cis_f_fh[nans] = 0
                cis_fh[nans] = 0
                cis_mfh[nans] = 0

            # Calculate mux (without prefactor)
            mux[0] = np.sum(cis_f_fh * (1 - H1 + 1j * H2))
            mux[1] = np.sum(cis_f_fh * (H2 + 1j * H1))
            mux[2] = np.sum(cis_f_fh * (H4 - 1j * H5))
            mux[3] = np.sum(cis_f_fh * (-H5 + 1j * (H3 - H4)))
            mux[4] = np.sum(cis_f_fh * (H6 - 1j * H7))

            mux[5] = 2 * (np.sum(cis_f * (1 - H1)).real - np.sum(cis_f * H2).imag)
            mux[6] = 2 * (np.sum(cis_f * H2).real - np.sum(cis_f * H1).imag)
            mux[7] = 2 * (np.sum(cis_f * H4).real + np.sum(cis_f * H5).imag)
            mux[8] = 2 * (np.sum(cis_f * -H5).real - np.sum(cis_f * (H3 - H4)).imag)
            mux[9] = 2 * (np.sum(cis_f * H6).real + np.sum(cis_f * H7).imag)

            mux[10] = np.sum(cis_fh_f * (1 - H1 - 1j * H2))
            mux[11] = np.sum(cis_fh_f * (H2 - 1j * H1))
            mux[12] = np.sum(cis_fh_f * (H4 + 1j * H5))
            mux[13] = np.sum(cis_fh_f * (-H5 + 1j * (H4 - H3)))
            mux[14] = np.sum(cis_fh_f * (H6 + 1j * H7))

            # Calculate muy (without prefactor)
            muy[0] = np.sum(cis_f_fh * (H2 + 1j * (H1 - 1)))
            muy[1] = np.sum(cis_f_fh * (H1 - 1j * H2))
            muy[2] = np.sum(cis_f_fh * (-H5 - 1j * H4))
            muy[3] = np.sum(cis_f_fh * (H3 - H4 + 1j * H5))
            muy[4] = np.sum(cis_f_fh * (-H7 - 1j * H6))

            muy[5] = 2 * (np.sum(cis_f * H2).real + np.sum(cis_f * (1 - H1)).imag)
            muy[6] = 2 * (np.sum(cis_f * H1).real + np.sum(cis_f * H2).imag)
            muy[7] = 2 * (-np.sum(cis_f * H5).real + np.sum(cis_f * H4).imag)
            muy[8] = 2 * (np.sum(cis_f * (H3 - H4)).real - np.sum(cis_f * H5).imag)
            muy[9] = 2 * (np.sum(cis_f * -H7).real + np.sum(cis_f * H6).imag)

            muy[10] = np.sum(cis_fh_f * (H2 + 1j * (1 - H1)))
            muy[11] = np.sum(cis_fh_f * (H1 + 1j * H2))
            muy[12] = np.sum(cis_fh_f * (-H5 + 1j * H4))
            muy[13] = np.sum(cis_fh_f * (H3 - H4 - 1j * H5))
            muy[14] = np.sum(cis_fh_f * (-H7 + 1j * H6))

            # Calculate muz (without prefactor)
            muz[0] = muz[1] = muz[5] = muz[6] = muz[10] = muz[11] = 0.

            muz[2] = np.sum(cis_mfh * H6)
            muz[3] = -np.sum(cis_mfh * H7)
            muz[4] = np.sum(cis_mfh * (1 - H3))

            muz[7] = np.sum(H6)
            muz[8] = -np.sum(H7)
            muz[9] = np.sum(1 - H3) - len(nans)

            muz[12] = np.sum(cis_fh * H6)
            muz[13] = -np.sum(cis_fh * H7)
            muz[14] = np.sum(cis_fh * (1 - H3))

            # Multiply by prefactors
            # (Note that we introduce the frequency factor later on)
            mux_prefactor = np.pi * R * np.sqrt(RHODM / 8)
            muy_prefactor = -mux_prefactor
            muz_prefactor = -2 * mux_prefactor
            mux *= mux_prefactor
            muy *= muy_prefactor
            muz *= muz_prefactor

            # Package into one array
            mu = np.ascontiguousarray(np.array([mux, muy, muz]))

            if VERBOSE:
                print('Mean computed')
                sys.stdout.flush()



            # Step 3: Covariance
            sigma = np.zeros((num_frequencies, 15, 15)) + 0j

            # Find which stationarity chunks this coherence chunk overlaps with
            overlapping_stationarity_chunks = find_overlap_chunks(coh, chunk)

            # Iterate over i,j elements of covariance matrix
            for i in range(5):
                for j in range(i, 5):

                    # Iterate over overlapping chunks
                    for (stationarity_chunk, overlap) in overlapping_stationarity_chunks:

                        # Load stationarity chunk
                        ijchunk_dir = get_stationarity_chunk_dir(stationarity_chunk)
                        ijchunk = np.load(f'{ijchunk_dir}/X{i+1}X{j+1}.npy')

                        # Find frequencies at which spectra were computed
                        spec_freqs = spectra_freqs(stationarity_chunk)

                        # Covariance matrix is block diagonal (and Hermitian)
                        # Blocks correspond to frequencies: f - fdhat, f, f + fdhat
                        # For each block, we interpolate the power to the appropriate frequencies
                        # Then add overlap * interpolated power to sigma

                        # Upper block: f - fdhat
                        interpolated_power = np.interp((coh_freqs - fdhat) % 1.0, spec_freqs, ijchunk)
                        if i == j:
                            sigma[:, i, i] += overlap * interpolated_power
                        else:
                            sigma[:, i, j] += overlap * interpolated_power
                            sigma[:, j, i] += overlap * np.conj(interpolated_power)

                        # Middle block: f
                        interpolated_power = np.interp(coh_freqs, spec_freqs, ijchunk)
                        if i == j:
                            sigma[:, i + 5, i + 5] += overlap * interpolated_power
                        else:
                            sigma[:, i + 5, j + 5] += overlap * interpolated_power
                            sigma[:, j + 5, i + 5] += overlap * np.conj(interpolated_power)

                        # Lower block: f + fdhat
                        interpolated_power = np.interp((coh_freqs + fdhat) % 1.0, spec_freqs, ijchunk)
                        if i == j:
                            sigma[:, i + 10, i + 10] += overlap * interpolated_power
                        else:
                            sigma[:, i + 10, j + 10] += overlap * interpolated_power
                            sigma[:, j + 10, i + 10] += overlap * np.conj(interpolated_power)

            if VERBOSE:
                print('Variance computed')
                sys.stdout.flush()



            # Step 4: Analysis Variables

            # Calculate Ainv by Cholesky decomposing Sigma and inverting
            # Shape: (num_frequencies, 15, 15)
            ainv = np.linalg.inv(np.linalg.cholesky(sigma))

            # Calculate Y_k = Ainv_k * X_k
            # Shape: (num_frequencies, 15)
            y = np.einsum('fij, jf -> fi', ainv, data_vector)

            # Calculate nu_k = Ainv_k * mu_k
            # Shape: (num_frequencies, 15, 3)
            nu = ainv @ mu.T

            # Singular value decompose nu_k = U_k * S_k * Vh_k
            # u shape: (num_frequencies, 15, 3)
            # s shape: (num_frequenceis, 3)
            # vh shape: (num_frequencies, 3, 3)
            u, s, vh = np.linalg.svd(nu, full_matrices = False)

            # Calculate Z_k = Uh_k * Y_k
            # Shape: (num_frequencies, 3)
            z = np.einsum('fji, fj -> fi', np.conj(u), y)

            # Write to s_chunks, z_chunks, and vh_chunks arrays
            s_chunks[chunk] = s * coh_freqs[:, None] # reintroduce frequency factor
            z_chunks[chunk] = z
            vh_chunks[chunk] = vh

            if VERBOSE:
                print(f'Chunk {chunk} analysis complete\n')
                sys.stdout.flush()

        # Swap axes and store variables
        # s_chunks shape: (num_frequencies, total_chunks, 3)
        # z_chunks shape: (num_frequencies, total_chunks, 3)
        # v_chunks shape: (num_frequencies, total_chunks, 3, 3) [v is Hermitian conjugate of vh]
        s_chunks = np.ascontiguousarray(np.transpose(s_chunks, (1, 0, 2)))
        z_chunks = np.ascontiguousarray(np.transpose(z_chunks, (1, 0, 2)))
        v_chunks = np.ascontiguousarray(np.transpose(np.conjugate(vh_chunks), (1, 0, 3, 2)))

        if VERBOSE:
            print(f'Storing results for coherence time {coh}')
            sys.stdout.flush()

        np.save(f'{analysis_vars_dir}/s_{coh}', s_chunks)
        np.save(f'{analysis_vars_dir}/z_{coh}', z_chunks)
        np.save(f'{analysis_vars_dir}/v_{coh}', v_chunks)

        if VERBOSE:
            print(f'Coherence time {coh} complete!\n\n\n')
            sys.stdout.flush()
        
    if VERBOSE: print('Done!')
