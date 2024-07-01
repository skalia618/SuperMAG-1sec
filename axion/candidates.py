import numpy as np
from params import *
from pathlib import Path
from scipy.stats import chi2
from scipy.special import erfcinv
import sys
from utils import *

# Whether to check validity of candidates (or just identify them)
CHECK_CANDIDATES = True

if __name__ == '__main__':
    if VERBOSE: print_params()

    # Analysis variable directories for full-dataset analysis and subset analyses
    full_vars_dir = get_analysis_vars_dir(subset = -1)
    subset_vars_dir = [get_analysis_vars_dir(subset = j) for j in range(4)]

    # Calculate coherence times and their respective frequency bins
    coherence_times = coherence_times(TOTAL_TIME)
    freq_bins = frequencies_from_coherence_times(coherence_times)

    # Identify coherence times corresponding to MIN_FREQ and MAX_FREQ
    start_ind = np.where([freq_bin[1] * freq_bin[2] > MIN_FREQ for freq_bin in freq_bins])[0][0]
    end_ind = np.where([freq_bin[0] * freq_bin[2] < MAX_FREQ for freq_bin in freq_bins])[0][-1] # inclusive

    # Compute number of frequencies between MIN_FREQ and MAX_FREQ
    # First compute number of frequencies in coherence times from start_ind to end_ind (inclusive)
    # Then remove frequencies below MIN_FREQ or above MAX_FREQ
    Nfreq = np.sum([freq_bin[1] - freq_bin[0] + 1 for freq_bin in freq_bins[start_ind:end_ind + 1]])
    Nfreq -= int(MIN_FREQ / freq_bins[start_ind][2]) - freq_bins[start_ind][0] + 1
    Nfreq -= freq_bins[end_ind][1] - int(MAX_FREQ / freq_bins[end_ind][2])

    # Compute threshold for global signficance of CONFIDENCE
    pcrit = 1 - CONFIDENCE ** (1 / Nfreq)

    # Initiate count/lists of candidate frequencies and their p-values,
    # as well as list of successful frequencies
    count = 0
    cand_freqs = []
    cand_p0s = []
    cand_sigmas = []
    cand_pjs = []
    cand_pfulls = []
    succ_freqs = []

    # Iterate over coherence times (within relevant range)
    for (coh, (lof, hif_inclusive, df)) in zip(coherence_times[start_ind:end_ind + 1],
                                               freq_bins[start_ind:end_ind + 1]):        

        # List of frequencies with given coherence time
        coh_freqs = (lof + np.arange(hif_inclusive - lof + 1)) * df

        # Load z variables
        z = np.load(f'{full_vars_dir}/z_{coh}.npy')

        # Build chi-squared variable, fit for degrees of freedom, and turn into p-value/significance
        Q0 = 2 * np.sum(np.abs(z) ** 2, axis = 1)
        dof = chi2.fit(Q0, floc = 0., fscale = 1.)[0]
        p0 = chi2.sf(Q0, dof)
        sigma = np.sqrt(2) * erfcinv(2 * (1 - (1 - p0) ** Nfreq))

        # Determine indices of candidate frequencies (where p-value is below threshold),
        # and remove candidates below MIN_FREQ and above MAX_FREQ
        freq_inds = np.where(p0 < pcrit)[0]
        freq_inds = np.delete(freq_inds, np.where(coh_freqs[freq_inds] < MIN_FREQ)[0])
        freq_inds = np.delete(freq_inds, np.where(coh_freqs[freq_inds] > MAX_FREQ)[0])

        # Add to candidate count/lists
        count += len(freq_inds)
        cand_freqs += list(coh_freqs[freq_inds])
        cand_p0s += list(p0[freq_inds])
        cand_sigmas += list(sigma[freq_inds])

        if VERBOSE:
            if coh != coherence_times[start_ind]: print('')
            print(f'Coherence time {coh} has {len(freq_inds)} candidates')
            sys.stdout.flush()
    
        # Check candidate frequencies
        if CHECK_CANDIDATES and len(freq_inds) > 0:

            # Load s variables for full-dataset analysis
            s = np.load(f'{full_vars_dir}/s_{coh}.npy')

            # Initiate list of resampling p-values (for each candidate and each subset)
            pj = np.zeros((len(freq_inds), 4))

            # Iterate over subsets
            for j in range(4):

                # Load s and z variables for subset analyses
                sj = np.load(f'{subset_vars_dir[j]}/s_{coh}.npy')
                zj = np.load(f'{subset_vars_dir[j]}/z_{coh}.npy')

                # Test statistic to compare subset analysis with full-dataset analysis
                # (Should cancel contribution from true signal)
                W = z / s - zj / sj

                # Variance of W (for no signal)
                Sigma = s ** -2 + sj ** -2

                # Normalize W by Sigma, and turn into chi-squared statistic
                w = W / np.sqrt(Sigma)
                Qj = 2 * np.sum(np.abs(w) ** 2, axis = 1)

                # For each candidate, turn Qj into p-value
                # Rather than using chi-squared distribution, we use empirical distribution of Qj
                # i.e., determine where this candidate's Qj lies in distribution of all Qj (excluding this candidate)
                for (i, freq_ind) in enumerate(freq_inds):
                    pj[i, j] = min(1 - (np.where(np.argsort(Qj) == freq_ind)[0][0] - 1) / (len(Qj) - 1), 1)

            # Combine p-values for individual tests into one p-value
            Qfull = -2 * np.sum(np.log(pj), axis = 1)
            pfull = chi2.sf(Qfull, 8)

            # Record p-values of candidates
            cand_pjs += list(pj)
            cand_pfulls += list(pfull)

            # Record frequency and significance of successful candidates (combined p-value above 0.01)
            for i in np.where(pfull > 0.01)[0]:
                freq_ind = freq_inds[i]
                succ_freqs.append(coh_freqs[freq_ind])

                if VERBOSE:
                    print(f'Successful candidate at {coh_freqs[freq_ind]:.4f} Hz ({sigma[freq_ind]:.2f}-sigma)!')
                    pstrings = []
                    for j in range(4):
                        if pj[i, j] >= 0.1:
                            pstrings.append(f'{pj[i, j]:.2f}')
                        elif pj[i, j] >= 0.01:
                            pstrings.append(f'{pj[i, j]:.3f}')
                        else:
                            pstrings.append(f'{pj[i, j]:.1e}')
                    print(f'P-values: [' + ', '.join(pstrings) + f'] (Combined: {pfull[i]:.2f})')
                    sys.stdout.flush()

    if VERBOSE:
        print('')
        print(f'Total Candidates: {count}')
        if CHECK_CANDIDATES:
            print(f'Storing results\n')
        else:
            print('')
        sys.stdout.flush()

    # Store results (only if checking candidates)
    if CHECK_CANDIDATES:
        Path('candidates').mkdir(exist_ok = True)
        filename = get_dirname()
        np.savez(f'candidates/{filename}',
                 cand_freqs = cand_freqs,
                 cand_p0s = cand_p0s,
                 cand_sigmas = cand_sigmas,
                 cand_pjs = cand_pjs,
                 cand_pfulls = cand_pfulls,
                 succ_freqs = succ_freqs)

    if VERBOSE: print(f'Done!')
