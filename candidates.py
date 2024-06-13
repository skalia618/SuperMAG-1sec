import mcint
import numpy as np
from pathlib import Path
from scipy.stats import chi2
from scipy.special import erfcinv
from utils import *

# Whether to check validity of candidates (or just identify them)
CHECK_CANDIDATES = True

if __name__ == "__main__":
    if VERBOSE: print_params()

    # Analysis variable directories for full-dataset analysis and subset analyses
    full_vars_dir = get_analysis_vars_dir()
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

    # Initiate count/lists of candidate frequencies and lists of successful candidates
    count = 0
    cand_freqs = []
    cand_p0s = []
    succ_freqs = []
    succ_sigmas = []
    succ_pjs = []
    succ_pfulls = []

    # Iterate over coherence times (within relevant range)
    for (coh, (lof, hif_inclusive, df)) in zip(coherence_times[start_ind:end_ind + 1],
                                               freq_bins[start_ind:end_ind + 1]):        

        # List of frequencies with given coherence time
        coh_freqs = (lof + np.arange(hif_inclusive - lof + 1)) * df

        # Load z variables and determine number of coherence chunks
        z = np.load(f'{full_vars_dir}/{coh}_all_z.npy')
        K0 = z.shape[1]

        # Build chi-squared variable, fit for degrees of freedom, and turn into p-value
        Q0 = 2 * np.sum(np.abs(z) ** 2, axis = (1, 2))
        dof = chi2.fit(Q0, floc = 0., fscale = 1.)[0]
        p0 = chi2.sf(Q0, dof)

        # Determine indices of candidate frequencies (where p-value is below threshold),
        # and remove candidates below MIN_FREQ and above MAX_FREQ
        freq_inds = np.where(p0 < pcrit)[0]
        freq_inds = np.delete(freq_inds, np.where(coh_freqs[freq_inds] < MIN_FREQ)[0])
        freq_inds = np.delete(freq_inds, np.where(coh_freqs[freq_inds] > MAX_FREQ)[0])

        # Add to candidate count/lists
        count += len(freq_inds)
        cand_freqs += list(coh_freqs[freq_inds])
        cand_p0s += list(p0[freq_inds])

        if VERBOSE: print(f'Coherence time {coh} has {len(freq_inds)} candidates\n')
    
        # Check candidate frequencies
        if CHECK_CANDIDATES and len(freq_inds) > 0:

            # Load s and vh variables for full-dataset analysis
            s = np.load(f'{full_vars_dir}/{coh}_all_s.npy')
            vh = np.load(f'{full_vars_dir}/{coh}_all_vh.npy')

            # Load s, z, and vh variables for subset analyses
            sj = np.zeros(np.insert(s.shape, 0, 4))
            zj = np.zeros(np.insert(z.shape, 0, 4), dtype = complex)
            vhj = np.zeros(np.insert(vh.shape, 0, 4), dtype = complex)
            for j in range(4):
                sj[j] = np.load(f'{subset_vars_dir[j]}/{coh}_all_s.npy')
                zj[j] = np.load(f'{subset_vars_dir[j]}/{coh}_all_z.npy')
                vhj[j] = np.load(f'{subset_vars_dir[j]}/{coh}_all_vh.npy')

            # Iterate over candidate frequencies
            for freq_ind in freq_inds:

                # s, z, and vh variables for candidate frequency (and full dataset)
                sf = s[freq_ind]
                zf = z[freq_ind]
                vhf = vh[freq_ind]

                # Calculate posterior cdf for full-dataset analysis
                int_grid, cdf = calculate_cdf(sf, zf)

                # Determine resampling p-value for each subset j
                pj = np.zeros(4)
                for j in range(4):

                    # s, z, and vh variables for candidate frequency (and subset)
                    sfj = sj[j][freq_ind]
                    zfj = zj[j][freq_ind]
                    vhfj = vhj[j][freq_ind]

                    def sampler():
                        """
                        Sampler for Monte Carlo integration
                        Draws epsilon from posterior of full-dataset analysis
                        Draws c variables from Gaussian distribution
                        Outputs: (real parts of c, imaginary parts of c, epsilon)
                        """
                        while True:
                            # Draw from posterior by drawing uniform random number and applying inverse CDF
                            eps = np.interp(np.random.random(), cdf, int_grid)

                            # Mean and variance of Gaussian distribution for d variables
                            mu = eps * sf * zf / (3 + eps ** 2 * sf ** 2)
                            var = 2 * (3 + eps ** 2 * sf ** 2)

                            # Draw d and apply v to convert to c
                            d = (np.random.randn(K0, 3) + 1j * np.random.randn(K0, 3)) / np.sqrt(var) + mu
                            c = np.einsum("kji, kj -> ki", np.conjugate(vhf), d)

                            yield np.concatenate((np.real(c).flatten(), np.imag(c).flatten(), [eps]))

                    def integrand(x):
                        """
                        Integrand used for Monte Carlo integration
                        Takes input from sampler
                        """
                        # Reconstruct epsilon and c variables
                        eps = x[-1]
                        c = x[:3 * K0].reshape(K0, 3) + 1j * x[3 * K0:-1].reshape(K0, 3)
                        
                        # Construct chi-squared variable comparing subset z's to
                        # sampled variables and output p-value (computed with 6 * K0 dof)
                        Qj = 2 * np.sum(np.abs(zfj - eps * sfj * np.einsum("kij, kj -> ki", vhfj, c)) ** 2)
                        return chi2.sf(Qj, 6 * K0)

                    # Compute Monte Carlo integration
                    integral, error = mcint.integrate(integrand, sampler(), n = MC_NUM)

                    # Result of integration is only recorded if it exceeds 1 / MC_NUM
                    if integral < 1 / MC_NUM:
                        break
                    else:
                        pj[j] = integral
                
                # Determine combined probability via Fisher's method
                # (only if integration completes for all subsets)
                if np.all(pj != 0):
                    Qfull = -2 * np.sum(np.log(pj))
                    pfull = chi2.sf(Qfull, 8)
                    
                    # Record data for successful candidates (combined p-value above 0.01)
                    if pfull > 0.01:
                        succ_freqs.append(coh_freqs[freq_ind])

                        # Convert full-dataset p-value to global significance
                        sigma = np.sqrt(2) * erfcinv(2 * (1 - (1 - p0[freq_ind]) ** Nfreq))
                        succ_sigmas.append(sigma)

                        succ_pjs.append(pj)
                        succ_pfulls.append(pfull)

                        if VERBOSE:
                            print(f'Successful candidate at {coh_freqs[freq_ind]} ({sigma:.2f}-sigma)!')
                            print(f'Individual p-values: {pj}')
                            print(f'Combined p-value: {pfull}\n')

    if VERBOSE:
        print(f'Total Candidates: {count}')
        print(f'Storing results\n')

    # Store results
    Path('candidates').mkdir(exist_ok = True)
    filename = get_dirname()
    np.savez(f'candidates/{filename}',
             cand_freqs = cand_freqs,
             cand_p0s = cand_p0s,
             succ_freqs = succ_freqs,
             succ_sigmas = succ_sigmas,
             succ_pjs = succ_pjs,
             succ_pfulls = succ_pfulls)

    if VERBOSE: print(f'Done!')
