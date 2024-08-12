import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import norm
import sys
from utils import *

# Plot parameters
DM = 'DPDM'
dirname1 = 'weekly_window15'
dirname2 = 'weekly_window60'
LOG_DOWNSAMPLE = 10000
LOG_WINDOW = 0.02

# Use latex
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 12})

# Initiate figure
fig, ax = plt.subplots(figsize = (6., 6.))

# Calculate coherence times and their respective frequency bins
coherence_times = coherence_times(TOTAL_TIME)
freq_bins = frequencies_from_coherence_times(coherence_times)

# Initiate lists of frequencies and powers of z
freqs = []
zsqrs1 = []
zquars1 = []
zsqrs2 = []
zquars2 = []

# Iterate over coherence chunks
for (coh, (lof, hif_inclusive, df)) in zip(coherence_times, freq_bins):

    # Compute frequencies for coherence time
    coh_freqs = (lof + np.arange(hif_inclusive - lof + 1)) * df
    freqs.append(coh_freqs)

    # Load z variables and compute powers
    z1 = np.load(f'../{DM}/analysis_vars/{dirname1}/z_{coh}.npy')
    z2 = np.load(f'../{DM}/analysis_vars/{dirname2}/z_{coh}.npy')
    if DM == 'DPDM':
        zsqrs1.append(np.mean(np.abs(z1) ** 2, axis = (1, 2)))
        zquars1.append(np.mean(np.abs(z1) ** 4, axis = (1, 2)))
        zsqrs2.append(np.mean(np.abs(z2) ** 2, axis = (1, 2)))
        zquars2.append(np.mean(np.abs(z2) ** 4, axis = (1, 2)))
    elif DM == 'axion':
        zsqrs1.append(np.mean(np.abs(z1) ** 2, axis = 1))
        zquars1.append(np.mean(np.abs(z1) ** 4, axis = 1))
        zsqrs2.append(np.mean(np.abs(z2) ** 2, axis = 1))
        zquars2.append(np.mean(np.abs(z2) ** 4, axis = 1))
    
    print(f'Coherence time {coh} loaded')
    sys.stdout.flush()

# Concatenate lists from separate coherence times
# (Remove f=0 Hz because we will take logarithms)
freqs = np.concatenate(freqs)[1:]
zsqrs1 = np.concatenate(zsqrs1)[1:]
zquars1 = np.concatenate(zquars1)[1:]
zsqrs2 = np.concatenate(zsqrs2)[1:]
zquars2 = np.concatenate(zquars2)[1:]

# Smooth powers of z
# Powers are smoothed by convolving (in log space) with Gaussian of width LOG_WINDOW
# Smoothed powers only computed at every LOG_DOWNSAMPLE-th point
# and only if filter has negligible support outside frequency range
smoothed_zsqrs1 = np.zeros(div_ceil(len(freqs), LOG_DOWNSAMPLE))
smoothed_zquars1 = np.zeros(div_ceil(len(freqs), LOG_DOWNSAMPLE))
smoothed_zsqrs2 = np.zeros(div_ceil(len(freqs), LOG_DOWNSAMPLE))
smoothed_zquars2 = np.zeros(div_ceil(len(freqs), LOG_DOWNSAMPLE))
for ind in np.arange(0, len(freqs), LOG_DOWNSAMPLE):
    filter = norm.pdf((np.log(freqs[1:]) + np.log(freqs[:-1])) / 2, loc = np.log(freqs[ind]), scale = LOG_WINDOW)
    dfreq = np.log(freqs[1:]) - np.log(freqs[:-1])
    if np.sum(filter * dfreq) >= 0.99:
        smoothed_zsqrs1[ind // LOG_DOWNSAMPLE] = np.sum((zsqrs1[1:] + zsqrs1[:-1]) / 2 * filter * dfreq)
        smoothed_zquars1[ind // LOG_DOWNSAMPLE] = np.sum((zquars1[1:] + zquars1[:-1]) / 2 * filter * dfreq)
        smoothed_zsqrs2[ind // LOG_DOWNSAMPLE] = np.sum((zsqrs2[1:] + zsqrs2[:-1]) / 2 * filter * dfreq)
        smoothed_zquars2[ind // LOG_DOWNSAMPLE] = np.sum((zquars2[1:] + zquars2[:-1]) / 2 * filter * dfreq)
mask = np.where(smoothed_zsqrs1 != 0)[0]
print('Smoothing complete')
sys.stdout.flush()

# Setup main axes
xlim1 = 1e-3 # in Hz
xlim2 = freqs[::LOG_DOWNSAMPLE][mask][-1] # in Hz
ylim1 = 0.9
ylim2 = 1.4
ax.set_xlim(xlim1, xlim2)
ax.set_ylim(ylim1, ylim2)
ax.set_xscale('log')
ax.set_xlabel(r'$f$\,[Hz]')
ax.set_ylabel(r'$\langle|z_{ik}|^2\rangle$ OR $\frac{\langle|z_{ik}|^4\rangle}{2\langle|z_{ik}|^2\rangle^2}$')
ax.xaxis.set_major_formatter(CustomTicker())
ax.tick_params(which = 'both', direction = 'in')

# Setup secondary axes
secxax = ax.secondary_xaxis('top', zorder = 1)
secxax.tick_params(which = 'both', direction = 'in')
plt.setp(secxax.get_xticklabels(), visible = False)
secyax = ax.secondary_yaxis('right', zorder = 1)
secyax.tick_params(which = 'both', direction = 'in')
plt.setp(secyax.get_yticklabels(), visible = False)

# Plot powers of z
ax.axhline(1., color = '0.6', zorder = 1.8)
ax.plot(freqs[::LOG_DOWNSAMPLE][mask], smoothed_zsqrs1[mask], zorder = 2,
        color = 'royalblue', label = r'$\langle|z_{ik}|^2\rangle\,(\sigma=15)$')
ax.plot(freqs[::LOG_DOWNSAMPLE][mask], smoothed_zquars1[mask] / (2 * smoothed_zsqrs1[mask] ** 2), zorder = 2,
        color = 'darkorange', label = r'$\frac{\langle|z_{ik}|^4\rangle}{2\langle|z_{ik}|^2\rangle^2}\,(\sigma=15)$')
ax.plot(freqs[::LOG_DOWNSAMPLE][mask], smoothed_zsqrs2[mask], zorder = 1.9,
        color = 'limegreen', label = r'$\langle|z_{ik}|^2\rangle\,(\sigma=60)$')
ax.plot(freqs[::LOG_DOWNSAMPLE][mask], smoothed_zquars2[mask] / (2 * smoothed_zsqrs2[mask] ** 2), zorder = 1.9,
        color = 'crimson', label = r'$\frac{\langle|z_{ik}|^4\rangle}{2\langle|z_{ik}|^2\rangle^2}\,(\sigma=60)$')

# Plot legend
ax.legend()

# Save figure
Path('gaussianity').mkdir(parents = True, exist_ok = True)
fig.tight_layout()
fig.savefig(f'gaussianity/{DM}_{dirname1}_{dirname2}.pdf')
