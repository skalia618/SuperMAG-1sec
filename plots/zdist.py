import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import chi2
from utils import *

# Plot parameters
DM = 'DPDM'
dirname = 'weekly_window15'
COH_IND = 70
PCRIT = 7.65543e-9

xlim1 = 200
xlim2 = 900
ylim1 = 2e-6
ylim2 = 0.03
BINSIZE = 5
LEGEND = True

# Use latex
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 12})

# Initiate figure
fig, ax = plt.subplots(figsize = (6., 6.))

# Calculate coherence times and their respective frequency bins
coherence_times = coherence_times(TOTAL_TIME)
freq_bins = frequencies_from_coherence_times(coherence_times)
print(f'Frequency range: ({freq_bins[COH_IND][0] * freq_bins[COH_IND][2]:.4f}, {freq_bins[COH_IND][1] * freq_bins[COH_IND][2]:.4f})')

# Load z variables
z = np.load(f'../{DM}/analysis_vars/{dirname}/z_{coherence_times[COH_IND]}.npy')
K0 = z.shape[1]

# Compute Q0
if DM == 'DPDM':
    Q0 = 2 * np.sum(np.abs(z) ** 2, axis = (1, 2))
if DM == 'axion':
    Q0 = 2 * np.sum(np.abs(z) ** 2, axis = 1)

# Fit Q0 to chi-squared and convert to p0
dof = chi2.fit(Q0, floc = 0., fscale = 1.)[0]
p0 = chi2.sf(Q0, dof)
if DM == 'DPDM':
    print(f'Degrees of freedom: {dof:.2f} (best-fit) vs {6 * K0} (naive)')
if DM == 'axion':
    print(f'Degrees of freedom: {dof:.2f} (best-fit) vs {2 * K0} (naive)')
upper = np.max(Q0)

# Setup main axes
ax.set_xlim(xlim1, xlim2)
ax.set_ylim(ylim1, ylim2)
ax.set_yscale('log')
ax.set_xlabel(r'$Q_0$')
ax.set_ylabel(r'Normalized Counts')
ax.yaxis.set_major_formatter(CustomTicker())
ax.tick_params(which = 'both', direction = 'in')

# Setup secondary axes
secxax = ax.secondary_xaxis('top', zorder = 1)
secxax.tick_params(which = 'both', direction = 'in')
plt.setp(secxax.get_xticklabels(), visible = False)
secyax = ax.secondary_yaxis('right', zorder = 1)
secyax.tick_params(which = 'both', direction = 'in')
plt.setp(secyax.get_yticklabels(), visible = False)

# Plot histogram of Q0s (candidates are plotted in crimson) and chi-squared distributions
ax.hist([Q0[np.where(p0 < PCRIT)], Q0[np.where(p0 > PCRIT)]],
        bins = np.arange(0, upper + BINSIZE, BINSIZE),
        density = True,
        stacked = True,
        color = ['crimson', 'royalblue'],
        label = ['Candidates'])
xcoor = np.linspace(xlim1, xlim2, 1000)
ax.plot(xcoor, chi2.pdf(xcoor, dof), color = 'darkorange', label = r'$\chi^2(\nu_\mathrm{fit})$')
if DM == 'DPDM':
    ax.plot(xcoor, chi2.pdf(xcoor, 6 * K0), color = 'limegreen', label = r'$\chi^2(6K_0)$')
elif DM == 'axion':
    ax.plot(xcoor, chi2.pdf(xcoor, 2 * K0), color = 'limegreen', label = r'$\chi^2(2K_0)$')

# Plot legend
if LEGEND: ax.legend()

# Save figure
Path('zdist').mkdir(parents = True, exist_ok = True)
fig.tight_layout()
fig.savefig(f'zdist/{DM}_{COH_IND}.png')
