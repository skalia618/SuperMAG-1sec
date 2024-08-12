import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import chi2
from utils import *

# Plot parameters
DM = 'DPDM'
j = 0
dirname = 'weekly_window15'
dirnamej = 'weekly_subset0_window15'
COH_IND = 70
PCRIT = 7.65543e-9

xlim1 = 100
xlim2 = 600
ylim1 = 2e-6
ylim2 = 0.03
BINSIZE = 5
LEGEND = False

# Use latex
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 12})

# Initiate figure
fig, ax = plt.subplots(figsize = (6., 6.))

# Calculate coherence times and their respective frequency bins
coherence_times = coherence_times(TOTAL_TIME)
freq_bins = frequencies_from_coherence_times(coherence_times)
print(f'Frequency range: ({freq_bins[COH_IND][0] * freq_bins[COH_IND][2]:.4f}, {freq_bins[COH_IND][1] * freq_bins[COH_IND][2]:.4f})')

# Load analysis variables (for full dataset and subset)
s = np.load(f'../{DM}/analysis_vars/{dirname}/s_{coherence_times[COH_IND]}.npy')
z = np.load(f'../{DM}/analysis_vars/{dirname}/z_{coherence_times[COH_IND]}.npy')
v = np.load(f'../{DM}/analysis_vars/{dirname}/v_{coherence_times[COH_IND]}.npy')
sj = np.load(f'../{DM}/analysis_vars/{dirnamej}/s_{coherence_times[COH_IND]}.npy')
zj = np.load(f'../{DM}/analysis_vars/{dirnamej}/z_{coherence_times[COH_IND]}.npy')
vj = np.load(f'../{DM}/analysis_vars/{dirnamej}/v_{coherence_times[COH_IND]}.npy')
K0 = z.shape[1]

# Compute Q0 and Qj
if DM == 'DPDM':
    Q0 = 2 * np.sum(np.abs(z) ** 2, axis = (1, 2))

    Zeta = (np.einsum('fkij, fkj, fkj -> fki', v, 1 / s, z)
            - np.einsum('fkij, fkj, fkj -> fki', vj, 1 / sj, zj))
    Xi = (np.einsum('fkab, fkb, fkcb -> fkac', v, 1 / s ** 2, np.conjugate(v))
          + np.einsum('fkab, fkb, fkcb -> fkac', vj, 1 / sj ** 2, np.conjugate(vj)))
    invB = np.linalg.inv(np.linalg.cholesky(Xi))
    zeta = np.einsum('fkij, fkj -> fki', invB, Zeta)
    Qj = 2 * np.sum(np.abs(zeta) ** 2, axis = (1, 2))

if DM == 'axion':
    Q0 = 2 * np.sum(np.abs(z) ** 2, axis = 1)

    Zeta = z / s - zj / sj
    Xi = s ** -2 + sj ** -2
    zeta = Zeta / np.sqrt(Xi)
    Qj = 2 * np.sum(np.abs(zeta) ** 2, axis = 1)

# Fit Q0 to chi-squared and convert to p0
dof0 = chi2.fit(Q0, floc = 0., fscale = 1.)[0]
p0 = chi2.sf(Q0, dof0)

# Fit Qj to chi-squared
dofj = chi2.fit(Qj, floc = 0., fscale = 1.)[0]
if DM == 'DPDM':
    print(f'Degrees of freedom: {dofj:.2f} (best-fit) vs {6 * K0} (naive)')
if DM == 'axion':
    print(f'Degrees of freedom: {dofj:.2f} (best-fit) vs {2 * K0} (naive)')
upper = np.max(Qj)

# Setup main axes
ax.set_xlim(xlim1, xlim2)
ax.set_ylim(ylim1, ylim2)
ax.set_yscale('log')
ax.set_xlabel(r'$Q_j$')
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

# Plot histogram of Qjs (candidates are plotted in crimson) and chi-squared distributions
ax.hist([Qj[np.where(p0 < PCRIT)], Qj[np.where(p0 > PCRIT)]],
        bins = np.arange(0, upper + BINSIZE, BINSIZE),
        density = True,
        stacked = True,
        color = ['crimson', 'royalblue'],
        label = ['Candidates'])
xcoor = np.linspace(xlim1, xlim2, 1000)
ax.plot(xcoor, chi2.pdf(xcoor, dofj), color = 'darkorange', label = r'$\chi^2(\nu_\mathrm{fit})$')
if DM == 'DPDM':
    ax.plot(xcoor, chi2.pdf(xcoor, 6 * K0), color = 'limegreen', label = r'$\chi^2(6K_0)$')
elif DM == 'axion':
    ax.plot(xcoor, chi2.pdf(xcoor, 2 * K0), color = 'limegreen', label = r'$\chi^2(2K_0)$')

# Plot legend
if LEGEND: ax.legend()

# Save figure
Path('zetadist').mkdir(parents = True, exist_ok = True)
fig.tight_layout()
fig.savefig(f'zetadist/{DM}_{COH_IND}_j{j}.png')
