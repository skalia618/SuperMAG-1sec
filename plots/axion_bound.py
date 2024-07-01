import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from scipy import stats

BOUND_PATH = '../axion/bounds/weekly_window15.npz'
DOWNSAMPLE = 10000
LOG_WINDOW = 0.02

# Use latex
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 12})

# Initiate figure
fig, ax = plt.subplots(figsize = (6., 6.))

# Setup main axes
Hz_to_eV = 2 * np.pi * 6.582e-16
xlim1 = 0.007 # in Hz
xlim2 = 0.7 # in Hz
ylim1 = 3e-13
ylim2 = 3e-7
ax.set_xlim(Hz_to_eV * xlim1, Hz_to_eV * xlim2)
ax.set_ylim(ylim1, ylim2)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r"$m_a$\,[eV]")
ax.set_ylabel(r'$g_{a\gamma}$')
ax.tick_params(which = 'both', direction = 'in')

# Setup secondary axes
secxax = ax.secondary_xaxis('top', functions = (lambda x: x / Hz_to_eV, lambda x: Hz_to_eV * x), zorder = 1)
secxax.tick_params(which = 'both', direction = 'in')
secxax.set_xlabel(r"$f_a$\,[Hz]")
secyax = ax.secondary_yaxis('right', zorder = 1)
secyax.tick_params(which = 'both', direction = 'in')
plt.setp(secyax.get_yticklabels(), visible = False)

# Import and plot bound
bounddata = np.load(BOUND_PATH)
masses = Hz_to_eV * bounddata['freqs']
bounds = 1.25 * bounddata['bounds'] # include degradation factor (see Sec. V.F of 2108.08852)
plt.plot(masses, bounds, color = 'royalblue', label = 'SuperMAG 1-sec')

# Plot smoothed bound
# Bound is smoothed by convolving (in log space) with Gaussian of width LOG_WINDOW
# Smoothed bound only computed at every DOWNSAMPLE-th point
smoothed = np.zeros(len(masses) // DOWNSAMPLE)
for ind in np.arange(0, len(masses), DOWNSAMPLE):
    filter = stats.norm.pdf((np.log(masses[1:]) + np.log(masses[:-1])) / 2, loc = np.log(masses[ind]), scale = LOG_WINDOW)
    dmass = np.log(masses[1:]) - np.log(masses[:-1])
    if np.sum(filter * dmass) >= 0.99:
        smoothed[ind // DOWNSAMPLE] = np.sum((bounds[1:] + bounds[:-1]) / 2 * filter * dmass)
mask = np.where(smoothed != 0)[0]
plt.plot(masses[::DOWNSAMPLE][mask], smoothed[mask], color = 'deepskyblue', label = 'SuperMAG 1-sec, smoothed')

# Plot constraints
supermag = np.loadtxt('axion_constraints/supermag_axion.txt').T
line, = ax.plot(Hz_to_eV * supermag[0], supermag[1], color = 'darkorange', label = 'SuperMAG 1-min, smoothed')
line.set_dashes([2, 2])
snipehunt = np.loadtxt('axion_constraints/snipehunt_axion.txt').T
line, = ax.plot(Hz_to_eV * snipehunt[0], snipehunt[1], color = 'limegreen', label = 'SNIPE Hunt')
line.set_dashes([10, 5])
chandra = np.loadtxt('axion_constraints/chandra.txt').T
line, = ax.plot(chandra[0], chandra[1], color = 'crimson', label = 'Chandra')
line.set_dashes([5, 5])
line = ax.axhline(5.2304e-12, color = 'darkviolet', label = 'SN1987A')
line.set_dashes([5, 2, 2, 2])
line = ax.axhline(6.6316e-11, color = 'sienna', label = 'CAST')
line.set_dashes([6, 2, 4, 2])

# Plot legend
plt.legend(loc = 'upper center', handlelength = 3.3)

# Save figure
plt.savefig('axion_bound.pdf')
