import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from utils import *

# DM candidate and window sizes (for annual, monthly, weekly, and daily spectra, respectively)
DM = 'DPDM'
WINDOWS = [10000, 1000, 200, 30]

# Use latex
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 12})

# Initiate figure
fig, ax = plt.subplots(figsize = (6., 6.))

# Choose random year and timeseries to plot
year = np.random.randint(START_YEAR, END_YEAR + 1)
start = year_start_sec(year)
end = year_start_sec(year + 1)
if DM == 'DPDM':
    i = np.random.randint(1, 6)
elif DM == 'axion':
    i = np.random.randint(1, 3)
filename = f'{DM}_{year}_X{i}X{i}'
print(f'Plotting {filename}')

# Construct spectrum frequencies and load annual spectrum
freqs = spectra_freqs(year - START_YEAR, 'annual', WINDOWS[0])
spectrum_annual = np.real(np.load(f'../{DM}/spectra/annual_window{WINDOWS[0]}/chunk_{year - START_YEAR:04d}/X{i}X{i}.npy'))

# Construct monthly spectrum (via weighting by overlaps)
monthly_overlapping_chunks = find_overlap_chunks(start, end, 'monthly')
spectrum_monthly = np.zeros(len(freqs))
for (chunk, overlap) in monthly_overlapping_chunks:
    spec_freqs = spectra_freqs(chunk, 'monthly', WINDOWS[1])
    spectrum = np.real(np.load(f'../{DM}/spectra/monthly_window{WINDOWS[1]}/chunk_{chunk:04d}/X{i}X{i}.npy'))
    spectrum_monthly += overlap * np.interp(freqs, spec_freqs, spectrum)
spectrum_monthly /= end - start

# Construct weeky spectrum (via weighting by overlaps)
weekly_overlapping_chunks = find_overlap_chunks(start, end, 'weekly')
spectrum_weekly = np.zeros(len(freqs))
for (chunk, overlap) in weekly_overlapping_chunks:
    spec_freqs = spectra_freqs(chunk, 'weekly', WINDOWS[2])
    spectrum = np.real(np.load(f'../{DM}/spectra/weekly_window{WINDOWS[2]}/chunk_{chunk:04d}/X{i}X{i}.npy'))
    spectrum_weekly += overlap * np.interp(freqs, spec_freqs, spectrum)
spectrum_weekly /= end - start

# Construct daily spectrum (via weighting by overlaps)
daily_overlapping_chunks = find_overlap_chunks(start, end, 'daily')
spectrum_daily = np.zeros(len(freqs))
for (chunk, overlap) in daily_overlapping_chunks:
    spec_freqs = spectra_freqs(chunk, 'daily', WINDOWS[3])
    spectrum = np.real(np.load(f'../{DM}/spectra/daily_window{WINDOWS[3]}/chunk_{chunk:04d}/X{i}X{i}.npy'))
    spectrum_daily += overlap * np.interp(freqs, spec_freqs, spectrum)
spectrum_daily /= end - start

# Setup main axes
xlim1 = 1e-3 # in Hz
xlim2 = 1. # in Hz
ylim1 = min(np.amin(spectrum_annual), np.amin(spectrum_monthly), np.amin(spectrum_weekly), np.amin(spectrum_daily))
ylim1 = 10 ** np.floor(np.log10(ylim1))
ylim2 = 1e6 * ylim1
ax.set_xlim(xlim1, xlim2)
ax.set_ylim(ylim1, ylim2)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$f$\,[Hz]')
ax.set_ylabel(f'$S_{{{i}{i}}}^{{{year}}}(f)\,[\mathrm{{nT}}^2/\mathrm{{Hz}}]$')
ax.xaxis.set_major_formatter(CustomTicker())
ax.yaxis.set_major_formatter(CustomTicker())
ax.tick_params(which = 'both', direction = 'in')

# Setup secondary axes
secxax = ax.secondary_xaxis('top', zorder = 1)
secxax.tick_params(which = 'both', direction = 'in')
plt.setp(secxax.get_xticklabels(), visible = False)
secyax = ax.secondary_yaxis('right', zorder = 1)
secyax.tick_params(which = 'both', direction = 'in')
plt.setp(secyax.get_yticklabels(), visible = False)

# Plot spectra
ax.plot(freqs, spectrum_annual, color = 'royalblue', label = r'Annual')
ax.plot(freqs, spectrum_monthly, color = 'darkorange', label = r'Monthly')
ax.plot(freqs, spectrum_weekly, color = 'limegreen', label = r'Weekly')
ax.plot(freqs, spectrum_daily, color = 'crimson', label = r'Daily')

# Plot legend
ax.legend()

# Save figure
Path('stationarity_times').mkdir(parents = True, exist_ok = True)
fig.tight_layout()
fig.savefig(f'stationarity_times/{filename}.pdf')
