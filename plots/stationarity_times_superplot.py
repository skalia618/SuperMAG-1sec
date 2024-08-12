from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from utils import *

# Plot parameters
DM = 'DPDM'
WINDOWS = [10000, 1000, 200, 30]
YEARS = np.array([[2006, 2009], [2010, 2014]])
IS = np.array([[2, 1], [4, 3]])
YLIMS = np.array([[[1e-4, 1e2], [1e-4, 1e2]], [[2e-5, 20], [2e-5, 20]]])

# Use latex
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 14})

# Initiate figure (with 2x2 grid of sub-figures)
fig = plt.figure(figsize = (12, 8))
mastergrid = GridSpec(2, 2, figure = fig)

# Plot each subfigure
for m in range(2):
    for n in range(2):
        ax = fig.add_subplot(mastergrid[m, n])

        # Get year and timeseries for subfigure
        year = YEARS[m, n]
        start = year_start_sec(year)
        end = year_start_sec(year + 1)
        i = IS[m, n]

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
        ylim1 = YLIMS[m, n, 0]
        ylim2 = YLIMS[m, n, 1]
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
        if m == 0:
            ax.get_xaxis().tick_top()
            ax.get_xaxis().set_label_position('top')
            secxax = ax.secondary_xaxis('bottom', zorder = 1)
        else:
            secxax = ax.secondary_xaxis('top', zorder = 1)
        secxax.tick_params(which = 'both', direction = 'in')
        plt.setp(secxax.get_xticklabels(), visible = False)
        if n == 1:
            ax.get_yaxis().tick_right()
            ax.get_yaxis().set_label_position('right')
            secyax = ax.secondary_yaxis('left', zorder = 1)
        else:
            secyax = ax.secondary_yaxis('right', zorder = 1)
        secyax.tick_params(which = 'both', direction = 'in')
        plt.setp(secyax.get_yticklabels(), visible = False)

        # Plot spectra
        ax.plot(freqs, spectrum_annual, color = 'royalblue', label = r'Annual')
        ax.plot(freqs, spectrum_monthly, color = 'darkorange', label = r'Monthly')
        ax.plot(freqs, spectrum_weekly, color = 'limegreen', label = r'Weekly')
        ax.plot(freqs, spectrum_daily, color = 'crimson', label = r'Daily')

        # Plot legend (only in upper right sub-figure)
        if m == 0 and n == 1: ax.legend(loc = 'upper right')

# Save figure
Path('stationarity_times').mkdir(parents = True, exist_ok = True)
fig.tight_layout()
fig.savefig(f'stationarity_times/superplot.pdf')
