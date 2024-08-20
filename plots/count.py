import matplotlib.pyplot as plt
import numpy as np
from utils import *

# Load indicator series
I = np.load('../axion/proj_aux/weekly_1998_2020/I.npy')

# Use latex (and set axes above plot)
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 12, 'axes.axisbelow': False})

# Initiate figure
fig, ax = plt.subplots(figsize = (6., 4.))

# Setup main axes
xlim1 = 0 # in s
xlim2 = len(I) - 1 # in s
ylim1 = 0
ylim2 = 140
ax.set_xlim(xlim1, xlim2)
ax.set_ylim(ylim1, ylim2)
ax.set_xticks([year_start_sec(year, start_year = 1998) for year in range(2000, 2021, 5)])
ax.set_xticklabels([year for year in range(2000, 2021, 5)])
ax.set_xticks([year_start_sec(year, start_year = 1998) for year in range(1998, 2021)], minor = True)
ax.set_ylabel(r'Stations reporting')
ax.tick_params(which = 'both', direction = 'in')

# Setup secondary axes
secxax = ax.secondary_xaxis('top', zorder = 2.5)
secxax.tick_params(which = 'both', direction = 'in')
secxax.set_xticks([year_start_sec(year, start_year = 1998) for year in range(2000, 2021, 5)])
secxax.set_xticklabels([''] * 5)
secxax.set_xticks([year_start_sec(year, start_year = 1998) for year in range(1999, 2021)], minor = True)
secyax = ax.secondary_yaxis('right', zorder = 2.5)
secyax.tick_params(which = 'both', direction = 'in')
plt.setp(secyax.get_yticklabels(), visible = False)

# Plot station count, shaded region, and grid lines
ax.plot(np.arange(len(I)), I, color = 'crimson')
ax.fill_between([xlim1, year_start_sec(2005, start_year = 1998)],
                [ylim1, ylim1], [ylim2, ylim2], color = '0.9', zorder = 1)
for year in range(1999, 2021):
    if year % 5 == 0:
        ax.axvline(year_start_sec(year, start_year = 1998), color = '0.7', lw = 0.75, zorder = 1.5)
    else:
        ax.axvline(year_start_sec(year, start_year = 1998), color = '0.7', ls = (0, (5, 7)), lw = 0.5, zorder = 1.5)
for y in range(20, 140, 20):
    ax.axhline(y, color = '0.7', lw = 0.75, zorder = 1.5)

# Save figure
fig.tight_layout()
fig.savefig(f'count.pdf')
