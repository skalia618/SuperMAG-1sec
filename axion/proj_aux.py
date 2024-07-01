from coordinate import *
import csv
from loader import *
import numpy as np
from params import *
from pathlib import Path
import sys
from utils import *

# Shorthands for mode sums appearing in projections/auxiliary timeseries
def mode_sum_theta(theta, phi, Cinterp):
    mode_sum = 0
    for l in range(1, LMAX + 1):
        for m in range(-l, l + 1):
            mode_sum += Cinterp[l][m] / l * Philm_theta(theta, phi, l, m)
    return np.real(mode_sum)

def mode_sum_phi(theta, phi, Cinterp):
    mode_sum = 0
    for l in range(1, LMAX + 1):
        for m in range(-l, l + 1):
            mode_sum += Cinterp[l][m] / l * Philm_phi(theta, phi, l, m)
    return np.real(mode_sum)

# Functions to compute contributions of a given station to projections,
# as given in Eqs. (C1) and (C2) in 2112.09620
def X1F(bn, wn, theta, phi, Cinterp):
    return np.nan_to_num(bn * wn * mode_sum_theta(theta, phi, Cinterp))

def X2F(be, we, theta, phi, Cinterp):
    return np.nan_to_num(be * we * mode_sum_phi(theta, phi, Cinterp))

# Functions for auxiliary timeseries [see Eq. (C8)]
def H1F(wn, theta, phi, Cinterp):
    return wn * mode_sum_theta(theta, phi, Cinterp) ** 2

def H2F(we, theta, phi, Cinterp):
    return we * mode_sum_phi(theta, phi, Cinterp) ** 2


# Functions to compute components of injected signal [see Eq. (C7)]
def signal_bn(theta, phi, Cinterp, start_sec, end_sec_exclusive, g_agamma = G_AGAMMA, fa = FA):
    # Prefactor in Eq. (C7)
    prefactor = g_agamma * R * np.sqrt(2 * RHODM)
    
    # Time dependence
    timedep = np.exp(-2 * np.pi * 1j * fa * np.arange(start_sec, end_sec_exclusive))
    
    return prefactor * np.imag(timedep) * mode_sum_theta(theta, phi, Cinterp)

def signal_be(theta, phi, Cinterp, start_sec, end_sec_exclusive, g_agamma = G_AGAMMA, fa = FA):
    # Prefactor in Eq. (C7)
    prefactor = g_agamma * R * np.sqrt(2 * RHODM)
    
    # Time dependence
    timedep = np.exp(-2 * np.pi * 1j * fa * np.arange(start_sec, end_sec_exclusive))
    
    return prefactor * np.imag(timedep) * mode_sum_phi(theta, phi, Cinterp)

if __name__ == '__main__':
    if VERBOSE:
        print_params(include_window = False,
                     include_threshold = False,
                     include_boundparams = False)

    # Get all stations and their coordinates
    stations = get_all_stations()
    station_coords = get_station_coordinates()

    # If using a subset, reduced the set of stations
    # The set of stations to be used for each subset
    # are given by each row of 'permutation.csv'
    if SUBSET != -1:
        file = open(PERMUTATION_PATH)
        reader = csv.reader(file, delimiter = ',')
        for i in range(SUBSET): next(reader)
        stations = next(reader)
        file.close()

    # Get stationarity periods
    periods = get_stationarity_chunks()

    # Get IGRF coefficients
    IGRFyears, g, h = load_IGRFcoeffs()
    C = gh_to_C(g, h)

    # Initialize all arrays up front
    # Projections
    X1 = np.zeros(TOTAL_TIME)
    X2 = np.zeros(TOTAL_TIME)
    # Auxiliary
    H1 = np.zeros(TOTAL_TIME)
    H2 = np.zeros(TOTAL_TIME)
    # Weights
    WN = np.zeros(TOTAL_TIME)
    WE = np.zeros(TOTAL_TIME)
    weights = {} # dictionary for individual station weights
    # Indicator
    I = np.zeros(TOTAL_TIME).astype(np.int32)

    if VERBOSE:
        print('Initialized workspace arrays\n')
        sys.stdout.flush()

    # We loop over stations first so that we can cache
    # station data from one period to the next
    for station in stations:
        if VERBOSE:
            print(f'Working on station {station}')
            sys.stdout.flush()

        # Get station coordinates
        (lat, lon) = station_coords[station]
        (theta, phi) = lat_lon_deg_to_theta_phi_radian(lat, lon)

        # Initialize station weights
        # Two weights (north and east) for each stationarity period
        station_weights = np.zeros((len(periods), 2))

        for period in periods:

            # Shorthands for start and end of period
            s = period[1]
            e = period[2]

            # Get period data
            period_data = load_period(station, s, e)

            if period_data is not None:
                # Unpack data
                bn, be = period_data

                # Clean data and get indicator
                period_indicator = clean_fields(bn, be)
                
                # Update global indicator
                I[s:e] += period_indicator.astype(np.int32)

                # Rotate data
                period_seconds = np.arange(s, e)
                rotation_angle = interpolate_angle(station, period_seconds)
                bn, be = rotate_fields(bn, be, rotation_angle)

                # Convert bn to b_theta
                bn = -bn

                # Interpolate IGRF coefficients to seconds for this period
                Cinterp = interpolate_C(C, period_seconds, IGRFyears)

                # Inject signal, if necessary (only at times where there is data)
                if INJECT:
                    bn += period_indicator * signal_bn(theta, phi, Cinterp, s, e)
                    be += period_indicator * signal_be(theta, phi, Cinterp, s, e)
                
                # Calculate and store weights
                wn = np.nan_to_num(period_indicator.sum() / (bn * bn).sum())
                we = np.nan_to_num(period_indicator.sum() / (be * be).sum())
                station_weights[period[0]] = [wn, we]

                # Add to total weights
                WN[s:e] += period_indicator * wn
                WE[s:e] += period_indicator * we

                # Add proj, aux contributions
                X1[s:e] += X1F(bn, wn, theta, phi, Cinterp)
                X2[s:e] += X2F(be, we, theta, phi, Cinterp)
                H1[s:e] += period_indicator * H1F(wn, theta, phi, Cinterp)
                H2[s:e] += period_indicator * H2F(we, theta, phi, Cinterp)

        weights[station] = station_weights
        if VERBOSE:
            print(f'Finished station {station}\n')
            sys.stdout.flush()
    
    # Normalize Xi, Hi by total weights
    np.seterr(invalid = 'ignore') # ignore division errors in cases where there is no data at some time
    X1 /= WN
    X2 /= WE

    H1 /= WN
    H2 /= WE

    if VERBOSE:
        print('Storing results\n')
        sys.stdout.flush()

    # Create output directory
    proj_aux_dir = get_proj_aux_dir()
    Path(proj_aux_dir).mkdir(parents = True, exist_ok = True)

    # Store proj, aux, and indicator
    np.save(f'{proj_aux_dir}/X1.npy', X1)
    np.save(f'{proj_aux_dir}/X2.npy', X2)

    np.save(f'{proj_aux_dir}/H1.npy', H1)
    np.save(f'{proj_aux_dir}/H2.npy', H2)

    np.save(f'{proj_aux_dir}/I.npy', I)

    # Weights stored as dictionary, where values are arrays of weights for each period
    np.savez(f'{proj_aux_dir}/weights', **weights)

    if VERBOSE: print('Done!')
