from coordinate import *
import csv
from loader import *
import numpy as np
from params import *
from pathlib import Path
import sys
from utils import *

# Functions to compute contributions of a given station to projections,
# as given in Eqs. (7) - (11) of 2108.08852
def X1F(bn, wn, theta, phi):
    return np.nan_to_num(bn * wn * np.sin(phi))

def X2F(bn, wn, theta, phi):
    return np.nan_to_num(bn * wn * np.cos(phi))

def X3F(be, we, theta, phi):
    return np.nan_to_num(be * we * np.cos(phi) * np.cos(theta))

def X4F(be, we, theta, phi):
    return np.nan_to_num(be * we * -np.sin(phi) * np.cos(theta))

def X5F(be, we, theta, phi):
    return np.nan_to_num(be * we * np.sin(theta))

# Functions for auxiliary timeseries [see Eqs. (28) - (34)]
def H1F(wn, theta, phi):
    return wn * np.cos(phi) ** 2

def H2F(wn, theta, phi):
    return wn * np.sin(phi) * np.cos(phi)

def H3F(we, theta, phi):
    return we * np.cos(theta) ** 2

def H4F(we, theta, phi):
    return we * np.cos(phi) ** 2 * np.cos(theta) ** 2

def H5F(we, theta, phi):
    return we * np.sin(phi) * np.cos(phi) * np.cos(theta) ** 2

def H6F(we, theta, phi):
    return we * np.cos(phi) * np.sin(theta) * np.cos(theta)

def H7F(we, theta, phi):
    return we * np.sin(phi) * np.sin(theta) * np.cos(theta)


# Functions to compute components of injected signal [see Eq. (1)]
def signal_bn(theta, phi, start_sec, end_sec_exclusive, eps = EPS, fA = FA, pol = POL):
    # Normalize polarization vector
    pol /= np.linalg.norm(pol)

    # Prefactor in Eq. (1)
    prefactor = np.sqrt(np.pi / 3) * eps * 2 * np.pi * fA * R * np.sqrt(2 * RHODM)

    # Theta component of Phi11 spherical harmonic
    Phi11_theta = np.sqrt(3 / 8 / np.pi) * np.exp(1j * phi) * 1j

    # Time dependences of m = -1,1 modes
    minus_timedep = np.exp(-2 * np.pi * 1j * (fA + FD) * np.arange(start_sec, end_sec_exclusive))
    plus_timedep = np.exp(-2 * np.pi * 1j * (fA - FD) * np.arange(start_sec, end_sec_exclusive))

    return prefactor * np.real(
        pol[0] * np.conj(Phi11_theta) * minus_timedep +
        pol[2] * Phi11_theta * plus_timedep)

def signal_be(theta, phi, start_sec, end_sec_exclusive, eps = EPS, fA = FA, pol = POL):
    # Normalize polarization vector
    pol /= np.linalg.norm(pol)

    # Prefactor in Eq. (1)
    prefactor = np.sqrt(np.pi / 3) * eps * 2 * np.pi * fA * R * np.sqrt(2 * RHODM)

    # Phi components of Phi10 and Phi11 spherical harmonics
    Phi10_phi = -np.sqrt(3 / 4 / np.pi) * np.sin(theta)
    Phi11_phi = -np.sqrt(3 / 8 / np.pi) * np.exp(1j * phi) * np.cos(theta)

    # Time dependences of m = -1,0,1 modes
    minus_timedep = np.exp(-2 * np.pi * 1j * (fA + FD) * np.arange(start_sec, end_sec_exclusive))
    zero_timedep = np.exp(-2 * np.pi * 1j * fA * np.arange(start_sec, end_sec_exclusive))
    plus_timedep = np.exp(-2 * np.pi * 1j * (fA - FD) * np.arange(start_sec, end_sec_exclusive))

    return prefactor * np.real(
        pol[0] * np.conj(Phi11_phi) * minus_timedep +
        pol[1] * Phi10_phi * zero_timedep +
        pol[2] * Phi11_phi * plus_timedep)


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

    # Initialize all arrays up front
    # Projections
    X1 = np.zeros(TOTAL_TIME)
    X2 = np.zeros(TOTAL_TIME)
    X3 = np.zeros(TOTAL_TIME)
    X4 = np.zeros(TOTAL_TIME)
    X5 = np.zeros(TOTAL_TIME)
    # Auxiliary
    H1 = np.zeros(TOTAL_TIME)
    H2 = np.zeros(TOTAL_TIME)
    H3 = np.zeros(TOTAL_TIME)
    H4 = np.zeros(TOTAL_TIME)
    H5 = np.zeros(TOTAL_TIME)
    H6 = np.zeros(TOTAL_TIME)
    H7 = np.zeros(TOTAL_TIME)
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

                # Inject signal, if necessary (only at times where there is data)
                if INJECT:
                    bn += period_indicator * signal_bn(theta, phi, s, e)
                    be += period_indicator * signal_be(theta, phi, s, e)
                
                # Calculate and store weights
                wn = np.nan_to_num(period_indicator.sum() / (bn * bn).sum())
                we = np.nan_to_num(period_indicator.sum() / (be * be).sum())
                station_weights[period[0]] = [wn, we]

                # Add to total weights
                WN[s:e] += period_indicator * wn
                WE[s:e] += period_indicator * we

                # Add proj, aux contributions
                X1[s:e] += X1F(bn, wn, theta, phi)
                X2[s:e] += X2F(bn, wn, theta, phi)
                X3[s:e] += X3F(be, we, theta, phi)
                X4[s:e] += X4F(be, we, theta, phi)
                X5[s:e] += X5F(be, we, theta, phi)
                H1[s:e] += period_indicator * H1F(wn, theta, phi)
                H2[s:e] += period_indicator * H2F(wn, theta, phi)
                H3[s:e] += period_indicator * H3F(we, theta, phi)
                H4[s:e] += period_indicator * H4F(we, theta, phi)
                H5[s:e] += period_indicator * H5F(we, theta, phi)
                H6[s:e] += period_indicator * H6F(we, theta, phi)
                H7[s:e] += period_indicator * H7F(we, theta, phi)

        weights[station] = station_weights
        if VERBOSE:
            print(f'Finished station {station}\n')
            sys.stdout.flush()
    
    # Normalize Xi, Hi by total weights
    np.seterr(invalid = 'ignore') # ignore division errors in cases where there is no data at some time
    X1 /= WN
    X2 /= WN
    X3 /= WE
    X4 /= WE
    X5 /= WE

    H1 /= WN
    H2 /= WN
    H3 /= WE
    H4 /= WE
    H5 /= WE
    H6 /= WE
    H7 /= WE

    if VERBOSE:
        print('Storing results\n')
        sys.stdout.flush()

    # Create output directory
    proj_aux_dir = get_proj_aux_dir()
    Path(proj_aux_dir).mkdir(parents = True, exist_ok = True)

    # Store proj, aux, and indicator
    np.save(f'{proj_aux_dir}/X1.npy', X1)
    np.save(f'{proj_aux_dir}/X2.npy', X2)
    np.save(f'{proj_aux_dir}/X3.npy', X3)
    np.save(f'{proj_aux_dir}/X4.npy', X4)
    np.save(f'{proj_aux_dir}/X5.npy', X5)

    np.save(f'{proj_aux_dir}/H1.npy', H1)
    np.save(f'{proj_aux_dir}/H2.npy', H2)
    np.save(f'{proj_aux_dir}/H3.npy', H3)
    np.save(f'{proj_aux_dir}/H4.npy', H4)
    np.save(f'{proj_aux_dir}/H5.npy', H5)
    np.save(f'{proj_aux_dir}/H6.npy', H6)
    np.save(f'{proj_aux_dir}/H7.npy', H7)

    np.save(f'{proj_aux_dir}/I.npy', I)

    # Weights stored as dictionary, where values are arrays of weights for each period
    np.savez(f'{proj_aux_dir}/weights', **weights)

    if VERBOSE: print('Done!')
