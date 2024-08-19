# SuperMAG-1sec
Analysis code used in [arXiv:24XX.XXXXX](https://arxiv.org/pdf/24XX.XXXXX) to search SuperMAG high-fidelity dataset for dark-photon and axion dark matter

(Based on analyses of low-fidelity dataset in [arXiv:2108.08852](https://arxiv.org/pdf/2108.08852) and [arXiv:2112.09620](https://arxiv.org/pdf/2112.09620))

Raw magnetometer data not included!

If you have any questions, please contact Saarik Kalia (kalias@umn.edu).

## Root directory

Root directory contains auxiliary data files:
- IGRF13coeffs.csv: Table of IGRF-13 coefficients (see Table 2 of [this reference](https://earth-planets-space.springeropen.com/articles/10.1186/s40623-020-01288-x))
- permutation.csv: List of stations used for subset analyses
- station_data.txt: Geographic coordinates and declination angles (as a function of year) for all stations

Note that this repository does not contain the raw magnetometer data.  Once obtained, this data should be added here as a directory raw/ (with subfolders for each year).

If raw data directory or auxiliary data files need to be moved, their paths should be changed in utils.py for both analyses (see below).

## Analysis directories
axion/ and DPDM/ contain analysis pipelines for each case.

Desired parameters are set by modifying params.py (e.g. perform subset analysis, inject signal, change stationarity time or filter window size, etc.)

Libraries used in other codes:
- coordinate.py: Loads station coordinates and declination angles
- loader.py: Loads raw magnetometer data
- utils.py: Main library with variety of useful methods (also contains paths to external data files)

To execute main analysis pipeline, the following codes should be run in sequential order with fixed set of parameters (each code will generate a results directory with same name as code, unless otherwise specified):
- proj_aux.py: Produces *X<sup>(i)</sup>* and *H<sup>(i)</sup>* time series
- spectra.py: Computes PSD *S<sup>a</sup><sub>mn</sub>* for each stationarity period
- analysis.py: Performs likelihood analysis (results directory: analysis_vars/)
- bound.py: Computes constraint on DPDM/axion DM coupling

In addition, candidates.py uses the results of analysis.py to identify and validate potential DM candidates.  In order to run candidates.py, the entire pipeline must be run on the full dataset, as well as four subsets.

## Plotting directory
plots/ contains codes to produce all the plots in the paper (each code will generate a results directory with same name as code, unless otherwise specified):
- axion_bound.py: Right plot of Fig. 3 (existing constraints found in axion_constraints/; results directory: bounds/)
- DPDM_bound.py: Left plot of Fig. 3 (existing constraints found in DPDM_constraints/; results directory: bounds/)
- count.py: Fig. 1 (no results directory)
- gaussianity.py: Fig. 5
- stationarity_times.py: A single subpanel of Fig. 4
- stationarity_times_superplot.py: Full Fig. 4 (results directory: stationarity_times/)
- utils.py: Library with useful methods
- zdist.py: Left plot of Fig. 2
- zetadist.py: Right plot of Fig. 2
