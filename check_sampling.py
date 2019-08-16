"""
check the params entered to mm_params for given telescope settings for correct sampling
KD 8/13/19

According to the Proper manual:
    tp.beam_ratio = tp.enterance_d/ init_grid_width
    where init_grid_width is the initial grid width (diameter) in meters
Thus
    init_grid_width = tp.enterence_d / tp.beam_ratio

The sampling of the beam is defined by
    sampling = grid_width / tp.grid_size
    :keyword
    sampling = tp.enterence_d * tp.grid_size / tp.beam_ratio
    where grid_width here is the initial grid width

"""
import numpy as np
from mm_params import tp

#######################################
# Definitions from inside mini-medis
#######################################
# Basics-not imported
tp.enterance_d = 8.2  # m
tp.beam_ratio = 25/64
tp.grid_size = 512

# From Params Buried in mini-medis
legs_frac = 0.05  # m
Fnum = 12.6  # focal ratio of the beam as it approaches the focus
lmbda = 800e-6  # m  wavelength of interest

#######################
# Initial Sampling
######################
init_grid_width = tp.enterance_d / tp.beam_ratio
init_sampling = init_grid_width / tp.grid_size
print(f"Initial Grid Width = {init_grid_width:.2f} m")

#################################
# Checking Sampling of Spiders
################################
spider_width = tp.enterance_d * legs_frac
npix_on_spiders = spider_width / init_sampling
print(f"Number of pixels across the spiders = {npix_on_spiders}")
print("********************************")

###################
# Ideal Sampling
###################
"""
see proper manual pg 36 for definitions
"""
print(f"Ideal Values")
samp_good = Fnum*lmbda*tp.enterance_d/tp.grid_size
samp_nyq = Fnum*lmbda/2
print(f"Ideal Sampling is {samp_good:.6f} m/pix \nNyquist Sampling is {samp_nyq:.6f} m/pix")

# selecting beam ratio
br_good = tp.enterance_d/(samp_good*tp.grid_size)
br_nyq = tp.enterance_d/samp_nyq/tp.grid_size
print(f"Good starter beam ratio is {br_good} \nNyquist beam ratio is {br_nyq}")

