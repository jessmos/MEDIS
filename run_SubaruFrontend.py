"""
run_SubaruFrontend
KD
Dec 9 2019

This is the starting point to run the Subaru_frontend prescription. From here, you can turn on/off toggles, change AO
settings, view different planes, and make other changes to running the prescription without changing the base
prescription or the default mm_params themselves.

This file is meant to 'cleanup' a lot of the functionality between having defaults saved in the mm_params file and
the prescription. Most changes necessary to run multiple instances of the base prescription will be made here. This is
the same layot/format as the Example files that Rupert was using in v1.0. These are now meant to be paired with a more
specific prescription than the original optics_propagate, which had more toggles and less sophisiticated optical train.

"""

from mm_params import iop, sp, ap, tp, cdip
from mm_utils import dprint
import mini_medis as mm

#################################################################################################
#################################################################################################
#################################################################################################
tp.prescription = 'Subaru_frontend'
tp.enterance_d = 7.9716
tp.flen_primary = tp.enterance_d * 13.612

sp.focused_sys = True
sp.beam_ratio = 0.14  # parameter dealing with the sampling of the beam in the pupil/focal plane
sp.grid_size = 512  # creates a nxn array of samples of the wavefront
sp.maskd_size = 256  # will truncate grid_size to this range (avoids FFT artifacts) # set to grid_size if undesired

# Toggles for Aberrations and Control
tp.obscure = False
tp.use_atmos = True
tp.use_aber = False
tp.use_ao = True
tp.ao_act = 14
cdip.use_cdi = False

# Plotting
sp.show_wframe = False  # Plot white light image frame
sp.show_spectra = False  # Plot spectral cube at single timestep
sp.spectra_cols = 3  # number of subplots per row in view_datacube
sp.show_tseries = False  # Plot full timeseries of white light frames
sp.tseries_cols = 5  # number of subplots per row in view_timeseries

# Saving
sp.save_obs = False  # save obs_sequence (timestep, wavelength, x, y)
sp.save_fields = True  # toggle to turn saving uniformly on/off
sp.save_list = ['atmosphere', 'enterance_pupil',  'ideal_wfs', 'woofer', 'detector']  # list of locations in optics train to save



if __name__ == '__main__':
    # testname = input("Please enter test name: ")
    testname = 'Subaru-test1'
    iop.update(testname)
    iop.makedir()
    mm.run_mmedis()
