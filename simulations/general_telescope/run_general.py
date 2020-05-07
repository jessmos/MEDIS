"""
run_general
RD
Jan 22 2020

This is the starting point to run the general system. This general system (formerly optics_propagate.py) is fully
customisable from this script from a series of toggles.

"""

import numpy as np

from medis.params import params
# from medis.params import iop, sp, ap, tp, cdip
from medis.utils import dprint
# import medis.optics as opx
from medis.plot_tools import view_spectra, view_timeseries, quick2D, plot_planes, body_spectra
import medis.medis_main as mm



#################################################################################################
#################################################################################################
#################################################################################################

params['tp'].prescription = 'general_telescope'

# Companion
params['ap'].companion = True
params['ap'].contrast = [1e-5]
params['ap'].companion_xy = [[15, -15]]  # units of this are in lambda/tp.entrance_d

params['sp'].numframes = 2
params['sp'].focused_sys = False
params['sp'].beam_ratio = 0.3  # parameter dealing with the sampling of the beam in the pupil/focal plane
params['sp'].grid_size = 512  # creates a nxn array of samples of the wavefront
params['sp'].maskd_size = 256  # will truncate grid_size to this range (avoids FFT artifacts) # set to grid_size if undesired
params['sp'].closed_loop = False

# Toggles for Aberrations and Control
params['tp'].entrance_d = 8
params['tp'].obscure = False
params['tp'].use_atmos = True
params['tp'].use_ao = True
params['tp'].ao_act = 60
params['tp'].rotate_atmos = False
params['tp'].rotate_sky = False
params['tp'].f_lens = 200.0 * params['tp'].entrance_d
params['tp'].occult_loc = [0,0]

# Saving
params['sp'].save_to_disk = False  # save obs_sequence (timestep, wavelength, x, y)
params['sp'].save_list = ['detector']  # list of locations in optics train to save
# params['sp'].skip_planes = ['coronagraph']  # ['wfs', 'deformable mirror']  # list of locations in optics train to save
params['sp'].quick_detect = True
params['sp'].debug = False
params['sp'].verbose = True

if __name__ == '__main__':
    # =======================================================================
    # Run it!!!!!!!!!!!!!!!!!
    # =======================================================================

    sim = mm.RunMedis(params=params, name='general_example', product='photons')
    observation = sim()

    fp_sampling = sim.camera.platescale
    stackcube = sim.camera.stackcube

    print(fp_sampling)
    for o in range(len(params['ap'].contrast)+1):
        print(stackcube.shape)
        datacube = stackcube[o]
        print(o, datacube.shape)
        body_spectra(datacube, logZ=True, title='Spectral Channels')
