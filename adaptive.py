"""
adaptive.py

functions relating to simulating an AO system with Proper
mostly code copied from the original MEDIS from Rupert

Code is broken into quick_AO and full_AO sections
"""

import numpy as np
from scipy import interpolate
import scipy.ndimage
from skimage.restoration import unwrap_phase
import pickle as pickle
from scipy import ndimage
import proper

from proper_mod import prop_dm
from mm_params import tp, ap, iop
from mm_utils import dprint


################################################################################
# Quick AO
################################################################################
def quick_ao(wfo, WFS_maps):
    # TODO address the kludge. Is it still necessary

    wf_array = wfo.wf_array
    beam_ratios = wfo.beam_ratios

    nact = tp.ao_act  # 49                       # number of DM actuators along one axis
    nact_across_pupil = nact-2  # 47          # number of DM actuators across pupil
    dm_xc = (nact / 2)-0.5
    dm_yc = (nact / 2)-0.5

    shape = wf_array.shape

    for iw in range(shape[0]):
        for io in range(shape[1]):
            d_beam = 2 * proper.prop_get_beamradius(wf_array[iw,io])  # beam diameter
            act_spacing = d_beam / nact_across_pupil  # actuator spacing
            # Compensating for chromatic beam size
            dm_map = WFS_maps[iw,
                     tp.grid_size//2 - np.int_(beam_ratios[iw]*tp.grid_size//2):
                     tp.grid_size//2 + np.int_(beam_ratios[iw]*tp.grid_size//2)+1,
                     tp.grid_size//2 - np.int_(beam_ratios[iw]*tp.grid_size//2):
                     tp.grid_size//2 + np.int_(beam_ratios[iw]*tp.grid_size//2)+1]
            f = interpolate.interp2d(list(range(dm_map.shape[0])), list(range(dm_map.shape[0])), dm_map)
            dm_map = f(np.linspace(0,dm_map.shape[0],nact), np.linspace(0,dm_map.shape[0], nact))
            # dm_map = proper.prop_magnify(CPA_map, map_spacing / act_spacing, nact)

            if tp.piston_error:
                mean_dm_map = np.mean(np.abs(dm_map))
                var = 1e-4  # 1e-11
                dm_map = dm_map + np.random.normal(0, var, (dm_map.shape[0], dm_map.shape[1]))


            dm_map = -dm_map * proper.prop_get_wavelength(wf_array[iw,io]) / (4 * np.pi)  # <--- here
            # dmap = proper.prop_dm(wfo, dm_map, dm_xc, dm_yc, N_ACT_ACROSS_PUPIL=nact, FIT=True)  # <-- here
            dmap = proper.prop_dm(wf_array[iw,io], dm_map, dm_xc, dm_yc, act_spacing, FIT=True)  # <-- here

    # kludge to help with spiders
    for iw in range(shape[0]):
        phase_map = proper.prop_get_phase(wf_array[iw,0])
        amp_map = proper.prop_get_amplitude(wf_array[iw,0])

        lowpass = ndimage.gaussian_filter(phase_map, 1, mode='nearest')
        smoothed = phase_map - lowpass

        wf_array[iw,0].wfarr = proper.prop_shift_center(amp_map*np.cos(smoothed)+1j*amp_map*np.sin(smoothed))

    return


def flat_outside(wf_array):
    for iw in range(wf_array.shape[0]):
        for io in range(wf_array.shape[1]):
            proper.prop_circular_aperture(wf_array[iw,io], 1, NORM=True)


def quick_wfs(wf_vec):

    sigma = [2, 2]
    WFS_maps = np.zeros((len(wf_vec), tp.grid_size, tp.grid_size))

    for iw in range(len(wf_vec)):
        WFS_maps[iw] = scipy.ndimage.filters.gaussian_filter(unwrap_phase(proper.prop_get_phase(wf_vec[iw])), sigma,
                                                             mode='constant')
    return WFS_maps

################################################################################
# Full AO
################################################################################
