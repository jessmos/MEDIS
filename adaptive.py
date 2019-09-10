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

from mm_params import tp, ap, cdip
from proper_mod import prop_dm
import CDI as cdi
from mm_utils import dprint


################################################################################
# Quick AO
################################################################################
def quick_ao(wfo, WFS_maps, tstep):
    """
    calculate the DM phase from the WFS map and apply it with proper.prop_dm

    The main idea is to apply the DM only to the region of the wavefront that contains the beam. The phase map from
    the wfs saved the whole wavefront, so that must be cropped. During the wavefront initialization in
    wavefront.initialize_proper, the beam ratio set in tp.beam_ratio is scaled per wavelength (to achieve constant
    sampling sto create white light images), so the cropped value must also be scaled by wavelength.

    Then, we interpolate the cropped beam onto a grid of (n_actuators,n_actuators), such that the DM can apply a
    actuator height to each represented actuator, not a over or sub-sampled form. There is a discrepancy between
    the sampling of the wavefront at this location (the size you cropped) vs the size of the DM. Presumably prop_dm
    handles this, so just plug in the n_actuator sized DM map with specified parameters, and assume that prop_dm
    handles the resampling correctly via the spacing or n_act_across_pupil flag.

    The scale factor becomes an distance term, so actuator heights are scaled by wavelength
     you can think of it as the phase term of dm_map is the phase delay in units of cycles, so you need
     to multiply by the wavelength to get the height of the DM you want to use to correct for the phase delay
     you need the 4pi because you are treating it as a reflection, so it travels that distance twice

    In the call to proper.prop_dm, we apply the flag tp.fit_dm, which switches between two 'modes' of proper's DM
    surface fitting. If FALSE, the DM is driven to the heights specified by dm_map, and the influence function will
    act on these heights to define the final surface shape applied to the DM, which may differ substantially from
    the initial heights specified by dm_map. If TRUE, proper will iterate applying the influence function to the
    input heights, and adjust the heights until the difference between the influenced-map and input map meets some
    proper-defined convergence criterea. Setting tp.fit_dm=TRUE will obviously slow down the code, but will (likely)
    more accurately represent a well-calibrated DM response function.

    much of this code copied over from example from Proper manual on pg 94

    :param wfo: wavefront object created by optics.Wavefronts() [n_wavelengths, n_objects] of tp.gridsize x tp.gridsize
    :param WFS_maps: returned from quick_wfs (as of 8/19, its an idealized image)
    :return:
    """
    beam_ratios = wfo.beam_ratios
    shape = wfo.wf_array.shape  # [n_wavelengths, n_astro_objects]

    nact = tp.ao_act                    # number of DM actuators along one axis
    nact_across_pupil = nact-2          # number of full DM actuators across pupil (oversizing DM extent)
    dm_xc = (nact / 2)-0.5              # The location of the optical axis (center of the wavefront) on the DM in
    dm_yc = (nact / 2)-0.5              #  actuator units. First actuator is centered on (0.0, 0.0). The 0.5 is a
                                        #  parameter introduced/tuned by Rupert to remove weird errors (address this).
                                        # KD verified this needs to be here or else suffer weird errors 9/19
                                        # TODO address/remove the 0.5 in DM x,y coordinates

    ############################
    # Creating DM Surface Map
    ############################
    for iw in range(shape[0]):
        for io in range(shape[1]):
            d_beam = 2 * proper.prop_get_beamradius(wfo.wf_array[iw,io])  # beam diameter
            act_spacing = d_beam / nact_across_pupil  # actuator spacing [m]
            ###################################
            # Cropping the Beam from WFS map
            ###################################
            # cropping here by beam_ratio rather than d_beam is valid since the beam size was initialized
            #  using the scaled beam_ratios when the wfo was created
            dm_map = WFS_maps[iw,
                     tp.grid_size//2 - np.int_(beam_ratios[iw]*tp.grid_size//2):
                     tp.grid_size//2 + np.int_(beam_ratios[iw]*tp.grid_size//2)+1,
                     tp.grid_size//2 - np.int_(beam_ratios[iw]*tp.grid_size//2):
                     tp.grid_size//2 + np.int_(beam_ratios[iw]*tp.grid_size//2)+1]
            samp = proper.prop_get_sampling(wfo.wf_array[iw, io])

            ########################################################
            # Interpolating the WFS map onto the actuator spacing
            # (tp.nact,tp.nact)
            ########################################################
            f = interpolate.interp2d(range(dm_map.shape[0]), range(dm_map.shape[0]), dm_map)
            dm_map = f(np.linspace(0,dm_map.shape[0],nact), np.linspace(0,dm_map.shape[0], nact))
            # dm_map = proper.prop_magnify(CPA_map, map_spacing / act_spacing, nact)

            #########################
            # Applying Piston Error
            #########################
            if tp.piston_error:
                mean_dm_map = np.mean(np.abs(dm_map))
                var = 1e-4  # 1e-11
                dm_map = dm_map + np.random.normal(0, var, (dm_map.shape[0], dm_map.shape[1]))

            ################################################
            # Converting phase delay to DM actuator height
            ################################################
            # Apply the inverse of the WFS image to the DM, so use -dm_map (dm_map is in phase units)
            surf_height = proper.prop_get_wavelength(wfo.wf_array[iw, io]) / (4 * np.pi)
            dm_map = -dm_map * surf_height

            #######
            # CDI
            ######
            if cdip.use_cdi == True:
                dprint(f"Using CDI")
                theta = np.pi  # change this to loop later
                if tstep == 0 and iw == 1:
                    probe = cdi.CDIprobe(theta, plot=True)
                else:
                    probe = cdi.CDIprobe(theta, plot=False)

                # Add Probe to DM map
                dm_map = dm_map + probe

            #########################
            # proper.prop_dm
            #########################
            proper.prop_dm(wfo.wf_array[iw,io], dm_map, dm_xc, dm_yc, act_spacing, FIT=tp.fit_dm)  #
            # proper.prop_dm(wfo, dm_map, dm_xc, dm_yc, N_ACT_ACROSS_PUPIL=nact, FIT=True)  #

    # kludge to help Rupert with weird phase discontinuities he was seeing
    # kludge is basically a low-pass filter?
    # TODO address the kludge. Is it still necessary
    # there is a similar type kludge in the proper.prop_dm code, lines ~249-252
    # for iw in range(shape[0]):
    #     phase_map = proper.prop_get_phase(wfo.wf_array[iw,0])
    #     amp_map = proper.prop_get_amplitude(wfo.wf_array[iw,0])
    #
    #     lowpass = ndimage.gaussian_filter(phase_map, 1, mode='nearest')
    #     smoothed = phase_map - lowpass
    #
    #     wfo.wf_array[iw,0].wfarr = proper.prop_shift_center(amp_map*np.cos(smoothed)+1j*amp_map*np.sin(smoothed))

    return


def quick_wfs(wf_vec):
    """
    saves the unwrapped phase [arctan2(imag/real)] of the wfo.wf_array at each wavelength

    so it is an idealized image (exact copy) of the wavefront phase per wavelength

    :param wf_vec: array containing wavefront array for each wavelength in the simulation shape=[n_wavelengths]
    :return: array containing only the unwrapped phase delay of the wavefront; shape=[n_wavelengths], units=radians
    """

    sigma = [2, 2]
    WFS_maps = np.zeros((len(wf_vec), tp.grid_size, tp.grid_size))

    for iw in range(len(wf_vec)):
        WFS_maps[iw] = scipy.ndimage.filters.gaussian_filter(unwrap_phase(proper.prop_get_phase(wf_vec[iw])), sigma,
                                                             mode='constant')
    return WFS_maps

################################################################################
# Full AO
################################################################################
# not implemented. Full AO implies a time delay, and maybe non-ideal WFS
