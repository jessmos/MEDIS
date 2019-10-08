"""
adaptive.py

functions relating to simulating an AO system with Proper
mostly code copied from the original MEDIS from Rupert

"""

import numpy as np
import scipy.ndimage
from skimage.restoration import unwrap_phase
from scipy import interpolate, ndimage
import proper

from mm_params import sp, tp, ap, cdip
from proper_mod import prop_dm
import CDI as cdi
from optics import check_sampling
from mm_utils import dprint


################################################################################
# Ideal AO
################################################################################
def deformable_mirror(wfo, WFS_map, theta):
    """
    combine different DM actuator commands into single map to send to prop_dm

    prop_dm needs an input map of n_actuators x n_actuators in units of actuator command height. quick_ao will handle
    the conversion to actuator command height, and the CDI probe must be scaled in cdip.probe_amp in mm_params in
    units of m. Each subroutine is also responsible for creating a map of n_actuators x n_actuators spacing. prop_dm
    handles the resampling of this map onto the wavefront, including the influence function. Its some wizardry that
    happens in c, and presumably it is taken care of so you don't have to worry about it.

    In the call to proper.prop_dm, we apply the flag tp.fit_dm, which switches between two 'modes' of proper's DM
    surface fitting. If FALSE, the DM is driven to the heights specified by dm_map, and the influence function will
    act on these heights to define the final surface shape applied to the DM, which may differ substantially from
    the initial heights specified by dm_map. If TRUE, proper will iterate applying the influence function to the
    input heights, and adjust the heights until the difference between the influenced-map and input map meets some
    proper-defined convergence criterea. Setting tp.fit_dm=TRUE will obviously slow down the code, but will (likely)
    more accurately represent a well-calibrated DM response function.

    much of this code copied over from example from Proper manual on pg 94

    :param wfo: wavefront object
    :param WFS_map: wavefront sensor map, should be in units of phase delay
    :param theta: phase of CDI probe (either radians or NAN if no probe to be applied at this timestep)
    :return:
    """

    # DM Coordinates
    nact = tp.ao_act  # number of DM actuators along one axis
    nact_across_pupil = nact - 2  # number of full DM actuators across pupil (oversizing DM extent)
    dm_xc = (nact / 2) - 0.5  # The location of the optical axis (center of the wavefront) on the DM in
    dm_yc = (nact / 2) - 0.5  # actuator units. First actuator is centered on (0.0, 0.0). The 0.5 is a
    #  parameter introduced/tuned by Rupert to remove weird errors (address this).
    # KD verified this needs to be here or else suffer weird errors 9/19
    # TODO address/remove the 0.5 in DM x,y coordinates

    ############################
    # Creating DM Surface Map
    ############################
    for iw in range(wfo.wf_array.shape[0]):
        for io in range(wfo.wf_array.shape[1]):
            d_beam = 2 * proper.prop_get_beamradius(wfo.wf_array[iw, io])  # beam diameter
            act_spacing = d_beam / nact_across_pupil  # actuator spacing [m]
            # map_spacing = proper.prop_get_sampling(wfo.wf_array[iw,0])

            #######
            # AO
            #######
            dm_map = quick_ao(wfo, WFS_map)

            #######
            # CDI
            ######
            if not np.isnan(theta):
                # dprint(f"Applying CDI probe, lambda = {wfo.wsamples[iw]*1e9:.2f} nm")
                probe = cdi.CDIprobe(theta, iw)
                # Add Probe to DM map
                dm_map = dm_map + probe

            #########################
            # Applying Piston Error
            #########################
            if tp.piston_error:
                mean_dm_map = np.mean(np.abs(dm_map))
                var = 1e-4  # 1e-11
                dm_map = dm_map + np.random.normal(0, var, (dm_map.shape[0], dm_map.shape[1]))

            #########################
            # proper.prop_dm
            #########################
            prop_dm(wfo.wf_array[iw, io], dm_map, dm_xc, dm_yc, act_spacing, FIT=tp.fit_dm)  #
            # proper.prop_dm(wfo, dm_map, dm_xc, dm_yc, N_ACT_ACROSS_PUPIL=nact, FIT=True)  #
            # check_sampling(0, wfo, "post-DM in quickAO")  # check sampling in optics.py

    return


def quick_ao(wfo, WFS_map):
    """
    calculate the offset map to send to the DM from the WFS map

    The main idea is to apply the DM only to the region of the wavefront that contains the beam. The phase map from
    the wfs saved the whole wavefront, so that must be cropped. During the wavefront initialization in
    wavefront.initialize_proper, the beam ratio set in sp.beam_ratio is scaled per wavelength (to achieve constant
    sampling sto create white light images), so the cropped value must also be scaled by wavelength. Note, beam ratio
    is scaled differently depending on if sp.OOPP is True or not. See mm_params-->sp.OOPP and Proper manual pg 36
    for more info.

    Then, we interpolate the cropped beam onto a grid of (n_actuators,n_actuators), such that the DM can apply a
    actuator height to each represented actuator, not a over or sub-sampled form. If the number of actuators is low
    compared to the beam size, you must anti-alias the WFS map via a lowpass filter before interpolating.
    There is a discrepancy between the sampling of the wavefront at this location (the size you cropped) vs the size
    of the DM. proper.prop_dm handles this, so just plug in the n_actuator sized DM map with specified parameters,
    and assume that prop_dm handles the resampling correctly via the spacing or n_act_across_pupil flag. FYI the
    resampling is done via a c library you installed/compiled when installing proper.

    The WFS map is a map of real values in units of phase delay in radians. However, the AO map that gets passed to
    proper.prop_dm wants input in nm height of each actuator. Therefore, you need to convert the phase delay to
    a DM height. For the ideal AO, you would do this individually for each wavelength. However, for a 'real' AO system
    you do this for the median wavelength. You also need to account for a factor of 2, since the DM is modeled as
    a mirror so it travels the length of the phase delay twice.

    much of this code copied over from example from Proper manual on pg 94

    :param wfo: wavefront object created by optics.Wavefronts() [n_wavelengths, n_objects] of tp.gridsize x tp.gridsize
    :param WFS_map: returned from quick_wfs (as of Aug 2019, its an idealized image)
    :return: ao_map: map of DM actuator command heights in units of m
    """
    beam_ratios = wfo.beam_ratios
    shape = wfo.wf_array.shape  # [n_wavelengths, n_astro_objects]

    nact = tp.ao_act                    # number of DM actuators along one axis
    nact_across_pupil = nact-2          # number of full DM actuators across pupil (oversizing DM extent)
    
    ############################
    # Creating AO Surface Map
    ############################
    for iw in range(shape[0]):
        for io in range(shape[1]):
            d_beam = 2 * proper.prop_get_beamradius(wfo.wf_array[iw,io])  # beam diameter
            act_spacing = d_beam / nact_across_pupil  # actuator spacing [m]
            # map_spacing = proper.prop_get_sampling(wfo.wf_array[iw,0])

            ###################################
            # Cropping the Beam from WFS map
            ###################################
            # cropping here by beam_ratio rather than d_beam is valid since the beam size was initialized
            #  using the scaled beam_ratios when the wfo was created
            ao_map = WFS_map[iw,
                     sp.grid_size//2 - np.int_(beam_ratios[iw]*sp.grid_size//2):
                     sp.grid_size//2 + np.int_(beam_ratios[iw]*sp.grid_size//2)+1,
                     sp.grid_size//2 - np.int_(beam_ratios[iw]*sp.grid_size//2):
                     sp.grid_size//2 + np.int_(beam_ratios[iw]*sp.grid_size//2)+1]

            ########################################################
            # Interpolating the WFS map onto the actuator spacing
            # (tp.nact,tp.nact)
            ########################################################
            # Lowpass Filter- prevents aliasing
            lowpass = ndimage.gaussian_filter(ao_map, 1, mode='nearest')
            ao_map = ao_map - lowpass

            f = interpolate.interp2d(range(ao_map.shape[0]), range(ao_map.shape[0]), ao_map, kind='cubic')
            ao_map = f(np.linspace(0,ao_map.shape[0],nact), np.linspace(0,ao_map.shape[0], nact))
            # dm_map = proper.prop_magnify(dm_map, map_spacing / act_spacing, nact)

            ################################################
            # Converting phase delay to DM actuator height
            ################################################
            # Apply the inverse of the WFS image to the DM, so use -dm_map (dm_map is in phase units, divide by k=2pi/lambda)
            surf_height = proper.prop_get_wavelength(wfo.wf_array[iw, io]) / (4 * np.pi)  # [m/rad]
            ao_map = -ao_map * surf_height  # Converts DM map to units of [m] of actuator heights
            
            return ao_map
            

def ideal_wfs(wf_vec):
    """
    saves the unwrapped phase [arctan2(imag/real)] of the wfo.wf_array at each wavelength

    so it is an idealized image (exact copy) of the wavefront phase per wavelength

    :param wf_vec: array containing wavefront array for each wavelength in the simulation shape=[n_wavelengths]
    :return: array containing only the unwrapped phase delay of the wavefront; shape=[n_wavelengths], units=radians
    """

    sigma = [2, 2]
    WFS_map = np.zeros((len(wf_vec), sp.grid_size, sp.grid_size))

    for iw in range(len(wf_vec)):
        WFS_map[iw] = scipy.ndimage.filters.gaussian_filter(unwrap_phase(proper.prop_get_phase(wf_vec[iw])), sigma,
                                                             mode='constant')
    return WFS_map

################################################################################
# Full AO
################################################################################
# not implemented. Full AO implies a time delay, and maybe non-ideal WFS
