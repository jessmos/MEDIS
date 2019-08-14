"""
optics.py

contains functionality related to the optical components needed to build up a proper prescription. This is a generic
module containing functions that don't have a home elsewhere. It contains the class structure to make the wavefront
object used in most prescriptions.
"""
import numpy as np
import proper

from mm_params import ap, tp


############################
# Create Wavefront Array
############################
class Wavefronts():
    """
    An object containing all of the complex E fields (for each sample wavelength and astronomical object) for this timestep

    :params

    :returns
    self.wf_array: a matrix of proper wavefront objects after all optic modifications have been applied
    self.save_E_fields: a matrix of E fields (proper.WaveFront.wfarr) at specified locations in the chain
    """
    def __init__(self):

        # Using Proper to propagate wavefront from primary through optical system, loop over wavelength
        self.wsamples = np.linspace(ap.wvl_range[0], ap.wvl_range[1], ap.nwsamp)  # units set in params (should be m)

        # wf_array is an array of arrays; the wf_array is (number_wavelengths x number_astro_objects)
        # each 2D field in the wf_array is the 2D array of complex E-field values at that wavelength, per object
        # the E-field size is given by (tp.grid_size x tp.grid_size)
        if ap.companion:
            self.wf_array = np.empty((len(self.wsamples), 1 + len(ap.contrast)), dtype=object)
        else:
            self.wf_array = np.empty((len(self.wsamples), 1), dtype=object)

        # Init Beam Ratios
        self.beam_ratios = np.zeros_like(self.wsamples)

        # Init Locations of saved E-field
        self.save_E_fields = np.empty((0, np.shape(self.wf_array)[0],
                                       np.shape(self.wf_array)[1],
                                       tp.grid_size,
                                       tp.grid_size), dtype=np.complex64)

    def initialize_proper(self):
        # Initialize the Wavefront in Proper
        for iw, w in enumerate(self.wsamples):
            # Scale beam ratio by wavelength to achieve consistent sampling across wavelengths
            # see Proper manual pg 37
            self.beam_ratios[iw] = tp.beam_ratio * ap.wvl_range[0] / w

            # Initialize the wavefront at entrance pupil
            wfp = proper.prop_begin(tp.enterance_d, w, tp.grid_size, self.beam_ratios[iw])

            wfs = [wfp]
            names = ['primary']
            # Initiate wavefronts for companion(s)
            if ap.companion:
                for ix in range(len(ap.contrast)):
                    wfc = proper.prop_begin(tp.enterance_d, w, tp.grid_size, self.beam_ratios[iw])
                    wfs.append(wfc)
                    names.append('companion_%i' % ix)

            for io, (iwf, wf) in enumerate(zip(names, wfs)):
                self.wf_array[iw, io] = wf

    def apply_func(self, func, *args, **kwargs):
        """
        For each wavelength and astronomical object apply a function to the wavefront.

        The wavefront object has dimentions of
        If func is in save_locs then append the E field to save_E_fields

        :param func: function to be applied e.g. ap.add_aber()
        :param args:
        :param kwargs:
        :return: self.save_E_fields
        """
        shape = self.wf_array.shape
        optic_E_fields = np.zeros((1, np.shape(self.wf_array)[0],
                                   np.shape(self.wf_array)[1],
                                   tp.grid_size,
                                   tp.grid_size), dtype=np.complex64)
        for iw in range(shape[0]):
            for iwf in range(shape[1]):
                func(self.wf_array[iw, iwf], *args, **kwargs)
                # Saving E-field
        #         if self.save_locs is not None and func.__name__ in self.save_locs:
        #             wf = proper.prop_shift_center(self.wf_array[iw, iwf].wfarr)
        #             optic_E_fields[0, iw, iwf] = copy.copy(wf)
        #
        # if self.save_locs is not None and func.__name__ in self.save_locs:
        #     self.save_E_fields = np.vstack((self.save_E_fields, optic_E_fields))


################################################################################################################
# Optics in Proper
################################################################################################################

def add_obscurations(wf, M2_frac=0, d_primary=0, d_secondary=0, legs_frac=0.05):
    """
    adds central obscuration (secondary shadow) and/or spider legs as spatial mask to the wavefront

    :param wf: proper wavefront
    :param M2_frac: ratio of tp.diam the M2 occupies
    :param d_primary: diameter of the primary mirror
    :param d_secondary: diameter of the secondary mirror
    :param legs_frac: fractional size of spider legs relative to d_primary
    :return: acts upon wfo, applies a spatial mask of s=circular secondary obscuration and possibly spider legs
    """
    # dprint('Including Obscurations')
    if M2_frac > 0 and d_primary > 0:
        proper.prop_circular_obscuration(wf, M2_frac * d_primary)
    elif d_secondary > 0:
        proper.prop_circular_aperture(wf, d_secondary)
    else:
        raise ValueError('must either specify M2_frac and d_primary or d_secondary')
    if legs_frac > 0:
        proper.prop_rectangular_obscuration(wf, legs_frac*d_primary, d_primary*1.3, ROTATION=20)
        proper.prop_rectangular_obscuration(wf, d_primary*1.3, legs_frac * d_primary, ROTATION=20)


def prop_mid_optics(wfo, fl_lens, dist):
    """
    pass the wavefront through a lens then propagate to the next surface

    :param wfo: wavefront object, shape=(n_wavelengths, n_astro_objects, grid_sz, grid_sz)
    :param fl_lens: focal length in m
    :param dist: distance in m
    """
    proper.prop_lens(wfo, fl_lens)
    proper.prop_propagate(wfo, dist)


def abs_zeros(wf_array):
    """
    zeros everything outside the pupil

    This function attempts to figure out where the edges of the pupil are by determining if either the real
     or imaginary part of the complex-valued E-field is zero at a gridpoint. If so, it sets the full cpmplex
     value to zero, so 0+0j
    """
    shape = wf_array.shape
    for iw in range(shape[0]):
        for io in range(shape[1]):
            bad_locs = np.logical_or(np.real(wf_array[iw,io].wfarr) == -0,
                                     np.imag(wf_array[iw,io].wfarr) == -0)
            wf_array[iw,io].wfarr[bad_locs] = 0 + 0j

    return wf_array





