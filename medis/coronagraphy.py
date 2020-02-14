"""
coronagraphy.py
Kristina Davis
Nov 2019

This script adds functionality related to coronagraphy of the telescope system. I have decided to add a class
of occulter to the module to take care of different types of coronagraphs


"""
import numpy as np
import proper

from medis.params import tp, sp, ap
import medis.optics as opx
from medis.utils import dprint


class Occulter():
    """
    produces an occulter of various modes

    valid modes:
    'Gaussian'
    'Solid'
    '8th_Order'

    most of this class is a modification of the chronograph routine found on the proper manual pg 85
    """
    def __init__(self, _mode):
        if _mode is 'Gaussian' or 'Solid' or '8th_Order':
            self.mode = _mode
        else:
            raise ValueError('please choose a valid mode: Gaussian, Solid, 8th_Order')

    def set_size(self, wf, size_in_lambda_d=0, size_in_m=0):
        """
        sets size of occulter in m, depending on if input is in units of lambda/D or in m

        :param wf: 2D wavefront
        :param size_in_lambda_d: desired occulter size in units of lambda/D
        :param size_in_m: desired occulter size in m
        :return: occulter size in m
        """
        lamda = proper.prop_get_wavelength(wf)
        dx_m = proper.prop_get_sampling(wf)
        dx_rad = proper.prop_get_sampling_radians(wf)

        if size_in_lambda_d is not 0:
            occrad_rad = size_in_lambda_d * lamda / tp.entrance_d  # occulter radius in radians
            self.size = occrad_rad * dx_m / dx_rad  # occulter radius in meters
        elif size_in_m is not 0:
            self.size = size_in_m
        else:
            raise ValueError('must set occulter size in either m or lambda/D units')

        # rescaling for focused system
        # if sp.focused_sys:
        #     self.size = wf.lamda / tp.entrance_d * ap.wvl_range[0] / wf.lamda

    def apply_occulter(self, wf):
        """
        applies the occulter by type spcified when class object was initiated

        :param wf: 2D wavefront
        :return:
        """
        # Code here pulled directly from Proper Manual pg 86
        if self.mode == "Gaussian":
            r = proper.prop_radius(wf)
            h = np.sqrt(-0.5 * self.size**2 / np.log(1 - np.sqrt(0.5)))
            gauss_spot = 1 - np.exp(-0.5 * (r/h)**2)
            # gauss_spot = shift(gauss_spot, shift=occult_loc, mode='wrap')  # ???
            proper.prop_multiply(wf, gauss_spot)
        elif self.mode == "Solid":
            proper.prop_circular_obscuration(wf, self.size)
        elif self.mode == "8th_Order":
            proper.prop_8th_order_mask(wf, self.size, CIRCULAR=True)

    def apply_lyot(self, wf):
        """
        applies the appropriately sized Lyot stop depending on the coronagraph type

        :param wf: 2D wavefront
        :return:
        """
        if not hasattr(tp, 'lyot_size'):
            raise ValueError("must set tp.lyot_size in units fraction of the beam radius at the current surface")
        if self.mode is 'Gaussian':
            proper.prop_circular_aperture(wf, tp.lyot_size, NORM=True)
        elif self.mode is 'Solid':
            proper.prop_circular_aperture(wf, tp.lyot_size, NORM=True)
        elif self.mode is '8th_Order':
            proper.prop_circular_aperture(wf, tp.lyot_size, NORM=True)


def coronagraph(wf, occulter_mode=None):
    """
    propagates a wavefront through a coronagraph system

    We implicitly assume here that the system begins at the location of the occulting mask. The optical system leading
    up to this point must be defined in the telescope perscription. This coronagraph routine also ends at the image
    plane after the Lyot stop, if used. If no Lyot stop is used, the reimaging optics just pass the pupil image
    unaffected.

    this function is modified from the Proper Manual: Stellar Coronagraph example found on pg 85

    :param wf: a single wavefront of complex data shape=(sp.grid_size, sp.grid_size)
    :param occulter_type string defining the occulter type. Accepted types are "Gaussian", "Solid", and "8th_Order"
    :return: noting is returned but wfo has been modified
    """
    if occulter_mode is None:
        pass
    else:
        # Initialize Occulter based on Mode
        occ = Occulter(occulter_mode)
        if not hasattr(tp, 'cg_size'):
            raise ValueError("must set tp.cg_size and tp.cg_size_units to use coronagraph() in coronography.py")
        elif tp.cg_size_units == "m":
            occ.set_size(wf, size_in_m=tp.cg_size)
        else:
            occ.set_size(wf, size_in_lambda_d=tp.cg_size)

        # Apply Occulter
        occ.apply_occulter(wf)  # size saved to self in class, so don't need to pass it

        proper.prop_propagate(wf, tp.fl_cg_lens)  # propagate to coronagraph pupil reimaging lens from occulter
        opx.prop_pass_lens(wf, tp.fl_cg_lens, tp.fl_cg_lens)  # go to the middle of the 2-lens system

        # Apply Lyot
        occ.apply_lyot(wf)

        proper.prop_propagate(wf, tp.fl_cg_lens)  # propagate to reimaging lens from lyot stop
        opx.prop_pass_lens(wf, tp.fl_cg_lens, tp.fl_cg_lens)  # go to the next image plane


def apodization(wf):
    raise NotImplementedError


def vortex(wf):
    raise NotImplementedError



