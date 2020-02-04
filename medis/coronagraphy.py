"""
coronagraphy.py
Kristina Davis
Nov 2019

This script adds functionality related to coronagraphy of the telescope system. I have decided to add a class
of occulter to the module to take care of different types of coronagraphs


"""
import numpy as np
import proper

from medis.params import tp
from medis.utils import dprint


class Occulter():
    """
    produces an occulter of various modes

    valid modes:
    'Gaussian'
    'Solid'
    '8th_Order'
    """
    def __init__(self, _mode):
        if type is not 'Gaussian' or 'Solid' or '8th_Order':
            raise ValueError('please choose a valid type: Gaussian, Solid, 8th_Order')
        self.mode = _mode

    def set_size(self, wfo, size_in_lambda_d=0, size_in_m=0):
        """
        sets size of occulter in m, depending on if input is in units of lambda/D or in m

        :param wfo: wavefront object
        :param size_in_lambda_d: desired occulter size in units of lambda/D
        :param size_in_m: desired occulter size in m
        :return: occulter size in m
        """
        lamda = proper.prop_get_wavelength(wfo)
        dx_m = proper.prop_get_sampling(wfo)
        dx_rad = proper.prop_get_sampling_radians(wfo)

        if size_in_lambda_d is not 0:
            occrad_rad = size_in_lambda_d * lamda / tp.entrance_d  # occulter radius in radians
            self.size = occrad_rad * dx_m / dx_rad  # occulter radius in meters
        elif size_in_m is not 0:
            self.size = size_in_m
        else:
            raise ValueError('must set size in either m or lambda/D units')

    def apply_occulter(self, wfo):
        # Code here pulled directly from Proper Manual pg 86
        if self.mode == "Gaussian":
            r = proper.prop_radius(wfo)
            h = np.sqrt(-0.5 * self.size**2 / np.log(1 - np.sqrt(0.5)))
            gauss_spot = 1 - np.exp(-0.5 * (r/h)**2)
            # gauss_spot = shift(gauss_spot, shift=occult_loc, mode='wrap')  # ???
            proper.prop_multiply(wfo, gauss_spot)
        elif self.mode == "Solid":
            proper.prop_circular_obscuration(wfo, self.size)
        elif self.mode == "8th_Order":
            proper.prop_8th_order_mask(wfo, self.size, CIRCULAR=True)

    def apply_lyot(self):
        if self.mode is "Gaussian":
            x=1
        elif self.mode is "Solid":
            x=2
        elif self.mode is "8th_Order":
            x=3


def coronagraph(wfo):
    """
    builds a coronagraph system

    :param wfo:
    :return:
    """
    pass

def apodization(wfo):
    raise NotImplementedError

def vortex(wfo):
    raise NotImplementedError



