"""
optics.py

contains functionality related to the optical components needed to build up a proper prescription. This is a generic
module containing functions that don't have a home elsewhere. It contains the class structure to make the wavefront
object used in most prescriptions.
"""
import numpy as np
import proper
import copy
from scipy.interpolate import interp1d


from mm_params import ap, tp, sp
from mm_utils import dprint


############################
# Create Wavefront Array
############################
class Wavefronts():
    """
    An object containing all of the complex E fields for each sampled wavelength and astronomical object at this timestep

    :params

    :returns
    self.wf_array: a matrix of proper wavefront objects after all optic modifications have been applied
    self.save_E_fields: a matrix of E fields (proper.WaveFront.wfarr) at specified locations in the chain
    """
    def __init__(self):

        # Using Proper to propagate wavefront from primary through optical system, loop over wavelength
        self.wsamples = np.linspace(ap.wvl_range[0], ap.wvl_range[1], ap.n_wvl_init)  # units set in params (should be m)

        # wf_array is an array of arrays; the wf_array is (number_wavelengths x number_astro_objects)
        # each 2D field in the wf_array is the 2D array of complex E-field values at that wavelength, per object
        # the E-field size is given by (sp.grid_size x sp.grid_size)
        if ap.companion:
            self.wf_array = np.empty((ap.n_wvl_init, 1 + len(ap.contrast)), dtype=object)
        else:
            self.wf_array = np.empty((ap.n_wvl_init, 1), dtype=object)

        # Init Beam Ratios
        self.beam_ratios = np.zeros_like(self.wsamples)

        # Init Locations of saved E-field
        if sp.save_fields:
            self.saved_planes = []  # string of locations where fields have been saved (should match sp.save_list after run is completed)
            self.Efield_planes = np.empty((0, np.shape(self.wf_array)[0],  # array of saved complex field data at
                                           np.shape(self.wf_array)[1],    # specified locations of the optical train
                                           sp.grid_size,
                                           sp.grid_size), dtype=np.complex64)
        else:
            self.Efield_planes = None  # save empty attribute so it can be returned by the proper perscription even
                                         # if it is empty

    def initialize_proper(self):
        # Initialize the Wavefront in Proper
        for iw, w in enumerate(self.wsamples):
            # Scale beam ratio by wavelength for polychromatic imaging
            # see Proper manual pg 37
            # Proper is devised such that you get a Nyquist sampled image in the focal plane. If you optical system
            #  goes directly from pupil plane to focal plane, then you need to scale the beam ratio such that sampling
            #  in the focal plane is constant. You can check this with check_sampling, which returns the value from
            #  prop_get_sampling. If the optical system does not go directly from pupil-to-object plane at each optical
            #  plane, the beam ratio does not need to be scaled by wavelength, because of some optics wizardry that
            #  I don't fully understand. KD 2019
            if sp.focused_sys:
                self.beam_ratios[iw] = sp.beam_ratio
            else:
                self.beam_ratios[iw] = sp.beam_ratio * ap.wvl_range[0] / w
                # dprint(f"iw={iw}, w={w}, beam ratio is {self.beam_ratios[iw]}")
            # Initialize the wavefront at entrance pupil
            wfp = proper.prop_begin(tp.enterance_d, w, sp.grid_size, self.beam_ratios[iw])

            wfs = [wfp]
            names = ['star']

            # Initiate wavefronts for companion(s)
            if ap.companion:
                for ix in range(len(ap.contrast)):
                    wfc = proper.prop_begin(tp.enterance_d, w, sp.grid_size, self.beam_ratios[iw])
                    wfs.append(wfc)
                    names.append('companion_%i' % ix)

            for io, (iwf, wf) in enumerate(zip(names, wfs)):
                self.wf_array[iw, io] = wf

    def loop_over_function(self, func, *args, **kwargs):
        """
        For each wavelength and astronomical object apply a function to the wavefront.

        The wavefront object has dimensions of shape=(n_wavelengths, n_astro_objects, grid_sz, grid_sz)

        To save, you must pass in the keyword argument plane_name when you call this function from the perscription.
        This function does not have a keyword argument for plane_name specifically, since you need to distinguish
        it from the **kwargs you want to pass to the function that you are looping over.
        If you are saving the plane at this location, keep in mind it is saved AFTER the function is applied. This
        is desirable for most functions but be careful when using it for prop_lens, etc

        :param func: function to be applied e.g. ap.add_aber()
        :param plane_name: name the plane where this is called if you want to save the complex field data via save_plane
        :param args: args to be passed to the function
        :param kwargs: kwargs to be passed to the function
        :return: everything is just applied to the wfo, so nothing is returned in the traditional sense
        """
        if 'plane_name' in kwargs:
            plane_name = kwargs.pop('plane_name')  # remove plane_name from **kwargs
        else:
            plane_name = None
        shape = self.wf_array.shape
        for iw in range(shape[0]):
            for io in range(shape[1]):
                func(self.wf_array[iw, io], *args, **kwargs)

        # Saving complex field data
        if plane_name is not None:
            self.save_plane(location=plane_name)

    def save_plane(self, location=None):
        """
        Saves the complex field at a specified location in the optical system.

        Note that the complex planes saved are not summed by object, interpolated over wavelength, nor masked down
        to the sp.maskd_size.

        :param location: name of plane where field is being saved
        :return: self.save_E_fields
        """
        if sp.save_fields is True and location is not None and location in sp.save_list:
            shape = self.wf_array.shape
            E_field = np.zeros((1, np.shape(self.wf_array)[0],
                                       np.shape(self.wf_array)[1],
                                       sp.grid_size,
                                       sp.grid_size), dtype=np.complex64)
            for iw in range(shape[0]):
                for io in range(shape[1]):
                    wf = proper.prop_shift_center(self.wf_array[iw, io].wfarr)
                    E_field[0, iw, io] = copy.copy(wf)

            dprint(f"saving plane at {location}")
            self.Efield_planes = np.vstack((self.Efield_planes, E_field))
            self.saved_planes.append(location)

    def focal_plane(self):
        """
        ends the proper perscription and prepares the datacube to be sent to run_mmedis

        Converting Array of Arrays (wfo) into 3D array
        sums over objects for composite image
        prop_end moves center of the wavefront from lower left corner (Fourier space) back to the center
        also takes square modulus of complex values, so gives units as intensity not field

        :param tstep: timestep of the iteration, mostly used for check_sampling
        :return:
        """
        datacube = []

        sampling = np.zeros(ap.n_wvl_init)
        shape = self.wf_array.shape

        # Saving Complex Data via save_plane
        if sp.save_fields and 'detector' in sp.save_list:  # save before prop_end so unaffected by EXTRACT flag and
            self.save_plane(location='detector')           # shifting, etc already done in save_plane function

        # Proper prop_end
        for iw in range(shape[0]):
            for io in range(shape[1]):
                # EXTRACT flag removes only middle portion of the array. Used to remove FFT wrap-around effects
                if sp.maskd_size != sp.grid_size:
                    wframes = np.zeros((sp.maskd_size, sp.maskd_size))
                    (wframe, w_sampling) = proper.prop_end(self.wf_array[iw, io], EXTRACT=np.int(sp.maskd_size))
                else:
                    wframes = np.zeros((sp.grid_size, sp.grid_size))
                    (wframe, w_sampling) = proper.prop_end(
                        self.wf_array[iw, io])  # Sampling returned by proper is in [m]
                wframes += wframe  # adds 2D wavefront from all astro_objects together into single wavefront, per wavelength
            sampling[iw] = w_sampling
            datacube.append(wframes)  # puts each wavlength's wavefront into an array
            # (number_wavelengths x sp.grid_size x sp.grid_size)

        datacube = np.array(datacube)
        cpx_planes = np.array(self.Efield_planes)
        # Conex Mirror-- cirshift array for off-axis observing
        if tp.pix_shift is not [0, 0]:
            datacube = np.roll(np.roll(datacube, tp.pix_shift[0], 1), tp.pix_shift[1], 2)

        # Interpolating spectral cube from ap.n_wvl_init discreet wavelengths to ap.n_wvl_final
        if ap.interp_wvl and 1 < ap.n_wvl_init < ap.n_wvl_final:
            wave_samps = np.linspace(0, 1, ap.n_wvl_init)
            f_out = interp1d(wave_samps, datacube, axis=0)
            new_heights = np.linspace(0, 1, ap.n_wvl_final)
            datacube = f_out(new_heights)
            sampling = np.linspace(sampling[0], sampling[-1], ap.n_wvl_final)

        return datacube, cpx_planes, sampling


################################################################################################################
# Optics in Proper
################################################################################################################

def add_obscurations(wf, M2_frac=0, d_primary=0, d_secondary=0, legs_frac=0.05, plane_name=None):
    """
    adds central obscuration (secondary shadow) and/or spider legs as spatial mask to the wavefront

    :param wf: proper wavefront
    :param M2_frac: ratio of tp.diam the M2 occupies
    :param d_primary: diameter of the primary mirror
    :param d_secondary: diameter of the secondary mirror
    :param legs_frac: fractional size of spider legs relative to d_primary
    :return: acts upon wfo, applies a spatial mask of s=circular secondary obscuration and possibly spider legs
    """
    if tp.obscure is False:
        pass
    else:
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


def offset_companion(wfo):
    """
    offsets the companion wavefront using the 2nd and 3rd order Zernike Polynomials (X,Y tilt)
    companion(s) contrast and location(s) set in mm_params

    :param wfo: wavefront object, shape=(n_wavelengths, n_astro_objects, grid_sz, grid_sz)
    :return:
    """
    cont_scaling = np.linspace(1. / ap.C_spec, 1, ap.n_wvl_init)

    for iw in range(wfo.wf_array.shape[0]):
        for io in range(wfo.wf_array.shape[1]):
            if io > 0:
                # Shifting the Array
                xloc = ap.companion_locations[io][0]
                yloc = ap.companion_locations[io][1]
                proper.prop_zernikes(wfo.wf_array[iw, io], [2, 3], np.array([xloc, yloc]))

                # Wavelength Scaling the Companion
                wfo.wf_array[iw, io].wfarr = wfo.wf_array[iw, io].wfarr * np.sqrt(ap.contrast[io] * cont_scaling[iw])


def check_sampling(tstep, wfo, location, units=None):
    """
    checks the sampling of the wavefront at the given location and prints to console

    :param tstep: timestep, will only check for first timestep, so when tstep==0
    :param wfo: wavefront object
    :param location: string that identifies where call is being made
    :param units: desired units of returned print statement; options are 'mm,'um','nm','arcsec','rad'
    :return:
    """
    if tstep == 0:
        print(f"{location}")
        for w in range(wfo.wf_array.shape[0]):
            check_sampling = proper.prop_get_sampling(wfo.wf_array[w,0])
            if units == 'mm':
                print(f"sampling at wavelength={wfo.wsamples[w]*1e9:.0f}nm is {check_sampling:.4f} m")
            elif units == 'um':
                print(f"sampling at wavelength={wfo.wsamples[w] * 1e9:.0f}nm is {check_sampling*1e6:.1f} um")
            elif units == 'nm':
                print(f"sampling at wavelength={wfo.wsamples[w] * 1e9:.0f}nm is {check_sampling*1e9:.1f} nm")
            elif units == 'arcsec':
                check_sampling = proper.prop_get_sampling_arcsec(wfo.wf_array[w, 0])
                print(f"sampling at wavelength={wfo.wsamples[w] * 1e9:.0f}nm is {check_sampling*1e3:.2f} mas")
            elif units == 'rad':
                check_sampling = proper.prop_get_sampling_radians(wfo.wf_array[w, 0])
                print(f"sampling at wavelength={wfo.wsamples[w] * 1e9:.0f}nm is {check_sampling:.3f} rad")
            else:
                print(f"sampling at wavelength={wfo.wsamples[w] * 1e9:.0f}nm is {check_sampling} m")
