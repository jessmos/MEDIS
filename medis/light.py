"""
Module for the different format of light that gets passed between the observatory components

"""

import os
import numpy as np
import proper
import h5py
from medis.params import iop, sp, tp, ap, atmp, cdip
from medis.controller import auto_load
from medis.utils import dprint
from medis.observatory import Aberrations, Coronagraph, Telescope, Detector

# sp = iop.simulation_config

class Wavefronts():
    """
    A multidimensional array of proper Wavefront objects

    """

class Fields():
    """
    Class responsible for getting complex electromagnetic fields array from Telescope data, Astrophysics data,
    Atmosphere maps

    """
    def __init__(self):
        self.ap=ap
        self.tp=tp
        self.atmp=atmp
        self.cdip=cdip
        self.sp=sp
        self.cdip.theta_series = np.ones((self.sp.numframes))*np.nan  # todo move this to its own object initialiser
        self.iop=iop
        self.use_cache = True
        self.time_parrelize = True
        self.chunk_time = True

    def generate(self):
        self.telescope = auto_load(Telescope)

        self.cpx_sequence = np.zeros((self.sp.numframes, len(self.sp.save_list), self.ap.n_wvl_init,
                                 1 + len(self.ap.contrast), self.sp.grid_size, self.sp.grid_size), dtype=np.complex)

        for t in range(self.sp.numframes):
            kwargs = {'iter': t,
                      'params': [self.ap, self.tp, self.iop, self.sp],
                      'theta': self.cdip.theta_series[t]}
            self.cpx_sequence[t], self.sampling = proper.prop_run(self.tp.prescription,
                                                                  1, self.sp.grid_size,
                                                                  PASSVALUE=kwargs, VERBOSE=False,
                                                                  TABLE=False)  # 1 is dummy wavelength

        return self.cpx_sequence, self.sampling

    def can_load(self):
        if self.use_cache:
            file_exists = os.path.exists(self.iop.fields)
            if file_exists:
                configs_match = self.configs_match()
                if configs_match:
                    return True

        return False

    def configs_match(self):
        cur_config = self.__dict__
        cache_config = self.load_config()
        configs_match = cur_config == cache_config

        return configs_match

    def load_config(self):
        """ Reads the relevant config data from the saved file """
        with h5py.File(iop.fields, 'r') as hf:
            # fields = hf.get('data')[:]
            config = hf.get('config')[:]
        return config

    def save(self):
        pass

    def load(self):
        pass

    def view(self):
        pass

class Photons():
    """
    Class responsible for getting photon lists from Detector and Fields

    """
    def __init__(self):
        self.config = yaml.load(iop.photons_config)

    def generate(self):
        self.dp = Detector().device_params
        self.fields = Fields().data
        return self.get_packets()

    def get_packets(self):
        intensity = np.abs(self.fields)**2

        if self.dp.resamp:
            nyq_sampling = self.dp.band[0] * 1e-9 * 360 * 3600 / (4 * np.pi * tp.diam)
            sampling = nyq_sampling * tp.beam_ratio * 2  # nyq sampling happens at tp.beam_ratio = 0.5
            x = np.arange(-self.dp.grid_size * sampling / 2, self.dp.grid_size * sampling / 2, sampling)
            xnew = np.arange(-self.dp.array_size[0] * self.dp.platescale / 2, self.dp.array_size[0] * self.dp.platescale / 2, self.dp.platescale)
            mkid_cube = np.zeros((len(intensity), self.dp.array_size[0], self.dp.array_size[1]))
            for s, slice in enumerate(intensity):
                f = interpolate.interp2d(x, x, slice, kind='cubic')
                mkid_cube[s] = f(xnew, xnew)
            mkid_cube = mkid_cube * np.sum(intensity) / np.sum(mkid_cube)
            intensity = mkid_cube

        intensity[intensity < 0] *= -1

        if self.dp.QE_var:
            intensity *= self.dp.QE_map[:intensity.shape[1], :intensity.shape[1]]

        if hasattr(self.dp, 'star_phot'): self.dp.star_photons_per_s = self.dp.star_phot
        num_events = int(self.dp.star_photons_per_s * self.dp.sample_time * np.sum(intensity))

        if sp.verbose:
            dprint(f'star flux: {self.dp.star_photons_per_s}, cube sum: {np.sum(intensity)}, num events: {num_events}')

        photons = self.sample_cube(intensity, num_events)
        # photons = spec.calibrate_phase(photons)
        # photons = temp.assign_calibtime(photons, step)

        if self.dp.dark_counts:
            dark_photons = MKIDs.get_bad_packets(self.dp, step, type='dark')
            photons = np.hstack((photons, dark_photons))

        if self.dp.hot_pix:
            hot_photons = MKIDs.get_bad_packets(self.dp, step, type='hot')
            photons = np.hstack((photons, hot_photons))

        if self.dp.phase_uncertainty:
            photons[1] *= self.dp.responsivity_error_map[np.int_(photons[2]), np.int_(photons[3])]
            photons, idx = MKIDs.apply_phase_offset_array(photons, self.dp.sigs)

        if self.dp.phase_background:
            thresh = -photons[1] > 3 * self.dp.sigs[-1, np.int_(photons[3]), np.int_(photons[2])]
            photons = photons[:, thresh]

        if self.dp.remove_close:
            stem = pipe.arange_into_stem(photons.T, (self.dp.array_size[0], self.dp.array_size[1]))
            stem = MKIDs.remove_close(stem)
            photons = pipe.ungroup(stem)

        return photons.T

    def sample_cube(self):
        photons = Dist(intensity)
        return photons

    def can_load(self):
        if self.use_cache:
            file_exists = os.path.exists(iop.fields)
            if file_exists:
                configs_match = self.configs_match()
                if configs_match:
                    return True

        return False

    def configs_match(self):
        cur_config = self.__dict__
        cache_config = self.load_config()
        configs_match = cur_config == cache_config

        return configs_match

    def load_config(self):
        """ Reads the relevant config data from the saved file """
        pass

    def save(self):
        pass

    def load(self):
        pass

    def view(self):
        pass
