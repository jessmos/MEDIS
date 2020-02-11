"""
Module for the different format of light that gets passed between the observatory components

"""

import numpy as np
import yaml
from medis.params import iop
from controller import get_data
from medis.Utils.misc import dprint
from observatory import Aberrations, Coronagraph, Telescope, Detector

sp = iop.simulation_config

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
        self.config = yaml.load(iop.fields_config)

    def generate(self):
        self.prescription = get_data(Telescope)


        cpx_sequence = np.zeros((sp.numframes, len(sp.save_list), ap.n_wvl_init, 1 + len(ap.contrast),
                                      sp.grid_size, sp.grid_size), dtype=np.complex)
        for t in range(sp.numframes):
            kwargs = {'iter': t, 'params': [ap, tp, iop, sp], 'theta': theta_series[t]}
            self.cpx_sequence[t], self.sampling = proper.prop_run(self.prescription, 1, sp.grid_size,
                                                                  PASSVALUE=kwargs,
                                                                  VERBOSE=False,
                                                                  TABLE=False)  # 1 is dummy wavelength

        return cpx_sequence

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

    def save(self):
        pass

    def load(self):
        pass

    def view(self):
        pass
