"""
CDI.py
Kristina Davis
8/12/19

This module is used to set the CDI parameters for mini-medis.
"""
import numpy as np
import warnings
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import proper

from medis.params import tp, sp
from medis.utils import dprint


class CDI_params():
    """
    contains the parameters of the CDI probes and phase sequence to apply to the DM


    """
    def __init__(self):
        # General
        self.use_cdi = False
        self.show_probe = False  # False , flag to plot phase probe or not
        self.which_DM = ''  # plane_name parameter of the DM to apply CDI probe to (must be a valid name for
        # the telescope sim you are running, eg 'tweeter'

        # Probe Dimensions (extent in pupil plane coordinates)
        self.probe_amp = 2e-6  # [m] probe amplitude, scale should be in units of actuator height limits
        self.probe_w = 10  # [actuator coordinates] width of the probe
        self.probe_h = 30  # [actuator coordinates] height of the probe
        self.probe_shift = (15, 15)  # [actuator coordinates] center position of the probe (should move off-center to
        # avoid coronagraph)
        self.probe_spacing = 10  # distance from the focal plane center to edge of the rectangular probed region

        # Phase Sequence of Probes
        self.phs_intervals = np.pi / 2  # [rad] phase interval over [0, 2pi]
        self.phase_integration_time = 0.01  # [s]  How long in sec to apply each probe in the sequence
        self.null_time = 0  # [s]  time between repeating probe cycles (data to be nulled using probe info)

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def __name__(self):
        return self.__str__().split(' ')[0].split('.')[-1]

    def gen_phaseseries(self):
        """
        generate an array of phases per timestep for the CDI algorithm

        phase_series is used to populate cdi.phase_series, which may be longer than cdi.phase_cycle if multiple cycles
        are run, or probes may last for longer than one single timestep

        currently, I assume the timestream is not that long. Should only use this for short timestreams, and use a more
        efficient code for long simulations (scale of minutes or more)

        :return: phase_series  array of phases of CDI probes to apply to DM
        """
        self.phase_series = np.zeros(sp.numframes) * np.nan

        if self.use_cdi:
            # Repeating Probe Phases for Integration time
            self.phase_cycle = np.arange(0, 2 * np.pi, self.phs_intervals)  # FYI not inclusive of 2pi endpoint
            self.n_probes = len(self.phase_cycle)  # number of phase probes
            if self.n_probes % 2 != 0:
                raise ValueError(f"must have even number of phase probes\n\tchange cdi.phs_intervals")

            if self.phase_integration_time > sp.sample_time:
                phase_hold = self.phase_integration_time / sp.sample_time
                phase_1cycle = np.repeat(self.phase_cycle, phase_hold)
            elif self.phase_integration_time == sp.sample_time:
                phase_1cycle = self.phase_cycle
            else:
                raise ValueError(f"Cannot have CDI phase probe integration time less than sp.sample_time")

            # Repeating Cycle of Phase Probes for Simulation Duration
            full_simulation_time = sp.numframes * sp.sample_time
            time_for_one_cycle = len(phase_1cycle) * self.phase_integration_time + self.null_time

            if time_for_one_cycle > full_simulation_time and self.phase_integration_time >= sp.sample_time:
                warnings.warn(f"\nLength of one full CDI probe cycle (including nulling) exceeds the "
                              f"full simulation time \n"
                              f"not all phases will be used\n"
                              f"phase reconstruction will be incomplete")
                self.phase_series = phase_1cycle[0:sp.numframes]
            elif time_for_one_cycle <= full_simulation_time and self.n_probes < sp.numframes:
                print(f"\nCDI Params\n\tThere will be {sp.numframes - self.n_probes} "
                      f"nulling steps after timestep {self.n_probes - 1}")
                self.phase_series[0:self.n_probes] = phase_1cycle
            else:
                warnings.warn(f"Haven't run into  CDI phase situation like this yet")
                raise NotImplementedError

        return self.phase_series

    def init_probes(self, nact):
        self.grid_size = sp.grid_size
        self.beam_ratio = sp.beam_ratio
        self.DM_probe_series = np.zeros((self.n_probes, nact, nact))

    def save_probe(self, ix, probe):
        self.DM_probe_series[ix] = probe


cdi = CDI_params()


def config_probe(theta, nact, iw=0, ib=0):
    """
    create a probe shape to apply to the DM for CDI processing

    The probe applied to the DM to achieve CDI is that originally proposed in Giv'on et al 2011, doi: 10.1117/12.895117;
    and was used with proper in Matthews et al 2018, doi:  10.1117/1.JATIS.3.4.045001.

    The pupil coordinates used in those equations relate to the sampling of the pupil (plane of the DM). However, in
    Proper, the prop_dm uses a DM map that is the size of (n_ao_act, n_ao_act). This map was resampled from its
    original size to the actuator spacing, and the spacing in units of [m] is supplied as a keyword to the prop_dm
    function during the call. All that is to say, we apply the CDI probe using the coordinates of the DM actuators,
    and supply the probe height as an additive height to the DM map, which is passed to the prop_dm function.

    :param theta: phase of the probe
    :param nact: number of actuators in the mirror, should change if 'woofer' or 'tweeter'
    :param iw: index of wavelength number in ap.wvl_range
    :return: height of phase probes to add to the DM map in adaptive.py
    """
    x = np.linspace(-1/2-cdi.probe_shift[0]/nact, 1/2-cdi.probe_shift[0]/nact, nact)
    y = np.linspace(-1/2-cdi.probe_shift[1]/nact, 1/2-cdi.probe_shift[1]/nact, nact)
    X,Y = np.meshgrid(x, y)

    probe = cdi.probe_amp * np.sinc(cdi.probe_w * X) * np.sinc(cdi.probe_h * Y) \
            * np.sin(2*np.pi*cdi.probe_spacing*X + theta)

    # Testing FF propagation
    if sp.verbose and iw == 0 and ib == 0:  # and theta == cdi.phase_series[0]
        probe_ft = (1/np.sqrt(2*np.pi)) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(probe)))

        # fig, ax = plt.subplots(1, 3, figsize=(12, 5))
        # fig.subplots_adjust(wspace=0.5)
        # ax1, ax2, ax3 = ax.flatten()
        #
        # fig.suptitle(f"spacing={cdi.probe_spacing}, Dimensions {cdi.probe_w}x{cdi.probe_h} "
        #              f"\nProbe Amp = {cdi.probe_amp}, " + r'$\theta$' + f"={theta/np.pi:.3f}" + r'$\pi$' + '\n')
        #
        # im1 = ax1.imshow(probe, interpolation='none', origin='lower')
        # ax1.set_title(f"Probe on DM \n(dm coordinates)")
        # cb = fig.colorbar(im1, ax=ax1)
        #
        # im2 = ax2.imshow(np.sqrt(probe_ft.imag ** 2 + probe_ft.real ** 2), interpolation='none', origin='lower')
        # ax2.set_title("Focal Plane Amplitude")
        # cb = fig.colorbar(im2, ax=ax2)
        #
        # ax3.imshow(np.arctan2(probe_ft.imag, probe_ft.real), interpolation='none', origin='lower', cmap='hsv')
        # ax3.set_title("Focal Plane Phase")

        # plt.show()

        # Fig 2
        fig, ax = plt.subplots(1, 3, figsize=(12, 5))
        fig.subplots_adjust(wspace=0.5)
        ax1, ax2, ax3 = ax.flatten()
        fig.suptitle(f'Real & Imaginary Probe Response in Focal Plane\n'
                     f' '+r'$\theta$'+f'={theta/np.pi:.3f}'+r'$\pi$'+f', n_actuators = {nact}\n')

        im1 = ax1.imshow(probe, interpolation='none', origin='lower')
        ax1.set_title(f"Probe on DM \n(dm coordinates)")
        cb = fig.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(probe_ft.real, interpolation='none', origin='lower')
        ax2.set_title(f"Real FT of Probe")

        im3 = ax3.imshow(probe_ft.imag, interpolation='none', origin='lower')
        ax3.set_title(f"Imag FT of Probe")

        plt.show()

    # Saving Probe in the series
    ip = np.argwhere(cdi.phase_series == theta)
    cdi.DM_probe_series[ip[0,0]] = probe

    return probe


def cdi_postprocess(fp_seq, sampling, plot=False):
    """
    this is the function that accepts the timeseries of intensity images from the simuation and returns the processed
    single image. This function calculates the speckle amplitude phase, and then corrects for it to create the dark
    hole over the specified region of the image.

    :param fp_seq: timestream of 2D images (intensity only) from the focal plane complex field
    :param sampling: focal plane sampling
    :return:
    """
    nd = cdi.n_probes//2  # number of deltas (probe differentials)
    delta = np.zeros((nd, sp.grid_size, sp.grid_size))
    import medis.optics as opx
    fp_seq = np.sum(opx.cpx_to_intensity(fp_seq), axis=1)  # sum over wavelength

    for ix in range(nd):
        delta[ix] = np.copy(fp_seq[ix] - fp_seq[ix+nd])

    # Fig 2
    if plot:
        fig, subplot = plt.subplots(1, nd, figsize=(12,5))
        fig.subplots_adjust(wspace=0.5)

        fig.suptitle('Deltas for CDI Probes')

        for ax, ix in zip(subplot.flatten(), range(nd)):
            im = ax.imshow(delta[ix]*1e6, interpolation='none', origin='lower', norm=LogNorm())
            ax.set_title(f"Diff Probe\n" + r'$\theta$' + f'={cdi.phase_series[ix]/np.pi:.3f}' +
                         r'$\pi$ -$\theta$' + f'={cdi.phase_series[ix+nd]/np.pi:.3f}' + r'$\pi$')

        cb = fig.colorbar(im, orientation='horizontal')  #
        cb.set_label('dB? maybe?')

        plt.show()


if __name__ == '__main__':
    dprint(f"Testing CDI probe")
    cdi.use_cdi = True; cdishow_probe = True

    cdiprobe_w = 10  # [actuator coordinates] width of the probe
    cdiprobe_h = 30  # [actuator coordinates] height of the probe
    cdiprobe_shift = [5, 5]  # [actuator coordinates] center position of the probe
    cdiprobe_spacing = 15
    cdiprobe_amp = 2e-6  # [m] probe amplitude, scale should be in units of actuator height limits

    sp.numframes = 10
    cdigen_phaseseries()
    probes = CDI_Probes(cp)
    # cdiphase_series = [-1*np.pi/4]
    probes.probe(cdiphase_series[0], 49)  #
