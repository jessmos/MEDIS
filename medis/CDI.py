"""
CDI.py
Kristina Davis
8/12/19

This module is used to set the CDI parameters for mini-medis.
"""
import numpy as np
import warnings
import proper

from medis.params import tp, sp
from medis.utils import dprint
from matplotlib import pyplot as plt


class CDI_params():
    def __init__(self):
        # General
        self.use_cdi = False
        self.show_probe = False  # False , flag to plot phase probe or not
        self.which_DM = ''  # plane_name parameter of the DM to apply CDI probe to (must be a valid name for
        # the telescope sim you are running, eg 'tweeter'

        # Probe Dimensions (extent in pupil plane coordinates)
        self.probe_w = 10  # [actuator coordinates] width of the probe
        self.probe_h = 30  # [actuator coordinates] height of the probe
        self.probe_center = (15, 15)  # [actuator coordinates] center position of the probe (should move off-center to
        # avoid coronagraph)
        self.probe_amp = 2e-6  # [m] probe amplitude, scale should be in units of actuator height limits

        # Phase Sequence of Probes
        self.phs_intervals = np.pi / 4  # [rad] phase interval over [0, 2pi]
        self.phase_integration_time = 0.01  # [s]  How long in sec to apply each probe in the sequence
        self.null_time = 0.1  # [s]  time between repeating probe cycles (data to be nulled using probe info)

    def probe_phasestream(self):
        """
        generate an array of phases per timestep for the CDI algorithm

        phase_series is used to populate cp.theta_series, which may be longer than cp.phase_list if multiple cycles
        are run, or probes may last for longer than one single timestep

        currently, I assume the timestream is not that long. Should only use this for short timestreams, and use a more
        efficient code for long simulations (scale of minutes or more)

        :return: phase_list  array of phases to use in CDI probes
        """
        self.phase_series = np.zeros(sp.numframes) * np.nan

        if cp.use_cdi:
            # Repeating Probe Phases for Integration time
            self.phase_cycle = np.arange(0, 2 * np.pi, cp.phs_intervals)  # FYI not inclusive of 2pi endpoint
            self.n_probes = len(self.phase_cycle)  # number of phase probes

            if cp.phase_integration_time > sp.sample_time:
                phase_hold = cp.phase_integration_time / sp.sample_time
                phase_1cycle = np.repeat(self.phase_cycle, phase_hold)
            elif cp.phase_integration_time == sp.sample_time:
                phase_1cycle = self.phase_cycle
            else:
                raise ValueError(f"Cannot have CDI phase probe integration time less than sp.sample_time")

            # Repeating Cycle of Phase Probes for Simulation Duration
            full_simulation_time = sp.numframes * sp.sample_time
            time_for_one_cycle = len(phase_1cycle) * cp.phase_integration_time + cp.null_time

            if time_for_one_cycle > full_simulation_time and cp.phase_integration_time > sp.sample_time:
                warnings.warn(f"\nLength of one full CDI probe cycle (including nulling) exceeds the "
                              f"full simulation time \n"
                              f"not all phases will be used\n")
                self.phase_series = phase_1cycle[0:sp.numframes]
            elif time_for_one_cycle > full_simulation_time and self.n_probes < sp.numframes:
                print(f"There will be {sp.numframes - self.n_probes} "
                      f"nulling steps after timestep {self.n_probes - 1}")
                self.phase_series[0:self.n_probes] = phase_1cycle
            else:
                warnings.warn(f"Haven't run into  CDI phase situation like this yet")
                raise NotImplementedError
cp = CDI_params()


def cprobe(theta, nact, iw=0, ib=0):
    """
    apply a probe shape to DM to achieve CDI

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
    x = np.linspace(-1/2, 1/2, nact)
    y = x
    # x = np.linspace(-1/2-cp.probe_center[0], 1/2-cp.probe_center[0], nact)
    # y = np.linspace(-1/2-cp.probe_center[1], 1/2-cp.probe_center[1], nact)
    X,Y = np.meshgrid(x, y)

    probe = cp.probe_amp * np.sinc(cp.probe_w * X) * np.sinc(cp.probe_h * Y) \
            * np.sin(2*np.pi*cp.probe_center[0]*X + theta) #\
            # * np.sin(2 * np.pi * cp.probe_center[1] * Y + theta)

    # Testing FF propagation
    if sp.verbose and iw == 0 and ib == 0 and theta == cp.theta_series[0]:
        probe_ft = (1/np.sqrt(2*np.pi)) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(probe)))

        fig, ax = plt.subplots(1, 3, figsize=(12, 5))
        fig.subplots_adjust(wspace=0.5)
        ax1, ax2, ax3 = ax.flatten()

        fig.suptitle(f"center={cp.probe_center}, Dimensions {cp.probe_w}x{cp.probe_h} "
                     f"\nProbe Amp = {cp.probe_amp}, " + r'$\theta$' + f"={theta/np.pi:.3f}" + r'$\pi$')

        im1 = ax1.imshow(probe, interpolation='none', origin='lower')
        ax1.set_title(f"Probe on DM \n(dm coordinates)")
        cb = fig.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(np.sqrt(probe_ft.imag ** 2 + probe_ft.real ** 2), interpolation='none', origin='lower')
        ax2.set_title("Focal Plane Amplitude")
        cb = fig.colorbar(im2, ax=ax2)

        ax3.imshow(np.arctan2(probe_ft.imag, probe_ft.real), interpolation='none', origin='lower', cmap='hsv')
        ax3.set_title("Focal Plane Phase")

        plt.show()

        # Fig 2
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        fig.subplots_adjust(wspace=0.5)
        ax1, ax2 = ax.flatten()
        fig.suptitle(f'Real & Imaginary Probe Response in Focal Plane, '+r'$\theta$'+f'={theta/np.pi:.3f}'+r'$\pi$')

        im1 = ax1.imshow(probe_ft.real, interpolation='none', origin='lower')
        ax1.set_title(f"Real FT of Probe")

        im2 = ax2.imshow(probe_ft.imag, interpolation='none', origin='lower')
        ax2.set_title(f"Imag FT of Probe")

        plt.show()

        # quick2D(probe_ft.imag,
        #         title=f"Imag FFT of CDI probe, " r'$\theta$' + f"={cp.phase_list[iw]/np.pi:.2f}" + r'$\pi$',
        #         vlim=(-1e-6, 1e-6),
        #         colormap="YlGnBu_r")
        # quick2D(np.arctan2(probe_ft.imag, probe.real),
        #         title="Phase of Probe",
        #         # vlim=(-1e-6, 1e-6),
        #         colormap="YlGnBu_r")

    return probe


def cdi_postprocess(fp_seq, sampling):
    """
    this is the function that accepts the timeseries of intensity images from the simuation and returns the processed
    single image. This function calculates the speckle amplitude phase, and then corrects for it to create the dark
    hole over the specified region of the image.

    :param fp_seq: timestream of 2D images (intensity only) from the focal plane complex field
    :param sampling: focal plane sampling
    :return:
    """
    a=cp.n_probes//2
    dprint(f"cp.nprobes={cp.n_probes}, divided {cp.n_probes//2}")
    delta = np.zeros((cp.n_probes//2, sp.grid_size, sp.grid_size))
    for ix in range(cp.n_probes//2):
        delta[ix, :, :] = fp_seq[-ix] - fp_seq[ix]

    # Fig 2
    fig, ax = plt.subplots(1,2)
    fig.subplots_adjust(wspace=0.5)
    ax1, ax2 = ax.flatten()
    fig.suptitle('Deltas for CDI Probes')

    im1 = ax1.imshow(delta[0], interpolation='none', origin='lower')
    ax1.set_title(f"Deff Probe pair 1")

    im2 = ax2.imshow(delta[1], interpolation='none', origin='lower')
    ax2.set_title(f"Diff Probe pair 2")


if __name__ == '__main__':
    dprint(f"Testing CDI probe")
    cp.use_cdi = True; cp.show_probe = True

    cp.probe_w = 15  # [actuator coordinates] width of the probe
    cp.probe_h = 30  # [actuator coordinates] height of the probe
    cp.probe_center = (15, 15)  # [actuator coordinates] center position of the probe
    cp.probe_amp = 2e-6  # [m] probe amplitude, scale should be in units of actuator height limits

    cp.theta_series = [-1*np.pi/4]
    cprobe(-1*np.pi/4, 50)
