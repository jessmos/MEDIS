"""
CDI.py
Kristina Davis
8/12/19

This module is used to set the CDI parameters for mini-medis.
"""
import numpy as np
import warnings
import proper

from medis.params import tp, ap, sp, cdip
from medis.utils import dprint
from medis.plot_tools import quick2D


def CDIprobe(theta, nact, iw):
    """
    apply a probe shape to DM to achieve CDI

    The probe applied to the DM to achieve CDI is that originally proposed in Giv'on et al 2011, doi: 10.1117/12.895117;
    and was used with proper in Matthews et al 2018, doi:  10.1117/1.JATIS.3.4.045001.

    The pupil coordinates used in those equations relate to the sampling of the pupil (plane of the DM). However, in
    Proper, the prop_dm uses a DM map that is the size of (n_ao_act, n_ao_act). This map was resampled from its
    original size to the actuator spacing, and the spacing in units of [m] is supplied as a keyword to the prop_dm
    function during the call. All that is to say, we apply the CDI probe using the coordinates of the DM actuators,
    and supply the probe height as an additive height to the DM map, which is passed to the prop_dm function.

    :param theta: desired phase of the probe
    :param nact: number of actuators in the mirror, should change if 'woofer' or 'tweeter'
    :param iw: index of wavelength number in ap.wvl_range
    :return: height of phase probes to add to the DM map in adaptive.py
    """
    x = np.linspace(-1/2, 1/2, nact)
    y = x
    X,Y = np.meshgrid(x, y)

    probe = cdip.probe_amp * np.sinc(cdip.probe_w * X) * np.sinc(cdip.probe_h * Y) \
                * np.sin(2*np.pi*cdip.probe_center*X + theta)

    # Testing FF propagation
    if cdip.show_probe and iw == 0 and theta == cdip.phase_list[0]:
        from matplotlib import pyplot as plt
        probe_ft = (1/np.sqrt(2*np.pi)) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(probe)))

        fig, ax = plt.subplots(1, 3, figsize=(12, 5))
        fig.subplots_adjust(wspace=0.5)
        ax1, ax2, ax3 = ax.flatten()

        fig.suptitle(f"Probe Amp = {cdip.probe_amp}, Dimensions {cdip.probe_w}x{cdip.probe_h}, "
                     f"center={cdip.probe_center}, " + r'$\theta$' + f"={theta/np.pi}" + r'$\pi$')

        im1 = ax1.imshow(probe, interpolation='none', origin='lower')
        ax1.set_title(f"Probe on DM \n(dm coordinates)")
        cb = fig.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(np.sqrt(probe_ft.imag ** 2 + probe_ft.real ** 2), interpolation='none', origin='lower')
        ax2.set_title("Focal Plane Amplitude")
        cb = fig.colorbar(im2, ax=ax2)

        ax3.imshow(np.arctan2(probe_ft.imag, probe_ft.real), interpolation='none', origin='lower', cmap='hsv')
        ax3.set_title("Focal Plane Phase")

        # quick2D(probe_ft.imag,
        #         title=f"Imag FFT of CDI probe, " r'$\theta$' + f"={cdip.phase_list[iw]/np.pi:.2f}" + r'$\pi$',
        #         vlim=(-1e-6, 1e-6),
        #         colormap="YlGnBu_r")
        # quick2D(np.arctan2(probe_ft.imag, probe.real),
        #         title="Phase of Probe",
        #         # vlim=(-1e-6, 1e-6),
        #         colormap="YlGnBu_r")

    return probe


def gen_CDI_phase_stream():
    """
    generate an array of phases per timestep for the CDI algorithm

    currently, I assume the timestream is not that long. Should only use this for short timestreams, and use a more
    efficient code for long simulations (scale of minutes or more)

    :return: phase_list  array of phases to use in CDI probes
    """
    phase_series = np.zeros(sp.numframes) * np.nan

    # Repeating Probe Phases for Integration time
    if cdip.phase_integration_time > sp.sample_time:
        phase_hold = cdip.phase_integration_time / sp.sample_time
        phase_1cycle = np.repeat(cdip.phase_list, phase_hold)
    elif cdip.phase_integration_time == sp.sample_time:
        phase_1cycle = cdip.phase_list
    else:
        raise ValueError(f"Cannot have CDI phase probe integration time less than sp.sample_time")

    # Repeating Cycle of Phase Probes for Simulation Duration
    full_simulation_time = sp.numframes * sp.sample_time
    time_for_one_cycle = len(phase_1cycle) * cdip.phase_integration_time + cdip.null_time
    n_phase_cycles = full_simulation_time / time_for_one_cycle
    dprint(f"number of phase cycles = {n_phase_cycles}")
    if n_phase_cycles < 0.5:
        if cdip.n_probes > sp.numframes:
            warnings.warn(f"Number of timesteps in sp.numframes is less than number of CDI phases \n"
                          f"not all phases will be used")
            phase_series = phase_1cycle[0:sp.numframes]
        else:
            warnings.warn(f"Total length of CDI integration time for all phase probes exceeds full simulation time \n"
                          f"Not all phase probes will be used")
            phase_series = phase_1cycle[0:sp.numframes]
    elif 0.5 < n_phase_cycles < 1:
        phase_series[0:len(phase_1cycle)] = phase_1cycle
        dprint(f"phase_seris  = {phase_series}")
    else:
        n_full = np.floor(n_phase_cycles)
        raise NotImplementedError(f"Whoa, not implemented yet. Hang in there")
        # TODO implement

    return phase_series


if __name__ == '__main__':
    dprint(f"Testing CDI probe")
    theta = cdip.phase_list[0]
    CDIprobe(theta, 0)
