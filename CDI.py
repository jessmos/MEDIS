"""
CDI.py
Kristina Davis
8/12/19

This module is used to set the CDI parameters for mini-medis.
"""
import numpy as np
import proper

from mm_params import tp, ap, sp, cdip
from mm_utils import dprint
from plot_tools import quick2D


def CDIprobe(theta, plot=False):
    """
    apply a probe shape to DM to achieve CDI

    The probe applied to the DM to achieve CDI is that originally proposed in Giv'on et al 2011, doi: 10.1117/12.895117;
    and was used with proper in Matthews et al 2018, doi:  10.1117/1.JATIS.3.4.045001.

    The pupil coordinates used in those equations relate to the sampling of the pupil (plane of the DM). However, in
    Proper, the prop_dm uses a DM map that is the size of (n_ao_act, n_ao_act). This map was resampled from its
    original size to the actuator spacing, and the spacing in units of [m] is supplied as a keyword to the prop_dm
    function during the call. All that is to say, we apply the CDI probe using the coordinates of the DM actuators,
    and supply the probe height as a addative height to the DM map, which is passed to the prop_dm function.

    :param theta: desired phase of the probe
    :param plot: flag to plot phase probe
    :return: height of phase probes to add to the DM map in adaptive.py
    """
    x = np.linspace(-tp.ao_act/2, tp.ao_act/2, tp.ao_act)
    y = x
    X,Y = np.meshgrid(x, y)

    probe = cdip.probe_amp * np.sinc(cdip.probe_w * X) \
                           * np.sinc(cdip.probe_h * Y) \
                           * np.sin(2*np.pi*cdip.probe_center*X + theta)
    # dprint(f"Min={np.min(probe)*1e5:.4f}e-5, Max={np.max(probe)*1e5:.3f}e-5")

    if plot:
        quick2D(probe, title=f"Phase Probe at theta={theta:.4f} rad", vlim=(-1e-5, 1e-5),
            colormap="YlGnBu_r")  # logAmp=True)

    # Testing FF propagation
    probe_ft = np.fft.fftshift(np.fft.fft2(probe))
    if plot:
        quick2D(np.abs(probe_ft), title=f"Real FFT of phase probe", vlim=(-1e-5, 1e-5),
            colormap="YlGnBu_r")
        quick2D(np.angle(probe_ft), title=f"Imaginary FFT of phase probe", vlim=(-1e-5, 1e-5),
                colormap="YlGnBu_r")

    return probe









