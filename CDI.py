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
    x = np.linspace(-0.045, 0.045, tp.grid_size)
    y = x
    X,Y = np.meshgrid(x, y)

    probe = cdip.probe_amp * np.sinc(cdip.probe_w * X) \
                             * np.sinc(cdip.probe_h * Y) \
                             * np.sin(2*np.pi*cdip.probe_center*X + theta)
    # dprint(f"Min={np.min(probe)*1e5:.4f}e-5, Max={np.max(probe)*1e5:.3f}e-5")

    if plot:
        quick2D(probe, title=f"Phase Probe at theta={theta:.4f} rad", vlim=(-1e-6, 1e-6),
            colormap="YlGnBu_r")  # logAmp=True)

    return probe









