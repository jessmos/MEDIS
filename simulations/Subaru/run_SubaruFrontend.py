"""
run_SubaruFrontend
KD
Dec 9 2019

This is the starting point to run the Subaru_frontend prescription. From here, you can turn on/off toggles, change AO
settings, view different planes, and make other changes to running the prescription without changing the base
prescription or the default params themselves.

This file is meant to 'cleanup' a lot of the functionality between having defaults saved in the params file and
the prescription. Most changes necessary to run multiple instances of the base prescription will be made here. This is
the same layot/format as the Example files that Rupert was using in v1.0. These are now meant to be paired with a more
specific prescription than the original optics_propagate, which had more toggles and less sophisiticated optical train.

"""
import numpy as np

from medis.params import iop, sp, ap, tp, cdip
from medis.utils import dprint
import medis.optics as opx
from medis.plot_tools import view_spectra, view_timeseries, quick2D, plot_planes
import medis.medis_main as mm

#################################################################################################
#################################################################################################
#################################################################################################
# Companion
ap.companion = True
ap.contrast = [1e-5]
ap.companion_xy = [[15, -15]]  # units of this still confuse me

tp.prescription = 'Subaru_frontend'
tp.enterance_d = 7.9716
tp.flen_primary = tp.enterance_d * 13.612

sp.focused_sys = True
sp.beam_ratio = 0.14  # parameter dealing with the sampling of the beam in the pupil/focal plane
sp.grid_size = 512  # creates a nxn array of samples of the wavefront
sp.maskd_size = 256  # will truncate grid_size to this range (avoids FFT artifacts) # set to grid_size if undesired

# Toggles for Aberrations and Control
tp.obscure = False
tp.use_atmos = False
tp.use_aber = False
tp.use_ao = True
tp.ao_act = 14
cdip.use_cdi = False

# Plotting
sp.show_wframe = True  # Plot white light image frame
sp.show_spectra = True  # Plot spectral cube at single timestep
sp.spectra_cols = 3  # number of subplots per row in view_datacube
sp.show_tseries = False  # Plot full timeseries of white light frames
sp.tseries_cols = 5  # number of subplots per row in view_timeseries
sp.show_planes = True

# Saving
sp.save_obs = False  # save obs_sequence (timestep, wavelength, x, y)
sp.save_fields = True  # toggle to turn saving uniformly on/off
sp.save_list = ['atmosphere', 'entrance_pupil', 'woofer', 'detector']  # list of locations in optics train to save


if __name__ == '__main__':
    # testname = input("Please enter test name: ")
    testname = 'Subaru-test1'
    iop.update(testname)
    iop.makedir()

    # =======================================================================
    # Run it!!!!!!!!!!!!!!!!!
    # =======================================================================
    cpx_sequence, sampling = mm.run_medis()

    # =======================================================================
    # Focal Plane Processing
    # =======================================================================
    # obs_sequence = np.array(obs_sequence)  # obs sequence is returned by gen_timeseries (called above)
    # (n_timesteps ,n_planes, n_waves_init, n_objects, nx ,ny)
    # cpx_sequence = mmu.interp_wavelength(cpx_sequence, 2)  # interpolate over wavelength
    focal_plane = opx.extract_plane(cpx_sequence, 'detector')
    focal_plane = np.sum(opx.cpx_to_intensity(focal_plane), axis=1)  # convert to intensity THEN sum over object
    dprint(f"focal plane shape is {focal_plane.shape}")

    # =======================================================================
    # Plotting
    # =======================================================================
    # White Light, Last Timestep
    if sp.show_wframe:
        # vlim = (np.min(spectralcube) * 10, np.max(spectralcube))  # setting z-axis limits
        # img = np.sum(focal_plane[sp.numframes - 1], axis=0)  # extract last tstep, sum over wavelength
        # dprint(f"shape of img is {img.shape}")
        quick2D(opx.extract_center(focal_plane[sp.numframes-1]),
                title=f"White light image at timestep {sp.numframes} \n"  # img
                           f"AO={tp.use_ao}, CDI={cdip.use_cdi} ",
                           # f"Grid Size = {sp.grid_size}, Beam Ratio = {sp.beam_ratio} ",
                           # f"sampling = {sampling*1e6:.4f} (um/gridpt)",
                logZ=True,
                dx=sampling[-1],
                vlim=(1e-6, 1e-3))

    # Plotting Spectra at last tstep
    if sp.show_spectra:
        tstep = sp.numframes-1
        view_spectra(focal_plane[sp.numframes-1],
                      title=f"Intensity per Spectral Bin at Timestep {tstep} \n"
                            f" AO={tp.use_ao}, CDI={cdip.use_cdi}"
                            f"Beam Ratio = {sp.beam_ratio:.4f}",#  sampling = {sampling*1e6:.4f} [um/gridpt]",
                      logZ=True,
                      subplt_cols=sp.spectra_cols,
                      vlim=(1e-8, 1e-3),
                      dx=sampling[-1])

    # Plotting Timeseries in White Light
    if sp.show_tseries:
        img_tseries = np.sum(focal_plane, axis=1)  # sum over wavelength
        view_timeseries(img_tseries, title=f"White Light Timeseries\n"
                                            f"AO={tp.use_ao}. CDI={cdip.use_cdi}",
                        subplt_cols=sp.tseries_cols,
                        logZ=True,
                        vlim=(1e-6, 1e-3))
                        # dx=sampling

    # Plotting Selected Plane
    if sp.show_planes:
        # vlim = ((None, None), (None, None), (None, None), (None, None), (None, None))
        vlim = ((None,None), (None,None), (None,None),  (1e-7,1e-3))  # (7e-4, 6e-4)
        logZ = (True, False, False,  True)
        if sp.save_list:
            plot_planes(cpx_sequence,
                        title=f"White Light through Optical System",
                        subplt_cols=2,
                        vlim=vlim,
                        logZ=logZ,
                        dx=sampling)

    test = 1
