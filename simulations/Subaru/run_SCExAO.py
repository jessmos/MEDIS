"""
run_SCExAO
KD
Feb 2020

This is the starting point to run the Subaru_SCExAO prescription. From here, you can turn on/off toggles, change AO
settings, view different planes, and make other changes to running the prescription without changing the base
prescription or the default params themselves.

"""
import numpy as np

from medis.params import params
from medis.utils import dprint
import medis.optics as opx
from medis.plot_tools import view_spectra, view_timeseries, quick2D, plot_planes
import medis.medis_main as mm

#################################################################################################
#################################################################################################
#################################################################################################
# iop.update_root(f"/home/captainkay/mazinlab/MKIDSim/CDIsim_data/")
# iop.update_testname('Subaru-test2')
# iop.makedir()

# Telescope
params['tp'].prescription = 'Subaru_SCExAO'
params['tp'].entrance_d = 7.9716
params['tp'].flen_primary = params['tp'].entrance_d * 13.612

# Simulation & Timing
params['sp'].numframes = 1
params['sp'].closed_loop = False

# Grid Parameters
params['sp'].focused_sys = True
params['sp'].beam_ratio = 0.2  # parameter dealing with the sampling of the beam in the pupil/focal plane
params['sp'].grid_size = 512  # creates a nxn array of samples of the wavefront
params['sp'].maskd_size = 256  # will truncate grid_size to this range (avoids FFT artifacts) # set to grid_size if undesired

# Companion
params['ap'].companion = False
params['ap'].contrast = [1e-1]
params['ap'].companion_xy = [[5, -5]]  # units of this are lambda/params['tp'].entrance_d
params['ap'].n_wvl_init = 3  # initial number of wavelength bins in spectral cube (later sampled by MKID detector)
params['ap'].n_wvl_final = None  # final number of wavelength bins in spectral cube after interpolation (None sets equal to n_wvl_init)
params['ap'].interp_wvl = False  # Set to interpolate wavelengths from ap.n_wvl_init to ap.n_wvl_final
params['ap'].wvl_range = np.array([800, 1400]) / 1e9  # wavelength range in [m] (formerly ap.band)
# eg. DARKNESS band is [800, 1500], J band =  [1100,1400])


# Toggles for Aberrations and Control
params['tp'].obscure = False
params['tp'].use_atmos = False
params['tp'].use_aber = False
params['tp'].use_ao = True
params['cdip'].use_cdi = False

# Plotting
params['sp'].show_wframe = False  # plot white light image frame
params['sp'].show_spectra = True  # Plot spectral cube at single timestep
params['sp'].spectra_cols = 3  # number of subplots per row in view_spectra
params['sp'].show_tseries = False  # Plot full timeseries of white light frames
params['sp'].tseries_cols = 5  # number of subplots per row in view_timeseries
params['sp'].show_planes = True

# Saving
params['sp'].save_to_disk = False  # save obs_sequence (timestep, wavelength, x, y)
params['sp'].save_list = ['atmosphere', 'entrance_pupil','woofer', 'focus', 'coronagraph', 'detector']  # list of locations in optics train to save

if __name__ == '__main__':
    # =======================================================================
    # Run it!!!!!!!!!!!!!!!!!
    # =======================================================================
    sim = mm.RunMedis(params=params, name='SCExAO', product='fields')
    observation = sim()
    cpx_sequence = observation['fields']
    sampling = observation['sampling']

    # =======================================================================
    # Focal Plane Processing
    # =======================================================================
    # cpx_sequence = (n_timesteps ,n_planes, n_waves_init, n_astro_bodies, nx ,ny)
    # cpx_sequence = mmu.interp_wavelength(cpx_sequence, 2)  # interpolate over wavelength
    focal_plane = opx.extract_plane(cpx_sequence, 'detector')  # eliminates astro_body axis
    # convert to intensity THEN sum over object, keeping the dimension of tstep even if it's one
    focal_plane = np.sum(opx.cpx_to_intensity(focal_plane), axis=2)
    fp_sampling = sampling[-1,:]

    # =======================================================================
    # Plotting
    # =======================================================================
    # White Light, Last Timestep
    if params['sp'].show_wframe:
        # vlim = (np.min(spectralcube) * 10, np.max(spectralcube))  # setting z-axis limits
        img = np.sum(focal_plane[params['sp'].numframes-1], axis=0)  # sum over wavelength
        quick2D(opx.extract_center(img), #focal_plane[params['sp'].numframes-1]),
                title=f"White light image at timestep {params['sp'].numframes} \n"  # img
                           f"AO={params['tp'].use_ao}, CDI={params['cdip'].use_cdi} ",
                           # f"Grid Size = {params['sp'].grid_size}, Beam Ratio = {params['sp'].beam_ratio} ",
                           # f"sampling = {sampling*1e6:.4f} (um/gridpt)",
                logZ=True,
                dx=fp_sampling[0],
                vlim=(None,None))  # (1e-3, 1e-1)

    # Plotting Spectra at last tstep
    if params['sp'].show_spectra:
        tstep = params['sp'].numframes-1
        view_spectra(focal_plane[params['sp'].numframes-1],
                      title=f"Intensity per Spectral Bin at Timestep {tstep} \n"
                            f" AO={params['tp'].use_ao}, CDI={params['cdip'].use_cdi}",
                            # f"Beam Ratio = {params['sp'].beam_ratio:.4f}",#  sampling = {sampling*1e6:.4f} [um/gridpt]",
                      logZ=True,
                      subplt_cols=params['sp'].spectra_cols,
                      vlim=(1e-10, 1e-4),
                      dx=fp_sampling)

    # Plotting Timeseries in White Light
    if params['sp'].show_tseries:
        img_tseries = np.sum(focal_plane, axis=1)  # sum over wavelength
        view_timeseries(img_tseries, title=f"White Light Timeseries\n"
                                            f"AO={params['tp'].use_ao}. CDI={params['cdip'].use_cdi}",
                        subplt_cols=params['sp'].tseries_cols,
                        logZ=True,
                        vlim=(1e-10, 1e-4))
                        # dx=fp_sampling[0])

    # Plotting Selected Plane
    if params['sp'].show_planes:
        vlim = [(None, None), (None, None), (None, None), (None, None), (None, None), (None, None)]
        # vlim = [(None,None), (None,None), (None,None), (None,None)]  # (1e-2,1e-1) (7e-4, 6e-4)
        logZ = [True, False, False, True, True, True]
        if params['sp'].save_list:
            plot_planes(cpx_sequence,
                        title=f"White Light through Optical System",
                        subplt_cols=2,
                        vlim=vlim,
                        logZ=logZ,
                        dx=sampling)

    test = 1
