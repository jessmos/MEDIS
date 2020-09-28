"""
run_SCExAO_contrast
RD
Sep 2020

This is the starting point to run the Subaru_SCExAO prescription. From here, you can turn on/off toggles, change AO
settings, view different planes, and make other changes to running the prescription without changing the base
prescription or the default params themselves.

"""
import numpy as np
import os
import copy
import matplotlib.pyplot as plt

from vip_hci import phot, metrics, pca
import vip_hci.metrics.contrcurve as contrcurve

from medis.params import sp, tp, iop, ap, mp
from medis.utils import dprint
import medis.optics as opx
from medis.plot_tools import view_spectra, view_timeseries, quick2D, plot_planes
import medis.medis_main as mm

#################################################################################################
#################################################################################################
#################################################################################################
testname = 'SCExAO-test2'
iop.update_datadir(f"/home/captainkay/mazinlab/MKIDSim/CDIsim_data/")
iop.update_testname(testname)
iop.makedir()

# Telescope
tp.prescription = 'Subaru_SCExAO'
tp.entrance_d = 7.9716
tp.flen_primary = tp.entrance_d * 13.612

# Simulation & Timing
sp.numframes = 1
sp.closed_loop = False

# Grid Parameters
sp.focused_sys = True
sp.beam_ratio = 0.08  # parameter dealing with the sampling of the beam in the pupil/focal plane
sp.grid_size = 512  # creates a nxn array of samples of the wavefront
sp.maskd_size = 256  # will truncate grid_size to this range (avoids FFT artifacts) # set to grid_size if undesired

# Companion
ap.companion = True
ap.contrast = [5e-2]
ap.companion_xy = [[5, -5]]  # units of this are lambda/tp.entrance_d
ap.n_wvl_init = 3  # initial number of wavelength bins in spectral cube (later sampled by MKID detector)
ap.n_wvl_final = None  # final number of wavelength bins in spectral cube after interpolation (None sets equal to n_wvl_init)
ap.interp_wvl = False  # Set to interpolate wavelengths from ap.n_wvl_init to ap.n_wvl_final
ap.wvl_range = np.array([800, 1400]) / 1e9  # wavelength range in [m] (formerly ap.band)
# eg. DARKNESS band is [800, 1500], J band =  [1100,1400])


# Toggles for Aberrations and Control
tp.obscure = False
tp.use_atmos = True
tp.use_aber = False
tp.use_ao = True
tp.use_cdi = False

# Plotting
sp.show_wframe = False  # plot white light image frame
sp.show_spectra = True  # Plot spectral cube at single timestep
sp.spectra_cols = 3  # number of subplots per row in view_spectra
sp.show_tseries = False  # Plot full timeseries of white light frames
sp.tseries_cols = 5  # number of subplots per row in view_timeseries
sp.show_planes = True
sp.verbose = True
sp.debug = False

# Saving
sp.save_to_disk = False  # save obs_sequence (timestep, wavelength, x, y)
sp.save_list = ['entrance_pupil','woofer', 'tweeter', 'post-DM-focus', 'coronagraph', 'detector']  # list of locations in optics train to save

def get_unoccult_psf(name):
    # sp_orig = copy.deepcopy(sp)
    # ap_orig = copy.deepcopy(ap)
    # tp_orig = copy.deepcopy(tp)
    # iop_orig = copy.deepcopy(iop)

    params = [sp, ap, tp, iop]
    save_state = [copy.deepcopy(param) for param in params]

    sp.save_fields = True
    ap.companion = False
    tp.cg_type = None
    sp.numframes = 1
    ap.sample_time = 1e-3
    sp.save_list = ['detector']
    sp.save_to_disk = True
    iop.telescope = name + '.pkl'
    iop.fields = name + '.h5'

    telescope = mm.Telescope(usesave=True)
    cpx_sequence = telescope()['fields']

    # sp.__dict__ = sp_orig.__dict__
    # ap.__dict__ = ap_orig.__dict__
    # tp.__dict__ = tp_orig.__dict__
    # iop.__dict__ = iop_orig.__dict__
    for i, param in enumerate(params):
        param.__dict__ = save_state[i].__dict__

    # psf_template = np.abs(fields[0, -1, :, 0, 1:, 1:]) ** 2
    psf_template = np.abs(cpx_sequence[0, -1, :, 0]) ** 2
    # grid(psf_template, logZ=True)

    return psf_template

def get_contrast(cpx_sequence):
    unoccultname = os.path.join(iop.testdir, f'telescope_unoccult')
    psf_template = get_unoccult_psf(unoccultname)

    wsamples = np.linspace(ap.wvl_range[0], ap.wvl_range[1], ap.n_wvl_final)
    scale_list = wsamples / (ap.wvl_range[1] - ap.wvl_range[0])

    algo_dict = {'scale_list': scale_list}

    median_fwhm = mp.lod
    median_wave = (wsamples[-1] + wsamples[0]) / 2
    fwhm = median_fwhm * wsamples / median_wave

    fourcube = np.abs(np.sum(cpx_sequence[:, -1], axis=2)) ** 2  # select detector plane and sum over objects
    fourcube = np.transpose(fourcube, axes=(1,0,2,3))
    fulloutput = contrcurve.contrast_curve(cube=fourcube,  # 4D cube
                              algo=pca.pca,  # does SDI (and ADI but no rotation so just median collapse)
                              nbranch=3,  # number of simultaneous fake companions used for the throughput calculation
                              fc_snr=10,   # brightness of the fake companions
                              angle_list=np.zeros((fourcube.shape[1])),  # angle of each time frame is 0 (no rotation)
                              psf_template=psf_template[:, 1:, 1:],  # unocculted
                              interp_order=2,  # interpolation of the throughput curve
                              fwhm=fwhm,  # determines distances between fake companions and sampling of x axis
                              pxscale=mp.platescale / 1000,  # units of x axis
                              starphot=1.1,  # scaling the y axis. use psf_template to get brightness
                              adimsdi='double',  # ADI and SDI
                              ncomp2=None,  # no PCAs used in ADI algo
                              debug=True, plot=True,  # debug plots are handy
                              theta=0,
                              full_output=True,
                              **algo_dict)

    return fulloutput[0]['distance_arcsec'], fulloutput[0]['sensitivity_student']

if __name__ == '__main__':
    # =======================================================================
    # Run it!!!!!!!!!!!!!!!!!
    # =======================================================================
    sim = mm.RunMedis(name=testname, product='fields')
    observation = sim()
    cpx_sequence = observation['fields']
    sampling = observation['sampling']

    # =======================================================================
    # Focal Plane Processing
    # =======================================================================
    # cpx_sequence = (n_timesteps ,n_planes, n_waves_init, n_astro_bodies, nx ,ny)
    cpx_sequence = opx.interp_wavelength(cpx_sequence, 2)  # interpolate over wavelength
    focal_plane = opx.extract_plane(cpx_sequence, 'detector')  # eliminates astro_body axis
    # convert to intensity THEN sum over object, keeping the dimension of tstep even if it's one
    focal_plane = np.sum(opx.cpx_to_intensity(focal_plane), axis=2)
    fp_sampling = sampling[-1,:]

    # =======================================================================
    # Differential Imaging, throughput calculation, and contrast plotting
    # =======================================================================
    sep, contrast = get_contrast(cpx_sequence)

    plt.plot(sep, contrast)
    plt.show()