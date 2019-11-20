"""
This is the main configuration file. It contains default global variables (as in they are read in by the relevant
modules) that define the parameters of the whole telescope system. These parameters can be redefined at the beginning
of the example module the user is running

Unless otherwise specified, units shall be given in:
distance: meters
time: seconds

"""

import numpy as np
import os
import proper

class IO_params:
    """
    Define file tree/structure to import and save data

    telescope.gridsize and telescope.beam_ratio affect the sampling of the simulation, so are included in filenames
    of files genereated based on the sampling parameters, and simulation_params.numframes affects how many files need
    to be generated, so are also included in filenames
    """

    def __init__(self, testname='Subaru-test1'):  # testname should be the name of the particular example you are running,
                                              # for example 'BetaPic' or 'simple_telescope'
        self.rootdir = '/home/captainkay/mazinlab/MKIDSim/miniMEDIS/'
        self.datadir = '/home/captainkay/mazinlab/MKIDSim/CDIsim_data/'

        # Unprocessed Science Data
        self.sciroot = 'mini_medis'
        self.scidir = os.path.join(self.datadir, self.sciroot)  # self.savedata
        self.testname = testname  # set this up in the definition line, but can update it with iop.update('newname')
        self.testdir = os.path.join(self.scidir,
                                  self.testname)  # Save results in new sub-directory
        self.obs_seq = os.path.join(self.testdir,
                                  'ObsSeq.h5')  # a x/y/t/w cube of data
        self.fields = os.path.join(self.testdir, 'fields.h5')
        self.obs_table = os.path.join(self.testdir,
                                    'ObsTable.h5')  # a photon table with 4 columns

        # Aberration Metadata
        self.aberroot = 'aberrations'
        self.aberdata = f"gridsz{sp.grid_size}_bmratio{sp.beam_ratio}_tsteps{sp.numframes}"
        self.aberdir = os.path.join(self.testdir, self.aberroot, self.aberdata)
        self.NCPA_meas = os.path.join(self.aberdir, 'NCPA_meas.pkl')  #
        self.CPA_meas = os.path.join(self.aberdir, 'CPA_meas.pkl')

        # Atmosphere Metadata
        self.atmosroot = f"atmos/gridsz{sp.grid_size}_bmratio{sp.beam_ratio}_tsteps{sp.numframes}"
        self.atmosdir = os.path.join(self.testdir, self.atmosroot)  # full path to FITS files

    def update(self, new_name='example1'):
        self.__init__(testname=new_name)

    def makedir(self):
        #print(self.datadir, self.testdir,  self.scidir)
        if not os.path.isdir(self.datadir):
            os.makedirs(self.datadir, exist_ok=True)
        if not os.path.isdir(self.testdir):
            os.makedirs(self.testdir, exist_ok=True)
        if not os.path.isdir(self.scidir):
            os.makedirs(self.scidir, exist_ok=True)
        if not os.path.isdir(self.aberdir):
            os.makedirs(self.aberdir, exist_ok=True)
        if not os.path.isdir(self.atmosdir):
            os.makedirs(self.atmosdir, exist_ok=True)

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def __name__(self):
        return self.__str__().split(' ')[0].split('.')[-1]


class Simulation_params:
    """
    Default parameters for outputs of the simulation. What plots you want to see etc

    """
    def __init__(self):
        self.timing = True  # True will print timing statements in run_medis()
        self.num_processes = 1  # multiprocessing.cpu_count()

        # Grid Sizing/Sampling Params
        self.beam_ratio = 0.14  # parameter dealing with the sampling of the beam in the pupil/focal
                                # plane vs grid size. See Proper manual pgs 36-37 and 71-74 for discussion
        self.grid_size = 512  # creates a nxn array of samples of the wavefront
                              # must be bigger than the beam size to avoid FT effects at edges; must be factor of 2
                              # NOT the size of your detector/# of pixels
        self.maskd_size = 512  # will truncate grid_size to this range (avoids FFT artifacts)
                               # set to grid_size if undesired
        self.focused_sys = True  # use this to turn scaling of beam ratio by wavelength on/off
                        # turn on/off as necessary to ensure optics in focal plane have same sampling at each
                        # wavelength. Can check focal plane sampling in the proper perscription with opx.check_sampling
                        # see Proper manual pg 36 for more info

        # Timing Params
        self.sample_time = 0.01  # [s] seconds per timestep/frame. used in atmosphere evolution, etc
        self.startframe = 0  # useful for things like RDI
        self.numframes = 1  # number of timesteps in the simulation

        # Plotting Params
        self.show_cube = False  # Plot datacube
        self.show_wframe = True  # Plot image frame
        self.show_tseries = False
        self.spectra_cols = 3  # number of subplots per row in view_datacube
        self.tseries_cols = 4  # number of subplots per row in view_timeseries

        # Reading/Saving Params
        self.save_obs = False
        self.save_cube = False  #
        self.get_ints = False
        self.save_locs = None

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def __name__(self):
        return self.__str__().split(' ')[0].split('.')[-1]


class Astro_params:
    """
    Default parameters for the astronomical system under investigation
    exposure_time, startframe and numframes may seem a bit out of place here. Perhaps this class could be renamed
    """
    def __init__(self):
        # Companion Object Params
        self.star_photons = int(1e5)  # A 5 apparent mag star 1e6 cts/cm^2/s
        self.companion = False
        self.contrast = [0.05]
        self.C_spec = 1.5  # the gradient of the increase in contrast towards shorter wavelengths
        self.companion_locations = [[-1.0e-6, 1.0e-6]]  # [m] initial location (no rotation)

        # Wavelength and Wavefront Array Settings
        # In optics_propagate(), proper initially takes N  discreet wavelengths evenly spaced in wvl_range, where N is
        # given by ap.n_wvl_init. Later, in gen_timeseries(), the 3rd axis of the spectral cube is interpolated so that
        # there are ap.n_wvl_final over the range in ap.wvl_range.
        self.n_wvl_init = 3  # initial number of wavelength bins in spectral cube (later sampled by MKID detector)
        self.n_wvl_final = 9  # final number of wavelength bins in spectral cube after interpolation
        self.interp_wvl = True  # Set to interpolate wavelengths from ap.n_wvl_init to ap.n_wvl_final
        self.wvl_range = np.array([810, 1500]) / 1e9  # wavelength range in [m] (formerly ap.band)
            # eg. DARKNESS band is [800, 1500], J band =  [1100,1400])

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def __name__(self):
        return self.__str__().split(' ')[0].split('.')[-1]


class Telescope_params:
    """
    This contains most of the parameters you will probably modify when running tests
    """
    def __init__(self):
        # Optics + Detector
        self.prescription = 'Subaru_frontend'
        self.enterance_d = 7.9716  # 7.971  # [m] telescope enterence pupil diameter in meters
        self.fnum_primary = 13.612  # f-number of primary
        self.flen_primary = 108.5124  # [m] focal length of primary

        # AO System Settings
        self.use_ao = True
        self.ao_act = 60  # number of actuators on the DM on one axis (proper only models nxn square DMs)
        self.piston_error = False  # flag for allowing error on DM surface shape
        self.fit_dm = True  # flag to use DM surface fitting (see proper manual pg 52, the FIT switch)

        # Ideal Detector Params (not bothering with MKIDs yet)
        self.detector = 'ideal'  # 'MKIDs'
        self.array_size = np.array([129, 129])  # np.array([125,80])
        self.wavecal_coeffs = [1. / 12, -157]  # assume linear for now 800nm = -90deg, 1500nm = -30deg
                                                # used to make phase cubes. I assume this has something to do with the
                                                # QE of MKIDs at different wavelengths?
        self.pix_shift = [0, 0]  # False?  Shifts the central star to off-axis (mimics conex mirror, tip/tilt error)
        # self.platescale = 13.61  # mas # have to run get_sampling at the focus to find this

        # Aberrations
        self.servo_error = [0, 1]  # [0,1] # False # No delay and rate of 1/frame_time
        self.abertime = 0.5  # time scale of optic aberrations in seconds
        self.aber_params = {'QuasiStatic': False,  # or 'Static'
                            'Phase': True,
                            'Amp': False}
                            # Coefficients used to calcuate PSD errormap in Proper (see pg 56 in manual)
                            # only change these if making new aberration maps
        self.aber_vals = {'a': [7.2e-17, 3e-17],  # power at low spatial frequencies (m4)
                           'b': [0.8, 0.2],  # correlation length (b/2pi defines knee)
                           'c': [3.1, 0.5],  #
                           'a_amp': [0.05, 0.01]}
        # Zernike Settings- see pg 192 for details
        self.zernike_orders = [2, 3, 4]  # Order of Zernike Polynomials to include
        self.zernike_vals = np.array([175, -150, 200])*1.0e-9  # value of Zernike order in nm,
                                                               # must be same length as zernike_orders

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def __name__(self):
        return self.__str__().split(' ')[0].split('.')[-1]


class CDI_params():
    def __init__(self):
        self.use_cdi = False
        self.show_probe = True  # False flag to plot phase probe or not

        self.phs_intervals = np.pi/4  # [rad] phase interval over [0, 2pi]
        self.phase_list = np.arange(0, 2 * np.pi, self.phs_intervals)  # FYI not inclusive of 2pi endpoint
        self.n_probes = len(self.phase_list)  # number of phase probes
        self.phase_integration_time = 0.01  # [s]
        self.null_time = 0.1  # [s]
        self.probe_type = "pairwise"

        # Probe Dimensions (extent in pupil plane coordinates)
        self.probe_w = 20  # [actuator coordinates] width of the probe
        self.probe_h = 20  # [actuator coordinates] height of the probe
        self.probe_center = 15  # [actuator coordinates] center position of the probe
        self.probe_amp = 2e-5  # [m] probe amplitude, scale should be in units of actuator height limits

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def __name__(self):
        return self.__str__().split(' ')[0].split('.')[-1]


class Atmos_params():
    """
    default params for making atmospheric models

    original use was to use CAOS, but has now been changed to use hcipy (2019)
    :model: 'single', 'hcipy_standard', 'evolving'
             evolving->apply variation to some parameter

    hcipy still assumes frozen flow as turbulent layers. more here: https://hcipy.readthedocs.io/en/latest/index.html
    """
    def __init__(self):
        self.model = 'single'  # single|hcipy_standard|evolving
        self.cn_sq = 0.22 * 1e-12  # magnitude of perturbations within single atmos layer, at single wavelength
        self.L0 = 10  # outer scale of the model that sets distance of layers (not boundary). used in Kalmogorov model
        self.vel = 5  # velocity of the layer in m/s
        self.h = 100  # scale height in m
        self.fried = 0.2  # m

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def __name__(self):
        return self.__str__().split(' ')[0].split('.')[-1]


# Creating class structures
ap = Astro_params()
tp = Telescope_params()
sp = Simulation_params()
atmp = Atmos_params()
cdip = CDI_params()
iop = IO_params()  # Must call this last since IO_params uses ap and tp

# Turning off messages from Proper
proper.print_it = False
proper.use_cubic_conv = True
# print(proper.__version__)
# proper.prop_init_savestate()


