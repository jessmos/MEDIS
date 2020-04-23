"""
MKIDs.py

is a function under run_medis class

Rupert adds all of his stuff here.
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
import pickle
import random

from .distribution import *
from medis.params import mp, ap, tp, iop, dp, sp
from medis.utils import dprint
from medis.plot_tools import view_spectra
from medis.telescope import Telescope


class Camera():
    def __init__(self, params):
        """
        Creates a simulation for the MKID Camera to create a series of photons

        During initialisation a backup of the pthon PROPER prescription is copied to the testdir, atmisphere maps and
        aberration maps are created, serialisation and memory requirements are tested, and the cdi vairable initialised

        Resulting file structure:
        datadir
            testdir
                params.pkl         <--- input
                fields.h5          <--- input
                camera             <--- new
                    devicesave.pkl <--- new
                photonlist.h5      <--- output


        input
        params dict
            collection of the objects in params.py
        fields ndarray complex
            complex tensor of dimensions (n_timesteps, n_saved_planes, n_wavelengths, n_stars/planets, grid_size, grid_size)

        :return:
        either photon list or rebinned cube

        """
        self.params = params

        self.save_exists = True if os.path.exists(self.params['iop'].camera) else False

        if self.save_exists:
            with open(self.params['iop'].camera, 'rb') as handle:
                load = pickle.load(handle)
                self.__dict__ = load.__dict__
                self.save_exists = True

        else:
            # create fields
            telescope_sim = Telescope(self.params)
            dataproduct = telescope_sim()
            self.fields = dataproduct['fields']

            # create device
            self.dp = self.create_device()


    def create_device(self):
        # det = Detector(params['mp'])  # either loads or initialises dp
        # output = det(fields)  # loops over time and calls get_packets on each one
        # self.QE_map = None
        # self.Rs = None
        # self.sigs = None
        # self.basesDeg = None
        # self.hot_pix = None
        # self.dark_pix_frac = None
        # dp = device()
        self.platescale = mp.platescale
        self.array_size = mp.array_size
        self.dark_pix_frac = mp.dark_pix_frac
        self.hot_pix = mp.hot_pix
        self.lod = mp.lod
        self.QE_map_all = array_QE(plot=False)
        self.responsivity_error_map = responvisity_scaling_map(plot=False)
        if mp.pix_yield == 1:
            mp.bad_pix = False
        if mp.bad_pix == True:
            self.QE_map = create_bad_pix(self.QE_map_all)
            # self.QE_map = create_hot_pix(self.QE_map)
            # quick2D(self.QE_map_all)
            if mp.dark_counts:
                self.dark_per_step = sp.sample_time * mp.dark_bright * mp.array_size[0] * mp.array_size[
                    1] * self.dark_pix_frac
                self.dark_locs = create_false_pix(mp, amount=int(
                    mp.dark_pix_frac * mp.array_size[0] * mp.array_size[1]))
            if mp.hot_pix:
                self.hot_per_step = sp.sample_time * mp.hot_bright * self.hot_pix
                self.hot_locs = create_false_pix(mp, amount=mp.hot_pix)
            # self.QE_map = create_bad_pix_center(self.QE_map)
        self.Rs = assign_spectral_res(plot=False)
        self.sigs = get_R_hyper(self.Rs, plot=False)
        # get_phase_distortions(plot=True)
        if mp.phase_background:
            self.basesDeg = assign_phase_background(plot=False)
        else:
            self.basesDeg = np.zeros((mp.array_size))
        with open(iop.device, 'wb') as handle:
            pickle.dump(dp, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('Initialized MKID array parameters')

        return dp


    def __call__(self, *args, **kwargs):
        if not self.save_exists:

            photons = np.empty((0, 4))
            stackcube = np.zeros((len(self.fields), self.params['ap'].n_wvl_final, mp.array_size[1], mp.array_size[0]))
            for step in range(len(self.fields)):
                print('step', step)
                spectralcube = np.abs(np.sum(self.fields[step, -1, :, :], axis=1)) ** 2
                # view_spectra(spectralcube, logZ=True)
                step_packets = self.get_packets(spectralcube, step, dp, mp)
                photons = np.vstack((photons, step_packets))
                cube = self.make_datacube_from_list(step_packets, (self.params['ap'].n_wvl_final, dp.array_size[0], dp.array_size[1]))
                stackcube[step] = cube

            with open(self.params['iop'].photons, 'wb') as handle:
                pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

        dataproduct = {'photons': self.photons, 'stackcube': self.stackcube}

        return dataproduct



class Detector():
    def __init__(self, mp):
        # self.QE_map = None
        # self.Rs = None
        # self.sigs = None
        # self.basesDeg = None
        # self.hot_pix = None
        # self.dark_pix_frac = None
        # dp = device()
        self.platescale = mp.platescale
        self.array_size = mp.array_size
        self.dark_pix_frac = mp.dark_pix_frac
        self.hot_pix = mp.hot_pix
        self.lod = mp.lod
        self.QE_map_all = array_QE(plot=False)
        self.responsivity_error_map = responvisity_scaling_map(plot=False)
        if mp.pix_yield == 1:
            mp.bad_pix = False
        if mp.bad_pix == True:
            self.QE_map = create_bad_pix(self.QE_map_all)
            # self.QE_map = create_hot_pix(self.QE_map)
            # quick2D(self.QE_map_all)
            if mp.dark_counts:
                self.dark_per_step = sp.sample_time * mp.dark_bright * mp.array_size[0] * mp.array_size[1] * self.dark_pix_frac
                self.dark_locs = create_false_pix(mp, amount = int(mp.dark_pix_frac*mp.array_size[0]*mp.array_size[1]))
            if mp.hot_pix:
                self.hot_per_step = sp.sample_time * mp.hot_bright * self.hot_pix
                self.hot_locs = create_false_pix(mp, amount = mp.hot_pix)
            # self.QE_map = create_bad_pix_center(self.QE_map)
        self.Rs = assign_spectral_res(plot=False)
        self.sigs = get_R_hyper(self.Rs, plot=False)
        # get_phase_distortions(plot=True)
        if mp.phase_background:
            self.basesDeg = assign_phase_background(plot=False)
        else:
            self.basesDeg = np.zeros((mp.array_size))
        with open(iop.device, 'wb') as handle:
            pickle.dump(dp, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('Initialized MKID array parameters')

        return dp

    def arange_into_cube(packets, size):
        # print 'Sorting packets into xy grid (no phase or time sorting)'
        cube = [[[] for i in range(size[0])] for j in range(size[1])]
        for ip, p in enumerate(packets):
            x = np.int_(p[2])
            y = np.int_(p[3])
            cube[x][y].append([p[0] ,p[1]])
            if len(packets) >= 1e7 and ip % 1000:
                misc.progressBar(value=ip, endvalue=len(packets))
        # print cube[x][y]
        # cube = time_sort(cube)
        return cube


    def responvisity_scaling_map(plot=False):
        """Assigns each pixel a phase responsivity between 0 and 1"""
        dist = Distribution(gaussian(mp.r_mean, mp.r_sig, np.linspace(0, 2, mp.res_elements)), interpolation=True)
        responsivity = dist(mp.array_size[0] * mp.array_size[1])[0]/float(mp.res_elements) * 2
        if plot:
            plt.xlabel('Responsivity')
            plt.ylabel('#')
            plt.hist(responsivity)
            plt.show()
        responsivity = np.reshape(responsivity, mp.array_size[::-1])
        if plot:
            quick2D(responsivity)#plt.imshow(QE)
            # plt.show()

        return responsivity

    def array_QE(plot=False):
        """Assigns each pixel a phase responsivity between 0 and 1"""
        dist = Distribution(gaussian(mp.g_mean, mp.g_sig, np.linspace(0, 1, mp.res_elements)), interpolation=True)
        QE = dist(mp.array_size[0] * mp.array_size[1])[0]/float(mp.res_elements)
        if plot:
            plt.xlabel('Responsivity')
            plt.ylabel('#')
            plt.hist(QE)
            plt.show()
        QE = np.reshape(QE, mp.array_size[::-1])
        if plot:
            quick2D(QE)#plt.imshow(QE)
            # plt.show()

        return QE

    def assign_spectral_res(plot=False):
        """Assigning each pixel a spectral resolution (at 800nm)"""
        dist = Distribution(gaussian(0.5, 0.25, np.linspace(-0.2, 1.2, mp.res_elements)), interpolation=True)
        # dprint(f"Mean R = {mp.R_mean}")
        Rs = (dist(mp.array_size[0]*mp.array_size[1])[0]/float(mp.res_elements)-0.5)*mp.R_sig + mp.R_mean#
        if plot:
            plt.xlabel('R')
            plt.ylabel('#')
            plt.hist(Rs)
            plt.show()
        Rs = np.reshape(Rs, mp.array_size)

        if plot:
            plt.figure()
            plt.imshow(Rs)
            plt.show()
        return Rs

    def get_R_hyper(Rs, plot=False):
        """Each pixel of the array has a matrix of probabilities that depends on the input wavelength"""
        print('Creating a cube of R standard deviations')
        m = (mp.R_spec*Rs)/(ap.wvl_range[1] - ap.wvl_range[0]) # looses R of 10% over the 700 wvl_range
        c = Rs-m*ap.wvl_range[0] # each c depends on the R @ 800
        waves = np.ones((np.shape(m)[1],np.shape(m)[0],ap.n_wvl_final+5))*np.linspace(ap.wvl_range[0],ap.wvl_range[1],ap.n_wvl_final+5)
        waves = np.transpose(waves) # make a tensor of 128x128x10 where every 10 vector is 800... 1500
        R_spec = m * waves + c # 128x128x10 tensor is now lots of simple linear lines e.g. 50,49,.. 45
        # probs = np.ones((np.shape(R_spec)[0],np.shape(R_spec)[1],np.shape(R_spec)[2],
        #                 mp.res_elements))*np.linspace(0, 1, mp.res_elements)
        #                         # similar to waves but 0... 1 using 128 elements
        # R_spec = np.repeat(R_spec[:,:,:,np.newaxis], mp.res_elements, 3) # creat 128 repeats of R_spec so (10,128,128,128)
        # mp.R_probs = gaussian(0.5, R_spec, probs) #each xylocation is gaussian that gets wider for longer wavelengths
        sigs_w = (waves/R_spec)/2.35 #R = w/dw & FWHM = 2.35*sig

        # plt.plot(range(0,1500),spec.phase_cal(np.arange(0,1500)))
        # plt.show()
        sigs_p = phase_cal(sigs_w) - phase_cal(np.zeros_like((sigs_w)))

        if plot:
            plt.figure()
            plt.plot(R_spec[:, 0, 0])
            plt.plot(R_spec[:,50,15])
            # plt.figure()
            # plt.plot(sigs_w[:,0,0])
            # plt.plot(sigs_w[:,50,15])
            # plt.figure()
            # plt.plot(sigs_p[:, 0, 0])
            # plt.plot(sigs_p[:, 50, 15])
            # plt.figure()
            # plt.imshow(sigs_p[:,:,0], aspect='auto')
            view_spectra(sigs_w, show=False)
            view_spectra(sigs_p, show=False)
            plt.figure()
            plt.plot(np.mean(sigs_w, axis=(1,2)))
            plt.figure()
            plt.plot(np.mean(sigs_p, axis=(1,2)))
            # plt.imshow(mp.R_probs[:,0,0,:])
            plt.show(block=True)
        return sigs_p

    def apply_phase_scaling(photons, ):
        """
        From things like resonator Q, bias power, quasiparticle losses

        :param photons:
        :return:
        """

    def apply_phase_offset_array(photons, sigs):
        """
        From things like IQ phase offset noise

        :param photons:
        :param sigs:
        :return:
        """
        wavelength = wave_cal(photons[1])

        idx = wave_idx(wavelength)

        bad = np.where(np.logical_or(idx>=len(sigs), idx<0))[0]

        photons = np.delete(photons, bad, axis=1)
        idx = np.delete(idx, bad)

        distortion = np.random.normal(np.zeros((photons[1].shape[0])),
                                      sigs[idx,np.int_(photons[3]), np.int_(photons[2])])

        photons[1] += distortion

        return photons, idx

    def apply_phase_distort(phase, loc, sigs):
        """
        Simulates phase height of a real detector system per photon
        proper will spit out the true phase of each photon it propagates. this function will give it
        a 'measured' phase based on the resolution of the detector, and a Gaussian distribution around
        the center of the resolution bin

        :param phase: real(exact) phase information from Proper
        :param loc:
        :param sigs:
        :return: distorted phase
        """
        # phase = phase + mp.phase_distortions[ip]
        wavelength = wave_cal(phase)
        idx = wave_idx(wavelength)

        if phase != 0 and idx<len(sigs):
            phase = np.random.normal(phase,sigs[idx,loc[0],loc[1]],1)[0]
        return phase

    def wave_idx(wavelength):
        m = float(ap.n_wvl_final-1)/(ap.wvl_range[1] - ap.wvl_range[0])
        c = -m*ap.wvl_range[0]
        idx = wavelength*m + c
        # return np.int_(idx)
        return np.int_(np.round(idx))

    def phase_cal(wavelengths):
        '''Wavelength in nm'''
        phase = mp.wavecal_coeffs[0]*wavelengths + mp.wavecal_coeffs[1]
        return phase

    def wave_cal(phase):
        wave = (phase - mp.wavecal_coeffs[1])/(mp.wavecal_coeffs[0])
        return wave

    def assign_phase_background(plot=False):
        """assigns each pixel a baseline phase"""
        dist = Distribution(gaussian(0.5, 0.25, np.linspace(-0.2, 1.2, mp.res_elements)), interpolation=True)

        basesDeg = dist(mp.array_size[0]*mp.array_size[1])[0]/float(mp.res_elements)*mp.bg_mean/mp.g_mean
        if plot:
            plt.xlabel('basesDeg')
            plt.ylabel('#')
            plt.title('Background Phase')
            plt.hist(basesDeg)
            plt.show(block=True)
        basesDeg = np.reshape(basesDeg, mp.array_size)
        if plot:
            plt.title('Background Phase--Reshaped')
            plt.imshow(basesDeg)
            plt.show(block=True)
        return basesDeg


    def create_bad_pix(QE_map_all, plot=False):
        amount = int(mp.array_size[0]*mp.array_size[1]*(1.-mp.pix_yield))

        bad_ind = random.sample(list(range(mp.array_size[0]*mp.array_size[1])), amount)

        dprint(f"Bad indices = {len(bad_ind)}, # MKID pix = { mp.array_size[0]*mp.array_size[1]}, "
               f"Pixel Yield = {mp.pix_yield}, amount? = {amount}")

        # bad_y = random.sample(y, amount)
        bad_y = np.int_(np.floor(bad_ind/mp.array_size[1]))
        bad_x = bad_ind % mp.array_size[1]

        # dprint(f"responsivity shape  = {responsivities.shape}")
        QE_map = np.array(QE_map_all)

        QE_map[bad_x, bad_y] = 0
        if plot:
            plt.xlabel('responsivities')
            plt.ylabel('?')
            plt.title('Something Related to Bad Pixels')
            plt.imshow(QE_map)
            plt.show()

        return QE_map

    def create_bad_pix_center(responsivities):
        res_elements=mp.array_size[0]
        # responsivities = np.zeros()
        for x in range(mp.array_size[1]):
            dist = Distribution(gaussian(0.5, 0.25, np.linspace(0, 1, mp.res_elements)), interpolation=False)
            dist = np.int_(dist(int(mp.array_size[0]*mp.pix_yield))[0])#/float(mp.res_elements)*np.int_(mp.array_size[0]) / mp.g_mean)
            # plt.plot(dist)
            # plt.show()
            dead_ind = []
            [dead_ind.append(el) for el in range(mp.array_size[0]) if el not in dist]
            responsivities[x][dead_ind] = 0

        return responsivities

    def get_bad_packets(dp, step, type='hot'):
        if type == 'hot':
            n_device_counts = self.hot_per_step
        elif type == 'dark':
            n_device_counts = self.dark_per_step
        else:
            print("type currently has to be 'hot' or 'dark'")
            raise AttributeError

        if n_device_counts % 1 > np.random.uniform(0,1,1):
            n_device_counts += 1

        n_device_counts = int(n_device_counts)
        photons = np.zeros((4, n_device_counts))
        if n_device_counts > 0:
            if type == 'hot':
                phases = np.random.uniform(-120, 0, n_device_counts)
                hot_ind = np.random.choice(range(len(self.hot_locs[0])), n_device_counts)
                bad_pix = self.hot_locs[:, hot_ind]
            elif type == 'dark':
                dist = Distribution(gaussian(0, 0.25, np.linspace(0, 1, mp.res_elements)), interpolation=False)
                phases = dist(n_device_counts)[0]
                max_phase = max(phases)
                phases = -phases*120/max_phase
                bad_pix_options = create_false_pix(dp, self.dark_pix_frac * self.array_size[0] * self.array_size[1])
                bad_ind = np.random.choice(range(len(bad_pix_options[0])), n_device_counts)
                bad_pix = bad_pix_options[:, bad_ind]

            meantime = step * sp.sample_time
            photons[0] = np.random.uniform(meantime - sp.sample_time / 2, meantime + sp.sample_time / 2, len(photons[0]))
            photons[1] = phases
            photons[2:] = bad_pix

        return photons

    def create_false_pix(dp, amount):
        # dprint(f"amount = {amount}")
        bad_ind = random.sample(list(range(self.array_size[0]*self.array_size[1])), int(amount))
        bad_y = np.int_(np.floor(bad_ind / self.array_size[1]))
        bad_x = bad_ind % self.array_size[1]

        return np.array([bad_x, bad_y])


    def remove_bad(frame, QE):
        bad_map = np.ones((sp.grid_size,sp.grid_size))
        bad_map[QE[:-1,:-1]==0] = 0
        # quick2D(QE, logAmp =False)
        # quick2D(bad_map, logAmp =False)
        frame = frame*bad_map
        return frame

    def sample_cube(datacube, num_events):
        # print 'creating photon data from reference cube'

        dist = Distribution(datacube, interpolation=True)

        photons = dist(num_events)
        return photons

    def assign_calibtime(photons, step):
        meantime = step*sp.sample_time
        # photons = photons.astype(float)#np.asarray(photons[0], dtype=np.float64)
        # photons[0] = photons[0] * ps.mp.frame_time
        timedist = np.random.uniform(meantime-sp.sample_time/2, meantime+sp.sample_time/2, len(photons[0]))
        photons = np.vstack((timedist, photons))
        return photons

    def calibrate_phase(photons):
        """
        idx -> phase

        :param photons:
        :return:
        """
        photons = np.array(photons)
        c = ap.wvl_range[0]
        m = (ap.wvl_range[1] - ap.wvl_range[0])/ap.n_wvl_final
        wavelengths = photons[0]*m + c
        # dprint(wavelengths[:5])
        # photons[0] = wavelengths*mp.wavecal_coeffs[0] + mp.wavecal_coeffs[1]
        photons[0] = phase_cal(wavelengths)
        # dprint(photons[0,:5])
        # exit()

        return photons

    def make_datacube_from_list(packets, shape):
        phase_band = phase_cal(ap.wvl_range)
        bins = [np.linspace(phase_band[0], phase_band[1], shape[0] + 1),
                range(shape[1]+1),
                range(shape[2]+1)]
        datacube, _ = np.histogramdd(packets[:,1:], bins)

        return datacube

    # def load_fields(fields_file):
    #     """
    #     Take the continuously saved sequence of fivecubes and load a sixcube into memory
    #     :return:
    #     """
    #     with h5py.File(fields_file, 'r') as hf:
    #         keys = list(hf.keys())
    #         if 'data' in keys:
    #             dprint('File in old format. Loading fields straight in')
    #             fields = hf.get('data')[:]
    #         else:
    #             try:
    #                 step_shape = hf.get('t0').shape
    #             except AttributeError:
    #                 print(f'Time 0 dataset does not exist for {fields_file}. Try recreating it')
    #                 raise AttributeError
    #             fields = np.zeros(
    #                 (len(keys), step_shape[0], step_shape[1], step_shape[2], step_shape[3], step_shape[4]),
    #                 dtype=np.complex64)
    #             for t in range(len(keys)):
    #                 timestep = hf.get(f't{t}')
    #                 # from medis.Utils.plot_tools import view_datacube
    #                 # view_datacube(np.abs(timestep[-1,:,0]) ** 2, logAmp=True)
    #                 fields[t] = timestep
    #         if sp.verbose: print(f'fields sixcube has shape: {fields.shape}')
    #     return fields
    #
    # def save_fields(e_fields_sequence, fields_file='hyper.pkl'):
    #
    #     dprint((fields_file, fields_file[-3:], fields_file[-3:] == '.h5'))
    #     if fields_file[-3:] == 'pkl':
    #         with open(fields_file, 'wb') as handle:
    #             pickle.dump(e_fields_sequence, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     elif fields_file[-3:] == 'hdf' or fields_file[-3:] == '.h5':
    #         with h5py.File(fields_file, 'w') as hf:
    #             hf.create_dataset('data', data=e_fields_sequence)
    #             # for param in [iop, cp, tp, mp, sp, iop, dp, fp]:
    #             #     for key, value in dict(param).items():
    #             #         if type(value) == str or value is None:
    #             #             value = np.string_(value)
    #             #         try:
    #             #             hf.attrs.create(f'{param.__name__()}.{key}', value)
    #             #         except TypeError:
    #             #             print('WARNING skipping some attributes - probably the aber dictionaries or save locs')
    #     else:
    #         print('Extension not recognised')

    def get_packets(datacube, step, dp, mp, plot=False):
        if plot: view_spectra(datacube, logZ=True, extract_center=False, title='pre')

        if mp.resamp:
            nyq_sampling = ap.wvl_range[0]*360*3600/(4*np.pi*tp.entrance_d)
            sampling = nyq_sampling*sp.beam_ratio*2  # nyq sampling happens at sp.beam_ratio = 0.5
            x = np.arange(-sp.grid_size*sampling/2, sp.grid_size*sampling/2, sampling)
            xnew = np.arange(-self.array_size[0]*self.platescale/2, self.array_size[0]*self.platescale/2, self.platescale)
            mkid_cube = np.zeros((len(datacube), self.array_size[0], self.array_size[1]))
            for s, slice in enumerate(datacube):
                f = interpolate.interp2d(x, x, slice, kind='cubic')
                mkid_cube[s] = f(xnew, xnew)
            mkid_cube = mkid_cube*np.sum(datacube)/np.sum(mkid_cube)
            # view_spectra(mkid_cube, logZ=True, show=True, extract_center=False, title='post')
            datacube = mkid_cube

        datacube[datacube < 0] *= -1

        if plot: view_spectra(datacube, logZ=True)
        if mp.QE_var:
            datacube *= self.QE_map[:datacube.shape[1],:datacube.shape[1]]
        if plot: view_spectra(datacube, logZ=True)
        # if mp.hot_pix:
        #     datacube = add_hot_pix(datacube, dp, step)

        # quick2D(self.QE_map)
        if plot: view_spectra(datacube, logZ=True, show=False)
        if hasattr(dp,'star_phot'): ap.star_flux = self.star_phot
        num_events = int(ap.star_flux * sp.sample_time * np.sum(datacube))

        if sp.verbose:
            dprint(f'star flux: {ap.star_flux}, cube sum: {np.sum(datacube)}, num events: {num_events}')

        photons = sample_cube(datacube, num_events)
        photons = calibrate_phase(photons)
        photons = assign_calibtime(photons, step)

        if plot:
            cube = make_datacube_from_list(photons.T, (ap.n_wvl_final, self.array_size[0], self.array_size[1]))
            dprint(cube.shape)
            # view_spectra(cube, logZ=True)

        if mp.dark_counts:
            dark_photons = get_bad_packets(dp, step, type='dark')
            photons = np.hstack((photons, dark_photons))

        if mp.hot_pix:
            hot_photons = get_bad_packets(dp, step, type='hot')
            photons = np.hstack((photons, hot_photons))
            # stem = add_hot(stem)

        # plt.hist(photons[3], bins=25)
        # plt.yscale('log')
        # plt.show(block=True)
        # stem = arange_into_stem(photons.T, (self.array_size[0], self.array_size[1]))
        # cube = make_datacube(stem, (self.array_size[0], self.array_size[1], ap.n_wvl_final))
        # view_spectra(cube, logZ=True, vmin=0.01)

        if plot:
            cube = make_datacube_from_list(photons.T, (ap.n_wvl_final, self.array_size[0], self.array_size[1]))
            dprint(cube.shape)
            view_spectra(cube, logZ=True, title='hot pix')

        if mp.phase_uncertainty:
            photons[1] *= self.responsivity_error_map[np.int_(photons[2]), np.int_(photons[3])]
            photons, idx = apply_phase_offset_array(photons, self.sigs)
            # stem = arange_into_stem(photons.T, (self.array_size[0], self.array_size[1]))
            # cube = make_datacube(stem, (self.array_size[0], self.array_size[1], ap.n_wvl_final))
            # view_spectra(cube, logZ=True, vmin=0.01)

        # stem = arange_into_stem(photons.T, (self.array_size[0], self.array_size[1]))
        # cube = make_datacube(stem, (self.array_size[0], self.array_size[1], ap.n_wvl_final))
        # view_spectra(cube, vmin=0.01, logZ=True)
        # plt.figure()
        # plt.imshow(cube[0], origin='lower', norm=LogNorm(), cmap='inferno', vmin=1)
        # plt.show(block=True)

        # thresh =  photons[1] < self.basesDeg[np.int_(photons[3]),np.int_(photons[2])]
        if mp.phase_background:
            thresh =  -photons[1] > 3*self.sigs[-1,np.int_(photons[3]), np.int_(photons[2])]
            photons = photons[:, thresh]
        # print(thresh)

        # stem = arange_into_stem(photons.T, (mp.array_size[0], mp.array_size[1]))
        # cube = make_datacube(stem, (mp.array_size[0], mp.array_size[1], ap.n_wvl_final))
        # quick2D(cube[0], vmin=1, logZ=True)
        # plt.figure()
        # plt.imshow(cube[0], origin='lower', norm=LogNorm(), cmap='inferno', vmin=1)
        # plt.show(block=True)

        # dprint(photons.shape)
        if mp.remove_close:
            stem = arange_into_stem(photons.T, (mp.array_size[0], mp.array_size[1]))
            stem = remove_close(stem)
            photons = ungroup(stem)

        if plot:
            cube = make_datacube_from_list(photons.T, (ap.n_wvl_final, self.array_size[0], self.array_size[1]))
            dprint(cube.shape)
            view_spectra(cube, logZ=True, use_axis=False, title='remove close')
        # This step was taking a long time
        # stem = arange_into_stem(photons.T, (mp.array_size[0], mp.array_size[1]))
        # cube = make_datacube(stem, (mp.array_size[0], mp.array_size[1], ap.n_wvl_final))
        # # ax7.imshow(cube[0], origin='lower', norm=LogNorm(), cmap='inferno', vmin=1)
        # cube /= self.QE_map
        # photons = ungroup(stem)

        # dprint(photons.shape)


        # dprint("Measured photons with MKIDs")

        return photons.T
