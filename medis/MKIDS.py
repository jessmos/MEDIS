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

from medis.distribution import *
# from medis.params import mp, ap, tp, iop, dp, sp
from medis.utils import dprint
from medis.plot_tools import view_spectra
from medis.telescope import Telescope


class Camera():
    def __init__(self, params, fields=None):
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

        self.name = self.params['iop'].camera

        self.save_exists = True if os.path.exists(self.name) else False

        if self.save_exists:
            with open(self.name, 'rb') as handle:
                load = pickle.load(handle)
                self.__dict__ = load.__dict__
                self.save_exists = True

        else:
            if fields is None:
                # make fields (or load if it already exists)
                telescope_sim = Telescope(self.params)
                dataproduct = telescope_sim()
                self.fields = dataproduct['fields']
            else:
                self.fields = fields

            # create device
            self.create_device()


    def create_device(self):
        # det = Detector(params['mp'])  # either loads or initialises dp
        # output = det(fields)  # loops over time and calls get_packets on each one

        # self.QE_map = None
        # self.Rs = None
        # self.sigs = None
        # self.basesDeg = None
        # self.hot_pix = None
        # self.dark_pix_frac = None
        self.platescale = self.params['mp'].platescale
        self.array_size = self.params['mp'].array_size
        self.dark_pix_frac = self.params['mp'].dark_pix_frac
        self.hot_pix = self.params['mp'].hot_pix
        self.lod = self.params['mp'].lod
        self.QE_map_all = self.array_QE(plot=False)
        self.responsivity_error_map = self.responvisity_scaling_map(plot=False)
        if self.params['mp'].pix_yield == 1:
            self.params['mp'].bad_pix = False
        if self.params['mp'].bad_pix == True:
            self.QE_map = self.create_bad_pix(self.QE_map_all)
            # self.QE_map = create_hot_pix(self.QE_map)
            # quick2D(self.QE_map_all)
            if self.params['mp'].dark_counts:
                self.dark_per_step = self.params['sp'].sample_time * self.params['mp'].dark_bright * self.array_size[0] * self.array_size[
                    1] * self.dark_pix_frac
                self.dark_locs = self.create_false_pix(amount=int(
                    self.params['mp'].dark_pix_frac * self.array_size[0] * self.array_size[1]))
            if self.params['mp'].hot_pix:
                self.hot_per_step = self.params['sp'].sample_time * self.params['mp'].hot_bright * self.hot_pix
                self.hot_locs = self.create_false_pix(amount=self.params['mp'].hot_pix)
            # self.QE_map = create_bad_pix_center(self.QE_map)
        self.Rs = self.assign_spectral_res(plot=False)
        self.sigs = self.get_R_hyper(self.Rs, plot=False)
        # get_phase_distortions(plot=True)
        if self.params['mp'].phase_background:
            self.basesDeg = self.assign_phase_background(plot=False)
        else:
            self.basesDeg = np.zeros((self.array_size))
        # with open(iop.device, 'wb') as handle:
        #     pickle.dump(dp, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('\nInitialized MKID device parameters\n')

    def __call__(self, *args, **kwargs):
        if not self.save_exists:

            self.photons = np.empty((0, 4))
            self.stackcube = np.zeros((len(self.fields), self.params['ap'].n_wvl_final, self.array_size[1], self.array_size[0]))
            for step in range(len(self.fields)):
                print('step', step)
                spectralcube = np.abs(np.sum(self.fields[step, -1, :, :], axis=1)) ** 2
                # view_spectra(spectralcube, logZ=True)
                step_packets = self.get_packets(spectralcube, step)
                self.photons = np.vstack((self.photons, step_packets))
                cube = self.make_datacube_from_list(step_packets)
                self.stackcube[step] = cube

            self.save()

        dataproduct = {'photons': self.photons, 'stackcube': self.stackcube}

        return dataproduct

    def save(self):
        with open(self.name, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # class Detector():
    #     def __init__(self, mp):
    #         # self.QE_map = None
    #         # self.Rs = None
    #         # self.sigs = None
    #         # self.basesDeg = None
    #         # self.hot_pix = None
    #         # self.dark_pix_frac = None
    #         # dp = device()
    #         self.platescale = self.params['mp'].platescale
    #         self.array_size = self.array_size
    #         self.dark_pix_frac = self.params['mp'].dark_pix_frac
    #         self.hot_pix = self.params['mp'].hot_pix
    #         self.lod = self.params['mp'].lod
    #         self.QE_map_all = array_QE(plot=False)
    #         self.responsivity_error_map = responvisity_scaling_map(plot=False)
    #         if self.params['mp'].pix_yield == 1:
    #             self.params['mp'].bad_pix = False
    #         if self.params['mp'].bad_pix == True:
    #             self.QE_map = create_bad_pix(self.QE_map_all)
    #             # self.QE_map = create_hot_pix(self.QE_map)
    #             # quick2D(self.QE_map_all)
    #             if self.params['mp'].dark_counts:
    #                 self.dark_per_step = self.params['sp'].sample_time * self.params['mp'].dark_bright * self.array_size[0] * self.array_size[1] * self.dark_pix_frac
    #                 self.dark_locs = create_false_pix(self.params['mp'], amount = int(self.params['mp'].dark_pix_frac*self.array_size[0]*self.array_size[1]))
    #             if self.params['mp'].hot_pix:
    #                 self.hot_per_step = self.params['sp'].sample_time * self.params['mp'].hot_bright * self.hot_pix
    #                 self.hot_locs = create_false_pix(self.params['mp'], amount = self.params['mp'].hot_pix)
    #             # self.QE_map = create_bad_pix_center(self.QE_map)
    #         self.Rs = assign_spectral_res(plot=False)
    #         self.sigs = get_R_hyper(self.Rs, plot=False)
    #         # get_phase_distortions(plot=True)
    #         if self.params['mp'].phase_background:
    #             self.basesDeg = assign_phase_background(plot=False)
    #         else:
    #             self.basesDeg = np.zeros((self.array_size))
    #         with open(iop.device, 'wb') as handle:
    #             pickle.dump(dp, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    #         print('Initialized MKID array parameters')
    #
    #         return dp

    def arange_into_cube(self, packets, size):
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


    def responvisity_scaling_map(self, plot=False):
        """Assigns each pixel a phase responsivity between 0 and 1"""
        dist = Distribution(gaussian(self.params['mp'].r_mean, self.params['mp'].r_sig, np.linspace(0, 2, self.params['mp'].res_elements)), interpolation=True)
        responsivity = dist(self.array_size[0] * self.array_size[1])[0]/float(self.params['mp'].res_elements) * 2
        if plot:
            plt.xlabel('Responsivity')
            plt.ylabel('#')
            plt.hist(responsivity)
            plt.show()
        responsivity = np.reshape(responsivity, self.array_size[::-1])
        if plot:
            quick2D(responsivity)#plt.imshow(QE)
            # plt.show()

        return responsivity

    def array_QE(self, plot=False):
        """Assigns each pixel a phase responsivity between 0 and 1"""
        dist = Distribution(gaussian(self.params['mp'].g_mean, self.params['mp'].g_sig, np.linspace(0, 1, self.params['mp'].res_elements)), interpolation=True)
        QE = dist(self.array_size[0] * self.array_size[1])[0]/float(self.params['mp'].res_elements)
        if plot:
            plt.xlabel('Responsivity')
            plt.ylabel('#')
            plt.hist(QE)
            plt.show()
        QE = np.reshape(QE, self.array_size[::-1])
        if plot:
            quick2D(QE)#plt.imshow(QE)
            # plt.show()

        return QE

    def assign_spectral_res(self, plot=False):
        """Assigning each pixel a spectral resolution (at 800nm)"""
        dist = Distribution(gaussian(0.5, 0.25, np.linspace(-0.2, 1.2, self.params['mp'].res_elements)), interpolation=True)
        # print(f"Mean R = {self.params['mp'].R_mean}")
        Rs = (dist(self.array_size[0]*self.array_size[1])[0]/float(self.params['mp'].res_elements)-0.5)*self.params['mp'].R_sig + self.params['mp'].R_mean#
        if plot:
            plt.xlabel('R')
            plt.ylabel('#')
            plt.hist(Rs)
            plt.show()
        Rs = np.reshape(Rs, self.array_size)

        if plot:
            plt.figure()
            plt.imshow(Rs)
            plt.show()
        return Rs

    def get_R_hyper(self, Rs, plot=False):
        """Each pixel of the array has a matrix of probabilities that depends on the input wavelength"""
        print('Creating a cube of R standard deviations')
        m = (self.params['mp'].R_spec*Rs)/(self.params['ap'].wvl_range[1] - self.params['ap'].wvl_range[0]) # looses R of 10% over the 700 wvl_range
        c = Rs-m*self.params['ap'].wvl_range[0] # each c depends on the R @ 800
        waves = np.ones((np.shape(m)[1],np.shape(m)[0],self.params['ap'].n_wvl_final+5))*np.linspace(self.params['ap'].wvl_range[0],self.params['ap'].wvl_range[1],self.params['ap'].n_wvl_final+5)
        waves = np.transpose(waves) # make a tensor of 128x128x10 where every 10 vector is 800... 1500
        R_spec = m * waves + c # 128x128x10 tensor is now lots of simple linear lines e.g. 50,49,.. 45
        # probs = np.ones((np.shape(R_spec)[0],np.shape(R_spec)[1],np.shape(R_spec)[2],
        #                 self.params['mp'].res_elements))*np.linspace(0, 1, self.params['mp'].res_elements)
        #                         # similar to waves but 0... 1 using 128 elements
        # R_spec = np.repeat(R_spec[:,:,:,np.newaxis], self.params['mp'].res_elements, 3) # creat 128 repeats of R_spec so (10,128,128,128)
        # self.params['mp'].R_probs = gaussian(0.5, R_spec, probs) #each xylocation is gaussian that gets wider for longer wavelengths
        sigs_w = (waves/R_spec)/2.35 #R = w/dw & FWHM = 2.35*sig

        # plt.plot(range(0,1500),spec.phase_cal(np.arange(0,1500)))
        # plt.show()
        sigs_p = self.phase_cal(sigs_w) - self.phase_cal(np.zeros_like((sigs_w)))

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
            # plt.imshow(self.params['mp'].R_probs[:,0,0,:])
            plt.show(block=True)
        return sigs_p

    def apply_phase_scaling(photons, ):
        """
        From things like resonator Q, bias power, quasiparticle losses

        :param photons:
        :return:
        """

    def apply_phase_offset_array(self, photons, sigs):
        """
        From things like IQ phase offset noise

        :param photons:
        :param sigs:
        :return:
        """
        wavelength = self.wave_cal(photons[1])

        idx = self.wave_idx(wavelength)

        bad = np.where(np.logical_or(idx>=len(sigs), idx<0))[0]

        photons = np.delete(photons, bad, axis=1)
        idx = np.delete(idx, bad)

        distortion = np.random.normal(np.zeros((photons[1].shape[0])),
                                      sigs[idx,np.int_(photons[3]), np.int_(photons[2])])

        photons[1] += distortion

        return photons, idx

    def apply_phase_distort(self, phase, loc, sigs):
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
        # phase = phase + self.params['mp'].phase_distortions[ip]
        wavelength = self.wave_cal(phase)
        idx = self.wave_idx(wavelength)

        if phase != 0 and idx<len(sigs):
            phase = np.random.normal(phase,sigs[idx,loc[0],loc[1]],1)[0]
        return phase

    def wave_idx(self, wavelength):
        m = float(self.params['ap'].n_wvl_final-1)/(self.params['ap'].wvl_range[1] - self.params['ap'].wvl_range[0])
        c = -m*self.params['ap'].wvl_range[0]
        idx = wavelength*m + c
        # return np.int_(idx)
        return np.int_(np.round(idx))

    def phase_cal(self, wavelengths):
        '''Wavelength in nm'''
        phase = self.params['mp'].wavecal_coeffs[0]*wavelengths + self.params['mp'].wavecal_coeffs[1]
        return phase

    def wave_cal(self, phase):
        wave = (phase - self.params['mp'].wavecal_coeffs[1])/(self.params['mp'].wavecal_coeffs[0])
        return wave

    def assign_phase_background(self, plot=False):
        """assigns each pixel a baseline phase"""
        dist = Distribution(gaussian(0.5, 0.25, np.linspace(-0.2, 1.2, self.params['mp'].res_elements)), interpolation=True)

        basesDeg = dist(self.array_size[0]*self.array_size[1])[0]/float(self.params['mp'].res_elements)*self.params['mp'].bg_mean/self.params['mp'].g_mean
        if plot:
            plt.xlabel('basesDeg')
            plt.ylabel('#')
            plt.title('Background Phase')
            plt.hist(basesDeg)
            plt.show(block=True)
        basesDeg = np.reshape(basesDeg, self.array_size)
        if plot:
            plt.title('Background Phase--Reshaped')
            plt.imshow(basesDeg)
            plt.show(block=True)
        return basesDeg


    def create_bad_pix(self, QE_map_all, plot=False):
        amount = int(self.array_size[0]*self.array_size[1]*(1.-self.params['mp'].pix_yield))

        bad_ind = random.sample(list(range(self.array_size[0]*self.array_size[1])), amount)

        print(f"Bad indices = {len(bad_ind)}, # MKID pix = { self.array_size[0]*self.array_size[1]}, "
               f"Pixel Yield = {self.params['mp'].pix_yield}, amount? = {amount}")

        # bad_y = random.sample(y, amount)
        bad_y = np.int_(np.floor(bad_ind/self.array_size[1]))
        bad_x = bad_ind % self.array_size[1]

        # print(f"responsivity shape  = {responsivities.shape}")
        QE_map = np.array(QE_map_all)

        QE_map[bad_x, bad_y] = 0
        if plot:
            plt.xlabel('responsivities')
            plt.ylabel('?')
            plt.title('Something Related to Bad Pixels')
            plt.imshow(QE_map)
            plt.show()

        return QE_map

    def create_bad_pix_center(self, responsivities):
        res_elements=self.array_size[0]
        # responsivities = np.zeros()
        for x in range(self.array_size[1]):
            dist = Distribution(gaussian(0.5, 0.25, np.linspace(0, 1, self.params['mp'].res_elements)), interpolation=False)
            dist = np.int_(dist(int(self.array_size[0]*self.params['mp'].pix_yield))[0])#/float(self.params['mp'].res_elements)*np.int_(self.array_size[0]) / self.params['mp'].g_mean)
            # plt.plot(dist)
            # plt.show()
            dead_ind = []
            [dead_ind.append(el) for el in range(self.array_size[0]) if el not in dist]
            responsivities[x][dead_ind] = 0

        return responsivities

    def get_bad_packets(self, step, type='hot'):
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
                dist = Distribution(gaussian(0, 0.25, np.linspace(0, 1, self.params['mp'].res_elements)), interpolation=False)
                phases = dist(n_device_counts)[0]
                max_phase = max(phases)
                phases = -phases*120/max_phase
                bad_pix_options = self.create_false_pix(self.dark_pix_frac * self.array_size[0] * self.array_size[1])
                bad_ind = np.random.choice(range(len(bad_pix_options[0])), n_device_counts)
                bad_pix = bad_pix_options[:, bad_ind]

            meantime = step * self.params['sp'].sample_time
            photons[0] = np.random.uniform(meantime - self.params['sp'].sample_time / 2, meantime + self.params['sp'].sample_time / 2, len(photons[0]))
            photons[1] = phases
            photons[2:] = bad_pix

        return photons

    def create_false_pix(self, amount):
        # print(f"amount = {amount}")
        bad_ind = random.sample(list(range(self.array_size[0]*self.array_size[1])), int(amount))
        bad_y = np.int_(np.floor(bad_ind / self.array_size[1]))
        bad_x = bad_ind % self.array_size[1]

        return np.array([bad_x, bad_y])


    def remove_bad(self, frame, QE):
        bad_map = np.ones((self.params['sp'].grid_size,self.params['sp'].grid_size))
        bad_map[QE[:-1,:-1]==0] = 0
        # quick2D(QE, logZ =False)
        # quick2D(bad_map, logZ =False)
        frame = frame*bad_map
        return frame

    def sample_cube(self, datacube, num_events):
        # print 'creating photon data from reference cube'

        dist = Distribution(datacube, interpolation=True)

        photons = dist(num_events)
        return photons

    def assign_calibtime(self, photons, step):
        meantime = step*self.params['sp'].sample_time
        # photons = photons.astype(float)#np.asarray(photons[0], dtype=np.float64)
        # photons[0] = photons[0] * ps.self.params['mp'].frame_time
        timedist = np.random.uniform(meantime-self.params['sp'].sample_time/2, meantime+self.params['sp'].sample_time/2, len(photons[0]))
        photons = np.vstack((timedist, photons))
        return photons

    def calibrate_phase(self, photons):
        """
        idx -> phase

        :param photons:
        :return:
        """
        photons = np.array(photons)
        c = self.params['ap'].wvl_range[0]
        m = (self.params['ap'].wvl_range[1] - self.params['ap'].wvl_range[0])/self.params['ap'].n_wvl_final
        wavelengths = photons[0]*m + c
        # print(wavelengths[:5])
        # photons[0] = wavelengths*self.params['mp'].wavecal_coeffs[0] + self.params['mp'].wavecal_coeffs[1]
        photons[0] = self.phase_cal(wavelengths)
        # print(photons[0,:5])
        # exit()

        return photons

    def make_datacube_from_list(self, packets):
        phase_band = self.phase_cal(self.params['ap'].wvl_range)
        bins = [np.linspace(phase_band[0], phase_band[1], self.params['ap'].n_wvl_final + 1),
                range(self.array_size[0]+1),
                range(self.array_size[1]+1)]
        datacube, _ = np.histogramdd(packets[:,1:], bins)

        return datacube

    def remove_close(self, stem):
        print('removing close photons')
        for x in range(len(stem)):
            for y in range(len(stem[0])):
                # print(x,y)
                if len(stem[x][y]) > 1:
                    events = np.array(stem[x][y])
                    timesort = np.argsort(events[:, 0])
                    events = events[timesort]
                    detected = [0]
                    idx = 0
                    while idx != None:
                        these_times = events[idx:, 0] - events[idx, 0]
                        detect, _ = next(((i, v) for (i, v) in enumerate(these_times) if v > mp.dead_time),
                                         (None, None))
                        if detect != None:
                            detect += idx
                            detected.append(detect)
                        idx = detect

                    missed = [ele for ele in range(detected[-1] + 1) if ele not in detected]
                    events = np.delete(events, missed, axis=0)
                    stem[x][y] = events
                    # dprint(x, len(missed))
        return stem

    def arange_into_stem(self, packets, size):
        # print 'Sorting packets into xy grid (no phase or time sorting)'
        stem = [[[] for i in range(size[0])] for j in range(size[1])]
        # dprint(np.shape(cube))
        # plt.hist(packets[:,1], bins=100)
        # plt.show()
        for ip, p in enumerate(packets):
            x = np.int_(p[2])
            y = np.int_(p[3])
            stem[x][y].append([p[0], p[1]])
            if len(packets) >= 1e7 and ip % 10000 == 0: misc.progressBar(value=ip, endvalue=len(packets))
        # print cube[x][y]
        # cube = time_sort(cube)
        return stem

    def ungroup(self, stem):
        photons = np.empty((0, 4))
        for x in range(mp.array_size[1]):
            for y in range(mp.array_size[0]):
                # print(x, y)
                if len(stem[x][y]) > 0:
                    events = np.array(stem[x][y])
                    xy = [[x, y]] * len(events) if len(events.shape) == 2 else [x, y]
                    events = np.append(events, xy, axis=1)
                    photons = np.vstack((photons, events))
                    # print(events, np.shape(events))
                    # timesort = np.argsort(events[:, 0])
                    # events = events[timesort]
                    # sep = events[:, 0] - np.roll(events[:, 0], 1, 0)
        return photons.T

    def get_packets(self, datacube, step, plot=False):
        if plot: view_spectra(datacube, logZ=True, extract_center=False, title='pre')

        if self.params['mp'].resamp:
            nyq_sampling = self.params['ap'].wvl_range[0]*360*3600/(4*np.pi*self.params['tp'].entrance_d)
            sampling = nyq_sampling*self.params['sp'].beam_ratio*2  # nyq sampling happens at self.params['sp'].beam_ratio = 0.5
            x = np.arange(-self.params['sp'].grid_size*sampling/2, self.params['sp'].grid_size*sampling/2, sampling)
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
        if self.params['mp'].QE_var:
            datacube *= self.QE_map[:datacube.shape[1],:datacube.shape[1]]
        if plot: view_spectra(datacube, logZ=True)

        # quick2D(self.QE_map)
        if plot: view_spectra(datacube, logZ=True, show=False)
        if hasattr(self,'star_phot'): self.params['ap'].star_flux = self.star_phot
        num_events = int(self.params['ap'].star_flux * self.params['sp'].sample_time * np.sum(datacube))

        if self.params['sp'].verbose:
            print(f"star flux: {self.params['ap'].star_flux}, cube sum: {np.sum(datacube)}, num events: {num_events}")

        photons = self.sample_cube(datacube, num_events)
        photons = self.calibrate_phase(photons)
        photons = self.assign_calibtime(photons, step)

        if plot:
            cube = self.make_datacube_from_list(photons.T)
            print(cube.shape)
            # view_spectra(cube, logZ=True)

        if self.params['mp'].dark_counts:
            dark_photons = self.get_bad_packets(step, type='dark')
            photons = np.hstack((photons, dark_photons))

        if self.params['mp'].hot_pix:
            hot_photons = self.get_bad_packets(step, type='hot')
            photons = np.hstack((photons, hot_photons))
            # stem = add_hot(stem)

        # plt.hist(photons[3], bins=25)
        # plt.yscale('log')
        # plt.show(block=True)
        # stem = arange_into_stem(photons.T, (self.array_size[0], self.array_size[1]))
        # cube = make_datacube(stem, (self.array_size[0], self.array_size[1], self.params['ap'].n_wvl_final))
        # view_spectra(cube, logZ=True, vmin=0.01)

        if plot:
            cube = self.make_datacube_from_list(photons.T)
            print(cube.shape)
            view_spectra(cube, logZ=True, title='hot pix')

        if self.params['mp'].phase_uncertainty:
            photons[1] *= self.responsivity_error_map[np.int_(photons[2]), np.int_(photons[3])]
            photons, idx = self.apply_phase_offset_array(photons, self.sigs)
            # stem = arange_into_stem(photons.T, (self.array_size[0], self.array_size[1]))
            # cube = make_datacube(stem, (self.array_size[0], self.array_size[1], self.params['ap'].n_wvl_final))
            # view_spectra(cube, logZ=True, vmin=0.01)

        # stem = arange_into_stem(photons.T, (self.array_size[0], self.array_size[1]))
        # cube = make_datacube(stem, (self.array_size[0], self.array_size[1], self.params['ap'].n_wvl_final))
        # view_spectra(cube, vmin=0.01, logZ=True)
        # plt.figure()
        # plt.imshow(cube[0], origin='lower', norm=LogNorm(), cmap='inferno', vmin=1)
        # plt.show(block=True)

        # thresh =  photons[1] < self.basesDeg[np.int_(photons[3]),np.int_(photons[2])]
        if self.params['mp'].phase_background:
            thresh =  -photons[1] > 3*self.sigs[-1,np.int_(photons[3]), np.int_(photons[2])]
            photons = photons[:, thresh]
        # print(thresh)

        # stem = arange_into_stem(photons.T, (self.array_size[0], self.array_size[1]))
        # cube = make_datacube(stem, (self.array_size[0], self.array_size[1], self.params['ap'].n_wvl_final))
        # quick2D(cube[0], vmin=1, logZ=True)
        # plt.figure()
        # plt.imshow(cube[0], origin='lower', norm=LogNorm(), cmap='inferno', vmin=1)
        # plt.show(block=True)

        # print(photons.shape)
        if self.params['mp'].remove_close:
            stem = self.arange_into_stem(photons.T, (self.array_size[0], self.array_size[1]))
            stem = self.remove_close(stem)
            photons = self.ungroup(stem)

        if plot:
            cube = self.make_datacube_from_list(photons.T)
            print(cube.shape)
            view_spectra(cube, logZ=True, use_axis=False, title='remove close')
        # This step was taking a long time
        # stem = arange_into_stem(photons.T, (self.array_size[0], self.array_size[1]))
        # cube = make_datacube(stem, (self.array_size[0], self.array_size[1], self.params['ap'].n_wvl_final))
        # # ax7.imshow(cube[0], origin='lower', norm=LogNorm(), cmap='inferno', vmin=1)
        # cube /= self.QE_map
        # photons = ungroup(stem)

        # print(photons.shape)


        # print("Measured photons with MKIDs")

        return photons.T

if __name__ == '__main__':
    from medis.params import params

    cam = Camera(params)
    dataproduct = cam()
    print(dataproduct.keys())
