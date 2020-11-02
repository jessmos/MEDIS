"""
CDI.py
Kristina Davis
8/12/19

This module is used to set the CDI parameters for mini-medis.
"""
import numpy as np
import warnings
import scipy.linalg as linalg
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

from medis.params import tp, sp
from medis.utils import dprint
from medis.optics import extract_plane

class CDIOut():
    '''Stole this idea from falco. Just passes an object you can slap stuff onto'''
    pass


class CDI_params():
    """
    contains the parameters of the CDI probes and phase sequence to apply to the DM
    """
    def __init__(self):
        # General
        self.use_cdi = False
        self.show_probe = False  # False , flag to plot phase probe or not
        self.which_DM = ''  # plane_name parameter of the DM to apply CDI probe to (must be a valid name for
        # the telescope sim you are running, eg 'tweeter'

        # Probe Dimensions (extent in pupil plane coordinates)
        self.probe_amp = 2e-6  # [m] probe amplitude, scale should be in units of actuator height limits
        self.probe_w = 10  # [actuator coordinates] width of the probe
        self.probe_h = 30  # [actuator coordinates] height of the probe
        self.probe_shift = (15, 15)  # [actuator coordinates] center position of the probe (should move off-center to
        # avoid coronagraph)
        self.probe_spacing = 10  # distance from the focal plane center to edge of the rectangular probed region

        # Phase Sequence of Probes
        self.phs_intervals = np.pi / 2  # [rad] phase interval over [0, 2pi]
        self.phase_integration_time = 0.01  # [s]  How long in sec to apply each probe in the sequence
        self.null_time = 0  # [s]  time between repeating probe cycles (data to be nulled using probe info)

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def __name__(self):
        return self.__str__().split(' ')[0].split('.')[-1]

    def gen_phaseseries(self):
        """
        generate an array of phases per timestep for the CDI algorithm

        phase_series is used to populate cdi.phase_series, which may be longer than cdi.phase_cycle if multiple cycles
        are run, or probes may last for longer than one single timestep

        currently, I assume the timestream is not that long. Should only use this for short timestreams, and use a more
        efficient code for long simulations (scale of minutes or more)

        :return: phase_series  array of phases of CDI probes to apply to DM
        """
        self.phase_series = np.zeros(sp.numframes) * np.nan

        if self.use_cdi:
            # Repeating Probe Phases for Integration time
            self.phase_cycle = np.arange(0, 2 * np.pi, self.phs_intervals)  # FYI not inclusive of 2pi endpoint
            self.n_probes = len(self.phase_cycle)  # number of phase probes
            if self.n_probes % 2 != 0:
                raise ValueError(f"must have even number of phase probes\n\tchange cdi.phs_intervals")

            if self.phase_integration_time > sp.sample_time:
                phase_hold = self.phase_integration_time / sp.sample_time
                phase_1cycle = np.repeat(self.phase_cycle, phase_hold)
            elif self.phase_integration_time == sp.sample_time:
                phase_1cycle = self.phase_cycle
            else:
                raise ValueError(f"Cannot have CDI phase probe integration time less than sp.sample_time")

            # Repeating Cycle of Phase Probes for Simulation Duration
            full_simulation_time = sp.numframes * sp.sample_time
            time_for_one_cycle = len(phase_1cycle) * self.phase_integration_time + self.null_time

            if time_for_one_cycle > full_simulation_time and self.phase_integration_time >= sp.sample_time:
                warnings.warn(f"\nLength of one full CDI probe cycle (including nulling) exceeds the "
                              f"full simulation time \n"
                              f"not all phases will be used\n"
                              f"phase reconstruction will be incomplete")
                self.phase_series = phase_1cycle[0:sp.numframes]
            elif time_for_one_cycle <= full_simulation_time and self.n_probes < sp.numframes:
                print(f"\nCDI Params\n\tThere will be {sp.numframes - self.n_probes} "
                      f"nulling steps after timestep {self.n_probes}")
                self.phase_series[0:self.n_probes] = phase_1cycle
            else:
                warnings.warn(f"Haven't run into  CDI phase situation like this yet")
                raise NotImplementedError

        return self.phase_series

    def init_cout(self, nact):
        self.cout = CDIOut()
        self.cout.nact = nact
        self.cout.grid_size = sp.grid_size
        self.cout.beam_ratio = sp.beam_ratio
        self.cout.DM_probe_series = np.zeros((self.n_probes, nact, nact))

    def save_probe(self, ix, probe):
        self.cout.DM_probe_series[ix] = probe

    def save_tseries(self, ts):
        """saves output of medis fields as 2D intensity images for CDI postprocessing"""
        self.cout.tseries = ts

    def save_cout_to_disk(self, plot=False):


        # Fig
        if plot:
            if self.n_probes >= 4:
                nrows = 2
                ncols = self.n_probes//2
                figheight = 5
            else:
                nrows = 1
                ncols = self.n_probes
                figheight = 12

            fig, subplot = plt.subplots(nrows, ncols, figsize=(12, figheight))
            fig.subplots_adjust(wspace=0.5)

            fig.suptitle('Probe Series')

            for ax, ix in zip(subplot.flatten(), range(self.n_probes)):
                # im = ax.imshow(self.DM_probe_series[ix], interpolation='none', origin='lower')
                im = ax.imshow(self.cout.probe_series[ix], interpolation='none', origin='lower')

                ax.set_title(f"Probe " + r'$\theta$=' + f'{self.DM_phase_series[ix]/np.pi:.2f}' + r'$\pi$')

            cb = fig.colorbar(im)  #
            cb.set_label('um')

            plt.show()

# Sneakily Instantiating Class Objects here
cdi = CDI_params()


def config_probe(theta, nact, iw=0, ib=0,tstep=0):
    """
    create a probe shape to apply to the DM for CDI processing

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
    x = np.linspace(-1/2-cdi.probe_shift[0]/nact, 1/2-cdi.probe_shift[0]/nact, nact)
    y = np.linspace(-1/2-cdi.probe_shift[1]/nact, 1/2-cdi.probe_shift[1]/nact, nact)
    X,Y = np.meshgrid(x, y)

    probe = cdi.probe_amp * np.sinc(cdi.probe_w * X) * np.sinc(cdi.probe_h * Y) \
            * np.sin(2*np.pi*cdi.probe_spacing*X + theta)

    # Testing FF propagation
    if sp.verbose and iw == 0 and ib == 0:  # and theta == cdi.phase_series[0]
        probe_ft = (1/np.sqrt(2*np.pi)) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(probe)))

        # fig, ax = plt.subplots(1, 3, figsize=(12, 5))
        # fig.subplots_adjust(wspace=0.5)
        # ax1, ax2, ax3 = ax.flatten()
        #
        # fig.suptitle(f"spacing={cdi.probe_spacing}, Dimensions {cdi.probe_w}x{cdi.probe_h} "
        #              f"\nProbe Amp = {cdi.probe_amp}, " + r'$\theta$' + f"={theta/np.pi:.3f}" + r'$\pi$' + '\n')
        #
        # im1 = ax1.imshow(probe, interpolation='none', origin='lower')
        # ax1.set_title(f"Probe on DM \n(dm coordinates)")
        # cb = fig.colorbar(im1, ax=ax1)
        #
        # im2 = ax2.imshow(np.sqrt(probe_ft.imag ** 2 + probe_ft.real ** 2), interpolation='none', origin='lower')
        # ax2.set_title("Focal Plane Amplitude")
        # cb = fig.colorbar(im2, ax=ax2)
        #
        # ax3.imshow(np.arctan2(probe_ft.imag, probe_ft.real), interpolation='none', origin='lower', cmap='hsv')
        # ax3.set_title("Focal Plane Phase")

        # plt.show()

        # Fig 2
        fig, ax = plt.subplots(1, 3, figsize=(12, 5))
        fig.subplots_adjust(wspace=0.5)
        ax1, ax2, ax3 = ax.flatten()
        fig.suptitle(f'Real & Imaginary Probe Response in Focal Plane\n'
                     f' '+r'$\theta$'+f'={theta/np.pi:.3f}'+r'$\pi$'+f', n_actuators = {nact}\n')

        im1 = ax1.imshow(probe, interpolation='none', origin='lower')
        ax1.set_title(f"Probe on DM \n(dm coordinates)")
        cb = fig.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(probe_ft.real, interpolation='none', origin='lower')
        ax2.set_title(f"Real FT of Probe")

        im3 = ax3.imshow(probe_ft.imag, interpolation='none', origin='lower')
        ax3.set_title(f"Imag FT of Probe")

        plt.show()

    # Saving Probe in the series
    if iw == 0 and ib == 0:
        ip = np.argwhere(cdi.phase_series == theta)
        cdi.save_probe(ip[0,0], probe)
        cdi.nact = nact

    return probe


def cdi_postprocess(cpx_seq, sampling, plot=False):
    """
    this is the function that accepts the timeseries of intensity images from the simulation and returns the processed
    single image. This function calculates the speckle amplitude phase, and then corrects for it to create the dark
    hole over the specified region of the image.

    :param cpx_seq: #timestream of 2D images (intensity only) from the focal plane complex field
    :param sampling: focal plane sampling
    :return:
    """
    import time
    tic = time.time()
    focal_plane = extract_plane(cpx_seq, 'detector')  # eliminates astro_body axis [tsteps,wvl,obj,x,y]
    fp_seq = np.sum(focal_plane, axis=(1,2))  # sum over wavelength,object


    n_pairs = cdi.n_probes//2  # number of deltas (probe differentials)
    n_nulls = sp.numframes - cdi.n_probes
    delta = np.zeros((n_pairs, sp.grid_size, sp.grid_size), dtype=float)
    abs_delta = np.zeros((n_nulls, sp.grid_size, sp.grid_size))
    phs_delta = np.zeros((n_pairs, sp.grid_size, sp.grid_size), dtype=float)
    Epupil = np.zeros((n_nulls, sp.grid_size, sp.grid_size), dtype=complex)
    H = np.zeros((n_pairs, 2), dtype=float)
    b = np.zeros((n_pairs, 1))


    for ip in range(n_pairs):
        # Compute deltas
        delta[ip] = np.abs(fp_seq[ip]) ** 2 - np.abs(fp_seq[ip + n_pairs]) ** 2

        # Phase Delta
        # FFT and Interpolate DM Map onto MEC coordinates
        fftA = (1 / np.sqrt(2 * np.pi) *
                np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(fp_seq[ip]))))
        fftB = (1 / np.sqrt(2 * np.pi) *
                np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(fp_seq[ip + n_pairs]))))
        phs_delta[ip, :, :] = np.arctan2(fftA.imag - fftB.imag, fftA.real - fftB.real)

    for i in range(sp.grid_size):
        for j in range(sp.grid_size):
            for xn in range(n_nulls):
                for ip in range(n_pairs):
                    # Amplitude Delta
                    Ip = np.abs(fp_seq[ip, i, j]) ** 2
                    Im = np.abs(fp_seq[ip + n_pairs, i, j]) ** 2
                    Io = np.abs(fp_seq[cdi.n_probes + xn, i, j]) ** 2
                    abs = (Ip + Im) / 2 - Io
                    if abs < 0:
                        abs = 0
                    absDeltaP = np.sqrt(abs)
                    # abs_delta[xn, :, :] = np.sqrt(abs)

                    # print(f'iteration i,j,xn,ip={i},{j},{xn},{ip}')
                    # absDeltaP = abs_delta[xn, i, j]
                    phsDeltaP = phs_delta[ip, i, j]
                    # phsDeltaP = np.arctan2(fp_seq[i,j,ip].imag - fp_seq[i,j, meta.ts.n_probes+xn].imag,
                    #                        fp_seq[i,j,ip].real - fp_seq[i,j, meta.ts.n_probes+xn].real)
                    cpxDeltaP = absDeltaP * np.exp(1j * phsDeltaP)

                    H[ip, :] = [-cpxDeltaP.imag, cpxDeltaP.real]  # [n_pairs, 2]
                    b[ip] = delta[ip, i, j]  # [n_pairs, 1]

                a = 2 * H
                # print(f'[a]\n{a};, \n[b]\n{b}')
                Exy, res, rnk, s = linalg.lstsq(a, b)  # returns tuple, not array
                # print(f'xn={xn},i={i},j={j},ip={ip}\nExy={Exy}')
                Epupil[xn, i, j] = Exy[0] + (1j * Exy[1])

    if plot:
        # ==================
        # Deltas
        # ==================
        fig, subplot = plt.subplots(1, n_pairs, figsize=(14,5))
        fig.subplots_adjust(wspace=0.5, right=0.85)
        fig.suptitle('Deltas for CDI Probes')

        for ax, ix in zip(subplot.flatten(), range(n_pairs)):
            im = ax.imshow(delta[ix]*1e6, interpolation='none', origin='lower',
                           norm=SymLogNorm(linthresh=1e-2),
                           vmin=-1, vmax=1) #, norm=SymLogNorm(linthresh=1e-5))
            ax.set_title(f"Diff Probe\n" + r'$\theta$' + f'={cdi.phase_series[ix]/np.pi:.3f}' +
                         r'$\pi$ -$\theta$' + f'={cdi.phase_series[ix+n_pairs]/np.pi:.3f}' + r'$\pi$')

        cax = fig.add_axes([0.9, 0.2, 0.03, 0.6])  # Add axes for colorbar @ position [left,bottom,width,height]
        cb = fig.colorbar(im, orientation='vertical', cax=cax)  #
        cb.set_label('Intensity')

        # ==================
        # E-filed Estimates
        # ==================
        fig, subplot = plt.subplots(1, n_nulls, figsize=(14, 5))
        fig.subplots_adjust(wspace=0.5, right=0.85)
        fig.suptitle('Estimated E-field')

        for ax, ix in zip(subplot.flatten(), range(n_nulls)):
            im = ax.imshow(np.abs(Epupil[ix])**2, interpolation='none', origin='lower',
                           norm=LogNorm())
                           # vmin=-1, vmax=1)  # , norm=SymLogNorm(linthresh=1e-5))
            ax.set_title(f'Null Step {ix}')

        cax = fig.add_axes([0.9, 0.2, 0.03, 0.6])  # Add axes for colorbar @ position [left,bottom,width,height]
        cb = fig.colorbar(im, orientation='vertical', cax=cax)  #
        cb.set_label('Intensity')

        # ==================
        # Subtracted E-field
        # ==================
        fig, subplot = plt.subplots(1, n_nulls, figsize=(14, 5))
        fig.subplots_adjust(wspace=0.5, right=0.85)
        fig.suptitle('Subtracted E-field')

        for ax, ix in zip(subplot.flatten(), range(n_nulls)):
            im = ax.imshow(np.abs(fp_seq[n_pairs+ix] - np.conjugate(Epupil[ix])) ** 2,
                           interpolation='none', origin='lower', norm=LogNorm())
            ax.set_title(f'Null Step {ix}')

        cax = fig.add_axes([0.9, 0.2, 0.03, 0.6])  # Add axes for colorbar @ position [left,bottom,width,height]
        cb = fig.colorbar(im, orientation='vertical', cax=cax)  #
        cb.set_label('Intensity')

        plt.show()
    toc = time.time()
    dprint(f'CDI post-processing took {(toc-tic)/60:.2} minutes')

if __name__ == '__main__':
    dprint(f"Testing CDI probe")
    cdi.use_cdi = True; cdishow_probe = True

    cdi.probe_amp = 2e-6  # [m] probe amplitude, scale should be in units of actuator height limits
    cdi.probe_w = 10  # [actuator coordinates] width of the probe
    cdi.probe_h = 30  # [actuator coordinates] height of the probe
    cdi.probe_shift = [5, 5]  # [actuator coordinates] center position of the probe
    cdi.probe_spacing = 15

    tp.act_tweeter = 49

    sp.numframes = 10
    cdi.gen_phaseseries()
    cdi.init_probes(tp.act_tweeter)
    # cdiphase_series = [-1*np.pi/4]
    config_probe(cdi.phase_series[0], tp.act_tweeter)  #
