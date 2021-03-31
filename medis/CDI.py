"""
CDI.py
Kristina Davis
8/12/19

This module is used to set the CDI parameters for mini-medis.
"""
##
import numpy as np
import warnings
from scipy import linalg, interpolate
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import time
import pickle

from medis.params import tp, sp, ap, iop
from medis.utils import dprint
from medis.optics import extract_plane, cpx_to_intensity
from medis.plot_tools import add_colorbar, view_timeseries


##
class Slapper:
    """Stole this idea from falco. Just passes an object you can slap stuff onto"""
    pass


class CDI_params:
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
        self.probe_ax = 'X'  # 'Y' # direction of the probe
        self.probe_amp = 2e-6  # [m] probe amplitude, scale should be in units of actuator height limits
        self.probe_w = 10  # [actuator coordinates] width of the probe
        self.probe_h = 30  # [actuator coordinates] height of the probe
        self.probe_shift = (15, 15)  # [actuator coordinates] center position of the probe (should move off-center to
        # avoid coronagraph)
        self.probe_spacing = 10  # distance from the focal plane center to edge of the rectangular probed region

        # Phase Sequence of Probes
        self.phs_intervals = np.pi / 2  # [rad] phase interval over [0, 2pi]
        self.probe_integration_time = 1  # [s]  How long in sec to apply each probe in the sequence
        self.null_time = 5 * sp.sample_time # [s]  time between repeating probe cycles (data to be nulled using probe info)

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

        # Repeating Probe Phases for Integration time
        self.phase_cycle = np.arange(0, 2 * np.pi, self.phs_intervals)  # FYI not inclusive of 2pi endpoint
        self.n_probes = len(self.phase_cycle)  # number of phase probes
        if self.n_probes % 2 != 0:
            raise ValueError(f"must have even number of phase probes\n\tchange cdi.phs_intervals")
        self.dummy_probe = np.zeros((self.n_probes, tp.act_tweeter, tp.act_tweeter))
        self.dummy_tstamp = np.zeros((sp.numframes,),  dtype='datetime64[ns]')

        if self.probe_integration_time > sp.sample_time:
            phase_hold = self.probe_integration_time / sp.sample_time
            phase_1cycle = np.repeat(self.phase_cycle, phase_hold)
        elif self.probe_integration_time == sp.sample_time:
            phase_1cycle = np.zeros(sp.numframes)
            phase_1cycle[0:self.n_probes] = self.phase_cycle
        else:
            raise ValueError(f"Cannot have CDI phase probe integration time less than sp.sample_time")

        # Repeating Cycle of Phase Probes for Simulation Duration
        full_simulation_time = sp.numframes * sp.sample_time
        self.time_for_one_cycle = len(phase_1cycle) * self.probe_integration_time + self.null_time

        if self.time_for_one_cycle > full_simulation_time and self.probe_integration_time >= sp.sample_time:
            warnings.warn(f"\nLength of one full CDI probe cycle (including nulling) exceeds the "
                          f"full simulation time \n"
                          f"not all phases will be used\n"
                          f"phase reconstruction will be incomplete")
            self.phase_series = phase_1cycle[0:sp.numframes]
        elif self.time_for_one_cycle <= full_simulation_time and self.n_probes < sp.numframes:
            print(f"\nCDI Params\n\tThere will be {sp.numframes - self.n_probes} "
                  f"nulling steps after timestep {self.n_probes}")
            self.phase_series[0:self.n_probes] = phase_1cycle
        else:
            warnings.warn(f"Haven't run into  CDI phase situation like this yet")
            raise NotImplementedError

        return self.phase_series

    def save_probe(self, ix, probe):
        self.dummy_probe[ix, :, :] = probe

    def save_tseries(self, ix, ts):
        """saves output of medis fields as 2D intensity images for CDI postprocessing"""
        self.dummy_tstamp[ix] = ts

    def save_out_to_disk(self, plot=False):
        out = Slapper()
        out.probe = Slapper()
        out.ts = Slapper()

        # Probe Info
        out.probe.direction = self.probe_ax
        out.probe.amp = self.probe_amp
        out.probe.width = self.probe_w
        out.probe.height = self.probe_h
        out.probe.shift = self.probe_shift
        out.probe.spacing = self.probe_spacing
        out.probe.DM_cmd_cycle = self.dummy_probe
        out.probe.phs_interval = self.phs_intervals

        # Timeseries Info
        out.ts.start = 0
        out.ts.n_probes = self.n_probes
        out.ts.phase_cycle = self.phase_cycle
        out.ts.probe_integration_time = self.probe_integration_time
        out.ts.t_one_cycle = self.time_for_one_cycle
        out.ts.null_time = self.null_time
        out.ts.elapsed_time = 0
        out.ts.n_cycles = 0
        out.ts.n_cmds = sp.numframes  # TODO verify this-this was just a late night hack
        out.ts.cmd_tstamps = self.dummy_tstamp

        save_location = iop.testdir + f"/{iop.testname}_CDIparams.pkl"
        dprint(f'save_location={save_location}')
        with open(save_location, 'wb') as handle:
            pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

        # Fig
        if plot:
            if self.n_probes >= 4:
                nrows = 2
                ncols = self.n_probes//2
                figheight = 8
            else:
                nrows = 1
                ncols = self.n_probes
                figheight = 4

            fig, subplot = plt.subplots(nrows, ncols, figsize=(10, figheight))
            fig.subplots_adjust(wspace=0.5, right=0.85, left=0.05)
            fig.suptitle('Probe Series')

            for ax, ix in zip(subplot.flatten(), range(out.ts.n_probes)):
                im = ax.imshow(out.probe.DM_cmd_cycle[ix], interpolation='none', origin='lower')
                ax.set_title(f"Probe " + r'$\theta$=' + f'{out.ts.phase_cycle[ix]/np.pi:.2f}' + r'$\pi$')

            cax = fig.add_axes([0.9, 0.2, 0.03, 0.6])  # Add axes for colorbar @ position [left,bottom,width,height]
            cb = fig.colorbar(im, orientation='vertical', cax=cax)  #
            cb.set_label('Probe Height (m)')

        return out

# Sneakily Instantiating Class Objects here
cdi = CDI_params()


##
def config_probe(theta, nact, iw=0, ib=0, tstep=0):
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
    :param iw: index of wavelength number in ap.wvl_range (used for plotting only)
    :param ib: index of astronomical body eg star or companion (used for plotting only)
    :return: height of phase probes to add to the DM map in adaptive.py
    """
    x = np.linspace(-1/2-cdi.probe_shift[0]/nact, 1/2-cdi.probe_shift[0]/nact, nact)
    y = np.linspace(-1/2-cdi.probe_shift[1]/nact, 1/2-cdi.probe_shift[1]/nact, nact)
    X, Y = np.meshgrid(x, y)

    wvl_samples = np.linspace(ap.wvl_range[0], ap.wvl_range[1], ap.n_wvl_init)
    # dprint(f'iw = {iw}, lambda = {wvl_samples[iw]}')
    mag = 4 * np.pi * wvl_samples[iw] * cdi.probe_amp

    if cdi.probe_ax == 'X' or cdi.probe_ax == 'x':
        dir = X
    elif cdi.probe_ax == 'Y' or cdi.probe_ax == 'y':
        dir = Y
    else:
        raise ValueError('probe direction value not understood; must be string "X" or "Y"')

    probe = mag * np.sinc(cdi.probe_w * X) * np.sinc(cdi.probe_h * Y) \
            * np.sin(2*np.pi*cdi.probe_spacing * dir + theta)

    # Testing FF propagation
    if sp.verbose and iw == 0 and ib == 0:  # and theta == cdi.phase_series[0]
        probe_ft = (1/np.sqrt(2*np.pi)) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(probe)))

        fig, ax = plt.subplots(1, 3, figsize=(12, 5))
        fig.subplots_adjust(wspace=0.5)
        ax1, ax2, ax3 = ax.flatten()

        fig.suptitle(f"spacing={cdi.probe_spacing}, Dimensions {cdi.probe_w}x{cdi.probe_h} "
                     f"\nProbe Amp = {cdi.probe_amp}, " + r'$\theta$' + f"={theta/np.pi:.3f}" + r'$\pi$' + '\n')

        im1 = ax1.imshow(probe, interpolation='none')
        ax1.set_title(f"Probe on DM \n(dm coordinates)")
        # cb = fig.colorbar(im1, ax=ax1)
        add_colorbar(im1)

        im2 = ax2.imshow(np.sqrt(probe_ft.imag ** 2 + probe_ft.real ** 2), interpolation='none')
        ax2.set_title("Focal Plane Amplitude")
        add_colorbar(im2)

        im3 = ax3.imshow(np.arctan2(probe_ft.imag, probe_ft.real), interpolation='none', cmap='hsv')
        ax3.set_title("Focal Plane Phase")
        add_colorbar(im3)

        # =========================
        # Fig 2  Real & Imag
        # fig, ax = plt.subplots(1, 3, figsize=(12, 5))
        # fig.subplots_adjust(wspace=0.5)
        # ax1, ax2, ax3 = ax.flatten()
        # fig.suptitle(f'Real & Imaginary Probe Response in Focal Plane\n'
        #              f' '+r'$\theta$'+f'={theta/np.pi:.3f}'+r'$\pi$'+f', n_actuators = {nact}\n')
        #
        # im1 = ax1.imshow(probe, interpolation='none', origin='lower')
        # ax1.set_title(f"Probe on DM \n(dm coordinates)")
        # cb = fig.colorbar(im1, ax=ax1)
        #
        # im2 = ax2.imshow(probe_ft.real, interpolation='none', origin='lower')
        # ax2.set_title(f"Real FT of Probe")
        #
        # im3 = ax3.imshow(probe_ft.imag, interpolation='none', origin='lower')
        # ax3.set_title(f"Imag FT of Probe")

        plt.show()

    # Saving Probe in the series
    if iw == 0 and ib == 0:
        ip = np.argwhere(cdi.phase_series == theta)
        cdi.save_probe(ip[0,0], probe)  #
        cdi.nact = nact

    return probe


##
def cdi_postprocess(cpx_sequence, sampling, plot=False):
    """
    this is the function that accepts the timeseries of intensity images from the simulation and returns the processed
    single image. This function calculates the speckle amplitude phase, and then corrects for it to create the dark
    hole over the specified region of the image.

    From Give'on et al 2011, we have in eq 10 two different values: DeltaP-the change in the focal plane due to the
    probe, and delta, the intensity difference measurements between pairs of phase probes. (It is unfortunate that
    both these terms use delta, so I have done my best to distinguish them in the variable names)

    Here I note that in the CDI phase stream generation, for n_probes there are n_pairs = n_probes/2 pairs of probes.
    These get applied to the DM in a series such that the two probes that form the conjugate pair are separated by
    n_pairs of probes. In other words, for n_probes = 6, the 0th and 3rd probes are a pair, the 1st and 4th are a pair,
    and so on. This is a choice made when creating cdi.phase_series.

    :param cpx_sequence: #timestream of 2D images (complex) from the focal plane complex field
    :param sampling: focal plane sampling
    :return:
    """
    ##
    tic = time.time()
    focal_plane = extract_plane(cpx_sequence, 'detector')  # eliminates astro_body axis [tsteps,wvl,obj,x,y]
    fp_seq = np.sum(focal_plane, axis=(1,2))  # sum over wavelength,object

    n_pairs = cdi.n_probes//2  # number of deltas (probe differentials)
    n_nulls = sp.numframes - cdi.n_probes
    delta = np.zeros((n_pairs, sp.grid_size, sp.grid_size), dtype=float)
    # absDelta = np.zeros((n_nulls, sp.grid_size, sp.grid_size))
    # phsDelta = np.zeros((n_pairs, sp.grid_size, sp.grid_size), dtype=float)
    E_pupil = np.zeros((n_nulls, sp.grid_size, sp.grid_size), dtype=complex)
    I_processed = np.zeros((n_nulls, sp.grid_size, sp.grid_size))
    H = np.zeros((n_pairs, 2), dtype=float)
    b = np.zeros((n_pairs, 1))

    # Get Masked Data
    mask2D, imsk, jmsk, irng, jrng, imx, imn, jmx, jmn = get_fp_mask(cdi)

    if sp.debug:
        fig, ax = plt.subplots(1,1)
        fig.suptitle(f'Masked FP in CDI probe Region')
        im = ax.imshow(cpx_to_intensity(fp_seq[0,:,:]*mask2D))

    for ip in range(n_pairs):
        # Compute deltas (I_ip+ - I_ip-)/2
        delta[ip] = (np.abs(fp_seq[ip])**2 - np.abs(fp_seq[ip + n_pairs])**2) / 2

    # for i,j in zip(imsk,jmsk):
    for i in irng:
        for j in jrng:
            for xn in range(n_nulls):
                for ip in range(n_pairs):
                    # Amplitude DeltaP
                    Ip = np.abs(fp_seq[ip, i, j]) ** 2
                    Im = np.abs(fp_seq[ip + n_pairs, i, j]) ** 2
                    Io = np.abs(fp_seq[cdi.n_probes + xn, i, j]) ** 2
                    abs = (Ip + Im) / 2 - Io
                    if abs < 0:
                        abs = 0
                    absDeltaP = np.sqrt(abs)
                    # absDeltaP = np.sqrt(np.abs((Ip + Im) / 2 - Io))

                    # phsDeltaP = phsDelta[ip, i, j]
                    # Phase DeltaP
                    # The phase of the change in the focal plane of the probe applied to the DM
                    # First subtract Eo vector from each probe phase to make new field vectors dEa, dEb,
                    # then take the angle between the two
                    dEp = fp_seq[ip, i, j] - fp_seq[cdi.n_probes + xn, i, j]
                    dEm = fp_seq[ip + n_pairs, i, j] - fp_seq[cdi.n_probes + xn, i, j]
                    phsDeltaP = np.arctan2(dEp.imag - dEm.imag, dEp.real - dEm.real)

                    cpxDeltaP = absDeltaP * np.exp(1j * phsDeltaP)

                    H[ip, :] = [-cpxDeltaP.imag, cpxDeltaP.real]  # [n_pairs, 2]
                    b[ip] = delta[ip, i, j]  # [n_pairs, 1]

                a = 2 * H
                Exy = linalg.lstsq(a, b)[0]  # returns tuple, not array
                E_pupil[xn, i, j] = Exy[0] + (1j * Exy[1])

    toc = time.time()
    dprint(f'CDI post-processing took {(toc-tic)/60:.2} minutes\n')

    ## ===========================
    # Contrast Ratios
    # ===========================
    intensity_probe        = np.zeros(n_nulls)
    intensity_DM_FFT       = np.zeros(n_nulls)
    intensity_pre_process  = np.zeros(n_nulls)
    intensity_post_process = np.zeros(n_nulls)
    E_ff = np.zeros((n_nulls, len(irng), len(jrng)), dtype=complex)

    for xn in range(n_nulls):
        E_ff[xn] = (1 / np.sqrt(2 * np.pi) *
                           np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E_pupil[xn, imn:imx+1, jmn:jmx+1]))))
        # E_ff[xn] = (1 / np.sqrt(2 * np.pi) *
        #             np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(np.conj(E_pupil[xn])))))
        # I_processed[xn] = np.sqrt(np.abs(fp_seq[n_pairs+xn])**2 - np.abs(E_ff[xn]*mask2D)**2)
        I_processed[xn] = np.abs(fp_seq[n_pairs+xn])**2 - np.abs(np.conj(E_ff[xn])*mask2D)**2
        # I_processed[xn] = np.sqrt(np.abs(np.abs(fp_seq[n_pairs+xn])**2 - np.abs(E_pupil[xn]*mask2D)**2))**2
        # I_processed[xn] = np.abs(fp_seq[n_pairs+xn] - np.conj(E_pupil[xn]*mask2D))**3

        # Contrast

        intensity_probe[xn] = np.sum(np.abs(fp_seq[xn]*mask2D)**2)
        intensity_pre_process[xn] = np.sum(np.abs(fp_seq[n_pairs + xn]*mask2D)**2)
        intensity_post_process[xn] = np.sum(I_processed[xn]*mask2D)  #np.sum(np.abs(E_processed[xn]*mask2D)**2)

        print(f'\nIntensity in probed region for null step {xn} is '
              f'\nprobe {intensity_probe[xn]}'
              f'\npre-processed {intensity_pre_process[xn]} '
              f'\npost-processed {intensity_post_process[xn]}'
              f'\n difference = {intensity_post_process[xn] - intensity_pre_process[xn]}'
              f'\n')

    if plot:
        # ==================
        # FFT of Tweeter Plane
        # ==================
        fig, subplot = plt.subplots(1, n_pairs, figsize=(14, 5))
        fig.subplots_adjust(wspace=0.5, right=0.85)
        fig.suptitle('FFT of Tweeter DM Plane')

        tweet = extract_plane(cpx_sequence, 'tweeter')  # eliminates astro_body axis [tsteps,wvl,obj,x,y]
        tweeter = np.sum(tweet, axis=(1, 2))
        for ax, ix in zip(subplot.flatten(), range(n_pairs)):
            fft_tweeter = (1 / np.sqrt(2 * np.pi) *
                           np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(tweeter[ix+n_pairs]))))
            intensity_DM_FFT = np.sum(np.abs(fft_tweeter*mask2D)**2)
            print(f'tweeter fft intensity = {intensity_DM_FFT}')
            # im = ax.imshow(np.abs(fft_tweeter*mask2D) ** 2,
            im = ax.imshow(np.abs(tweeter[ix+n_pairs, imn:imx, jmn:jmx]) ** 2,
                           interpolation='none', norm=LogNorm())  # ,
            # vmin=1e-3, vmax=1e-2)
            ax.set_title(f'Probe Phase ' r'$\theta$' f'={cdi.phase_cycle[ix] / np.pi:.2f}' r'$\pi$')

        cax = fig.add_axes([0.9, 0.2, 0.03, 0.6])  # Add axes for colorbar @ position [left,bottom,width,height]
        cb = fig.colorbar(im, orientation='vertical', cax=cax)  #
        cb.set_label('Intensity')

        # # ==================
        # # Deltas
        # # ==================
        fig, subplot = plt.subplots(1, n_pairs, figsize=(14,5))
        fig.subplots_adjust(wspace=0.5, right=0.85)
        fig.suptitle('Deltas for CDI Probes')

        for ax, ix in zip(subplot.flatten(), range(n_pairs)):
            im = ax.imshow(delta[ix]*1e6*mask2D, interpolation='none',
                           norm=SymLogNorm(linthresh=1),
                           # vmin=-1, vmax=1
                           ) #, norm=SymLogNorm(linthresh=1e-5))
            ax.set_title(f"Diff Probe\n" + r'$\theta$' + f'={cdi.phase_series[ix]/np.pi:.3f}' +
                         r'$\pi$ -$\theta$' + f'={cdi.phase_series[ix+n_pairs]/np.pi:.3f}' + r'$\pi$')

        cax = fig.add_axes([0.9, 0.2, 0.03, 0.6])  # Add axes for colorbar @ position [left,bottom,width,height]
        cb = fig.colorbar(im, orientation='vertical', cax=cax)  #
        cb.set_label('Intensity')

        # # ==================
        # # Original E-Field
        # # ==================
        # fig, subplot = plt.subplots(1, n_nulls, figsize=(14, 5))
        # fig.subplots_adjust(wspace=0.5, right=0.85)
        # fig.suptitle('Original (Null-Probe) E-field')
        #
        # for ax, ix in zip(subplot.flatten(), range(n_nulls)):
        #     im = ax.imshow(np.abs(fp_seq[n_pairs + ix, 250:270, 150:170]) ** 2,  # , 250:270, 150:170  *mask2D
        #                    interpolation='none', norm=LogNorm(),
        #                    vmin=1e-8, vmax=1e-2)
        #     ax.set_title(f'Null Step {ix}')
        #
        # cax = fig.add_axes([0.9, 0.2, 0.03, 0.6])  # Add axes for colorbar @ position [left,bottom,width,height]
        # cb = fig.colorbar(im, orientation='vertical', cax=cax)  #
        # cb.set_label('Intensity')

        # ===============================
        # E-filed Estimate--Pupil Plane
        # ===============================
        fig, subplot = plt.subplots(1, n_nulls, figsize=(14, 5))
        fig.subplots_adjust(wspace=0.5, right=0.85)
        fig.suptitle('DM Plane Estimated E-field')

        for ax, ix in zip(subplot.flatten(), range(n_nulls)):
            im = ax.imshow(np.abs(E_pupil[ix, irng[0]:irng[-1], jrng[0]:jrng[-1]])**2,  # , 250:270, 150:170  *mask2D
                           interpolation='none',
                           norm=LogNorm(),
                           vmin=1e-8, vmax=1
                           )  # , norm=SymLogNorm(linthresh=1e-5))
            ax.set_title(f'Null Step {ix}')

        cax = fig.add_axes([0.9, 0.2, 0.03, 0.6])  # Add axes for colorbar @ position [left,bottom,width,height]
        cb = fig.colorbar(im, orientation='vertical', cax=cax)  #
        cb.set_label('Intensity')

        # ===============================
        # E-filed Estimate--Focal Plane
        # ===============================
        fig, subplot = plt.subplots(1, n_nulls, figsize=(14, 5))
        fig.subplots_adjust(wspace=0.5, right=0.85)
        fig.suptitle('Focal Plane (rough FFT) Estimated E-field')

        for ax, ix in zip(subplot.flatten(), range(n_nulls)):
            im = ax.imshow(np.abs(E_ff[ix]) ** 2,  # , 250:270, 150:170  *mask2D
                           interpolation='none',
                           norm=LogNorm(),
                           # vmin=1e-8, vmax=1e-2
                           )  # , norm=SymLogNorm(linthresh=1e-5))
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
            # im = ax.imshow(np.abs(fp_seq[n_pairs+ix] - np.conj(E_pupil[ix]*mask2D))**2,
            im = ax.imshow(I_processed[ix],  # I_processed[ix, 250:270, 180:200]  I_processed[ix]
                           interpolation='none', norm=SymLogNorm(1e4),  # ,
                           # vmin=-1e-6, vmax=1e-6
                           )
            ax.set_title(f'Null Step {ix}')

        cax = fig.add_axes([0.9, 0.2, 0.03, 0.6])  # Add axes for colorbar @ position [left,bottom,width,height]
        cb = fig.colorbar(im, orientation='vertical', cax=cax)  #
        cb.set_label('Intensity')

        # ==================
        # View Time Series
        # ==================
        view_timeseries(cpx_to_intensity(fp_seq), cdi, title=f"White Light Timeseries",
                        subplt_cols=sp.tseries_cols,
                        logZ=False,
                        # vlim=(1e-7, 1e-4),
                        )

        plt.show()


##
def get_fp_mask(cdi, thresh=1e-7):
    """
    returns a mask of the CDI probe pattern in focal plane coordinates

    :param cdi: structure containing all CDI probe parameters
    :param thresh: intensity threshold for determining probed coordinates
    :return: fp_mask: boolean array where True marks the probed coordinates
             imsk, jmsk:
             irng, jrng:

    """
    nx = sp.grid_size
    ny = sp.grid_size
    dm_act = cdi.nact

    fftA = (1 / np.sqrt(2 * np.pi) *
            np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(cdi.out.DM_probe_series[0]))))

    Ar = interpolate.interp2d(range(dm_act), range(dm_act), fftA.real, kind='cubic')
    Ai = interpolate.interp2d(range(dm_act), range(dm_act), fftA.imag, kind='cubic')
    ArI = Ar(np.linspace(0, dm_act, ny), np.linspace(0, dm_act, nx))
    AiI = Ai(np.linspace(0, dm_act, ny), np.linspace(0, dm_act, nx))

    fp_probe = np.sqrt(ArI**2 + AiI**2)
    fp_mask = (fp_probe > 1e-7)
    (imsk, jmsk) = (fp_probe > 1e-7).nonzero()

    irng = range(min(imsk), max(imsk), 1)
    jrng = range(min(jmsk), max(jmsk), 1)

    imx = max(irng)-1  # -1 is to get index values for plotting purposes
    imn = min(irng)-1
    jmx = max(jrng)-1
    jmn = min(jrng)-1

    return fp_mask, imsk, jmsk, irng, jrng, imx, imn, jmx, jmn


##
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


