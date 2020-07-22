"""
This module creates an observation and tests the speckle statistics

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib
import pickle

# from mkidpipeline.speckle.binned_rician import plotLogLMap
from mpmath import mp, hyp1f1

from medis.medis_main import RunMedis
from medis.utils import dprint
from medis.plot_tools import quick2D, grid
from medis.twilight_colormaps import sunlight
from medis.params import sp, ap, tp, iop, mp, atmp

sp.numframes = 100 #200 # 2000  # 1000
ap.companion_xy = [[2,0], [-2,0]]
ap.companion = True
ap.spectra = [None, None, None]
ap.contrast = [1e-3, 1e-2]
ap.n_wvl_init = 1
ap.n_wvl_final = 1
tp.cg_type = 'Solid'
sp.sample_time = 0.5e-3  #0.5
sp.grid_size = 512
ap.star_flux = 1e9
tp.satelite_speck['apply'] = False
sp.beam_ratio = 0.15
tp.prescription = 'general_telescope' # 'Subaru_SCExAO'  #
tp.obscure = False
tp.use_ao = True
sp.save_to_disk = True
sp.debug = False
atmp.correlated_sampling = False
atmp.model = 'single'
tp.ao_act = 50
# sp.debug = True
# sp.skip_planes = ['coronagraph']

TESTDIR = 'GJ876'

def binMRlogL(n, Ic, Is):
    '''
    Given a light curve, calculate the Log likelihood that
    its intensity distribution follows a blurred modified Rician with Ic, Is.

    INPUTS:
        n: 1d array containing the (binned) intensity as a function of time, i.e. a lightcurve [counts/sec]. Bin size must be fixed.
        Ic: Coherent portion of MR [1/second]
        Is: Speckle portion of MR [1/second]
    OUTPUTS:
        [float] the Log likelihood.

    '''
    lnL = np.zeros(len(n))
    tmp = np.zeros(len(n))
    for ii in range(len(n)):  # hyp1f1 can't do numpy arrays because of its data type, which is mpf
        tmp[ii] = float(hyp1f1(n[ii] + 1, 1, Ic / (Is ** 2 + Is)))
    # dprint(np.log(1. / (Is + 1)))
    # dprint(- n * np.log(1 + 1. / Is))
    # dprint(- Ic / Is)
    # dprint(np.log(tmp))
    lnL = np.log(1. / (Is + 1)) - n * np.log(1 + 1. / Is) - Ic / Is + np.log(tmp)
    # dprint(np.sum(lnL))

    return np.sum(lnL)


def plotLogLMap(n, Ic_list, Is_list, effExpTime):
    """
    plots a map of the MR log likelihood function over the range of Ic, Is

    INPUTS:
        n - light curve [counts]
        Ic_list - list of Ic values [photons/second]
        Is_list - list
    OUTPUTS:

    """

    Ic_list_countsperbin = Ic_list * effExpTime  # convert from cps to counts/bin
    Is_list_countsperbin = Is_list * effExpTime

    im = np.zeros((len(Ic_list), len(Is_list)))

    for i, Ic in enumerate(Ic_list_countsperbin):  # calculate maximum likelihood for a grid of
        for j, Is in enumerate(Is_list_countsperbin):  # Ic,Is values using counts/bin
            print('Ic,Is = ', Ic / effExpTime, Is / effExpTime)
            lnL = binMRlogL(n, Ic, Is)
            im[i, j] = lnL

    Ic_ind, Is_ind = np.unravel_index(im.argmax(), im.shape)
    print('Max at (' + str(Ic_ind) + ', ' + str(Is_ind) + ')')
    print("Ic=" + str(Ic_list[Ic_ind]) + ", Is=" + str(Is_list[Is_ind]))
    print(im[Ic_ind, Is_ind])

    #    l_90 = np.percentile(im, 90)
    #    l_max=np.amax(im)
    #    l_min=np.amin(im)
    #    levels=np.linspace(l_90,l_max,int(len(im.flatten())*.1))

    plt.figure()
    #    plt.contourf(Ic_list, Is_list,im.T,levels=levels,extend='min')

    X, Y = np.meshgrid(Ic_list, Is_list)
    sigmaLevels = np.array([8.36, 4.78, 2.1])
    levels = np.amax(im) - sigmaLevels

    MYSTYLE = {'contour.negative_linestyle': 'solid'}
    oldstyle = {key: matplotlib.rcParams[key] for key in MYSTYLE}
    matplotlib.rcParams.update(MYSTYLE)

    #    plt.contourf(X,Y,im.T)
    plt.imshow(im.T, extent=[np.amin(Ic_list), np.amax(Ic_list), np.amin(Is_list), np.amax(Is_list)], aspect='auto',
               origin='lower')
    plt.contour(X, Y, im.T, colors='black', levels=levels)
    plt.plot(Ic_list[Ic_ind], Is_list[Is_ind], "xr")
    plt.xlabel('Ic [/s]')
    plt.ylabel('Is [/s]')
    plt.title('Map of log likelihood')

    matplotlib.rcParams.update(oldstyle)

    return X, Y, im

class Stats_Visualiser():
    def __init__(self, monofields, steps, I_bins, t_bins, pupil_xys, focal_xys=None, savefile=None):
        plt.ion()
        plt.show(block=True)

        self.savefile = savefile

        # self.planes = [np.where(sp.save_list == 'atmosphere')[0][0], np.where(sp.save_list == 'detector')[0][0]]
        self.planes = [np.where(sp.save_list == 'detector')[0][0]]
        self.fig, self.axes = plt.subplots(len(self.planes), 5, figsize=(17, 5))
        if len(self.planes) == 1:
            self.axes = self.axes[np.newaxis]

        if not focal_xys:
            focal_xys = pupil_xys
        # self.all_xys = [pupil_xys, focal_xys]
        self.all_xys = [pupil_xys]

        props = dict(boxstyle='square', facecolor='k', alpha=0.5)
        self.colors = [f'C{i}' for i in range(len(pupil_xys))]
        xlabels = ['x', r'$E_{real}$', 'time', 'time', 'intensity']
        ylabels = ['y', r'$E_{imag}$', 'phase', 'intensity', 'amount']

        for i, (plane, xys) in enumerate(zip(self.planes, self.all_xys)):

            self.axes[i, 0].imshow(np.sum(np.abs(monofields[0, plane, :]) ** 2, axis=0), origin='lower', norm=LogNorm())
            self.axes[i, 0].text(0.1, 0.1, sp.save_list[plane], transform=self.axes[i, 0].transAxes, fontweight='bold',
                            color='w', fontsize=16, bbox=props)

            for ip, xy in enumerate(xys):
                x, y = xy
                circle = plt.Circle((y, x), 9, color=self.colors[ip], fill=False, linewidth=2)
                self.axes[i, 0].add_artist(circle)

            [self.axes[i, ix].set_ylabel(ylabel) for ix, ylabel in enumerate(ylabels)]
        [self.axes[0, ix].set_xlabel(xlabel) for ix, xlabel in enumerate(xlabels)]
        self.axes[0, 1].legend()

        self.ih = 0
        self.iv = 0
        self.iz = 0
        self.ims = []
        self.save = False
        self.plot = False

        def on_key(event):
            if event.key == 'right':
                self.ih += 1
            elif event.key == 'left':
                self.ih -= 1
            elif event.key == 'up':
                self.iv += 1
            elif event.key == 'down':
                self.iv -= 1
            elif event.key == 'x':
                self.iz += 1
            elif event.key == 'z':
                self.iz -= 1
            elif event.key == 'd':
                self.save = True
            elif event.key == 'p':
                self.plot = True
            step = steps[self.ih]
            I_bin = I_bins[self.iv]
            t_bin = t_bins[self.iz]
            print('ih: ', self.ih, 'iv: ', self.iv, 'iz: ', self.iz, 'step num: ', step, 'intensity bin width: ', I_bin,
                  'time bin width: ', t_bin)
            self.draw(monofields, step, I_bin, t_bin, self.save, self.plot)
            self.save = False
            self.plot = False

        self.fig.canvas.mpl_connect('key_press_event', on_key)
        plt.tight_layout()

    def draw(self, monofields, step, I_bin, t_bin, save, plot):
        if len(self.ims) > 0:
            [im.remove() for im in self.ims]
            self.ims = []

        dprint(step)
        for i, (plane, xys) in enumerate(zip(self.planes, self.all_xys)):

            for ip, xy in enumerate(xys):
                x, y = xy
                amplitude_series = np.sum(monofields[:, plane, :, x, y], axis=1)[:step]  # containing all the objects
                # amplitude_series = self.sum_chunk(amplitude_series, t_bin)

                timesteps = (np.arange(sp.numframes) * sp.sample_time)[:step]
                # timesteps = timesteps[::t_bin]

                intensity = np.abs(amplitude_series) ** 2

                binned_intensity = self.sum_chunk(intensity, t_bin)
                I, bins = np.histogram(binned_intensity, bins=np.arange(np.min(binned_intensity), np.max(binned_intensity), I_bin))

                self.ims.append(self.axes[i, 1].plot(amplitude_series.real, amplitude_series.imag, marker='o',
                                label=str(xy), color=self.colors[ip])[0])
                self.ims.append(self.axes[i, 2].plot(timesteps, np.angle(amplitude_series),
                                marker='o', color=self.colors[ip])[0])
                self.ims.append(self.axes[i, 3].plot(timesteps, intensity, marker='o', color=self.colors[ip])[0])

                self.ims.append(self.axes[i, 4].step(bins[:-1], I, color=self.colors[ip])[0])

                if plot:
                    plotLogLMap(intensity, np.arange(0,250,25), np.arange(0,250,25), effExpTime=t_bin)

                # print(plotLogLMap(I))
                if save:
                    with open(self.savefile, 'ab') as handle:
                        # pickle.dump([bins[:-1], I, t_bin], handle, protocol=pickle.HIGHEST_PROTOCOL)
                        pickle.dump((intensity, t_bin), handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.fig.canvas.draw()

    def sum_chunk(self, x, chunk_size, axis=-1):
        shape = x.shape
        if axis < 0:
            axis += x.ndim
        shape = shape[:axis] + (-1, chunk_size) + shape[axis + 1:]
        x = x.reshape(shape)
        return x.sum(axis=axis + 1)

def investigate_quantized():

    sp.quick_detect = True
    mp.array_size = np.array([100,100])
    sp.verbose = True
    sp.save_to_disk = True
    center = mp.array_size[0]//2
    contrasts = range(-1, -8, -3)
    print(contrasts)
    locs = [[center+33, center], [center, center], [center, center-33], [center-24, center+24]]
    objects = ['satelite', 'star', 'planet', 'speckle']
    # fig, axes = plt.subplots(2, len(contrasts))
    name = f'{TESTDIR}/subaru/'

    axes_list = []
    atmosdir = iop.atmosdir.split('/')[-1]
    iop.atmosdir = os.path.join(iop.datadir, name, iop.atmosroot, atmosdir)
    for c, contrast in enumerate(contrasts):
        axes_list.append( plt.subplots(2, len(locs))[1] )
        ap.contrast = 10 ** np.array([float(contrast)])
        sim = RunMedis(name=name+f'1e{contrast}', product='photons')
        iop.atmosdir = os.path.join(iop.datadir, name, iop.atmosroot, atmosdir)
        iop.aberdir = os.path.join(iop.datadir, name, iop.aberroot, atmosdir)
        observation = sim()
        print(observation.keys(), observation['photons'].shape, observation['rebinned_photons'].shape)
        grid(observation['rebinned_photons'][:10], logZ=True, nstd=10, show=False, vlim=(0,2))
        timecube = np.sum(observation['rebinned_photons'], axis=1)
        # plot_stats(timecube, locs, names)
        for l, (loc, obj) in enumerate(zip(locs, objects)):
            print(l, loc, obj)
            bins_list, I_list, timesamps, lc_list = pixel_stats(timecube, loc)
            axes_list[c][0, l].plot(timesamps, lc_list[0])
            axes_list[c][1, l].plot(bins_list[0][:-1], I_list[0])
            axes_list[c][0, l].set_xlabel('time samples')
            axes_list[c][0, l].set_ylabel('intensity')
            axes_list[c][1, l].set_xlabel('intensity')
            axes_list[c][1, l].set_ylabel('amount')
            axes_list[c][0, l].set_title(f'{obj} contrast 10^{contrast}')
    plt.show()

if __name__ == '__main__':
    name = f'{TESTDIR}/frames={sp.numframes}_tau={atmp.tau}_comp_multiproc'

    planet1_x = 189
    planet1_xy = [256, planet1_x]

    planet2_x = sp.grid_size - planet1_x
    planet2_xy = [256, planet2_x]

    steps = range(0, sp.numframes + 1, 10)
    I_bins = np.logspace(-6, -10, 7)
    t_bins = [1, 2, 5, 10, 20, 50, 100, 200, 500]  # 2**np.arange(0,10)
    pupil_xys = [[sp.grid_size // 2, sp.grid_size // 2], [275, 275], [290, 260], [375, 375], [310, 310], planet1_xy]

    iop.update_datadir('/mnt/data0/dodkins/MKIDSim/')
    iop.update_testname(name)
    lightcurvefile = os.path.join(iop.testdir, 'lightcurves.pkl')
    print(lightcurvefile)
    lightcurves = []
    if os.path.exists(lightcurvefile):
        with open(lightcurvefile, 'rb') as handle:
            for i in range(len(pupil_xys)):
                # for i in range(2):
                print(i)
                lightcurves.append(pickle.load(handle))

        for lightcurve, t_bin in lightcurves:
            lightcurve *= 1e8
            # mean = np.mean(lightcurve)
            # std = np.std(lightcurve)
            print(lightcurve)
            plt.figure()
            plt.plot(lightcurve)
            # plt.show()
            # plotLogLMap(lightcurve, np.linspace(0, max(lightcurve), 10), np.linspace(1, max(lightcurve) + 1, 10),
            #             effExpTime=t_bin)
            plotLogLMap(lightcurve, np.arange(0, 25, 5), np.arange(1, 26, 5), effExpTime=t_bin)
            plt.show()

        plt.show()
    else:
        sp.save_list = np.array(['atmosphere', 'detector'])
        sim = RunMedis(name=name, product='fields')
        observation = sim()

        fields = observation['fields']

        # spectral_train_grid = np.concatenate((fields[0, :, :, 0].imag, fields[0, :, :, 0].real), axis=1)
        # nplanes = len(sp.save_list)
        # fig, axes = plt.subplots(2, nplanes, figsize=(14, 7))
        # print(axes.shape, spectral_train_grid.shape)
        # for i in range(nplanes):
        #     for j in range(2):
        #         axes[j,i].imshow(spectral_train_grid[i,j])
        #     axes[0, i].set_title(sp.save_list[i])
        # plt.tight_layout()

        monofields = fields[:, :, 0]  # keep all dimensions apart from wavelength

        Stats_Visualiser(monofields, steps, I_bins, t_bins, pupil_xys, savefile=lightcurvefile)

        plt.show(block=True)
