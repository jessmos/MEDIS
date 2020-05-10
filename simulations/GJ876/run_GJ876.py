"""
This module creates an observation of the GJ 876 system as sampled with Subaru/SCExAO


"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from medis.medis_main import RunMedis
from medis.utils import dprint
from medis.plot_tools import quick2D, grid
from medis.twilight_colormaps import sunlight
from medis.params import sp, ap, tp, iop, mp

sp.numframes = 500  # 2000  # 1000
ap.companion_xy = [[2,0], [-2,0]]
ap.companion = True
ap.spectra = [None, None, None]
ap.contrast = [1e-3, 1e-2]
ap.n_wvl_init = 1
ap.n_wvl_final = 1
tp.cg_type = 'Solid'
sp.sample_time = 0.5e-3
sp.grid_size = 512
ap.star_flux = 1e9
tp.satelite_speck['apply'] = False
sp.beam_ratio = 0.15
tp.prescription = 'general_telescope' # 'Subaru_SCExAO'  #
tp.obscure = False
tp.use_ao = True
sp.save_to_disk = True
sp.debug = False
tp.ao_act = 50
# sp.debug = True
# sp.skip_planes = ['coronagraph']

TESTDIR = 'GJ876'

class Stats_Visualiser():
    def __init__(self, monofields, steps, I_bins, pupil_xys, focal_xys=None):
        plt.ion()
        plt.show(block=True)

        self.planes = [np.where(sp.save_list == 'atmosphere')[0][0], np.where(sp.save_list == 'detector')[0][0]]
        self.fig, self.axes = plt.subplots(len(self.planes), 5, figsize=(17, 5))

        if not focal_xys:
            focal_xys = pupil_xys
        self.all_xys = [pupil_xys, focal_xys]
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
        [self.axes[1, ix].set_xlabel(xlabel) for ix, xlabel in enumerate(xlabels)]
        self.axes[0, 1].legend()

        self.ih = 0
        self.iv = 0
        self.ims = []

        def on_key(event):
            if event.key == 'right':
                self.ih += 1
            elif event.key == 'left':
                self.ih -= 1
            elif event.key == 'up':
                self.iv += 1
            elif event.key == 'down':
                self.iv -= 1
            step = steps[self.ih]
            I_bin = I_bins[self.iv]
            print('ih: ', self.ih, 'iv: ', self.iv, 'step num: ', step, 'intensity bin width: ', I_bin)
            self.draw(monofields, step, I_bin)

        self.fig.canvas.mpl_connect('key_press_event', on_key)
        plt.tight_layout()

    def draw(self, monofields, step, I_bin):
        if len(self.ims) > 0:
            [im.remove() for im in self.ims]
            self.ims = []

        for i, (plane, xys) in enumerate(zip(self.planes, self.all_xys)):

            for ip, xy in enumerate(xys):
                x, y = xy
                timecube = np.sum(monofields[:step, plane, :, x, y], axis=1)  # containing all the objects
                self.ims.append(self.axes[i, 1].plot(timecube.real, timecube.imag, marker='o',
                                label=str(xy), color=self.colors[ip])[0])
                self.ims.append(self.axes[i, 2].plot(np.arange(sp.numframes)[:step] * sp.sample_time,
                                                     np.angle(timecube),
                                marker='o', color=self.colors[ip])[0])
                intensity = np.abs(timecube) ** 2

                self.ims.append(self.axes[i, 3].plot(np.arange(sp.numframes)[:step] * sp.sample_time,
                                                     intensity, marker='o', color=self.colors[ip])[0])

                I, bins = np.histogram(intensity, bins=np.arange(np.min(intensity), np.max(intensity), I_bin))
                self.ims.append(self.axes[i, 4].step(bins[:-1], I, color=self.colors[ip])[0])

        self.fig.canvas.draw()

def investigate_fields():

    sp.save_list = np.array(['atmosphere', 'detector'])
    sim = RunMedis(name=f'{TESTDIR}/fieldscomp', product='fields')
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

    planet_x = 189
    anti_planet_x = sp.grid_size - planet_x
    monofields = fields[:, :, 0]  # keep all dimensions apart from wavelength

    steps = range(0,500,100)
    I_bins = np.logspace(-6, -9, 7)
    pupil_xys = [[sp.grid_size//2, sp.grid_size//2], [275, 275], [290, 260], [375, 375], [256,189], [256, anti_planet_x]]

    Stats_Visualiser(monofields, steps,I_bins, pupil_xys)

    plt.show(block=True)

def investigate_stats():
    sp.grid_size = 512
    sim = RunMedis(name=f'{TESTDIR}/fields', product='fields')
    observation = sim()

    fields = np.abs(observation['fields'][:,-1])**2
    timecube = np.sum(fields[:,0], axis=1)
    grid([np.sum(timecube, axis=0)], show=False)

    locs = [[210,210], [256,256], [256,206], [256,512-206]]
    names = ['satelite', 'star', 'planet', 'speckle']

    plot_stats(timecube, locs, names)

def plot_stats(timecube, xys, names):

    bins_list, I_list, timesamps, lc_list = pixel_stats(timecube, xys)

    fig, axes = plt.subplots(2, len(bins_list))

    for i, (bins, I, lightcurve, name) in enumerate(zip(bins_list, I_list, lc_list, names)):

        axes[0, i].plot(timesamps, lightcurve)
        axes[1, i].plot(bins[:-1], I)
        axes[0, i].set_xlabel('time samples')
        axes[0, i].set_ylabel('intensity')
        axes[1, i].set_xlabel('intensity')
        axes[1, i].set_ylabel('amount')
        axes[0, i].set_title(name)
    plt.show()

def pixel_stats(timecube, xys):
    assert timecube.ndim == 3
    xys = np.array(xys)
    print(xys.shape)
    if xys.ndim == 1:
        xys = [xys]

    timesamps = np.arange(len(timecube))*sp.sample_time

    lc_list, bins_list, I_list = [], [], []
    for xy in xys:
        lightcurve = timecube[:, xy[0], xy[1]]
        I, bins = np.histogram(lightcurve, bins=75)
        lc_list.append(lightcurve)
        bins_list.append(bins)
        I_list.append(I)

    return bins_list, I_list, timesamps, lc_list

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
        print(observation.keys(), observation['photons'].shape, observation['stackcube'].shape)
        grid(observation['stackcube'][:10], logZ=True, nstd=10, show=False, vlim=(0,2))
        timecube = np.sum(observation['stackcube'], axis=1)
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
    investigate_fields()
    # investigate_stats()
    # investigate_quantized()