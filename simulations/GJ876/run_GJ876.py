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

sp.numframes = 5  # 2000  # 1000
ap.companion_xy = [[2,0]]
ap.companion = True
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
# sp.skip_planes = ['coronagraph']

TESTDIR = 'GJ876'

def investigate_fields():

    ap.contrast = 10**np.array([-1.])
    if tp.prescription == 'general_telescope':
        sp.save_list = np.array(
            ['atmosphere', 'CPA', 'deformable mirror', 'NCPA', 'pre_coron', 'detector'])
    else:
        sp.save_list = np.array(
            ['atmosphere', 'effective-primary', 'ao188-OAP1', 'woofer', 'ao188-OAP2',  'tweeter', 'focus', 'detector'])
    sim = RunMedis(name=f'{TESTDIR}/fields', product='fields')
    observation = sim()
    print(observation.keys(), )

    fields = observation['fields']
    grid(fields, logZ=True, nstd=2, show=False)
    spectral_train_phase = np.angle(fields[0, :-2, :, 0])
    spectral_train_amp = np.abs(fields[0, -2:, :, 0] ** 2)
    spectral_train_grid = np.concatenate((spectral_train_phase,spectral_train_amp), axis=0)
    nplanes = len(sp.save_list)
    fig, axes = plt.subplots(1, nplanes, figsize=(14,7))

    for i in range(nplanes-2):
        im1 = axes[i].imshow(spectral_train_grid[i, 0], cmap=sunlight, vmin=-np.pi, vmax=np.pi)
        axes[i].set_title(sp.save_list[i])
    for i in range(nplanes-2,nplanes):
        im3 = axes[i].imshow(spectral_train_grid[i,0], cmap='inferno', norm=LogNorm(), vmin=1e-8, vmax=1e-3)
        axes[i].set_title(sp.save_list[i])
    plt.tight_layout()
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
    # investigate_fields()
    # investigate_stats()
    investigate_quantized()
