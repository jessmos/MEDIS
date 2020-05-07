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
from medis.params import params

params['sp'].numframes = 1  # 2000  # 1000
params['ap'].companion_xy = [[0.5,0]]
params['ap'].n_wvl_init = 1
params['ap'].n_wvl_final = 1
params['tp'].cg_type = 'Vortex'
params['sp'].sample_time = 0.5e-3
params['sp'].grid_size = 512
# params['sp'].grid_size = 1024
params['sp'].star_flux = 1e9
params['tp'].satelite_speck = False
params['sp'].beam_ratio = 0.15
params['tp'].prescription = 'general_telescope' # 'Subaru_SCExAO'  #
params['tp'].obscure = False
params['tp'].use_ao = True
params['sp'].save_to_disk = False
params['sp'].debug = True

TESTDIR = 'GJ876'

def investigate_fields():
    params['ap'].contrast = 10**np.array([-1.])
    if params['tp'].prescription == 'general_telescope':
        params['sp'].save_list = np.array(
            ['atmosphere', 'CPA', 'deformable mirror', 'NCPA', 'pre_coron', 'detector'])
    else:
        params['sp'].save_list = np.array(
            ['atmosphere', 'effective-primary', 'ao188-OAP1', 'woofer', 'ao188-OAP2',  'tweeter', 'focus', 'detector'])
    sim = RunMedis(params=params, name=f'{TESTDIR}/fields', product='fields')
    observation = sim()
    print(observation.keys(), )

    fields = observation['fields']
    grid(fields, logZ=True, nstd=2, show=False)
    spectral_train_phase = np.angle(fields[0, :-2, :, 0])
    spectral_train_amp = np.abs(fields[0, -2:, :, 0] ** 2)
    spectral_train_grid = np.concatenate((spectral_train_phase,spectral_train_amp), axis=0)
    nplanes = len(params['sp'].save_list)
    fig, axes = plt.subplots(1, nplanes, figsize=(14,7))

    for i in range(nplanes-2):
        im1 = axes[i].imshow(spectral_train_grid[i, 0], cmap=sunlight, vmin=-np.pi, vmax=np.pi)
        axes[i].set_title(params['sp'].save_list[i])
    for i in range(nplanes-2,nplanes):
        im3 = axes[i].imshow(spectral_train_grid[i,0], cmap='inferno', norm=LogNorm(), vmin=1e-8, vmax=1e-3)
        axes[i].set_title(params['sp'].save_list[i])
    plt.tight_layout()
    plt.show()

def investigate_stats():
    params['sp'].grid_size = 512
    sim = RunMedis(params=params, name=f'{TESTDIR}/fields', product='fields')
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

    timesamps = np.arange(len(timecube))*params['sp'].sample_time

    lc_list, bins_list, I_list = [], [], []
    for xy in xys:
        lightcurve = timecube[:, xy[0], xy[1]]
        I, bins = np.histogram(lightcurve, bins=20)
        lc_list.append(lightcurve)
        bins_list.append(bins)
        I_list.append(I)

    return bins_list, I_list, timesamps, lc_list

def investigate_quantized():

    params['sp'].quick_detect = True
    params['mp'].array_size = np.array([100,100])
    center = params['mp'].array_size//2
    contrasts = range(-1, -8, -3)
    print(contrasts)
    locs = [[center-24, center-24], [center, center], [center, center-5], [center, center+5]]
    fig, axes = plt.subplots(2, len(contrasts))

    name = f'{TESTDIR}/subaru/'

    atmosdir = params['iop'].atmosdir.split('/')[-1]
    params['iop'].atmosdir = os.path.join(params['iop'].datadir, name, params['iop'].atmosroot, atmosdir)
    for i, contrast in enumerate(contrasts):
        params['ap'].contrast = 10 ** np.array([float(contrast)])
        sim = RunMedis(params=params, name=name+f'1e{contrast}', product='photons')
        params['iop'].atmosdir = os.path.join(params['iop'].datadir, name, params['iop'].atmosroot, atmosdir)
        params['iop'].aberdir = os.path.join(params['iop'].datadir, name, params['iop'].aberroot, atmosdir)
        observation = sim()
        print(observation.keys(), observation['photons'].shape, observation['stackcube'].shape)
        grid(observation['stackcube'][0], logZ=True, nstd=10, show=False, vlim=(0,2))
        timecube = np.sum(observation['stackcube'], axis=1)
        # plot_stats(timecube, locs, names)
        bins_list, I_list, timesamps, lc_list = pixel_stats(timecube, locs[2])
        axes[0, i].plot(timesamps, lc_list[0])
        axes[1, i].plot(bins_list[0][:-1], I_list[0])
        axes[0, i].set_xlabel('time samples')
        axes[0, i].set_ylabel('intensity')
        axes[1, i].set_xlabel('intensity')
        axes[1, i].set_ylabel('amount')
        axes[0, i].set_title(f'planet contrast 10^{contrast}')
    plt.show()

if __name__ == '__main__':
    investigate_fields()
    # investigate_stats()
    # investigate_quantized()
