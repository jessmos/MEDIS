"""
Barebones

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from medis.medis_main import RunMedis
from medis.utils import dprint
from medis.plot_tools import quick2D, grid
from medis.params import sp, ap, tp, mp, iop, atmp

sp.numframes = 2000 #2000 #500
sp.checkpointing = 20
sp.num_processes = 10
ap.companion_xy = [[2,0]]
ap.contrast = [1e-4]
ap.companion = True
ap.n_wvl_init = 1
ap.n_wvl_final = 1
tp.cg_type = 'Solid'
sp.sample_time = 0.5e-3
sp.grid_size = 512
ap.star_flux = 1.1*1e8
tp.satelite_speck['apply'] = True
tp.satelite_speck['amp'] = 12e-10
sp.beam_ratio = 0.15
tp.prescription = 'general_telescope' #'Subaru_SCExAO'  #
tp.obscure = False
tp.use_ao = True
sp.save_to_disk = False
sp.debug = False
tp.ao_act = 50
mp.array_size = np.array([140, 144])
atmp.cn_sq = 0.5e-11
# sp.skip_planes = ['coronagraph']
atmp.correlated_sampling = False
atmp.model = 'hcipy_standard'
# atmp.model = 'single'


TESTDIR = 'SSD'

sp.save_sim_object = False
sp.save_to_disk = True
sp.debug = False

# iop.update_datadir('/mnt/data0/dodkins/MKIDSim/')
iop.update_datadir('/mnt/data0/dodkins/MEDIS_photonlists/')

if __name__ == '__main__':
    sp.quick_detect = False
    mp.hot_counts = False
    mp.dark_counts  = False
    mp.bad_pix = True
    sim = RunMedis(name=f'{TESTDIR}/hcipy_standard', product='photons')
    # sim = RunMedis(name=f'{TESTDIR}/test3', product='rebinned_cube')
    observation = sim()
    print(observation.keys(), )

    # grid(observation['rebinned_cube'], vlim= (0,3))
    # plt.hist(observation['photonls'][0])
    # plt.show()

    grid(sim.sim.rebin_list(observation['photons'])[::100], vlim= (0,3))

    from mkidpipeline.hdf.photontable import ObsFile

    # to look at the photontable
    obs = ObsFile(iop.photonlist)
    print(obs.photonTable)
    # to plot an image of all the photons on the array
    image = obs.getPixelCountImage(integrationTime=None)['image']
    plt.imshow(image)
    plt.show(block=True)