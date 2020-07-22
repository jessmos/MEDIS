"""
Generates test data for Enrico eb15500.2015@my.bristol.ac.uk

"""

import numpy as np
import matplotlib.pyplot as plt

from medis.medis_main import RunMedis
from medis.utils import dprint
from medis.plot_tools import grid, quick2D
# from mkidpipeline.hdf.photontable import Photontable

from medis.params import sp, mp, iop, atmp, tp, ap


sp.checkpointing = None
sp.grid_size = 512
sp.beam_ratio = 0.15
sp.debug = False
# sp.skip_planes = ['coronagraph']
sp.quick_detect = False
sp.save_sim_object = False
sp.save_to_disk = True
sp.numframes = 100
sp.sample_time = 0.1
sp.num_processes = 10

ap.n_wvl_init = 10
ap.n_wvl_final = 10
ap.star_flux = 1.1*1e8
ap.companion = False
ap.wvl_range = np.array([400, 1400]) / 1e9

atmp.correlated_sampling = False
atmp.model = 'single'
# atmp.cn_sq = 0.5e-11
atmp.cn_sq = 1e-11

tp.cg_type = 'Solid'
tp.satelite_speck['apply'] = True
tp.satelite_speck['amp'] = 12e-10
tp.prescription = 'general_telescope'
tp.obscure = False
tp.use_ao = True
tp.ao_act = 50

TESTDIR = f'/mnt/data0/dodkins/MEDIS_photonlists/PhotonStatistics/200708/{sp.numframes}'

if __name__ == '__main__':
    for cg_type in ['Solid', None]:
        input('rename fields before proceeding')
        tp.cg_type = cg_type
        sim = RunMedis(name=TESTDIR, product='fields')
        observation = sim()

        grid(observation['fields'], show=True)  #, vlim=(-2e-7,2e-7))
