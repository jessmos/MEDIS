import numpy as np
import matplotlib.pylab as plt
from scipy import special

from mkidpipeline.speckle.genphotonlist_IcIsIr import MRicdf, corrsequence

from medis.medis_main import RunMedis
from medis.utils import dprint
from medis.plot_tools import quick2D, grid
from medis.params import sp, ap, tp, mp, iop, atmp

sp.numframes = 2000 #500
# sp.checkpointing = 1
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
atmp.cn_sq = 0.1e-11
atmp.correlated_sampling = True
# sp.skip_planes = ['coronagraph']


TESTDIR = 'corr_atmos'

sp.save_sim_object = False
sp.save_to_disk = True
sp.debug = False
atmp.vel = np.array([3,4,3,4,2])

if __name__ == '__main__':
    # sp.debug = False
    # sp.numframes = 3
    # ap.n_wvl_init = 2
    # ap.companion = False
    # sp.save_to_disk = True
    # sp.sample_time = 0.1
    # tp.ao_act = 40
    sp.quick_detect = False
    mp.hot_counts = False
    mp.dark_counts  = False
    mp.bad_pix = True
    sim = RunMedis(name=f'{TESTDIR}/test', product='photons')
    observation = sim()

    normal = corrsequence(1000,1)[1]
    uniform = 0.5*(special.erf(normal/np.sqrt(2)) + 1)

# f = MRicdf(1000,300)
#
# plt.plot(normal)
# plt.plot(uniform)
#
# plt.figure()
#
# MR1 = f(uniform)
# MR2 = f(np.linspace(np.min(uniform),np.max(uniform),1000))
#
# plt.plot(MR1)
# plt.plot(MR2)
#
# plt.figure()
#
# plt.hist(MR1, alpha=0.6)
# plt.hist(MR2, alpha=0.6)
#
# plt.show()

