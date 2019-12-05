"""
model the Subaru optics system

This is a code modified from Rupert's original optics_propagate.py. This code adds more optics to the system,
as well as puts the AO, coronagraphs, etc in order for Subaru.

Here, we will add the basic functionality of the Subaru Telescope, including the primary, secondary, and AO188.
The SCExAO system sits behind the AO188 instrument of Subaru, which is a 188-element AO system located at the
Nasmyth focus (IR) of the telescope. AO188 uses laser guide-star technology. More info here:
https://subarutelescope.org/Introduction/instrument/AO188.html
We then will use just a basic focal lens and coronagraph in this example. A more detailed model of SCExAO will
be modelled in a SCExAO_optics.py code. However, this routine is designed for simple simulations that need to
optimize runtime but still have relevance to the Subaru Telescope.

Here, we do not include the final micro-lens array of MEC or any other device.

This script is meant to override any Subaru/SCExAO-specific parameters specified in the user's params.py
"""

import numpy as np
import proper

from mm_params import ap, tp, sp
from mm_utils import dprint
import optics as opx
import aberrations as aber
import adaptive as ao
import atmosphere as atmos


#################################################################################################
#################################################################################################
#################################################################################################
#iop.update('Subaru-basic-test1')

# Defining Subaru parameters
# ----------------------------
# According to Iye-et.al.2004-Optical_Performance_of_Subaru:AstronSocJapan, the AO188 uses the IR-Cass secondary,
# but then feeds it to the IR Nasmyth f/13.6 focusing arrangement. So instead of simulating the full Subaru system,
# we can use the effective focal length at the Nasmyth focus, and simulate it as a single lens.
tp.d_nsmyth = 7.9716  # m pupil diameter
tp.fn_nsmyth = 13.612  # f# Nasmyth focus
tp.flen_nsmyth = tp.d_nsmyth * tp.fn_nsmyth  # m focal length
tp.dist_nsmyth_ao1 = tp.flen_nsmyth + 1.14  # m distance secondary to M1 of AO188 (hand-tuned, could update with
                                            # data from literature)

#  Below are the actual dimenstions of the Subaru telescope.
# --------------------------------
# tp.enterence_d = 8.2  # m diameter of primary
# tp.flen_primary = 15  # m focal length of primary
# tp.dist_pri_second = 12.652  # m distance primary -> secondary
# Secondary
tp.d_secondary = 1.265  # m diameter secondary, used for central obscuration
# tp.fn_secondary = 12.6
# tp.flen_secondary = - tp.d_secondary * tp.fn_secondary  # m focal length of secondary

# ----------------------------
# AO188 OAP1
# Paramaters taken from "Design of the Subaru laser guide star adaptive optics module"
#  Makoto Watanabe et. al. SPIE doi: 10.1117/12.551032
tp.d_ao1 = 0.20  # m  diamater of AO1
tp.fl_ao1 = 1.201  # m  focal length OAP1
tp.dist_ao1_dm = 1.345  # m distance OAP1 to DM
# ----------------------------
# AO188 OAP2
tp.dist_dm_ao2 = 2.511-tp.dist_ao1_dm  # m distance DM to OAP2
tp.d_ao2 = 0.2  # m  diamater of AO2
tp.fl_ao2 = 1.201  # m  focal length AO2
tp.dist_oap2_focus = 1.261

# Wavelength
# ap.wvl_range = np.array([810, 1500]) / 1e9
# sp.subplt_cols = 3
#################################################################################################
#################################################################################################
#################################################################################################
sp.focused_sys = True

# Toggles for Aberrations and Control
tp.obscure = True
tp.use_atmos = True
tp.use_aber = True
tp.use_ao = True

# Plotting
sp.show_cube = True  # Plot spectral cube at single timestep
sp.show_wframe = True  # Plot white light image frame
sp.show_tseries = True  # Plot full timeseries of white light frames

# Saving
sp.save_obs = False  # save obs_sequence (timestep, wavelength, x, y)
sp.save_fields = True  # toggle to turn saving uniformly on/off
sp.save_list = ['atmosphere', 'ideal_wfs', 'woofer', 'detector']  # list of locations in optics train to save


#################################################################################################
#################################################################################################
#################################################################################################

def Subaru_frontend(empty_lamda, grid_size, PASSVALUE):
    """
    propagates instantaneous complex E-field thru Subaru from the primary through the AO188
        AO system in loop over wavelength range

    this function is called a 'prescription' by proper

    uses PyPROPER3 to generate the complex E-field at the source, then propagates it through atmosphere,
        then telescope, to the focal plane
    the AO simulator happens here
    this does not include the observation of the wavefront by the detector
    :returns spectral cube at instantaneous time in the focal_plane()
    """
    # print("Propagating Broadband Wavefront Through Subaru")

    # Initialize the Wavefront in Proper
    wfo = opx.Wavefronts()
    wfo.initialize_proper()

    # Atmosphere
    # atmos has only effect on phase delay, not intensity
    wfo.loop_over_function(atmos.add_atmos, PASSVALUE['iter'], plane_name='atmosphere')

    if ap.companion:
        # offset companion here after running prop_define_enterance (to normalize intensity)
        #  if you did it in wfo.initialise()
        opx.offset_companion(wfo)

    ########################################
    # Subaru Propagation
    #######################################
    # Defines aperture (baffle-before primary)
    wfo.loop_over_function(proper.prop_circular_aperture, **{'radius': tp.enterance_d/2})  # clear inside, dark outside
    # Obscurations (Secondary and Spiders)
    wfo.loop_over_function(opx.add_obscurations, d_primary=tp.d_nsmyth, d_secondary=tp.d_secondary, legs_frac=0.01)
    wfo.loop_over_function(proper.prop_define_entrance)  # normalizes the intensity

    # Test Sampling
    # opx.check_sampling(PASSVALUE['iter'], wfo, "initial", units='mm')
    # Testing Primary Focus (instead of propagating to focal plane)
    # wfo.loop_over_function(opx.prop_mid_optics, tp.flen_nsmyth, tp.flen_nsmyth)  # test only going to prime focus

    # Effective Primary
    # CPA from Effective Primary
    wfo.loop_over_function(aber.add_aber, tp.enterance_d, tp.aber_params, step=PASSVALUE['iter'], lens_name='effective-primary')
    # Zernike Aberrations- Low Order
    # wfo.loop_over_function(aber.add_zern_ab, tp.zernike_orders, aber.randomize_zern_values(tp.zernike_orders))
    wfo.loop_over_function(opx.prop_mid_optics, tp.flen_nsmyth, tp.dist_nsmyth_ao1)

    ########################################
    # AO188 Propagation
    # # #######################################
    # # AO188-OAP1
    wfo.loop_over_function(aber.add_aber, tp.d_ao1, tp.aber_params, step=PASSVALUE['iter'], lens_name='ao188-OAP1')
    wfo.loop_over_function(opx.prop_mid_optics, tp.fl_ao1, tp.dist_ao1_dm)

    # AO System
    if tp.use_ao:
        WFS_map = ao.ideal_wfs(wfo)
        ao.deformable_mirror(wfo, WFS_map, PASSVALUE['theta'], plane_name='woofer')
    # ------------------------------------------------
    wfo.loop_over_function(proper.prop_propagate, tp.dist_dm_ao2)

    # AO188-OAP2
    wfo.loop_over_function(aber.add_aber, tp.d_ao2, tp.aber_params, step=PASSVALUE['iter'], lens_name='ao188-OAP2')
    # wfo.loop_over_function(aber.add_zern_ab, tp.zernike_orders, aber.randomize_zern_values(tp.zernike_orders)/2)
    wfo.loop_over_function(opx.prop_mid_optics, tp.fl_ao2, tp.dist_oap2_focus)

    ########################################
    # Focal Plane
    # #######################################
    # Check Sampling in focal plane
    opx.check_sampling(PASSVALUE['iter'], wfo, "focal plane", units='nm')
    # opx.check_sampling(PASSVALUE['iter'], wfo, "focal plane", units='arcsec')

    # wfo.focal_plane 1) sums the wfo over objects(companions) 2) fft-shifts wfo from Fourier Space (origin==lower left
    #  corner) to object space (origin==center) 3) converts complex-valued field into intensity units 4) interpolates
    #  over wavelength
    cpx_planes, sampling = wfo.focal_plane()

    print(f"Finished datacube at timestep = {PASSVALUE['iter']}")

    # import plot_tools
    # pn = 'detector'
    # ip = sp.save_list.index(pn)
    img_plane = np.sum(cpx_planes, axis=2)  # sum over objects
    img_plane = np.sum(img_plane, axis=1)  # sum over wavelengths
    # dprint(f"img_plane.shape is {img_plane.shape}")
    # plot_tools.quick2D(np.abs(img_plane[ip]) ** 2,
    #                 title=f"White Light at  {pn}",
    #                 logAmp=True
    #                 )

    return cpx_planes, sampling

