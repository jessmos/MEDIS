"""
aberrations.py
mostly copied over from the original by Rupert in MEDIS

collection of functions that deal with making aberration maps in proper for a given optical element.

Two initialize functions set up the ques for creating, saving, and reading FITs files of aberration maps. In general,
for an optical element in the telescope optical train, a aberration map is generated in Proper using prop_psd_errormap.
The map is saved as a FITs file and used for every wavelength and timestep in the observation sequence. Ideally this
will be updated for quasi-static aberrations, where the aberrations evolve over some user-defined timescale.
"""
import numpy as np
import proper
import os
import pickle

from mm_params import tp, iop, ap, sp
from mm_utils import *

################################################################################################################
# Aberrations
################################################################################################################
def generate_maps(lens_diam, lens_name='lens'):
    """
    generate PSD-defined aberration maps for a lens(mirror) using Proper

    Use Proper to generate an 2D aberration pattern across an optical element. The amplitude of the error per spatial
     frequency (cycles/m) across the surface is taken from a power spectral density (PSD) of statistical likelihoods
     for 'real' aberrations of physical optics.
    parameters defining the PSD function are specified in tp.aber_vals. These limit the range for the constants of the
     governing equation given by PSD = a/ [1+(k/b)^c]. This formula assumes the Terrestrial Planet Finder PSD, which is
     set to TRUE unless manually overridden line-by-line. As stated in the proper manual, this PSD function general
      under-predicts lower order aberrations, and thus Zernike polynomials can be added to get even more realistic
      surface maps.
    more information on a PSD error map can be found in the Proper manual on pgs 55-60

    Note: Functionality related to OOPP (out of pupil plane) optics has been removed. There is only one surface
    simulated for each optical surface

    :param lens_diam: diameter of the lens/mirror to generate an aberration map for
    #:param Loc: either CPA or NCPA, depending on where the optic is relative to the DM/AO system
    :param lens_name: name of the lens, for file naming
    :return: will create a FITs file in the folder specified by iop.quasi for each optic (and  timestep in the case
     of quasi-static aberrations)
    """
    # TODO add different timescale aberations
    dprint('Generating optic aberration maps using Proper')
    iop.aberdata = f"gridsz{sp.grid_size}_bmratio{sp.beam_ratio}_tsteps{sp.numframes}"
    iop.aberdir = os.path.join(iop.testdir, iop.aberroot, iop.aberdata)
    if not os.path.isdir(iop.aberdir):
        os.makedirs(iop.aberdir, exist_ok=True)
    dprint(f"Abberation directory = {iop.aberdir}")

    # create blank lens wavefront for proper to add phase to
    wfo = proper.prop_begin(lens_diam, 1., sp.grid_size, sp.beam_ratio)
    aber_cube = np.zeros((sp.numframes, sp.grid_size, sp.grid_size   ))

    # Randomly select a value from the range of values for each constant
    rms_error = np.random.normal(tp.aber_vals['a'][0], tp.aber_vals['a'][1])
    c_freq = np.random.normal(tp.aber_vals['b'][0], tp.aber_vals['b'][1])  # correlation frequency (cycles/meter)
    high_power = np.random.normal(tp.aber_vals['c'][0], tp.aber_vals['c'][1])  # high frequency falloff (r^-high_power)

    perms = np.random.rand(sp.numframes, sp.grid_size, sp.grid_size)-0.5
    perms *= 1e-7

    phase = 2 * np.pi * np.random.uniform(size=(sp.grid_size, sp.grid_size)) - np.pi
    aber_cube[0] = proper.prop_psd_errormap(wfo, rms_error, c_freq, high_power, TPF=True, PHASE_HISTORY=phase)
        # PHASE_HISTORY stuff is a kwarg Rupert added to a proper.prop_pds_errormap in proper_mod that helps
        #  ennable the small perturbations to the phase aberrations over time (quasi-static aberration evolution)
        #  however, this may not be implemented here, and the functionality may not be robust. It has yet to be
        #  verified in a robust manner. However, I am not sure it is being used....? KD 10-15-19
    # TODO verify this and add qusi-static functionality

    filename = f"{iop.aberdir}/t{0}_{lens_name}.fits"
    #dprint(f"filename = {filename}")
    if not os.path.isfile(filename):
        saveFITS(aber_cube[0], filename)

    # I think this part does quasi-static aberrations, but not sure if the random error is correct. On 7-10-19
    for a in range(1, sp.numframes):
        perms = np.random.rand(sp.grid_size, sp.grid_size) - 0.5
        perms *= 0.05
        phase += perms
        aber_cube[a] = proper.prop_psd_errormap(wfo, rms_error, c_freq, high_power,
                             MAP="prim_map", TPF=True, PHASE_HISTORY=phase)

        filename = f"{iop.aberdir}/t{a}_{lens_name}.fits"
        if not os.path.isfile(filename):
            saveFITS(aber_cube[0], filename)


def add_aber(wf, d_lens, aber_params, step=0, lens_name=None):
    """
    loads a phase error map and adds aberrations using proper.prop_add_phase
    if no aberration file exists, creates one for specific lens using generate_maps

    :param wf: a single (2D) wfo.wf_array[iw,io] at one wavelength and object
    :param d_lens: diameter (in m) of lens (only used when generating new aberrations maps)
    :param aber_params: parameters specified by tp.aber_params
    :param step: is the step number for quasistatic aberrations
    :param lens_name: name of the lens, used to save/read in FITS file of aberration map
    :return returns nothing but will act upon a given wavefront and apply new or loaded-in aberration map
    """
    # TODO this does not currently loop over time, so it is not using quasi-static abberations.
    # dprint("Adding Abberations")
    if tp.use_aber is False:
        pass  # don't do anything. Putting this type of check here allows universal toggling on/off rather than
              # commenting/uncommenting in the proper perscription
    else:
        # Load in or Generate Aberration Map
        iop.aberdata = f"gridsz{sp.grid_size}_bmratio{sp.beam_ratio}_tsteps{sp.numframes}"
        iop.aberdir = os.path.join(iop.testdir, iop.aberroot, iop.aberdata)
        filename = f"{iop.aberdir}/t{step}_{lens_name}.fits"
        if not os.path.isfile(filename):
            generate_maps(d_lens, lens_name)

        if aber_params['Phase']:
            phase_map = readFITS(filename)

            # Add Phase Map
            proper.prop_add_phase(wf, phase_map)

        if aber_params['Amp']:
            dprint("Outdated code-please update")
            raise NotImplementedError


def add_zern_ab(wfo, zern_order=[2,3,4], zern_vals=np.array([175,-150,200])*1.0e-9):
    """
    adds low-order aberrations from Zernike polynomials

    see Proper Manual pg 192 for full details
    see good example in Proper manual pg 51
    quote: These [polynomials] form an orthogonal set of aberrations that include:
     wavefront tilt, defocus, coma, astigmatism, spherical aberration, and others

    Orders are:
    1 Piston
    2 X tilt
    3 Y tilt
    4 Focus
    5 45º astigmatism
    6 0º astigmatism
    7 Y coma
    8 X coma
    9 Y clover (trefoil)
    10 X clover (trefoil)
    """
    proper.prop_zernikes(wfo, zern_order, zern_vals)


def randomize_zern_values(zern_orders):
    """
    selects a value at random for the selected zernike orders

    The ranges specified are all in units of m. From Proper manual pg 192-193
    'The values must be in meters of RMS phase error or dimensionless RMS amplitude error.'

    :param zern_orders: array containing integers of zernike orders to apply
    :return: array with randomly selected values in a pre-determined range, per zernike order specified in zern_orders
    """
    # Set Range
    z_range = {
        'r1': [30e-9, 60e-9],  # m of RMS amplitude error
        'r2': [30e-9, 40e-9],  # X tilt
        'r3': [30e-9, 40e-9],  # Y tilt
        'r4': [60e-9, 120e-9],  # focus
        'r5': [30e-9, 1200e-9],
        'r6': [30e-9, 1200e-9],
        'r7': [30e-9, 1200e-9],
        'r8': [30e-9, 1200e-9],
        'r9': [30e-9, 1200e-9],
        'r10': [30e-9, 1200e-9],
    }

    zern_vals = np.zeros(len(zern_orders))
    for iw, w in enumerate(zern_orders):
        if w == 1:
            zern_vals[iw] = np.random.normal(z_range['r1'][0], z_range['r1'][1])
        elif w == 2:
            zern_vals[iw] = np.random.normal(z_range['r2'][0], z_range['r2'][1])
        elif w == 3:
            zern_vals[iw] = np.random.normal(z_range['r3'][0], z_range['r3'][1])
        elif w == 4:
            zern_vals[iw] = np.random.normal(z_range['r4'][0], z_range['r4'][1])
        elif w == 5:
            zern_vals[iw] = np.random.normal(z_range['r5'][0], z_range['r5'][1])
        elif w == 6:
            zern_vals[iw] = np.random.normal(z_range['r6'][0], z_range['r6'][1])
        elif w == 7:
            zern_vals[iw] = np.random.normal(z_range['r7'][0], z_range['r7'][1])
        elif w == 8:
            zern_vals[iw] = np.random.normal(z_range['r8'][0], z_range['r8'][1])
    return zern_vals



###############################################################################################
# Depricated--but potenetially useful for quasi-static aberration implementation later
###############################################################################################
# def initialize_CPA_meas():
#     required_servo = int(tp.servo_error[0])
#     required_band = int(tp.servo_error[1])
#     required_nframes = required_servo + required_band + 1
#     CPA_maps = np.zeros((required_nframes, ap.n_wvl_init, sp.grid_size, sp.grid_size))
#
#     with open(iop.CPA_meas, 'wb') as handle:
#         pickle.dump((CPA_maps, np.arange(0, -required_nframes, -1)), handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#
# def initialize_NCPA_meas():
#     Imaps = np.zeros((4, sp.grid_size, sp.grid_size))
#     phase_map = np.zeros((tp.ao_act, tp.ao_act))  # np.zeros((sp.grid_size,sp.grid_size))
#     with open(iop.NCPA_meas, 'wb') as handle:
#         pickle.dump((Imaps, phase_map, 0), handle, protocol=pickle.HIGHEST_PROTOCOL)


