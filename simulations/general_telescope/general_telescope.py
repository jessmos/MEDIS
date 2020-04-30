"""
Prescription containing a full system that can be fully toggled using switches in the parent module

*** This module is currently unverified ***

TODO
    * verify this script using all the modes
    * reimplement tiptilt again?

"""

import proper

import medis.atmosphere as atmos
import medis.adaptive as ao
import medis.aberrations as aber
import medis.optics as opx
from medis.coronagraphy import coronagraph

class Telescope_Params(object):
    def __init__(self, dict):
        self.__dict__ = dict

tp = Telescope_Params({'lens_params': [{'aber_vals': [7.2e-17, 0.8, 3.1], 'diam': 0.2,  'focal_length': 1.2, 'dist' : 1.345,
                       'name': 'CPA'},
                    {'aber_vals': [7.2e-17, 0.8, 3.1], 'diam': 0.2, 'focal_length': 1.2, 'dist': 1.345,
                     'name': 'NCPA'}]})

def general_telescope(empty_lamda, grid_size, PASSVALUE):
    """
    #TODO pass complex datacube for photon phases

    propagates instantaneous complex E-field through the optical system in loop over wavelength range

    this function is called as a 'prescription' by proper

    uses PyPROPER3 to generate the complex E-field at the source, then propagates it through atmosphere, then telescope, to the focal plane
    currently: optics system "hard coded" as single aperture and lens
    the AO simulator happens here
    this does not include the observation of the wavefront by the detector
    :returns spectral cube at instantaneous time
    """
    # print("Propagating Broadband Wavefront Through Telescope")

    # datacube = []
    params = PASSVALUE['params']

    wfo = opx.Wavefronts()
    wfo.initialize_proper()

    ###################################################
    # Aperture, Atmosphere, and Secondary Obscuration
    ###################################################
    # Defines aperture (baffle-before primary)
    wfo.loop_collection(proper.prop_circular_aperture, **{'radius': params['tp'].entrance_d/2})
    wfo.loop_collection(proper.prop_define_entrance)  # normalizes the intensity

    # Pass through a mini-atmosphere inside the telescope baffle
    #  The atmospheric model used here (as of 3/5/19) uses different scale heights,
    #  wind speeds, etc to generate an atmosphere, but then flattens it all into
    #  a single phase mask. The phase mask is a real-valued delay lengths across
    #  the array from infinity. The delay length thus corresponds to a different
    #  phase offset at a particular frequency.
    if params['tp'].use_atmos:
        wfo.loop_collection(atmos.add_atmos, PASSVALUE['iter'],
                            (params['iop'].atmosdir, params['sp'].sample_time, params['atmp'].model),
                            spatial_zoom=True, plane_name='atmosphere', zero_outside=True)

    wfo.abs_zeros()

    #TODO rotate atmos not yet implementid in 2.0
    # if params['tp'].rotate_atmos:
    #     wfo.loop_collection(aber.rotate_atmos, *(PASSVALUE['iter']))

    # Both offsets and scales the companion wavefront
    if wfo.wf_collection.shape[1] > 1:
        wfo.loop_collection(opx.offset_companion)
        wfo.loop_collection(proper.prop_circular_aperture,
                            **{'radius': params['tp'].entrance_d / 2})  # clear inside, dark outside

    # TODO rotate atmos not yet implementid in 2.0
    # if params['tp'].rotate_sky:
    #     wfo.loop_collection(opx.rotate_sky, *PASSVALUE['iter'])

    ########################################
    # Telescope Primary-ish Aberrations
    #######################################
    # Abberations before AO

    wfo.loop_collection(aber.add_aber, params['tp'].aber_params, params['iop'].aberdir, PASSVALUE['iter'],
                        lens_name='CPA', zero_outside=True)
    # wfo.loop_collection(proper.prop_circular_aperture, **{'radius': params['tp'].entrance_d / 2})
    # wfo.wf_collection = aber.abs_zeros(wfo.wf_collection)
    wfo.abs_zeros()

    #######################################
    # AO
    #######################################
    if params['tp'].use_ao:

        if params['sp'].closed_loop:
            previous_output = ao.retro_wfs(PASSVALUE['AO_field'], wfo)  # unwrap a previous steps phase map
            wfo.loop_collection(ao.deformable_mirror, WFS_map=None, iter=PASSVALUE['iter'],
                                previous_output=previous_output, plane_name='deformable mirror', zero_outside=True)
        elif params['sp'].ao_delay > 0:
            WFS_map = ao.retro_wfs(PASSVALUE['WFS_field'], wfo)  # unwrap a previous steps phase map
            wfo.loop_collection(ao.deformable_mirror, WFS_map, iter=PASSVALUE['iter'], previous_output=None,
                                plane_name='deformable mirror', zero_outside=True)
        else:
            WFS_map = ao.open_loop_wfs(wfo)  # just uwraps this steps measured phase_map
            wfo.loop_collection(ao.deformable_mirror, WFS_map, iter=PASSVALUE['iter'], previous_output=None,
                                tp=params['tp'], plane_name='deformable mirror', zero_outside=True)

    # Obscure Baffle
    if params['tp'].obscure:
        wfo.loop_collection(opx.add_obscurations, M2_frac=1/8, d_primary=params['tp'].entrance_d, legs_frac=params['tp'].legs_frac)

   ########################################
    # Post-AO Telescope Distortions
    # #######################################
    # Abberations after the AO Loop

    wfo.loop_collection(aber.add_aber, params['tp'].aber_params, params['iop'].aberdir, PASSVALUE['iter'],
                        lens_name='NCPA', zero_outside=True)
    wfo.loop_collection(proper.prop_circular_aperture, **{'radius': params['tp'].entrance_d / 2})
    # TODO does this need to be here?
    # wfo.loop_collection(opx.add_obscurations, params['tp'].entrance_d/4, legs=False)
    # wfo.wf_collection = aber.abs_zeros(wfo.wf_collection)
    wfo.loop_collection(opx.prop_pass_lens, params['tp'].lens_params[0]['focal_length'],
                        params['tp'].lens_params[0]['focal_length'], plane_name='pre_coron')

    ########################################
    # Coronagraph
    ########################################
    # there are additional un-aberated optics in the coronagraph module

    wfo.loop_collection(coronagraph, occulter_mode=params['tp'].cg_type, plane_name='coronagraph')

    ########################################
    # Focal Plane
    ########################################
    cpx_planes, sampling = wfo.focal_plane()

    print(f"Finished datacube at timestep = {PASSVALUE['iter']}")

    return cpx_planes, wfo.plane_sampling
