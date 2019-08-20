"""
run_mini_medis
Kristina Davis, Rupert Dodkins

This is code that K.Davis exported from the full MEDIS simulator written by Rupert Dodkins. It is the basic
pieces of the full simulator, excluding the atmospheric parts. Most of the functionality here is running proper
through a simple telescope system, to generate an observation sequence of spectral cubes. As does MEDIS, this code
begins with a course wavelength sampling to fully run through the telescope optics, and then interpolates to a finer
spectral cube at each time step. I have added some changes to the code to make it easier to build different telescope
designs, as this was one of the largest disadvantages of the full MEDIS code. There is no GUI in this mini-version
of MEDIS either, as this is meant as a streamlined version.

"""
import numpy as np
import multiprocessing
import time
import traceback
import proper_mod as pm
import glob

from mm_params import iop, ap, tp, sp
from plot_tools import view_datacube, quick2D
import atmosphere as atmos
import mm_utils as mmu

################################################################################################################
################################################################################################################
################################################################################################################

sentinel = None

def run_mmedis():
    """
    main script to organize calls to various aspects of the simulation

    initialize different sub-processes, such as atmosphere and aberration maps, MKID device parameters
    sets up the multiprocessing features
    returns the observation sequence created by gen_timeseries

    :return: obs_sequence
    """
    print('Beginning Simulation on Mini-MEDIS')
    print('***************************************')
    start = time.time()

    # Check for Existing File
    check = mmu.check_exists_obs_sequence(False)
    if check:
        if iop.obs_seq[-3:] == '.h5':
            obs_sequence = mmu.open_obs_sequence_hdf5(iop.obs_seq)
        else:
            obs_sequence = mmu.open_obs_sequence(iop.obs_seq)

        return obs_sequence

    # Initialize Obs Sequence
    if tp.maskd_size != tp.grid_size:
        obs_sequence = np.zeros((sp.numframes, ap.w_bins, tp.maskd_size, tp.maskd_size))
    else:
        obs_sequence = np.zeros((sp.numframes, ap.w_bins, tp.grid_size, tp.grid_size))
    if ap.companion is False:
        ap.contrast = []

    # Initialize Atmosphere
    # mmu.dprint("Atmosdir = %s " % iop.atmosdir)
    if glob.glob(iop.atmosdir + '/*.fits') == []:
        atmos.gen_atmos(plot=True)

    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    # =======================================================================================================
    # Multiprocessing with gen_timeseries
    # =======================================================================================================
    # Multiprocessing Settings
    inqueue = multiprocessing.Queue()
    spectral_queue = multiprocessing.Queue()
    photons_queue = multiprocessing.Queue()
    jobs = []

    # Sending Queues to gen_timeseries
    for i in range(sp.num_processes):
        p = multiprocessing.Process(target=gen_timeseries,
                                    args=(inqueue, photons_queue, spectral_queue))
        jobs.append(p)
        p.start()

    # Fun With Sentinels
    for t in range(sp.startframe, sp.startframe + sp.numframes):
        inqueue.put(t)

    for i in range(sp.num_processes):
        inqueue.put(sentinel)

    # Getting Returned Variables from gen_timeseries
    for t in range(sp.numframes):
        t, spectralcube, sampling = spectral_queue.get()
        obs_sequence[t - sp.startframe] = spectralcube  # should be in the right order now because of the identifier

    # Ending the gen_timeseries loop via multiprocessing protocol
    for i, p in enumerate(jobs):
        p.join()  # Send the sentinel to tell Simulation to end?

    photons_queue.put(None)
    spectral_queue.put(None)
    # =======================================================================

    obs_sequence = np.array(obs_sequence)  # obs sequence is returned by gen_timeseries (called above)
                                           # (n_timesteps , n_wavelength_bins , x_grid , y_grid)

    # Plotting Datacube
    if sp.show_cube:
        tstep = sp.numframes-1
        view_datacube(obs_sequence[tstep], logAmp=True,
                      title=f"Intensity per Spectral Bin at Timestep {tstep} \n"
                            f"Beam Ratio = {tp.beam_ratio:.4f}, sampling = {sampling*1e6:.4f} [um/gridpt]",
                      subplt_cols=sp.subplt_cols, vlim=(1e-8, 1e-3))

    print('mini-MEDIS Data Run Completed')
    print('**************************************')
    finish = time.time()
    print(f'Time elapsed: {(finish - start) / 60:.2f} minutes')

    print(f"Number of timesteps = {np.shape(obs_sequence)[0]}")
    print(f"Number of wavelength bins = {np.shape(obs_sequence)[1]}")

    # Saving
    if sp.save_obs:
        mmu.dprint("Saving obs_sequence:")
        mmu.save_obs_sequence(obs_sequence, obs_seq_file=iop.obs_seq)
        print(f"Data saved: {iop.obs_seq}")


def gen_timeseries(inqueue, photons_queue, spectral_queue):  # conf_obj_tuple
    """
    generates observation sequence by calling optics_propagate in time series

    is the time loop wrapper for the proper perscription, so multiple calls to the proper perscription as aberrations
        or atmosphere evolve
    this is where the detector observes the wavefront created by proper, thus creating the observation sequence
        of spectral cubes at each timestep (for MKIDs, the probability distribution of the observed parameters
        is saved instead)

    :param inqueue: time index for parallelization (used by multiprocess)
    :param photons_queue: photon table (list of photon packets) in the multiprocessing format
    :param spectral_queue: series of intensity images (spectral image cube) in the multiprocessing format

    :return: returns the observation sequence, but through the multiprocessing tools, not through more standard
      return protocols
    """
    try:
        start = time.time()

        for it, t in enumerate(iter(inqueue.get, sentinel)):

            kwargs = {'iter': t, 'params': [ap, tp, iop, sp]}
            spectralcube, sampling = pm.prop_run(tp.perscription, 1, tp.grid_size, PASSVALUE=kwargs,
                                                   VERBOSE=False)

            # for cx in range(len(ap.contrast) + 1):
            #     mmu.dprint(f"E-field shape is {save_E_fields.shape}")
            #     cube = np.abs(save_E_fields[-1, :, cx]) ** 2

            # Returning variables to run_mmedis
            if sp.save_cube or sp.show_cube:
                spectral_queue.put((t, spectralcube, sampling))

        now = time.time()
        elapsed = float(now - start) / 60.
        each_iter = float(elapsed) / (it + 1)

        print('***********************************')
        mmu.dprint(f'{elapsed:.2f} minutes elapsed, each time step took {each_iter:.2f} minutes')

        if tp.detector == 'ideal':
            image = np.sum(spectralcube, axis=0)  # sum 3D spectralcube over wavelength (at this timestep)

        # Plotting
        if sp.show_wframe:
            # vlim = (np.min(spectralcube) * 10, np.max(spectralcube))  # setting z-axis limits
            quick2D(image, title=f"White light image at timestep {it} \n"
                                 f"Grid Size = {tp.grid_size}, Beam Ratio = {tp.beam_ratio}, "
                                 f"sampling = {sampling*1e6:.4f} (um/gridpt)", logAmp=True)
        # loop_frames(obs_sequence[:, 0])
        # loop_frames(obs_sequence[0])

    except Exception as e:
        traceback.print_exc()
        # raise e
        pass


if __name__ == '__main__':
    # testname = input("Please enter test name: ")
    testname = 't1'
    iop.update(testname)
    iop.makedir()
    run_mmedis()
