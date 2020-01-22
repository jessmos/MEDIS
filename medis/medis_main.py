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
import glob

from medis.params import iop, ap, tp, sp, cdip
import proper
import medis.atmosphere as atmos
import medis.CDI as cdi
import medis.utils as mmu

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

    # =======================================================================
    # Intialization
    # =======================================================================
    # Check for Existing File
    check = mmu.check_exists_obs_sequence(plot=False)
    if check:
        if iop.obs_seq[-3:] == '.h5':
            obs_sequence = mmu.open_obs_sequence_hdf5(iop.obs_seq)
        else:
            obs_sequence = mmu.open_obs_sequence(iop.obs_seq)

        return obs_sequence

    if ap.companion is False:
        ap.contrast = []

    # Initialize Obs Sequence
    # obs_sequence = np.zeros((sp.numframes, ap.n_wvl_final, sp.maskd_size, sp.maskd_size))
    # cpx_seq = (tsteps, planes, wavelengths, objects, x, y)
    cpx_sequence = np.zeros((sp.numframes, len(sp.save_list), ap.n_wvl_init, 1 + len(ap.contrast),
                                 sp.grid_size, sp.grid_size), dtype=np.complex)

    # =======================================================================================================
    # Multiprocessing with gen_timeseries
    # =======================================================================================================
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    # Multiprocessing Settings
    inqueue = multiprocessing.Queue()
    out_queue = multiprocessing.Queue()
    jobs = []

    # Sending Queues to gen_timeseries
    for i in range(sp.num_processes):
        p = multiprocessing.Process(target=gen_timeseries,
                                    args=(inqueue, out_queue))
        jobs.append(p)
        p.start()

    # Fun With Sentinels
    for t in range(sp.startframe, sp.startframe + sp.numframes):
        inqueue.put(t)

    for i in range(sp.num_processes):
        inqueue.put(sentinel)

    # ========================
    # Generate Obs Sequence
    # ========================
    # Getting Returned Variables from gen_timeseries
    for t in range(sp.numframes):
        t, planes, sampling = out_queue.get()
        cpx_sequence[t - sp.startframe, :, :, :, :, :] = planes
        # cpx_seq dimensions (tsteps, planes, wavelengths, objects, x, y)
        # size is (sp.numframes, len(sp.save_list), ap.n_wavl_init, 1+len(ap.contrast), sp.grid_size, sp.grid_size)

    # Ending the gen_timeseries loop via multiprocessing protocol
    for i, p in enumerate(jobs):
        p.join()  # Send the sentinel to tell Simulation to end?
    out_queue.put(None)

    print('mini-MEDIS Data Run Completed')
    print('**************************************')
    finish = time.time()
    print(f'Time elapsed: {(finish - start) / 60:.2f} minutes')
    # print(f"Number of timesteps = {np.shape(cpx_sequence)[0]}")

    # =======================================================================
    # Saving
    # =======================================================================
    if sp.save_obs:
        mmu.dprint("Saving obs_sequence:")
        mmu.save_obs_sequence(cpx_sequence, obs_seq_file=iop.obs_seq)
        print(f"Data saved: {iop.obs_seq}")

    return cpx_sequence, sampling


def gen_timeseries(inqueue, out_queue):  # conf_obj_tuple
    """
    generates observation sequence by calling optics_propagate in time series

    is the time loop wrapper for the proper prescription, so multiple calls to the proper prescription as aberrations
        or atmosphere evolve
    this is where the detector observes the wavefront created by proper, thus creating the observation sequence
        of spectral cubes at each timestep (for MKIDs, the probability distribution of the observed parameters
        is saved instead)

    :param inqueue: time index for parallelization (used by multiprocess)
    :param out_queue: series of intensity images (spectral image cube) in the multiprocessing format

    :return: returns the observation sequence, but through the multiprocessing tools, not through more standard
      return protocols.
      :intensity_seq is the intensity data in the focal plane only, with shape [timestep, wavelength, x, y]
      :cpx_seq is the complex-valued E-field in the planes specified by sp.save_list.
            has shape [timestep, planes, wavelength, objects, x, y]
      :sampling is the final sampling per wavelength in the focal plane
    """
    try:
        start = time.time()

        # Initialize CDI probes
        if cdip.use_cdi is True:
            theta_series = cdi.gen_CDI_phase_stream()
        else:
            theta_series = np.zeros(sp.numframes) * np.nan  # string of Nans
        # mmu.dprint(f"Theta={theta_series}")

        for it, t in enumerate(iter(inqueue.get, sentinel)):
            kwargs = {'iter': t, 'params': [ap, tp, iop, sp], 'theta': theta_series[t]}
            cpx_seq, sampling = proper.prop_run(tp.prescription, 1, sp.grid_size, PASSVALUE=kwargs,
                                                 VERBOSE=False, TABLE=True)  # 1 is dummy wavelength

            cpx_seq = np.array(cpx_seq)

            # Returning variables to run_mmedis
            out_queue.put((t, cpx_seq, sampling))

        now = time.time()
        elapsed = float(now - start) / 60.
        each_iter = float(elapsed) / (it + 1)

        print('***********************************')
        mmu.dprint(f'{elapsed:.2f} minutes elapsed, each time step took {each_iter:.4f} minutes')

    except Exception as e:
        traceback.print_exc()
        # raise e
        pass


if __name__ == '__main__':
    # testname = input("Please enter test name: ")
    testname = 'Subaru-test1'
    iop.update(testname)
    iop.makedir()
    run_mmedis()
