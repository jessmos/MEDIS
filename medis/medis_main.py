"""
run_medis_main
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
from medis.plot_tools import view_spectra
import medis.CDI as cdi
import medis.utils as mmu

################################################################################################################
################################################################################################################
################################################################################################################
sentinel = None


def run_medis():
    """
    main script to organize calls to various aspects of the simulation

    initialize different sub-processes, such as atmosphere and aberration maps, MKID device parameters
    sets up the multiprocessing features
    returns the observation sequence created by gen_timeseries

    :return: obs_sequence
    """
    print('Beginning Simulation on MEDIS')
    print('***************************************')
    start = time.time()

    # =======================================================================
    # Intialization
    # =======================================================================
    # Check for Existing File

    #todo update this automatic load if exists functionality

    # check = mmu.check_exists_obs_sequence(plot=False)
    # if check:
    #     if iop.obs_seq[-3:] == '.h5':
    #         obs_sequence = mmu.open_obs_sequence_hdf5(iop.obs_seq)
    #     else:
    #         obs_sequence = mmu.open_obs_sequence(iop.obs_seq)
    #
    #     return obs_sequence

    if ap.companion is False:
        ap.contrast = []

    # =======================================================================================================
    # Multiprocessing with gen_timeseries
    # =======================================================================================================
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    # Multiprocessing Settings
    time_idx = multiprocessing.Queue()  # time indicies that begin each timestep calculation
    out_chunk = multiprocessing.Queue()  # tuple of planes and sampling after each chunk is calculated
    jobs = []

    # Everything initialised in Timeseries is available to us in this obj. planes etc need to be accessed using a queue
    series = Timeseries(time_idx, out_chunk, (tp, ap, sp, iop))

    # Create the processes
    for i in range(sp.num_processes):
        p = multiprocessing.Process(target=series.gen_timeseries, args=())
        jobs.append(p)
        p.start()

    # Populate the time indicies and start the simulation
    for t in range(sp.startframe, sp.startframe + sp.numframes):
        time_idx.put(t)

    # Tell the simulation to stop after the final time index is reached
    for i in range(sp.num_processes):
        time_idx.put(sentinel)

    # ========================
    # Generate Obs Sequence
    # ========================
    # Getting Returned Variables from gen_timeseries

    if sp.return_fields:
        # Initialize Obs Sequence
        cpx_sequence = np.zeros((sp.numframes, len(sp.save_list), ap.n_wvl_init, 1 + len(ap.contrast),
                                 sp.grid_size, sp.grid_size), dtype=np.complex)

        samplings = []
        for it in range(int(np.ceil(series.num_chunks))):
            fields_chunk, sampling = out_chunk.get()
            cpx_sequence[it*series.chunk_steps - sp.startframe :
                         (it+1)*series.chunk_steps - sp.startframe, :, :, :, :, :] = fields_chunk
            samplings.append(sampling)
            # cpx_seq dimensions (tsteps, planes, wavelengths, objects, x, y)
            # size is (sp.numframes, len(sp.save_list), ap.n_wavl_init, 1+len(ap.contrast), sp.grid_size, sp.grid_size)

        if sp.verbose:
            print(display_sequence_shape(cpx_sequence))

    # # Ending the gen_timeseries loop via multiprocessing protocol
    for i, p in enumerate(jobs):
        p.join()  # Wait for all the jobs to finish and consolidate them here

    print('MEDIS Data Run Completed')
    print('**************************************')
    finish = time.time()
    print(f'Time elapsed: {(finish - start) / 60:.2f} minutes')
    # print(f"Number of timesteps = {np.shape(cpx_sequence)[0]}")

    if sp.return_fields:
        return cpx_sequence, samplings

def display_sequence_shape(cpx_sequence):
    """

    :param cpx_sequence: sixcube to get shape for
    :return: nicely parsed string of sixcube shape
    """
    samps = ['timesteps', 'save planes', 'wavelengths', 'num obj', 'x', 'y']
    delim = ', '
    return f"Shape of cpx_sequence = {delim.join([samp + ':' + str(length) for samp, length in zip(samps, cpx_sequence.shape)])}"

class Timeseries():
    """
    generates observation sequence by calling optics_propagate in time series

    is the time loop wrapper for the proper prescription, so multiple calls to the proper prescription as aberrations
        or atmosphere evolve
    this is where the detector observes the wavefront created by proper, thus creating the observation sequence
        of spectral cubes at each timestep (for MKIDs, the probability distribution of the observed parameters
        is saved instead)

    :param time_ind: time index for parallelization (used by multiprocess)
    :param out_chunk: used for returning complex planes and sampling
    :param conf_obj_tup : "global" configuration parameters need to be passed as args to gen_timesteries and
                            proper.prop_run for multiprocessing reasons
    :param sp.memory_limit : number of bytes for sixcube of complex fields before chunking happens
    :param checkpointing : int or None number of timesteps before complex fields sixcube is saved
                            minimum of this and max allowed steps for memory reasons takes priority
    :type time_ind: mp.Queue
    :type conf_obj_tup: tuple

    :return: returns the observation sequence, but through the multiprocessing tools, not through more standard
      return protocols.
      :intensity_seq is the intensity data in the focal plane only, with shape [timestep, wavelength, x, y]
      :cpx_seq is the complex-valued E-field in the planes specified by sp.save_list.
            has shape [timestep, planes, wavelength, objects, x, y]
      :sampling is the final sampling per wavelength in the focal plane
    """
    def __init__(self, time_idx, out_chunk, conf_obj_tup):
        self.time_idx = time_idx
        self.out_chunk = out_chunk
        self.conf_obj_tup = conf_obj_tup

        self.sampling = None
        required_servo = int(tp.servo_error[0])
        required_band = int(tp.servo_error[1])
        required_nframes = required_servo + required_band

        # Initialize CDI probes and
        self.cpa_maps = np.zeros((required_nframes, ap.n_wvl_init, sp.grid_size, sp.grid_size))
        if cdip.use_cdi is True:
            self.theta_series = cdi.gen_CDI_phase_stream()
        else:
            self.theta_series = np.zeros(sp.numframes) * np.nan  # string of Nans

        max_steps = self.max_chunk()
        checkpoint_steps = max_steps if sp.checkpointing is None else sp.checkpointing
        self.chunk_steps = min([max_steps, sp.numframes, checkpoint_steps])
        print(f'Using chunks of size {self.chunk_steps}')
        self.num_chunks = sp.numframes/self.chunk_steps
        self.init_fields_chunk()
        self.final_chunk_size = sp.numframes % self.chunk_steps

    def init_fields_chunk(self):
        self.fields_chunk = np.empty((self.chunk_steps, len(sp.save_list), ap.n_wvl_init, 1 + len(ap.contrast),
                                      sp.grid_size, sp.grid_size), dtype=np.complex64)
        self.seen_substeps = np.zeros((self.chunk_steps))

    def max_chunk(self):
        timestep_size = len(sp.save_list) * ap.n_wvl_final * (1 + len(ap.contrast)) * sp.grid_size**2 * 8

        max_chunk = sp.memory_limit // timestep_size
        print(f'Each timestep is predicted to be {timestep_size/1000.} MB, requiring sim to be split into '
              f'{max_chunk} chunks')

        return max_chunk

    def gen_timeseries(self):
        """
        Time loop wrapper for prescriptions.

        :param inqueue: time index for parallelization (used by multiprocess)
        :param out_queue: series of intensity images (spectral image cube) in the multiprocessing format

        :return: returns the observation sequence, but through the multiprocessing tools, not through more standard
          return protocols.

        :timestep_field is the complex-valued E-field in the planes specified by sp.save_list.
            has shape [timestep, planes, wavelength, objects, x, y]
        :sampling is the final sampling per wavelength in the focal plane
        """

        (tp, ap, sp, iop) = self.conf_obj_tup  # This is neccessary

        start = time.time()

        for it, t in enumerate(iter(self.time_idx.get, sentinel)):

            # if sp.verbose: print('timestep %i, using process %i' % (it, i))
            kwargs = {'iter': t, 'params': [ap, tp, iop, sp], 'theta': self.theta_series[t], 'CPA_maps': self.cpa_maps}
            timestep_field, sampling = proper.prop_run(tp.prescription, 1, sp.grid_size, PASSVALUE=kwargs,
                                                       VERBOSE=False, TABLE=True)  # 1 is dummy wavelength

            chunk_ind = it % self.chunk_steps

            self.fields_chunk[chunk_ind] = timestep_field
            self.seen_substeps[chunk_ind]=1

            chunk_seen = np.all(self.seen_substeps)
            final_chunk_seen = it == sp.numframes-1 and np.all(self.seen_substeps[:self.final_chunk_size])
            while (chunk_ind == self.chunk_steps-1 and not chunk_seen) or \
                  (chunk_ind == self.final_chunk_size-1 and not final_chunk_seen):
                print(f'Waiting for chunk {it//self.chunk_steps} to finish being poplulated before saving') #if sp.verbose
                time.sleep(1)

            if chunk_seen or final_chunk_seen:  # chunk completed or simulation finished

                if sp.return_fields:
                    self.out_chunk.put((self.fields_chunk, sampling))
                if sp.save_fields:
                    self.save(self.fields_chunk, sampling)

                if sp.debug:
                    view_spectra(np.sum(np.abs(self.fields_chunk) ** 2, axis=(0, 1))[:, 0],
                                 title='Chunk Spectral Cube')

                self.init_fields_chunk()


        now = time.time()
        elapsed = float(now - start) / 60.
        each_iter = float(elapsed) / (sp.numframes + 1)

        print('***********************************')
        print(f'{elapsed:.2f} minutes elapsed, each time step took {each_iter:.2f} minutes')

    def save(self, fields, sampling):
        """
        :param fields: np.ndarray of any shape

        :return:
        """
        raise NotImplementedError

        np.save(iop.fields+np.str(sampling), fields)

    def load(self):
        return np.load(iop.fields)

if __name__ == '__main__':
    # testname = input("Please enter test name: ")
    testname = 'Subaru-test1'
    iop.update(testname)
    iop.makedir()
    run_medis()
