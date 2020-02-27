"""
run_medis_main
Kristina Davis, Rupert Dodkins

This is the version of code compiled by K. Davis and R. Dodkins updated and streamlined from the MEDISv0.0 version of
the code written by R. Dodkins for his graduate career. This code contains the same functionality as MEDISv0.0 and is
modular to continue adding features and functionality.

MEDIS is split into two main functionalities; the first is to do an end-to-end simulation of a telescope. This is more
or less a large wrapper for proper with some new capabilities built into it to enable faster system design.The second
main functionality is to convert the complex field data generated by the telescope simulator to MKID-type photon lists.

For the telescope simulator, the user builds a prescription "name.py" of the optical system of the telescope system as
a separate module in the Simulations subdirectory of MEDIS. They can then update the default params file in a "
run_name.py" type script, which also makes the call to run_medis to start the simulation. The output is a 6D complex
field observation sequence.

Rupert will fill in the overview of the MKIDS part.

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
import medis.utils as mu
from medis.controller import auto_load
from medis.light import Fields

################################################################################################################
################################################################################################################
################################################################################################################
sentinel = None


def run_medis(mode='Fields'):
    """
    Get either a multidimensional cps_sequence or photon set wrapped in an object. Modified version of auto_load_single

    Parameters
    ----------
    mode : str, optional
           'Fields' or 'Photons'

    Returns
    -------
    The light object with data atribute loaded/generated
    """
    data_product = eval(mode)()
    if data_product.can_load():
        data_product.data = data_product.load()
    else:
        data_product.data = data_product.generate()
        if data_product.use_cache:
            data_product.save()

    return data_product


class RunMedis():
    """
    collects both the telescope simulator (returns complex fields) and the MKIDs simulator (returns photon lists) into
    a single class


    """
    def __init__(self):
        # Check params before memory is allocated
        mu.check_telescope_params()

        # Initialize Obs Sequence & Sampling
        self.cpx_sequence = np.zeros((sp.numframes, len(sp.save_list), ap.n_wvl_init, 1 + len(ap.contrast),
                                      sp.grid_size, sp.grid_size), dtype=np.complex)
        self.sampling = np.zeros((len(sp.save_list), ap.n_wvl_init))

    def telescope(self):
        """
        main script to organize calls to various aspects of the telescope simulation

        initialize different sub-processes, such as atmosphere and aberration maps, MKID device parameters
        sets up the multiprocessing features
        returns the observation sequence

        :return: obs_sequence [n_timesteps, n_saved_planes, n_wavelengths, n_stars/planets, grid_size, grid_size]
        """
        print('Beginning Telescope Simulation with MEDIS')
        print('***************************************')
        start = time.time()

        # =======================================================================
        # Intialization
        # =======================================================================
        # Check for Existing File
        #todo update this automatic load if exists functionality

        # check = mu.check_exists_obs_sequence(plot=False)
        # if check:
        #     if iop.obs_seq[-3:] == '.h5':
        #         obs_sequence = mu.open_obs_sequence_hdf5(iop.obs_seq)
        #     else:
        #         obs_sequence = mu.open_obs_sequence(iop.obs_seq)
        #
        #     return obs_sequence


        # Initialize CDI probes
        # if cdip.use_cdi is True:
        #     theta_series = cdi.gen_CDI_phase_stream()
        # else:
        #     theta_series = np.zeros(sp.numframes) * np.nan  # string of Nans

        # =======================================================================================================
        # Closed-Loop- No Multiprocessing
        # =======================================================================================================
        if sp.closed_loop:
            ##########################
            # Generating Timeseries
            ##########################
            for t in range(sp.numframes):
                kwargs = {'iter': t, 'params': [ap, tp, iop, sp],
                          'WFS_map': self.cpx_sequence[t-sp.ao_delay]}
                self.cpx_sequence[t], self.sampling = proper.prop_run(tp.prescription, 1, sp.grid_size,
                                                                         PASSVALUE=kwargs,
                                                                         VERBOSE=False,
                                                                         TABLE=False)  # 1 is dummy wavelength

            print('MEDIS Telescope Run Completed')
            print('**************************************')
            finish = time.time()
            print(f'Time elapsed: {(finish - start) / 60:.2f} minutes')
            mu.display_sequence_shape(self.cpx_sequence)

            return self.cpx_sequence, self.sampling

        # =======================================================================================================
        # Open-Loop- Uses Multiprocessing
        # =======================================================================================================
        else:
            try:
                multiprocessing.set_start_method('spawn')
            except RuntimeError:
                pass

            # Multiprocessing Settings
            time_idx = multiprocessing.Queue()  # time indicies that begin each timestep calculation
            out_chunk = multiprocessing.Queue()  # tuple of planes and sampling after each chunk is calculated
            jobs = []

            # Everything initialised in Timeseries is available to us in this obj. planes etc need to be accessed using a queue
            mt = MutliTime(time_idx, out_chunk)

            # Create the processes
            for i in range(sp.num_processes):
                p = multiprocessing.Process(target=mt.gen_timeseries, args=())
                jobs.append(p)
                p.start()

            # Populate the time indicies and start the simulation
            for t in range(sp.startframe, sp.startframe + sp.numframes):
                time_idx.put(t)

            # Tell the simulation to stop after the final time index is reached
            for i in range(sp.num_processes):
                time_idx.put(sentinel)

            # ========================
            # Read Obs Sequence
            # ========================
            # Getting Returned Variables from gen_timeseries
            for it in range(int(np.ceil(mt.num_chunks))):
                fields_chunk, sampling = out_chunk.get()
                self.cpx_sequence[it*mt.chunk_steps - sp.startframe :
                             (it+1)*mt.chunk_steps - sp.startframe, :, :, :, :, :] = fields_chunk
            self.sampling = sampling

            # # Ending the gen_timeseries loop via multiprocessing protocol
            for i, p in enumerate(jobs):
                p.join()  # Wait for all the jobs to finish and consolidate them here

        print('MEDIS Telescope Run Completed')
        print('**************************************')
        finish = time.time()
        print(f'Total Time elapsed: {(finish - start) / 60:.2f} minutes')
        # print(f"Number of timesteps = {np.shape(cpx_sequence)[0]}")

        mu.display_sequence_shape(self.cpx_sequence)

        return self.cpx_sequence, self.sampling

    def MKIDs(self):
        """
        this is where Rupert begins the MKIDs related stuff

        :return:
        """
        #TODO Rupert integrates the MKIDs stuff here


class MutliTime():
    """
    multiprocessing class for running open-loop telescope simulations

    generates observation sequence by...

    is the time loop wrapper for the proper prescription, so multiple calls to the proper prescription as aberrations
        or atmosphere evolve
    this is where the detector observes the wavefront created by proper, thus creating the observation sequence
        of spectral cubes at each timestep (for MKIDs, the probability distribution of the observed parameters
        is saved instead)

    :param time_ind: time index for parallelization (used by multiprocess)
    :param out_chunk: used for returning complex planes and sampling
    :param sp.memory_limit : number of bytes for sixcube of complex fields before chunking happens
    :param checkpointing : int or None number of timesteps before complex fields sixcube is saved
                            minimum of this and max allowed steps for memory reasons takes priority
    :type time_ind: mp.Queue

    :return: returns the observation sequence, but through the multiprocessing tools, not through more standard
      return protocols.
      :intensity_seq is the intensity data in the focal plane only, with shape [timestep, wavelength, x, y]
      :cpx_seq is the complex-valued E-field in the planes specified by sp.save_list.
            has shape [timestep, planes, wavelength, astronomical bodies, x, y]
      :sampling is the final sampling per wavelength in the focal plane
    """
    def __init__(self, time_idx, out_chunk):
        self.time_idx = time_idx
        self.out_chunk = out_chunk

        self.sampling = None
        max_steps = self.max_chunk()
        checkpoint_steps = max_steps if sp.checkpointing is None else sp.checkpointing
        self.chunk_steps = min([max_steps, sp.numframes, checkpoint_steps])
        if sp.verbose:
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
        start = time.time()

        for it, t in enumerate(iter(self.time_idx.get, sentinel)):
            kwargs = {'iter': t, 'params': [iop, sp, ap, tp, cdip]}
            timestep_field, sampling = proper.prop_run(tp.prescription, 1, sp.grid_size, PASSVALUE=kwargs,
                                                       VERBOSE=False, TABLE=False)  # 1 is dummy wavelength

            chunk_ind = it % self.chunk_steps

            self.fields_chunk[chunk_ind] = timestep_field
            self.seen_substeps[chunk_ind] = 1

            chunk_seen = np.all(self.seen_substeps)
            final_chunk_seen = it == sp.numframes-1 and np.all(self.seen_substeps[:self.final_chunk_size])
            while (chunk_ind == self.chunk_steps-1 and not chunk_seen) or \
                  (chunk_ind == self.final_chunk_size-1 and not final_chunk_seen):
                print(f'Waiting for chunk {it//self.chunk_steps} to finish being poplulated before saving') #if sp.verbose
                time.sleep(1)

            if chunk_seen or final_chunk_seen:  # chunk completed or simulation finished
                self.out_chunk.put((self.fields_chunk, sampling))
                if sp.save_to_disk:
                    self.save(self.fields_chunk, sampling)

                if sp.debug:
                    view_spectra(np.sum(np.abs(self.fields_chunk) ** 2, axis=(0, 1))[:, 0],
                                 title='Chunk Spectral Cube')

                self.init_fields_chunk()


        now = time.time()
        elapsed = float(now - start)
        each_iter = float(elapsed) / (sp.numframes + 1)

        print('***********************************')
        print(f'{elapsed/60.:.2f} minutes elapsed in gen_timeseries \n each time step took {each_iter:.2f} seconds')


if __name__ == '__main__':
    # testname = input("Please enter test name: ")
    testname = 'dummy1'
    iop.update(testname)
    iop.makedir()
    RunMedis.telescope()
