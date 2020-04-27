import os
import sys
import numpy as np
import importlib
import multiprocessing
import time
import glob
import pickle
import shutil
import h5py


import proper
import medis.atmosphere as atmos
from medis.plot_tools import view_spectra
import medis.CDI as cdi
import medis.utils as mu
import medis.optics as opx
import medis.aberrations as aber


class Telescope():
    """
    Creates a simulation for the telescope to create a series of complex electric fields

    During initialisation a backup of the pthon PROPER prescription is copied to the testdir, atmisphere maps and
    aberration maps are created, serialisation and memory requirements are tested, and the cdi vairable initialised

    Resulting file structure:
    datadir
        testdir
            params.pkl                     <--- input
            prescription                   <--- new
                {prescriptionname}         <--- new
                    {prescriptionname}.py  <--- new
            aberrations                    <--- new
                {aberationparams}          <--- new
                    {lensname}0.fits       <--- new
                    ...                    <--- new
            atmosphere                     <--- new
                {atmosphereparams}         <--- new
                    {atmos}0.fits          <--- new
                    ...                    <--- new
            fields.h5                      <--- output


    input
    params dict
        collection of the objects in params.py

    :return:
    self.cpx_sequence ndarray
        complex tensor of dimensions (n_timesteps, n_saved_planes, n_wavelengths, n_stars/planets, grid_size, grid_size)
    """

    def __init__(self, params):
        # if not initialise atmosphere
        # aberrations etc

        self.params = params

        self.save_exists = True if os.path.exists(self.params['iop'].telescope) else False

        if self.save_exists:
            print(f"\nLoading telescope instance from\n\n\t{self.params['iop'].telescope}\n")
            with open(self.params['iop'].telescope, 'rb') as handle:
                load = pickle.load(handle)
                self.__dict__ = load.__dict__
                self.save_exists = True
        else:
            print(f"\nInitialising new telescope instance\n")
            # copy over the prescription
            self.params['iop'].prescopydir = self.params['iop'].prescopydir.format(self.params['tp'].prescription)
            self.target = os.path.join(self.params['iop'].prescopydir, self.params['tp'].prescription+'.py')

            prescriptions = self.params['iop'].prescriptions_root
            fullprescription = glob.glob(os.path.join(prescriptions, '**', self.params['tp'].prescription+'.py'),
                                         recursive=True)
            if len(fullprescription) == 0:
                raise FileNotFoundError
            elif len(fullprescription) > 1:
                print(f'Multiple precriptions at {fullprescription}')
                raise FileExistsError

            fullprescription = fullprescription[0]
            print(f'Using prescription {fullprescription}')
            if os.path.exists(self.target):
                print(f"Prescription already exists at \n\n\t{self.target} \n\n... skipping copying\n\n")
            else:
                print(f"Copying over prescription {fullprescription}")

                if not os.path.isdir(self.params['iop'].prescopydir):
                    os.makedirs(self.params['iop'].prescopydir, exist_ok=True)
                shutil.copyfile(fullprescription, self.target)

            #import prescription params
            # sys.path.insert(0, os.path.dirname(self.target))
            sys.path.insert(0, os.path.dirname(fullprescription))
            pres_module = importlib.import_module(params['tp'].prescription)
            self.params['tp'].__dict__.update(pres_module.tp.__dict__)

            # initialize atmosphere
            self.params['iop'].atmosdir = self.params['iop'].atmosdir.format(params['sp'].grid_size,
                                                                             params['sp'].beam_ratio,
                                                                             params['sp'].numframes)
            if glob.glob(self.params['iop'].atmosdir+'/*.fits'):
                print(f"Atmosphere maps already exist at \n\n\t{self.params['iop'].atmosdir}"
                      f" \n\n... skipping generation\n\n")
            else:
                if not os.path.isdir(self.params['iop'].atmosdir):
                    os.makedirs(self.params['iop'].atmosdir, exist_ok=True)
                atmos.gen_atmos(params)

            # initialize aberrations
            self.params['iop'].aberdir = self.params['iop'].aberdir.format(params['sp'].grid_size,
                                                                           params['sp'].beam_ratio,
                                                                           params['sp'].numframes)
            if glob.glob(self.params['iop'].aberdir + '/*.fits'):
                print(f"Aberration maps already exist at \n\n\t{self.params['iop'].aberdir} "
                      f"\n\n... skipping generation\n\n")
            else:
                if not os.path.isdir(self.params['iop'].aberdir):
                    os.makedirs(self.params['iop'].aberdir, exist_ok=True)
                for lens in params['tp'].lens_params:
                    aber.generate_maps(lens['aber_vals'], lens['diam'], lens['name'])

            # check if can do parrallel
            if params['sp'].closed_loop or params['sp'].ao_delay:
                print(f"closed loop or ao delay means sim can't be parrallelized in time domain. Forcing serial mode")
                params['sp'].parrallel = False

            # determine if can/should do all in memory
            max_steps = self.max_chunk()
            checkpoint_steps = max_steps if self.params['sp'].checkpointing is None else self.params['sp'].checkpointing
            self.chunk_steps = int(min([max_steps, self.params['sp'].numframes, checkpoint_steps]))
            if self.params['sp'].verbose: print(f'Using time chunks of size {self.chunk_steps}')
            self.num_chunks = self.params['sp'].numframes / self.chunk_steps

            if self.num_chunks > 1:
                print('Simulated data too large for dynamic memory. Storing to disk as the sim runs')
                self.params['sp'].chunking = True

            #todo remove the hard coding
            self.params['sp'].chunking = True
            self.params['sp'].parrallel = False
            self.params['sp'].ao_delay = False
            self.params['sp'].closed_loop = False

            self.markov = self.params['sp'].chunking or self.params['sp'].parrallel  # independent timesteps
            assert np.logical_xor(self.markov, self.params['sp'].ao_delay or self.params['sp'].closed_loop), \
                "Confliciting modes. Request requires the timesteps be both dependent and independent"

            modes = [self.params['sp'].chunking, self.params['sp'].ao_delay, self.params['sp'].parrallel,
                     self.params['sp'].closed_loop]

            # ensure contrast is set properly
            if self.params['ap'].companion is False:
                self.params['ap'].contrast = []

            # Initialize CDI probes
            if self.params['cdip'].use_cdi is True:
                self.theta_series = cdi.gen_CDI_phase_stream()
            else:
                self.theta_series = np.zeros(self.params['sp'].numframes) * np.nan  # string of Nans

    def __call__(self, *args, **kwargs):
        if not self.save_exists:
            print('\n\n\tBeginning Telescope Simulation with MEDIS\n\n')
            start = time.time()

            self.create_fields()

            print('\n\n\tMEDIS Telescope Run Completed\n')
            finish = time.time()
            print(f'Time elapsed: {(finish - start) / 60:.2f} minutes')

            self.pretty_sequence_shape()

            self.save_exists = True
            print(f"\nSaving telescope instance at\n\n\t{self.params['iop'].telescope}\n")
            with open(self.params['iop'].telescope, 'wb') as handle:
                pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

        dataproduct = {'fields': np.array(self.cpx_sequence), 'sampling': self.sampling}
        return dataproduct

    def max_chunk(self):
        """
        Determines the maximum duration each chunk can be to fit within the memory limit

        :return: integer
        """
        timestep_size = len(self.params['sp'].save_list) * self.params['ap'].n_wvl_final * \
                        (1 + len(self.params['ap'].contrast)) * self.params['sp'].grid_size**2 * 32

        max_chunk = self.params['sp'].memory_limit // timestep_size
        print(f'Each timestep is predicted to be {timestep_size/1.e6} MB, requiring sim to be split into '
              f'{max_chunk} time chunks')

        return max_chunk

    def create_fields(self):

        t0 = self.params['sp'].startframe
        self.kwargs = {'params': self.params, 'theta_series': self.theta_series}
        self.cpx_sequence = None

        if self.markov:  # time steps are independent
            for ichunk in range(int(np.ceil(self.num_chunks))):
                cpx_sequence = np.empty((self.chunk_steps, len(self.params['sp'].save_list),
                                        self.params['ap'].n_wvl_init, 1 + len(self.params['ap'].contrast),
                                        self.params['sp'].grid_size, self.params['sp'].grid_size),
                                        dtype=np.complex64)
                chunk_range = ichunk * self.chunk_steps + t0 + np.arange(self.chunk_steps)
                if self.params['sp'].num_processes == 1:
                    seq_samp_list = [self.run_timestep(t) for t in chunk_range]
                else:
                    pool = multiprocessing.Pool(processes=self.params['sp'].num_processes)
                    seq_samp_list = [pool.apply(self.run_timestep, args=(t,)) for t in chunk_range]
                self.cpx_sequence = [tup[0] for tup in seq_samp_list]
                self.sampling = seq_samp_list[0][1]

                if self.params['ap'].n_wvl_init < self.params['ap'].n_wvl_final:
                    self.cpx_sequence = opx.interp_wavelength(self.cpx_sequence, ax=2)
                    self.sampling = opx.interp_sampling(self.sampling)

                if self.params['sp'].save_to_disk: self.save_fields(self.cpx_sequence)

        else:
            print('*** This is untested ***')
            self.cpx_sequence = np.zeros((self.params['sp'].numframes, len(self.params['sp'].save_list),
                                          self.params['ap'].n_wvl_init, 1 + len(self.params['ap'].contrast),
                                          self.params['sp'].grid_size, self.params['sp'].grid_size), dtype=np.complex)
            self.sampling = np.zeros((len(self.params['sp'].save_list), self.params['ap'].n_wvl_init))

            for it, t in enumerate(range(t0, self.params['sp'].numframes + t0)):
                WFS_ind = ['wfs' in plane for plane in self.params['sp'].save_list]
                if t > self.params['sp'].ao_delay:
                    self.kwargs['WFS_field'] = self.cpx_sequence[it - self.params['sp'].ao_delay, WFS_ind, :, 0]
                    self.kwargs['AO_field'] =  self.cpx_sequence[it - self.params['sp'].ao_delay, AO_ind, :, 0]
                else:
                    self.kwargs['WFS_field'] = np.zeros((self.params['ap'].n_wvl_init, self.params['sp'].grid_size,
                                                    self.params['sp'].grid_size), dtype=np.complex)
                    self.kwargs['AO_field'] = np.zeros((self.params['ap'].n_wvl_init, self.params['sp'].grid_size,
                                                        self.params['sp'].grid_size), dtype=np.complex)
                self.cpx_sequence[it], sampling = self.run_timestep(t)

            print('************************')
            if self.params['sp'].save_to_disk: self.save_fields(self.cpx_sequence)



        # return {'fields': np.array(self.cpx_sequence), 'sampling': self.sampling}

    def run_timestep(self, t):
        self.kwargs['iter'] = t
        return proper.prop_run(self.params['tp'].prescription, 1, self.params['sp'].grid_size, PASSVALUE=self.kwargs)

    def pretty_sequence_shape(self):

        """
        displays data format easier

        :param cpx_sequence: the 6D complex sequence generated by run_medis.telescope
        :return: nicely parsed string of 6D shape--human readable output
        """
        samps = ['timesteps', 'save planes', 'wavelengths', 'num obj', 'x', 'y']
        delim = ', '
        print(f"Shape of cpx_sequence = " \
            f"{delim.join([samp + ':' + str(length) for samp, length in zip(samps, np.shape(self.cpx_sequence))])}")

    def save_fields(self, fields):
        """
        Option to save fields separately from the class pickle save since fields can be huge if the user requests

        :param fields:
            dtype ndarray of complex or float
            fields can be any shape but the h5 dataset can only extended along axis 0
        :return:
        """
        with h5py.File(self.params['iop'].fields, mode='a') as hdf:
            print(f"Saving observation data at {self.params['iop'].fields}")
            dims = np.shape(fields)
            keys = list(hdf.keys())
            fields = np.array(fields)
            print('dims', dims, keys, fields.dtype)
            if 'data' not in keys:
                dset = hdf.create_dataset('data', dims, maxshape=(None,) + dims[1:],
                                          dtype=fields.dtype, chunks=dims, compression="gzip")
                dset[:] = fields
            else:
                hdf['data'].resize((hdf["data"].shape[0] + len(fields)), axis = 0)
                hdf["data"][-len(fields):] = fields

    def load_fields(self):
        with h5py.File(self.params['iop'].fields, 'r') as hdf:
            keys = list(hdf.keys())
            if 'data' in keys:
                self.cpx_sequence = hdf.get('data')[:]
        self.pretty_sequence_shape()

        return {'fields': self.cpx_sequence, 'sampling': None}

if __name__ == '__main__':
    from medis.params import params

    telescope_sim = Telescope(params)
    dataproduct = telescope_sim()
    print(dataproduct.keys())