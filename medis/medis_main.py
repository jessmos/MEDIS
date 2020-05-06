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
import os
import sys
import numpy as np
import importlib
import multiprocessing
import time
import glob
import pickle
import shutil
from datetime import datetime
import h5py


import proper
import medis.atmosphere as atmos
from medis.plot_tools import view_spectra
import medis.CDI as cdi
import medis.utils as mu
import medis.MKIDS as MKIDs
from medis.telescope import Telescope
from medis.MKIDS import Camera
import medis.optics as opx
import medis.aberrations as aber

################################################################################################################
################################################################################################################
################################################################################################################
sentinel = None


class RunMedis():
    """
    Creates a simulation for calling Telescope or MKIDs to return a series of complex electric fields or photons,
    respectively.

    This class is a wrapper for Telescope and Camera that handles the checking of testdir and params existance

    Upon creation the code checks if a testdir of this name already exists, if it does it then checks if the params
    match. If the params are identifcal and the desired products are not already created, if it will create them.
    If the params are different or the testdir does not already exist a new testdir and simulation is created

    """
    def __init__(self, params, name='test', product='fields'):
        """
        File structure:
        datadir
            testdir          <--- output
                params.pkl   <--- output

        :param params:
        :param name:
        :param product:
        """

        self.params = params
        self.name = name
        self.product = product
        assert self.product in ['fields', 'photons'], f"Requested data product {self.product} not supported"

        self.params['iop'].update(self.name)

        if self.params['sp'].debug:
            for param in [self.params['iop']]:
                pprint(param.__dict__)

        # always make the top level directory if it doesn't exist yet
        if not os.path.isdir(self.params['iop'].datadir):
            print(f"Top level directory... \n\n\t{self.params['iop'].datadir} \n\ndoes not exist yet. Creating")
            os.makedirs(self.params['iop'].datadir, exist_ok=True)

        if not os.path.exists(self.params['iop'].testdir) or not os.path.exists(self.params['iop'].params_logs):
            print(f"No simulation data found at... \n\n\t{self.params['iop'].testdir} \n\n A new test simulation"
                  f" will be started")
            self.make_testdir()
        else:
            params_match = self.check_params()
            exact_match = all(params_match.values())
            if exact_match:
                print(f"Configuration files match. Initialization over")
            else:
                print(f"Configuration files differ. Creating a new test directory")
                now = datetime.now().strftime("%m:%d:%Y_%H-%M-%S")
                self.params['iop'].update(self.name+'_newsim_'+now)
                self.make_testdir()

    def make_testdir(self):
        if not os.path.isdir(self.params['iop'].testdir):
            os.makedirs(self.params['iop'].testdir, exist_ok=True)

        with open(self.params['iop'].params_logs, 'wb') as handle:
            pickle.dump(self.params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def check_params(self):
        """ Check all param classes apart from mp since that is not relevant at this stage """

        with open(self.params['iop'].params_logs, 'rb') as handle:
            loaded_params = pickle.load(handle)

        match_params = {}
        for p in ['ap','tp','atmp','cdip','iop','sp']:  # vars(self.params).keys()
            matches = []
            for (this_attr, this_val), (load_attr, load_val) in zip(self.params[p].__dict__.items(),
                                                                    self.params[p].__dict__.items()):
                matches.append(this_attr == load_attr and np.all(load_val == this_val))

            match = np.all(matches)
            print(f"param: {p}, match: {match}")
            match_params[p] = match

        return match_params

        # return {'ap':False, 'tp':True, 'atmp':True, 'cdip':True, 'iop':True, 'sp':True}

    def __call__(self, *args, **kwargs):
        if self.product == 'fields':
            telescope_sim = Telescope(self.params)  # checking of class's cache etc is left to the class
            dataproduct = telescope_sim()

        if self.product == 'photons':
            camera_sim = Camera(self.params)  # creating fields is left to Camera since fields only needs to be created
                                              # if camera.pkl does not exist
            dataproduct = camera_sim()

        return dataproduct


if __name__ == '__main__':
    from medis.params import params

    sim = RunMedis(params=params, name='example1', product='photons')
    observation = sim()
    print(observation.keys())


