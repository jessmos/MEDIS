"""
Module for the different observatory components

"""

import os
import yaml
from medis.params import iop
from medis.atmosphere import Atmosphere
from medis.controller import get_data, configs_match, can_load
from medis.utils import dprint

# class Astrophysics():
#     def __init__(self):

class Aberrations():
    """
    Class responsible for getting aberration maps

    """
    def __init__(self):
        self.config = yaml.load(iop.coron_config)

    def generate(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def view(self):
        pass

class Coronagraph():
    """
    Class responsible for getting atmosphere phase maps

    """
    def __init__(self):
        self.config = yaml.load(iop.coron_config)

    def generate(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def view(self):
        pass

class Telescope():
    """
    Class responsible for initialising telescope

    """
    def __init__(self):
        self.config = yaml.load(iop.telescope_config)

    def generate(self):
        get_data([Atmosphere, Aberrations, Coronagraph])
        if not os.exists(self.config.prescription):
            print(f'No prescription found with name {self.config.prescription}. Create it first. See ../simulations/ '
                  f'for examples')
            raise AssertionError

    def save(self):
        pass

    def load(self):
        pass

    def view(self):
        pass

class Detector():
    """
    Class responsible for getting MKID instrument data

    """
    def __init__(self):
        self.config = yaml.load(iop.detector_config)

    def generate(self):
        """ mkids device params code ported to here """
        pass

    def save(self):
        pass

    def load(self):
        pass

    def view(self):
        pass

