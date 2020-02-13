"""
Module for the different observatory components

"""

import os
import yaml
from medis.params import iop, tp
from medis.atmosphere import Atmosphere
from medis.controller import auto_load
from medis.utils import dprint

# class Astrophysics():
#     def __init__(self):

class Aberrations():
    """
    Class responsible for getting aberration maps

    """
    def __init__(self):
        self.config = tp
        self.use_cache = True
        self.debug = True

    def generate(self):
        pass

    def can_load(self):
        if self.use_cache:
            file_exists = os.path.exists(iop.fields)
            if file_exists:
                configs_match = self.configs_match()
                if configs_match:
                    return True

        return False

    def configs_match(self):
        cur_config = self.__dict__
        cache_config = self.load_config()
        configs_match = cur_config == cache_config

        return configs_match

    def load_config(self):
        """ Reads the relevant config data from the saved file """
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
        self.config = tp
        self.use_cache = True
        self.debug = True

    def generate(self):
        pass

    def can_load(self):
        if self.use_cache:
            file_exists = os.path.exists(iop.fields)
            if file_exists:
                configs_match = self.configs_match()
                if configs_match:
                    return True

        return False

    def configs_match(self):
        cur_config = self.__dict__
        cache_config = self.load_config()
        configs_match = cur_config == cache_config

        return configs_match

    def load_config(self):
        """ Reads the relevant config data from the saved file """
        pass

    def save(self):
        pass

    def load(self):
        pass

    def view(self):
        pass

class Telescope():
    """
    Class responsible for initialising telescope.


    Attributes
    ----------
    prescription is the data in this case its a python script containing a function
    configuration of that prescription is the toggles used in that prescription
    """
    def __init__(self):
        self.config = tp
        self.use_cache = True
        self.debug = True

    def generate(self):
        auto_load([Atmosphere, Aberrations, Coronagraph])
        if not os.path.exists(f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/simulations/{tp.prescription}'):
            print(f'No prescription found with name {self.config.prescription}. Create it first. See ../simulations/ '
                  f'for examples')
            raise AssertionError

    def can_load(self):
        if self.use_cache:
            file_exists = os.path.exists(iop.fields)
            if file_exists:
                configs_match = self.configs_match()
                if configs_match:
                    return True

        return False

    def configs_match(self):
        cur_config = self.__dict__
        cache_config = self.load_config()
        configs_match = cur_config == cache_config

        return configs_match

    def load_config(self):
        """ Reads the relevant config data from the saved file """
        pass

    def save(self):
        pass

    def load(self):


    def view(self):
        pass

class Detector():
    """
    Class responsible for getting MKID instrument data

    """
    def __init__(self):
        self.config = yaml.load(iop.detector_config)
        self.debug = True

    def generate(self):
        """ mkids device params code ported to here """
        pass

    def can_load(self):
        if self.use_cache:
            file_exists = os.path.exists(iop.fields)
            if file_exists:
                configs_match = self.configs_match()
                if configs_match:
                    return True

        return False

    def configs_match(self):
        cur_config = self.__dict__
        cache_config = self.load_config()
        configs_match = cur_config == cache_config

        return configs_match

    def load_config(self):
        """ Reads the relevant config data from the saved file """
        pass

    def save(self):
        pass

    def load(self):
        pass

    def view(self):
        pass

