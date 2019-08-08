
import numpy as np
from inspect import getframeinfo, stack
import pickle
import tables as pt
import astropy.io.fits as afits

from mm_params import ap, tp, iop


def dprint(message):
    caller = getframeinfo(stack()[1][0])
    print("%s:%d - %s" % (caller.filename, caller.lineno, message))


def make_datacube(cube, size):
    # print 'Making an xyw cube'
    datacube = np.zeros((size[2], size[1], size[0]))
    phase_band = phase_cal(ap.wvl_range)
    bins = np.linspace(phase_band[0], phase_band[1], size[2 ] +1)

    for x in range(size[1]):
        for y in range(size[0]):
            if cube[x][y] == []:
                datacube[:, x, y] = np.zeros((size[2]))
            else:
                datacube[:, x, y] = np.histogram(np.array(cube[x][y])[:, 1], bins=bins)[0]  # [::-1]
                datacube[0, x, y] += len(np.where(np.array(cube[x][y])[:, 1] < phase_band[0])[0])
                datacube[-1, x, y] += len(np.where(np.array(cube[x][y])[:, 1] > phase_band[1])[0])

    return datacube


def phase_cal(wavelengths):
    """Wavelength in nm"""
    phase = tp.wavecal_coeffs[0] * wavelengths + tp.wavecal_coeffs[1]
    return phase

####################################################################################################
# Functions Relating to Reading, Loading, and Saving Data #
####################################################################################################


def save_obs_sequence(obs_sequence, obs_seq_file='obs_seq.pkl'):
    """saves obs sequence as a .pkl file

    :param obs_sequence- Observation sequence, 4D data structure
    :param obs_seq_file- filename for saving, including directory tree
    """
    #dprint((obs_seq_file, obs_seq_file[-3:], obs_seq_file[-3:] == '.h5'))
    if obs_seq_file[-3:] == 'pkl':
        with open(obs_seq_file, 'wb') as handle:
            pickle.dump(obs_sequence, handle, protocol=pickle.HIGHEST_PROTOCOL)
    elif obs_seq_file[-3:] == 'hdf' or obs_seq_file[-3:] == '.h5':
        f = pt.open_file(obs_seq_file, 'w')
        # atom = pt.Atom.from_dtype(hypercube.dtype)
        # ds = f.createCArray(f.root, 'data', atom, hypercube.shape)
        ds = f.create_array(f.root, 'data', obs_sequence)
        # ds[:] = hypercube
        f.close()
    else:
        dprint('Extension not recognised')


def check_exists_obs_sequence(plot=False):
    """
    This code checks to see if there is already
    an observation sequence saved with the output of the run in the
    location specified by the iop.

    :return: boolean flag if it can find a file or not
    """
    import os
    if os.path.isfile(iop.obs_seq):
        dprint(f"File already exists at {iop.obs_seq}")
        return True
    else:
        return False


def open_obs_sequence(obs_seq_file='hyper.pkl'):
    """opens existing obs sequence .pkl file and returns it"""
    with open(obs_seq_file, 'rb') as handle:
        obs_sequence =pickle.load(handle)
    return obs_sequence


def open_obs_sequence_hdf5(obs_seq_file='hyper.h5'):
    """opens existing obs sequence .h5 file and returns it"""
    # hdf5_path = "my_data.hdf5"
    read_hdf5_file = pt.open_file(obs_seq_file, mode='r')
    # Here we slice [:] all the data back into memory, then operate on it
    obs_sequence = read_hdf5_file.root.data[:]
    # hdf5_clusters = read_hdf5_file.root.clusters[:]
    read_hdf5_file.close()
    return obs_sequence

####################################################################################################
# Functions Relating to Reading, Loading, and Saving Images #
####################################################################################################


def saveFITS(image, name):
    header = afits.Header()
    header["PIXSIZE"] = (0.16, " spacing in meters")

    hdu = afits.PrimaryHDU(image, header=header)
    hdu.writeto(name)


def readFITS(filename='dummy1.fits'):
    """
    reads a fits file and returns data fields only

    :param filename: must specify full filepath
    """
    hdulist = afits.open(filename)
    header = hdulist[0].header
    scidata = hdulist[0].data

    # xnum = np.shape(scidata)[0]
    # ynum = np.shape(scidata)[1]

    return scidata

