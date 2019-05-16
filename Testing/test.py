import h5py
import numpy as np
from scipy import io


def loadmat(matfile):
    try:
        f = h5py.File(matfile)
    except (IOError, OSError):
        return io.loadmat(matfile)
    else:
        return {name: np.transpose(f.get(name)) for name in f.keys()}


target_file = '../Training_Data/92_Image_Set/target_fmri.mat'
target = loadmat(target_file)
print(target['EVC_RDMs'][0])
print('Seperation')
print(target['IT_RDMs'][0])
