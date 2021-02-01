import numpy as np
import h5py as h5
import os
from scipy.interpolate import RegularGridInterpolator as interp3D
import matplotlib.pyplot as plt
import scipy
# import yt

from write_params import *


overwrite = False
filename = "data_samples/test.hdf5"


if overwrite:
    if os.path.exists(filename):
        os.remove(filename)
else:
    if os.path.exists(filename):
        raise Exception(">> The file already exists, and set not to overwrite.")

h5file = h5.File(filename, 'w')  # New hdf5 file I want to create

# base attributes
for key in base_attrb.keys():
    h5file.attrs[key] = base_attrb[key]

# group: Chombo_global
chg = h5file.create_group('Chombo_global')
for key in chombogloba_attrb.keys():
    chg.attrs[key] = chombogloba_attrb[key]

# group: levels
for il in range(base_attrb['num_levels']):
    lev = h5file.create_group('level_{}'.format(int(il)))
    for key in level_attrb.keys():
        lev.attrs[key] = level_attrb[key]
    sl = lev.create_group('data_attributes')
    sl.attrs['ghost'] = data_attributes['ghost']
    sl.attrs['outputGhost'] = data_attributes['outputGhost']
    sl.attrs['comps'] = base_attrb['num_components']
    sl.attrs['objectType'] = data_attributes['objectType']

    # # level datasets
    # N = params["N"]
    # dataset = np.zeros((base_attrb['num_components'], N, N, N))
    # for i, comp in enumerate(components):
    #     if comp in data.keys():                             #TODO
    #         dataset[i] = data[comp].T                       #TODO
    #     else:
    #         raise Exception(">> Component {} not found in the data dictionary".format(comp))
    # fdset = []
    # for c in range(base_attrb['num_components']):
    #     fc = dataset[c].T.flatten()
    #     fdset.extend(fc)
    # fdset = np.array(fdset)

    lev.create_dataset("Processors", data=np.array([0]))
    lev.create_dataset("boxes", data=boxes)

    # lev.create_dataset("data:datatype=0", data=     )  #TODO
    # lev.create_dataset("data:offsets=0", data=      )  #TODO

h5file.close()