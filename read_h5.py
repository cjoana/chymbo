import numpy as np
import h5py as h5
import os
from scipy.interpolate import RegularGridInterpolator as interp3D
import matplotlib.pyplot as plt
import scipy
# import yt

def load_dataset(  # self,
                          filename, component_names='default',
                          domain='default', data='default',
                          mode="r", overwrite=False):
    """
    First attemps
    """

    # if isinstance(component_names, str):
    #     if component_names == "default":   component_names = self._get_components()
    # if domain == 'default':
    #     domain = self._get_domain()
    # if data == 'default':
    #     data = self._get_data(component_names)

    f5 = h5.File(filename, 'r')

    """
    Mesh and Other Params
    """
    out=dict()
    all_attrb = dict()

    # def base attributes
    base_attrb = dict()
    base_attrb['time'] = f5['/'].attrs['time']
    base_attrb['iteration'] = f5['/'].attrs['iteration']
    base_attrb['max_level'] = f5['/'].attrs['max_level']
    base_attrb['num_components'] = f5['/'].attrs['num_components']
    base_attrb['num_levels'] = f5['/'].attrs['num_levels']
    num_levels = base_attrb['num_levels']
    base_attrb['regrid_interval_0'] = f5['/'].attrs['regrid_interval_0']
    base_attrb['steps_since_regrid_0'] = f5['/'].attrs['steps_since_regrid_0']

    num_components = f5['/'].attrs['num_components']
    for ic in range(num_components):
        key = 'component_' + str(ic)
        name = f5['/'].attrs[key].decode('UTF-8')  # decode removes the byte b'string'
        tt = 'S' + str(len(name))
        base_attrb[key] = np.array(name, dtype=tt)

    all_attrb["base"] = base_attrb

    # def Chombo_global attributes
    chomboglobal_attrb = dict()
    chomboglobal_attrb['testReal'] = f5['/Chombo_global'].attrs['testReal']
    chomboglobal_attrb['SpaceDim'] = f5['/Chombo_global'].attrs['SpaceDim']
    all_attrb["chombo_global"] = chomboglobal_attrb

    # def level0 attributes
    for level in range(num_levels):
        level_attrb = dict()
        level_attrb['dt'] = f5['/level_0'].attrs['dt']
        level_attrb['dx'] = f5['/level_0'].attrs['dx']
        level_attrb['time'] = f5['/level_0'].attrs['time']
        level_attrb['is_periodic_0'] = f5['/level_0'].attrs['is_periodic_0']
        level_attrb['is_periodic_1'] = f5['/level_0'].attrs['is_periodic_1']
        level_attrb['is_periodic_2'] = f5['/level_0'].attrs['is_periodic_2']
        level_attrb['ref_ratio'] = f5['/level_0'].attrs['ref_ratio']
        level_attrb['tag_buffer_size'] = f5['/level_0'].attrs['tag_buffer_size']
        level_attrb['prob_domain'] = f5['/level_0'].attrs['prob_domain']

        if level == 0:
            N = level_attrb['prob_domain'][-1] + 1

        all_attrb["level_{}".format(level)] = level_attrb.copy()


        out["level_{}".format(level)] = dict()
        boxes = np.array(f5['/level_0/boxes'][:])
        out["level_{}".format(level)]["boxes"] = boxes
        out["level_{}".format(level)]["Processors"] = f5['/level_0/Processors'][:]
        out["level_{}".format(level)]["offsets"] = f5['/level_0/data:offsets=0'][:]
        out["level_{}".format(level)]["data"] = f5['/level_0/data:datatype=0'][:]
        atts = out["level_{}".format(level)]["data_attrb"] = dict()
        atts['ghost'] = f5['/level_{}'.format(level)]['data_attributes'].attrs['ghost']
        atts['outputGhost'] = f5['/level_{}'.format(level)]['data_attributes'].attrs['outputGhost']
        atts["num_components"] = f5['/level_{}'.format(level)]['data_attributes'].attrs['comps']

    out["all_attrb"] = all_attrb

    return out  # all_attrb



# Params
fn = "data_samples/level2_dataset_SF.hdf5"
comp = 25  # ||  25 = \phi
verbose = 1
lev_max = 2
res = 32

# loading attrb and refs.
out = load_dataset(fn)
box0 = out["level_0"]["boxes"][0]
b0 = np.array([int(el) for el in box0], dtype=int)
num_ghost = out["level_0"]["data_attrb"]["outputGhost"][-1]
offsets = out["level_0"]["offsets"]
num_components = out["level_0"]["data_attrb"]["num_components"]
data = out["level_0"]["data"]
prob_dom = out["all_attrb"]["level_0"]["prob_domain"]
N = prob_dom[-1]+1
Ngrid = N * 2**(lev_max)
dims = np.array([Ngrid,Ngrid, Ngrid])
mask_grid = np.zeros(dims, dtype=bool)

# print out basic attrb
if verbose:
    print("making grid::")
    print("level max ", lev_max)
    print("dim 0",  N)
    print("dim_max ", Ngrid)

# Gridding: extract data and coords
xcords=[]
ycords=[]
zcords=[]
gdata=[]
for il in range(lev_max+1):

    boxes = out["level_{}".format(il)]["boxes"]
    offsets = out["level_{}".format(il)]["offsets"]
    num_ghost = out["level_{}".format(il)]["data_attrb"]["outputGhost"][-1]
    num_components = out["level_{}".format(il)]["data_attrb"]["num_components"]
    data = out["level_{}".format(il)]["data"]
    Nlev = N * 2 ** (il)

    if verbose:
        print("num boxes ", len(boxes))
        print("starting level ", il)

    for ib, box in enumerate(boxes):
        box = np.array([el for el in box], dtype=int)
        bdims = box[3:] - box[:3]+1
        shape = bdims + 2 * num_ghost
        boxsize = shape.prod()

        start = offsets[ib] + comp * boxsize
        stop = start + boxsize
        boxdata = data[start:stop]
        data_w_ghost = boxdata.reshape(shape, order='F')
        ghost_slice = tuple(
            [slice(g, d + g , None) for g, d in zip([num_ghost, num_ghost, num_ghost], bdims)])
        data_no_ghost = data_w_ghost[ghost_slice]

        # for some reasons it has to be in this order, y, x, z
        yi = np.linspace(box[0], box[3], bdims[0]) / Nlev
        xi = np.linspace(box[1], box[4], bdims[1]) / Nlev
        zi = np.linspace(box[2], box[5], bdims[2]) / Nlev
        xi, yi, zi = np.meshgrid(xi, yi, zi)

        xcords.extend(xi.flatten())
        ycords.extend(yi.flatten())
        zcords.extend(zi.flatten())
        gdata.extend(data_no_ghost.flatten())



# Const grid
xcords = np.array(xcords)
ycords = np.array(ycords)
zcords= np.array(zcords)
gdata = np.array(gdata)
x = np.linspace(0, 1, res)
y = np.linspace(0, 1, res)
z = np.linspace(0, 1, res)
x, y, z = np.meshgrid(x, y, z)

grid = scipy.interpolate.griddata((xcords, ycords, zcords),
                                                  gdata,
                                                  (x, y, z),
                                                  # method='linear')  # very very slow
                                                  method='nearest')

# plot result
plt.imshow(grid[res // 2, :, :], interpolation='spline36')
plt.show()
