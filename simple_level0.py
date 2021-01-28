import numpy as np
import h5py as h5
import os
from scipy.interpolate import RegularGridInterpolator as interp3D
import matplotlib.pyplot as plt
import scipy
import yt

def load_dataset_level0(  # self,
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

    f5 = h5.File(filename, mode)

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

    keys = f5['/'].attrs
    for comp, name in enumerate(component_names):
        key = 'component_' + str(comp)
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
        # prob_dom = (0, 0, 0, N - 1, N - 1, N - 1)
        # prob_dt = np.dtype([('lo_i', '<i4'), ('lo_j', '<i4'), ('lo_k', '<i4'),
        #                     ('hi_i', '<i4'), ('hi_j', '<i4'), ('hi_k', '<i4')])
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

        # sl0.attrs['ghost'] = np.array((3, 3, 3), dtype=dadt)
        # sl0.attrs['outputGhost'] = np.array((0, 0, 0), dtype=dadt)
        # sl0.attrs['comps'] = base_attrb['num_components']
        # sl0.attrs['objectType'] = np.array('FArrayBox', dtype='S9')


    out["all_attrb"] = all_attrb


    """"
    CREATE HDF5
    """
    def create():
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
        for key in chomboglobal_attrb.keys():
            chg.attrs[key] = chomboglobal_attrb[key]

        # group: levels
        l0 = h5file.create_group('level_0')
        for key in level_attrb.keys():
            l0.attrs[key] = level_attrb[key]
        sl0 = l0.create_group('data_attributes')
        dadt = np.dtype([('intvecti', '<i4'), ('intvectj', '<i4'), ('intvectk', '<i4')])
        sl0.attrs['ghost'] = np.array((3, 3, 3), dtype=dadt)
        sl0.attrs['outputGhost'] = np.array((0, 0, 0), dtype=dadt)
        sl0.attrs['comps'] = base_attrb['num_components']
        sl0.attrs['objectType'] = np.array('FArrayBox', dtype='S9')

        # level datasets
        dataset = np.zeros((base_attrb['num_components'], N, N, N))
        for i, comp in enumerate(component_names):
            if comp in data.keys():
                dataset[i] = data[comp].T
            else:
                raise Exception(">> Component {} not found in the data dictionary".format(comp))
        fdset = []
        for c in range(base_attrb['num_components']):
            fc = dataset[c].T.flatten()
            fdset.extend(fc)
        fdset = np.array(fdset)

        l0.create_dataset("Processors", data=np.array([0]))
        l0.create_dataset("boxes", data=boxes)
        l0.create_dataset("data:offsets=0", data=np.array([0, (base_attrb['num_components']) * N ** 3]))
        l0.create_dataset("data:datatype=0", data=fdset)

        h5file.close()

    return out #all_attrb



# Params
fn = "data_samples/level2_dataset_SF.hdf5"
comp = 25  # ||  25 = \phi
verbose = 1
lev_max = 2
res = 32




out = load_dataset_level0(fn)
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








if verbose:
    print("making grid::")
    print("level max ", lev_max)
    print("dim 0",  N)
    print("dim_max ", Ngrid)


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
        print("sizes:: ",  np.size(data)/num_components, np.size(grid))
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

plt.imshow(grid[res // 2, :, :], interpolation='spline36')
plt.show()
