import numpy as np
import h5py as h5
import os

import yt




def create_dataset_level0(self,
                          filename, component_names='default',
                          domain='default', data='default', overwrite=False):

    if isinstance(component_names, str):
        component_names = self._get_components()
    if domain == 'default':
        domain = self._get_domain()
    if data == 'default':
        data = self._get_data(component_names)

    N = int(domain['N'][0])
    if 'L0' not in domain.keys(): domain['L0'] = np.array([0, 0, 0])
    L = np.max(domain['L'] - domain['L0'])
    dt_multiplier = domain['dt_multiplier']

    """
    Mesh and Other Params
    """
    # def base attributes
    base_attrb = dict()
    base_attrb['time'] = self._handle['/'].attrs['time']
    base_attrb['iteration'] = self._handle['/'].attrs['iteration']
    base_attrb['max_level'] = self._handle['/'].attrs['max_level']
    base_attrb['num_components'] = len(component_names)
    base_attrb['num_levels'] = 1
    base_attrb['regrid_interval_0'] = 1
    base_attrb['steps_since_regrid_0'] = 0
    for comp, name in enumerate(component_names):
        key = 'component_' + str(comp)
        tt = 'S' + str(len(name))
        base_attrb[key] = np.array(name, dtype=tt)

    # def Chombo_global attributes
    chombogloba_attrb = dict()
    chombogloba_attrb['testReal'] = self._handle['/Chombo_global'].attrs['testReal']
    chombogloba_attrb['SpaceDim'] = self._handle['/Chombo_global'].attrs['SpaceDim']

    # def level0 attributes
    level_attrb = dict()
    level_attrb['dt'] = float(L) / N * dt_multiplier
    level_attrb['dx'] = float(L) / N
    level_attrb['time'] = self._handle['/level_0'].attrs['time']
    level_attrb['is_periodic_0'] = self._handle['/level_0'].attrs['is_periodic_0']
    level_attrb['is_periodic_1'] = self._handle['/level_0'].attrs['is_periodic_1']
    level_attrb['is_periodic_2'] = self._handle['/level_0'].attrs['is_periodic_2']
    level_attrb['ref_ratio'] = self._handle['/level_0'].attrs['ref_ratio']
    level_attrb['tag_buffer_size'] = self._handle['/level_0'].attrs['tag_buffer_size']
    prob_dom = (0, 0, 0, N - 1, N - 1, N - 1)
    prob_dt = np.dtype([('lo_i', '<i4'), ('lo_j', '<i4'), ('lo_k', '<i4'),
                        ('hi_i', '<i4'), ('hi_j', '<i4'), ('hi_k', '<i4')])
    level_attrb['prob_domain'] = np.array(prob_dom, dtype=prob_dt)
    boxes = np.array([(0, 0, 0, N - 1, N - 1, N - 1)],
                     dtype=[('lo_i', '<i4'), ('lo_j', '<i4'), ('lo_k', '<i4'), ('hi_i', '<i4'), ('hi_j', '<i4'),
                            ('hi_k', '<i4')])

    """"
    CREATE HDF5
    """

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

    return





def load(name, mode="r"):  #TODO *args, **kwargs
    """
        name
            Name of the file on disk.  Note: for files created with the 'core'
            driver, HDF5 still requires this be non-empty.
        mode
            r        Readonly, file must exist
            r+       Read/write, file must exist
            w        Create file, truncate if exists
            w- or x  Create file, fail if exists
            a        Read/write if exists, create otherwise (default)

    """
    f5 = h5.File(name, mode)



def _read_data(self, grid, field):
    lstring = 'level_%i' % grid.Level
    lev = self._handle[lstring]
    dims = grid.ActiveDimensions
    shape = dims + 2*self.ghost
    boxsize = shape.prod()

    if self._offsets is not None:
        grid_offset = self._offsets[grid.Level][grid._level_id]
    else:
        grid_offset = lev[self._offset_string][grid._level_id]
    start = grid_offset+self.field_dict[field]*boxsize
    stop = start + boxsize
    data = lev[self._data_string][start:stop]
    data_no_ghost = data.reshape(shape, order='F')
    ghost_slice = tuple(
        [slice(g, d+g, None) for g, d in zip(self.ghost, dims)])
    ghost_slice = ghost_slice[0:self.dim]
    return data_no_ghost[ghost_slice]