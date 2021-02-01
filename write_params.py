import numpy as np

params = dict()
all_attrb = dict()
base_attrb = dict()
chombogloba_attrb = dict()
level_attrb = dict()
boxes = dict()
data_attributes = dict()

# basic params  (MANUAL)
params['N'] = 64
params['L'] = 10
params['dt_multiplier'] = 0.01
params['is_periodic'] = [1, 1, 1]
params['ghosts'] = [0, 0, 0]

# Set components  (MANUAL)
components = np.array([
    'phi'
])

# Set boxes, for each level (MANUAL)
boxes["level_0"] = np.array([
   [0, 0, 0, 63, 63, 63],
])
boxes["level_1"] = np.array([
   [40, 40, 40, 87, 87, 87],
])
boxes["level_2"] = np.array([
   [104, 104, 104, 151, 151, 151],
])


# set base attibutes (MANUAL)
base_attrb['time'] = 0
base_attrb['iteration'] = 0
base_attrb['max_level'] = 3
base_attrb['num_levels'] = 3
base_attrb['num_components'] = components.size
base_attrb['regrid_interval_0'] = 2
base_attrb['steps_since_regrid_0'] = 0
for comp, name in enumerate(components):
    key = 'component_' + str(comp)
    tt = 'S' + str(len(name))
    base_attrb[key] = np.array(name, dtype=tt)


# def Chombo_global attributes (AUTO)
chombogloba_attrb['testReal'] = 0.0
chombogloba_attrb['SpaceDim'] = 3

# set level attributes and boxes (AUTO)
for il in range(base_attrb['num_levels']):
    level_attrb['level_{}'.format(il)] = dict()
    ldict = level_attrb['level_{}'.format(il)]
    ldict['ref_ratio'] = 2
    ldict['dt'] = float(params['L']) / params['N'] * params['dt_multiplier'] / (float(ldict['ref_ratio']) ** il)
    ldict['dx'] = float(params['L']) / params['N'] / (float(ldict['ref_ratio']) ** il)
    ldict['time'] = base_attrb['time']
    ldict['is_periodic_0'] = params['is_periodic'][0]
    ldict['is_periodic_1'] = params['is_periodic'][1]
    ldict['is_periodic_2'] = params['is_periodic'][2]
    ldict['tag_buffer_size'] = 3
    Nlev = int(params['N'] * (int(ldict['ref_ratio']) ** il))
    prob_dom = (0, 0, 0, Nlev - 1, Nlev - 1, Nlev - 1)
    prob_dt = np.dtype([('lo_i', '<i4'), ('lo_j', '<i4'), ('lo_k', '<i4'),
                        ('hi_i', '<i4'), ('hi_j', '<i4'), ('hi_k', '<i4')])
    ldict['prob_domain'] = np.array(prob_dom, dtype=prob_dt)

    prob_dt = np.dtype([('lo_i', '<i4'), ('lo_j', '<i4'), ('lo_k', '<i4'),
                        ('hi_i', '<i4'), ('hi_j', '<i4'), ('hi_k', '<i4')])
    lev_box = np.array([ tuple(elm) for elm in  boxes["level_{}".format(il)]], dtype=prob_dt)
    boxes["level_{}".format(il)] = lev_box


# set "data attributes" directory in levels, always the same.  (AUTO)
dadt = np.dtype([('intvecti', '<i4'), ('intvectj', '<i4'), ('intvectk', '<i4')])
data_attributes['ghost'] = np.array(tuple(params['ghosts']), dtype=dadt)
data_attributes['outputGhost'] = np.array((0, 0, 0), dtype=dadt)
data_attributes['comps'] = base_attrb['num_components']
data_attributes['objectType'] = np.array('FArrayBox', dtype='S9')