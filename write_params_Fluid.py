import numpy as np

params = dict()
all_attrb = dict()
base_attrb = dict()
chombogloba_attrb = dict()
levels_attrb = dict()
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
    "chi",
    "h11", "h12", "h13", "h22", "h23", "h33",
    "K",
    "A11", "A12", "A13", "A22", "A23", "A33",
    "Theta",
    "Gamma1", "Gamma2", "Gamma3",
    "lapse",
    "shift1", "shift2", "shift3",
    "B1", "B2", "B3",
    "density", "energy", "pressure", "enthalpy",
    "D", "E", "W",
    "Z1", "Z2", "Z3",
    "V1", "V2", "V3",
])

# Set boxes, for each level (MANUAL)
boxes["level_0"] = np.array([
    [0, 0, 0, 15, 15, 15],
    [0, 0, 16, 15, 15, 31],
    [0, 0, 32, 15, 15, 47],
    [0, 0, 48, 15, 15, 63],
    [0, 16, 0, 15, 31, 15],
    [0, 16, 16, 15, 31, 31],
    [0, 16, 32, 15, 31, 47],
    [0, 16, 48, 15, 31, 63],
    [0, 32, 0, 15, 47, 15],
    [0, 32, 16, 15, 47, 31],
    [0, 32, 32, 15, 47, 47],
    [0, 32, 48, 15, 47, 63],
    [0, 48, 0, 15, 63, 15],
    [0, 48, 16, 15, 63, 31],
    [0, 48, 32, 15, 63, 47],
    [0, 48, 48, 15, 63, 63],
    [16, 0, 0, 31, 15, 15],
    [16, 0, 16, 31, 15, 31],
    [16, 0, 32, 31, 15, 47],
    [16, 0, 48, 31, 15, 63],
    [16, 16, 0, 31, 31, 15],
    [16, 16, 16, 31, 31, 31],
    [16, 16, 32, 31, 31, 47],
    [16, 16, 48, 31, 31, 63],
    [16, 32, 0, 31, 47, 15],
    [16, 32, 16, 31, 47, 31],
    [16, 32, 32, 31, 47, 47],
    [16, 32, 48, 31, 47, 63],
    [16, 48, 0, 31, 63, 15],
    [16, 48, 16, 31, 63, 31],
    [16, 48, 32, 31, 63, 47],
    [16, 48, 48, 31, 63, 63],
    [32, 0, 0, 47, 15, 15],
    [32, 0, 16, 47, 15, 31],
    [32, 0, 32, 47, 15, 47],
    [32, 0, 48, 47, 15, 63],
    [32, 16, 0, 47, 31, 15],
    [32, 16, 16, 47, 31, 31],
    [32, 16, 32, 47, 31, 47],
    [32, 16, 48, 47, 31, 63],
    [32, 32, 0, 47, 47, 15],
    [32, 32, 16, 47, 47, 31],
    [32, 32, 32, 47, 47, 47],
    [32, 32, 48, 47, 47, 63],
    [32, 48, 0, 47, 63, 15],
    [32, 48, 16, 47, 63, 31],
    [32, 48, 32, 47, 63, 47],
    [32, 48, 48, 47, 63, 63],
    [48, 0, 0, 63, 15, 15],
    [48, 0, 16, 63, 15, 31],
    [48, 0, 32, 63, 15, 47],
    [48, 0, 48, 63, 15, 63],
    [48, 16, 0, 63, 31, 15],
    [48, 16, 16, 63, 31, 31],
    [48, 16, 32, 63, 31, 47],
    [48, 16, 48, 63, 31, 63],
    [48, 32, 0, 63, 47, 15],
    [48, 32, 16, 63, 47, 31],
    [48, 32, 32, 63, 47, 47],
    [48, 32, 48, 63, 47, 63],
    [48, 48, 0, 63, 63, 15],
    [48, 48, 16, 63, 63, 31],
    [48, 48, 32, 63, 63, 47],
    [48, 48, 48, 63, 63, 63],
])

# boxes["level_0"] = np.array([
#     [0, 0, 0, 63, 63, 63],
# ])

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
    levels_attrb['level_{}'.format(il)] = dict()
    ldict = levels_attrb['level_{}'.format(il)]
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
    lev_box = np.array([tuple(elm) for elm in boxes["level_{}".format(il)]], dtype=prob_dt)
    boxes["level_{}".format(il)] = lev_box

# set "data attributes" directory in levels, always the same.  (AUTO)
dadt = np.dtype([('intvecti', '<i4'), ('intvectj', '<i4'), ('intvectk', '<i4')])
data_attributes['ghost'] = np.array(tuple(params['ghosts']), dtype=dadt)
data_attributes['outputGhost'] = np.array((0, 0, 0), dtype=dadt)
data_attributes['comps'] = base_attrb['num_components']
data_attributes['objectType'] = np.array('FArrayBox', dtype='S9')


###################################
###   DATA TEMPLATE        ########
###################################


def _rho_fl(x, y, z):
    """
    x,y,z : These are the cell-centered conformal physical coordinates  ( grid-cords-centered * N_lev/ L )
            usually they are given as 3D arrays. size :(Dim, Nx_box, Ny_box, Nz_box)
    """
    L = params['L']
    vec = np.array([x, y, z])
    rc = np.zeros_like(vec) + L / 2
    cvec = vec - rc

    A_gauss = 1
    S_gauss = L / 10
    S_gauss_x = L / 10
    S_gauss_y = L / 9
    S_gauss_z = L / 8

    dot_prod = cvec[0, :] ** 2 / S_gauss_x ** 2 + cvec[1, :] ** 2 / S_gauss_y ** 2 + cvec[2, :] ** 2 / S_gauss_z ** 2

    return A_gauss * np.exp(- 0.5 * dot_prod)
    # return A_gauss * (np.sin(x *  2*np.pi / L) + np.sin(y * 2*np.pi / L) + np.sin(z * 2*np.pi / L))


def _chi(x, y, z):
    L = params['L']
    vec = np.array([x, y, z])
    # rc = np.zeros_like(vec) + L/2
    out = np.zeros_like(x) + 1
    return out


def _K(x, y, z):
    L = params['L']
    vec = np.array([x, y, z])
    # rc = np.zeros_like(vec) + L/2
    out = np.zeros_like(x)
    return out


def _D(x, y, z):
    L = params['L']
    vec = np.array([x, y, z])
    # rc = np.zeros_like(vec) + L/2
    out = np.zeros_like(x)+ 0.1
    return out


def _E(x, y, z):
    L = params['L']
    vec = np.array([x, y, z])
    # rc = np.zeros_like(vec) + L/2
    out = np.zeros_like(x)
    return out + 0.01


components_vals = [
    ['chi', _chi],
    ['h11', 1], ['h22', 1], ['h33', 1],
    ['h12', 0], ['h13', 0], ['h23', 0],
    ['K', _K],
    ['A11', 0], ['A22', 0], ['A33', 0],
    ['A12', 0], ['A13', 0], ['A23', 0],
    ['Theta', 0],
    ['Gamma1', 0], ['Gamma2', 0], ['Gamma3', 0],
    ['lapse', 1],
    ['shift1', 0], ['shift2', 0], ['shift3', 0],
    ['B1', 0], ['B2', 0], ['B3', 0],
    ['density', 0], ['energy', 0], ['pressure', 0], ['enthalpy', 0],
    ['D', _D], ['E', _E], ['W', 1],
    ['Z1', 0], ['Z2', 0], ['Z3', 0],
    ['V1', 0], ['V2', 0], ['V3', 0],
]
components_vals = np.array(components_vals)

# "chi",
# "h11",    "h12",    "h13",    "h22", "h23", "h33",
# "K",
# "A11",    "A12",    "A13",    "A22", "A23", "A33",
# "Theta",
# "Gamma1", "Gamma2", "Gamma3",
# "lapse",
# "shift1", "shift2", "shift3",
# "B1",     "B2",     "B3",
# "density",  "energy", "pressure", "enthalpy",
# "D",  "E", "W",
# "Z1", "Z2", "Z3",
# "V1", "V2","V3",

