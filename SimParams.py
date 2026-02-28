import numpy as np
import math

class SimParams:
    def __init__(self,t_max:float,n_max:int,points_to_plot:int,Lattice_dimensions:tuple,runs:int=1,rng_seed=None):
        self.t_max,self.n_max,self.runs = t_max,n_max,runs
        self.t_step , self.t_points = t_max/points_to_plot , points_to_plot+1
        self.lat = np.zeros((math.prod(Lattice_dimensions),2),dtype=int) # each row of the form [site type, occupancy]
        self.lat_dimensions = Lattice_dimensions
        self.lat_occ,self.lat_type,self.sys_type = 'Empty','square','ideal'
        self.rng,self.rng_seed = np.random.default_rng(rng_seed),rng_seed
        self._bool_build = False

    def what_params(self):
        y,x=self.lat_dimensions
        out = f'{self.runs} run(s) on {y}x{x} {self.lat_occ} lattice\n' 
        out += f'Stop conditions: t={self.t_max},n={self.n_max}\n'
        out += f'Plotting interval: t_step={self.t_step} ({self.t_points} steps)\n'
        out += f'Generator seed: {self.rng_seed}'
        return print(out)
    
    def build_lat(sys,system_type:str,lattice_type:str,**sys_attr):
        """ Choose system and lattice geometry
        Systems: \n
        1. Ideal = flat, square lattice, one site type \n
        2. SAA = flat, square lattice, two site types -> attr = density, default is 1 dopant in supercell \n
        3. Stepped = square lattice with single step, three site types -> attr = step location, default is middle of supercell \n
        Lattices: \n
        1. Square (100) -> 4 nearest neighbours \n
        2. Triangular (111) -> 6 nearest neighbours \n
        Imposes helical boundary conditions - see Newman, Barkema (1999)
        """
        if system_type.upper() == 'SAA':
            system_type = system_type.upper()
            density = sys_attr.get('density')
            if density == None:
                density = 1/len(sys.lat[:,0])
            sites = int(sys.lat[:,0].size)
            dopants = math.floor(sites*density)
            if dopants < 1: raise ValueError('dopant too dilute for supercell, try including more sites')
            dope_sites = sys.rng.choice(range(sites),size=dopants,replace=False)
            for site in dope_sites: sys.lat[site,0] = 1 # 1 represents dopant and 0 represents host
        elif system_type.lower() == 'stepped':
            system_type = system_type.lower()
            rows,cols = sys.lat_dimensions
            step_site = math.floor(cols/2)-1 # indexing starts at zero
            for row in range(rows):
                sys.lat[step_site,0] = 1
                sys.lat[step_site+1,0] = 2
                step_site += cols
        elif system_type.lower() == 'ideal':
            system_type = system_type.lower()
            sys.lat = np.zeros((sys.lat[:,0].size,2),dtype=int)
        else: raise ValueError('unrecognised system type, check spelling')
        sys.sys_type = system_type
        # Neighbour key
        sites = sys.lat[:,0].size
        if lattice_type.lower() == 'square':
            sys.neigh_key = np.empty((sys.lat[:,0].size,4),dtype=int)
            # neighs 1-4 from site 0, starting top
            #
            #         0
            #         |
            #     3---i---1
            #         |
            #         2
            #
            for site_index in range(sites):
                _,L = sys.lat_dimensions
                sys.neigh_key[site_index,:] = [
                    (site_index-L)%sites, # 0 = +y
                    (site_index+1)%sites, # 1 = +x
                    (site_index+L)%sites, # 2 = -y
                    (site_index-1)%sites # 3 = -x
                ]
        elif lattice_type.lower() == 'triangular':
            sys.neigh_key = np.empty((sys.lat[:,0].size,6),dtype=int)
            # neighs 1-6 from site 0, starting top left
            #
            #       0---1
            #      /     \
            #     5   i   2
            #      \     /
            #       4---3
            #
            for site_index in range(sites):
                _,L = sys.lat_dimensions
                sys.neigh_key[site_index,:] = [
                    (site_index-L-1)%sites, # 0
                    (site_index-L)%sites, # 1
                    (site_index+1)%sites, # 2
                    (site_index+L+1)%sites, # 3
                    (site_index+L)%sites, # 4
                    (site_index-1)%sites # 5
                ]
        else: raise ValueError('unrecognised lattice type, check spelling')
        sys.lat_type = lattice_type
        print(f'Built {sys.lat_type} lattice for {sys.sys_type} system')
        sys._bool_build = True
        return

    def set_lat_occ(self,method:str,lat_template=np.empty((1,1)),**kwargs):
        """Methods: \n
        1. random (needs a num_species kwarg)\n
        2. saturated (needs a fill_species kwarg)\n
        3. custom (note need to supply site types as well) \n
        4. many species (need to supply a {species:coverage} dict as 'theta_key' kwarg)\n
        Make sure this key is consistent with the KMC reaction species identities \n
        Many species method populates lattice randomly according to distribution supplied, so will be subject to some variation depending on lattice size
        """
        if method.lower() == 'random':
            num_species = kwargs['num_species']
            for site in range(math.prod(self.lat_dimensions)):
                self.lat[site,1] = self.rng.integers(0,num_species,endpoint=True)

        if method.lower() == 'saturated':
            fill_species = kwargs['fill_species']
            if fill_species>num_species: raise ValueError('Invalid fill species given')
            self.lat[:,1] = np.full((self.lat[:,1].size),fill_value=fill_species,dtype=int)
            method += f'({fill_species})'

        if method.lower() == 'empty':
            self.lat[:,1] = np.zeros((self.lat[:,1].size),dtype=int)

        if method.lower() == 'custom':
            if self.lat.shape != lat_template.shape: raise ValueError(f'Lattice supplied is wrong dimensions, should be: {self.lat.shape}')
            self.lat = lat_template

        if method.lower() == 'many species':
            theta_key = dict(kwargs['theta_key'])
            if sum(theta_key.values()) != 1: raise ValueError('Total fractional coverage must be 1, note the empty site coverage is also required')
            species_dist = np.empty((len(theta_key)))
            for species in theta_key.keys():
                species_dist[species] = theta_key[species]
            species_dist = np.cumsum(species_dist)
            for site in range(math.prod(self.lat_dimensions)):
                adatom = np.searchsorted(species_dist,self.rng.uniform())
                self.lat[site,1] = adatom
        
        self.lat_occ = method
        print(f'Lattice populated according to {method} occupancy')
        return 

    def what_lat_occ(self,see_lattice=False):
        if see_lattice: print(f'Initial lattice:\n{self.lat}')
        return print(f'Lattice occupancy: {self.lat_occ}')

    def to_KMC(self):
        if not self._bool_build: raise AttributeError('Latttice not built!')
        param_dict = {
            't_max':self.t_max,
            'n_max':self.n_max,
            't_step':self.t_step,
            'runs':self.runs,
            'lattice':self.lat,
            'lattice_info':(self.lat_type,self.sys_type,self.lat_dimensions),
            'neighbours':self.neigh_key,
            't_points':self.t_points,
            'generator':self.rng_seed
        }
        return param_dict
