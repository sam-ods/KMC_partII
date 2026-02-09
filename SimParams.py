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
        Lattices:
        1. Square (100) -> 4 nearest neighbours
        2. Triangular (111) -> 6 nearest neighbours
        """
        if system_type.upper() == 'SAA':
            system_type = system_type.upper()
            density = sys_attr.get('density')
            if density == None:
                density = 1/len(sys.lat[:,0])
            sites = int(sys.lat[:,0].size)
            dopants = math.floor(sites*density)
            if dopants < 1: raise ValueError('dopant too dilute for supercell, try including more sites')
            dope_sites = sys.rng.choice(range(sites),size=dopants,replace=False) # are they really randomly distributed?
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
        rows,cols = sys.lat_dimensions
        def _get_index(location:tuple):
            row,col = location
            _,cols = sys.lat_dimensions
            return int(col + row*cols)
        def _get_coords(index:int):
            _,cols = sys.lat_dimensions
            return int(index // cols), int(index % cols)
        if lattice_type.lower() == 'square':
            sys.neigh_key = np.empty((sys.lat[:,0].size,4),dtype=int)
            # neighs 1-4 from site 0, starting top
            #
            #         1
            #         |
            #     4---0---2
            #         |
            #         3
            #
            for site_index in range(sites):
                row,col = _get_coords(site_index)
                sys.neigh_key[site_index,:] = [
                    _get_index(((row-1),col)) if row!=0 else _get_index((rows-1,col)), # +y
                    _get_index((row,(col+1)%cols)), # +x
                    _get_index(((row+1)%rows,col)), # -y
                    _get_index((row,col-1)) if col!=0 else _get_index((row,cols-1)) # -x
                ]
        elif lattice_type.lower() == 'triangular':
            sys.neigh_key = np.empty((sys.lat[:,0].size,6),dtype=int)
            # neighs 1-6 from site 0, starting top left
            #
            #       1---2
            #      /     \
            #     6   0   3
            #      \     /
            #       5---4
            #
            for site_index in range(sites):
                row,col = _get_coords(site_index)
                if row == 0:
                    site1_row = rows-1
                if col == 0:
                    site1_col = cols-1
                else:
                    site1_row,site1_col = row-1,col-1
                sys.neigh_key[site_index,:] = [
                    _get_index((site1_row,site1_col)), # 1
                    _get_index(((row-1),col)) if row!=0 else _get_index((rows-1,col)), # 2
                    _get_index((row,(col+1)%cols)), # 3
                    _get_index(((row+1)%rows,(col+1)%cols)), # 4
                    _get_index(((row+1)%rows,col)), # 5
                    _get_index((row,(col-1))) if col!=0 else _get_index((row,cols-1)) # 6
                ]
        else: raise ValueError('unrecognised lattice type, check spelling')
        sys.lat_type = lattice_type
        print(f'Built {sys.lat_type} lattice for {sys.sys_type} system')
        sys._bool_build = True
        return

    def set_lat_occ(self,method:str,num_species:int=1,fill_species:int=1,lat_template=np.empty((1,1))):
        """Methods: \n
        1. random \n
        2. saturated \n
        3. custom (note need to supply site types as well)
        """
        if method == 'random':
            for site in range(math.prod(self.lat_dimensions)):
                self.lat[site,1] = self.rng.integers(0,num_species,endpoint=True)
        if method == 'saturated':
            if fill_species>num_species: raise ValueError('Invalid fill species given')
            self.lat[:,1] = np.full((self.lat[:,1].size),fill_value=fill_species,dtype=int)
            method += f'({fill_species})'
        if method == 'empty':
            self.lat[:,1] = np.zeros((self.lat[:,1].size),dtype=int)
        if method == 'custom':
            if self.lat.shape != lat_template.shape: raise ValueError(f'Lattice supplied is wrong dimensions, should be: {self.lat.shape}')
            self.lat = lat_template
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
