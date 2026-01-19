import numpy as np
import pandas as pd
import math
from sortedcontainers.sortedlist import SortedList
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from scipy.special import lambertw
from scipy.constants import Boltzmann as k_B

## ## ######################## ## ##
## ## ######################## ## ##
## ## ### Simulation setup ### ## ##
## ## ######################## ## ##
## ## ######################## ## ##
class SimParams:
    def __init__(self,t_max:float,n_max:int,points_to_plot:int,Lattice_dimensions:tuple,runs:int=1,rng_seed=None):
        self.t_max,self.n_max,self.runs = t_max,n_max,runs
        self.t_step , self.t_points = t_max/points_to_plot , points_to_plot+1
        self.lat = np.zeros((math.prod(Lattice_dimensions),2),dtype=int) # each row of the form [site type, occupancy]
        self.lat_dimensions = Lattice_dimensions
        self.lat_occ,self.lat_type = 'Empty','ideal'
        self.rng,self.rng_seed = np.random.default_rng(rng_seed),rng_seed
        self._bool_build = False

    def what_params(self):
        y,x=self.lat_dimensions
        out = f'{self.runs} run(s) on {y}x{x} {self.lat_occ} lattice\n' 
        out += f'Stop conditions: t={self.t_max},n={self.n_max}\n'
        out += f'Plotting interval: t_step={self.t_step} ({self.t_points} steps)\n'
        out += f'Generator seed: {self.rng_seed}'
        return print(out)
    
    def build_lat(sys,system_type:str,**sys_attr):
        """only SAA so far, density = fractional amount of dopant in host supercell (rounded down to an integer number of dopant atoms) \n
        Systems: \n
        1. Ideal = flat, square lattice, one site type \n
        2. SAA = flat, square lattice, two site types -> attr = density, default is 1 dopant in supercell \n
        3. Stepped = square lattice with single step, three site types -> attr = step location, default is middle of supercell
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
        sys.lat_type = system_type
        # Neighbour key -> currently only square lattice
        sys.neigh_key = np.empty((sys.lat[:,0].size,4),dtype=int) # 4 neighbours in square lattice
        rows,cols = sys.lat_dimensions
        def _get_index(location:tuple):
            row,col = location
            _,cols = sys.lat_dimensions
            return int(col + row*cols)
        def _get_coords(index:int):
            _,cols = sys.lat_dimensions
            return int(index // cols), int(index % cols)
        for site_index in range(sys.lat[:,0].size):
            row,col = _get_coords(site_index)
            sys.neigh_key[site_index,:] = [
                _get_index((row,(col+1)%cols)), # +x
                _get_index((row,col-1)) if col!=0 else _get_index((row,cols-1)), # -x
                _get_index(((row-1),col)) if row!=0 else _get_index((rows-1,col)), # +y
                _get_index(((row+1)%rows,col)) # -y
            ]
        print(f'Built {sys.lat_type} lattice')
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
            'lattice_info':(self.lat_type,self.lat_dimensions),
            'neighbours':self.neigh_key,
            't_points':self.t_points,
            'generator':self.rng_seed
        }
        return param_dict

## ## ##################### ## ##
## ## ##################### ## ##
## ## ### KMC execution ### ## ##
## ## ##################### ## ##
## ## ##################### ## ##
class KMC:
    def __init__(sys,sim_param:dict,E_a:np.ndarray,Pre_exp:np.ndarray,Temp_function:callable):
        """E_a and pre-exponetial required as specific shape (n_site_types,2+n_site_types) \n
        T-dependent pre-exponentials not supported yet \n
        [ads0,des0,diff00,diff01,diff02,...] \n
        [ads1,des1,diff10,diff11,diff12,...] \n
        [ads2,des2,diff20,diff21,diff22,...] \n
        [...]
        """
        sys.rng = np.random.default_rng(sim_param['generator'])
        sys.t_max,sys.n_max,sys.t_step,sys.t_points = sim_param['t_max'],sim_param['n_max'],sim_param['t_step'],sim_param['t_points']
        sys.runs = sim_param['runs']
        sys.lat = sim_param['lattice']
        sys.lat_type,sys.lat_dimensions = sim_param['lattice_info']
        sys.neighbour_key = sim_param['neighbours']
        sys.FRM_sortlist = SortedList(key=lambda tup:tup[0]) # sort list based on first tuple entry (time)
        sys.FRM_id_key = {} # ID -> time
        sys.FRM_site_keys = {site:[] for site in range(len(sys.lat[:,0]))} # site -> IDs
        sys.rxn_log,sys.int_log = [],[]
        print(f'Intialising {sys.lat_type} lattice system on a {sys.lat_dimensions} supercell')
        
        sys.T = Temp_function
        
        # check array dimensions
        E_a = np.atleast_2d(E_a)
        Pre_exp = np.atleast_2d(Pre_exp)
        n_site_types = {'ideal':1,'SAA':2,'stepped':3} # SAA
        expect_shape = (n_site_types[sys.lat_type],2+n_site_types[sys.lat_type])
        if np.shape(E_a) != expect_shape:
            raise IndexError(f'Actvation energies input wrong, should be {expect_shape} but E_a input is {np.shape(E_a)}')
        if np.shape(E_a) != np.shape(Pre_exp):
            raise IndexError(f'Pre-exp and E_a array dimensions dont match: {np.shape(Pre_exp)} and {np.shape(E_a)}')
        # build base E_a array
        sys.E_a = np.empty((len(sys.lat[:,0]),6),dtype=float) # only 6 process: [ads,des,diff+x,diff-x,diff+y,diff-y]
        for site,site_type in enumerate(sys.lat[:,0]):
            sys.E_a[site,0:2] = E_a[site_type,0:2]
            for neigh_ind,neigh in enumerate(sys.neighbour_key[site,:]):
                sys.E_a[site,2+neigh_ind] = E_a[site_type,2+sys.lat[neigh,0]]
        # build base pre_exp array, will be tricky for T-dependent pre-exponentials
        sys.A = np.empty((len(sys.lat[:,0]),6),dtype=float) # only 6 process: [ads,des,diff+x,diff-x,diff+y,diff-y]
        for site,site_type in enumerate(sys.lat[:,0]):
            sys.A[site,0:2] = Pre_exp[site_type,0:2]
            for neigh_ind,neigh in enumerate(sys.neighbour_key[site,:]):
                sys.A[site,2+neigh_ind] = Pre_exp[site_type,2+sys.lat[neigh,0]]
        

    def what_params(params,see_lattice=False):
        print(f'Reaction channels = {len(params.E_a[0,:])}')
        print(f'Simulation: t_max={params.t_max}, n_max={params.n_max}, grid_points={params.t_points}, runs={params.runs}')
        if see_lattice: print(f'Initial lattice:\n{params.lat}')
    
    def change_params(sys,**params_to_change):
        for keyword in params_to_change.keys():
            setattr(sys,keyword,params_to_change[keyword])
    
    def report_logs(sys,clear=True):
        int_counts = [0,0,0,0,0]
        for entry in sys.int_log:
            if entry[0] == '1':
                int_counts[0] += 1
            elif entry[0] == '2':
                int_counts[1] += 1
            elif entry[0] == '3':
                int_counts[2] += 1
        for entry in sys.rxn_log:
            if entry[0] == '1':
                int_counts[3] += 1
            elif entry[0] == 2:
                int_counts[4] += 1
        print('t generation errors:')
        print(f'Intial guess too high = {int_counts[0]}')
        print(f'Failed to find brentq bracket = {int_counts[1]}')
        print(f'Newton method failed = {int_counts[2]}')
        print(f'Null reactions: blocked hop={int_counts[4]}, infinite wait times={int_counts[3]}')
        if clear: sys.rxn_log,sys.int_log = [],[]

    def _get_index(grid,location:tuple):
        row,col = location
        _,cols = np.shape(grid.lat)
        return int(col + row*cols)
    
    def _get_coords(grid,index:int):
        _,cols = grid.lat_dimensions
        return int(index // cols), int(index % cols)
    
    #####################################
    ### FRM_data structure definition ###
    #####################################

    def _FRM_insert(tree,time:float,ID:tuple):
        if not np.isfinite(time):
            tree.rxn_log.append(f'1. Null rxn: t={time} ID={ID}')
            return
        if ID in tree.FRM_id_key:
            old_time = tree.FRM_id_key[ID]
            try:
                tree.FRM_sortlist.remove((old_time,ID))
            except Exception:
                pass
            tree.FRM_id_key.pop(ID)
            tree.FRM_site_keys[ID[0]].remove(ID)
        data = (time,ID)
        tree.FRM_sortlist.add(data)
        tree.FRM_id_key[ID] = time
        tree.FRM_site_keys[ID[0]].append(ID)
        return
    
    def _FRM_remove(tree,ID:tuple):
        if ID in tree.FRM_id_key:
            old_time = tree.FRM_id_key.pop(ID)
            try:
                tree.FRM_sortlist.remove((old_time,ID))
                tree.FRM_site_keys[ID[0]].remove(ID)
            except ValueError:
                pass

    #####################
    ### KMC functions ###
    #####################
    
    def _t_gen(self,prop_func:callable,time:float,random_number:float,other_args:tuple,improved_guess=True):
        """Generates a new absolute time from a random time step by solving: \n
        $int_{t}^{t+delta_t}(a0(t,other_args)) + ln(r) == 0$ \n
        Uses newton root-finding method with x0 = -ln(r)/a0(t) \n
        If newton fails resorts to brentq method \n
        Relative tolerance: 10**-6
        """
        if improved_guess:
            # Setup initial guess (improved)
            rxn,_,site = other_args
            guess = time+self._improved_guess(time,random_number,self.E_a[site,rxn],self.A[site,rxn])
        else:
            # Setup intial guess (naive)
            a0_t = prop_func(time,*other_args)
            if a0_t<=0: raise ValueError(f'Negative or zero propensity:\na0(t)={a0_t},t={time},r={random_number}\nother={other_args}') 
            guess = time-np.log(random_number)/a0_t
        rel_tol = 10**-6
        max_tau = 5 * self.t_max
        if guess > max_tau:
            guess_bool = ('Y' if improved_guess else 'N')
            self.int_log.append(f'1. Guess greater than max time: Improved={guess_bool}')
            return np.inf # is this needed?
        # Generate A0(t)
        t_lim = min(2*guess,self.t_max)
        sol_prop = solve_ivp(
            fun=lambda tt, y : np.array([prop_func(tt,*other_args)],dtype=float),
            t_span=(time,t_lim),
            y0=[0.0], # int from time -> time+time_step is zero when time_step=0
            method='Radau',
            dense_output=True,
            rtol=rel_tol
        )
        # Define functions
        def f(new_time:float):
            return float(sol_prop.sol(new_time)[0] + np.log(random_number))
        def fprime(new_time:float):
            return prop_func(new_time,*other_args)
        # Newton method
        try:
            x0 = max(guess,10**-12)
            sol = root_scalar(f,method='newton',x0=x0,fprime=fprime,rtol=rel_tol)
            if sol.converged:
                return sol.root
        except Exception:
            pass
        if sol.root > max_tau: return np.inf
        # If Newton method fails use brentq backup
        newt_attempt = sol.root,sol.flag
        tau_lo = 0 if random_number > 0 else -10**-12 
        tau_hi = min(guess,max_tau)
        if f(tau_hi)<0:
            loop_count = 0
            while f(tau_hi)<0:
                tau_hi*=2
                loop_count += 1
                if loop_count>60: self.int_log.append('2. Failed to find appropriate bracket') ; return np.inf
        sol = root_scalar(f,method='brentq',bracket=[tau_lo,tau_hi],rtol=rel_tol)
        if not sol.converged: raise RuntimeError(f'Both root finding methods failed, check prop_func behaviour (t={time})')
        self.int_log.append(f'3. Newton failed: step={newt_attempt[0]}, Brentq converged: step={sol.root}, flag: {newt_attempt[1]}')
        return sol.root # absolute time of next reaction
    
    def _improved_guess(self,time:float,random_number:float,E_a:float,Pre_exp:float):
        """Michail's improved intial guess for Newton root-finding
        only applies to linear temperature ramps and time-indpendent Pre-exponential factors
        """
        sim_temp = self.T(time)
        beta = self.T(1)-self.T(0)

        C = Pre_exp * np.exp(-E_a/(k_B*sim_temp)) * (k_B*sim_temp**2)/E_a + beta*np.log(1/random_number)
        temp_guess = (E_a/k_B) / (2*lambertw(1/2*np.sqrt((Pre_exp*E_a)/(C*k_B))))
        if np.imag(temp_guess) != 0: return ValueError('Complex valued initial guess!')
        # lambertw returns complex types but k=0 branch is real valued for all z>-1/e so can safely ignore imaginary part
        return ((temp_guess - sim_temp)/(beta)).real

    def _rxn_step(sys,lattice:np.ndarray,site:int,rxn_ind:int,event_count:int,adatom_count:int):
        """Updates the lattice according to the chosen reaction \n
        returns the updated lattice
        """
        # the actual forms of the process are the same, its the propensities that differ i think
        rxn_key = {
            0 : (1,0,site), # adsorption
            1 : (0,0,site), # desorption
            2 : (0,1,sys.neighbour_key[site,0]), # hop +x
            3 : (0,1,sys.neighbour_key[site,1]), # hop -x
            4 : (0,1,sys.neighbour_key[site,2]), # hop +y
            5 : (0,1,sys.neighbour_key[site,3]) # hop -y
        }
        species,new_species,new_site = rxn_key[rxn_ind]
        lattice[site,1] = species
        if new_site != site:
            lattice[new_site,1] = new_species
        # Counters
        if rxn_ind == 0: adatom_count += 1
        if rxn_ind == 1: adatom_count -= 1 ; event_count += 1
        return lattice,new_site,event_count,adatom_count

    def _k(sys,site:int,rxn:int,time:float)->np.ndarray:
        return np.multiply(sys.A[site,rxn],np.exp(-sys.E_a[site,rxn]/(k_B*sys.T(time))))
    
    ## DM funcs ##
    def _k_array(sys,time):
        return np.multiply(sys.A,np.exp(-sys.E_a/(k_B*sys.T(time))))

    def _DM_total_prop(sys,time:float,c_arr:np.ndarray,dummy=None)->float:
        return np.sum(np.multiply(sys._k_array(time),c_arr))
        
    def _DM_site_c(
            sys,
            lattice:np.ndarray,
            site:tuple,
        )->np.ndarray:
        if int(lattice[site,1]) == 0:
            c = np.array([1,0,0,0,0,0],dtype=int)
        elif int(lattice[site,1]) == 1:
            c = np.array([0,1,0,0,0,0],dtype=int)
            neighs = sys.neighbour_key[site,:]
            for ind,neighbour in enumerate(neighs):
                if int(lattice[neighbour,1]) == 0:
                    c[2+ind] = 1
        return c
    
    def _DM_c_change(
            sys,
            lattice:np.ndarray,
            c_array:np.ndarray,
            site:int,
            new_site:int
        ):
        # Origin
        c_array[site,:] = sys._DM_site_c(lattice,site)
        for site_n in sys.neighbour_key[site,:]:
            c_array[site_n,:] = sys._DM_site_c(lattice,site_n) # update original neighbours
        if new_site != site: # hopping
            for site_n in sys.neighbour_key[new_site,:]:
                c_array[site_n,:] = sys._DM_site_c(lattice,site_n) # update new_location neighbours
        return c_array
    
    def _DM_gen_c_array(sys,lattice):
        c = np.empty((np.shape(sys.E_a)),dtype=int)
        for site in range(len(lattice[:,0])):
            c[site,:] = sys._DM_site_c(lattice,site)
        return c
    
    def _DM_get_prop_array(sys,c_array,time):
        a_array = np.multiply(c_array,sys._k_array(time))
        a_acc = np.cumsum(a_array)
        return a_acc

    ## FRM funcs ##
    def _FRM_site_prop(
            sys,
            time:float,
            rxn:int,
            lattice:np.ndarray,
            site:int,
        )->float:
        if int(lattice[site,1]) == 0: # adsorption
            c = 1
        elif int(lattice[site,1]) == 1:
            if rxn == 1: # desorption
                c = 1
            elif rxn > 1:
                neigh_rxn = rxn - 2
                c = 1 - int(lattice[sys.neighbour_key[site,neigh_rxn],1]) # still only applicable to 1 adsorbate
        k = sys._k(site,rxn,time)
        return  k*c
    
    def _FRM_generate_queue(sys):
        for site in range(len(sys.lat[:,0])):
            if sys.lat[site,1] == 0:
                t_k = sys._t_gen(sys._FRM_site_prop,0,sys.rng.random(),(0,sys.lat,site)) # adsorption
                sys._FRM_insert(t_k,(site,0))
            elif sys.lat[site,1] == 1:
                t_k = sys._t_gen(sys._FRM_site_prop,0,sys.rng.random(),(1,sys.lat,site)) # desorption
                sys._FRM_insert(t_k,(site,1))
                for ind,neigh in enumerate(sys.neighbour_key[site,:]):
                    if sys.lat[neigh,1] == 0:
                        t_k = sys._t_gen(sys._FRM_site_prop,0,sys.rng.random(),(ind+2,sys.lat,site))
                        sys._FRM_insert(t_k,(site,ind+2))
        return
    
    def _FRM_update(sys,time:float,site:int,new_site:int,lattice:np.ndarray):
        # This site
        to_remove = sys.FRM_site_keys[site].copy()
        for ID in to_remove:
            sys._FRM_remove(ID)
        
        lat_at_site = lattice[site,1]
        if lat_at_site == 0:
            t_ads = sys._t_gen(sys._FRM_site_prop,time,sys.rng.random(),(0,lattice,site))
            sys._FRM_insert(t_ads,ID=(site,0)) # adsorption
        elif lat_at_site == 1:
            t_des = sys._t_gen(sys._FRM_site_prop,time,sys.rng.random(),(1,lattice,site))
            sys._FRM_insert(t_des,ID=(site,1)) # desorption
            for ind,neigh in enumerate(sys.neighbour_key[site,:]):
                if lattice[neigh,1] == 0:
                    t_hop = sys._t_gen(sys._FRM_site_prop,time,sys.rng.random(),(ind+2,lattice,site))
                    sys._FRM_insert(t_hop,ID=(site,ind+2)) # hops
        # New_site
        if new_site != site:
            to_remove = sys.FRM_site_keys[new_site].copy() #[ID for ID in list(sys.FRM_id_key.keys()) if ID[0] == new_site]
            for ID in to_remove:
                sys._FRM_remove(ID)

            lat_at_new_site = lattice[new_site,1]
            if lat_at_new_site == 0:
                t_ads = sys._t_gen(sys._FRM_site_prop,time,sys.rng.random(),(0,lattice,new_site))
                sys._FRM_insert(t_ads,ID=(new_site,0)) # adsorption
            elif lat_at_new_site == 1:
                t_des = sys._t_gen(sys._FRM_site_prop,time,sys.rng.random(),(1,lattice,new_site))
                sys._FRM_insert(t_des,ID=(new_site,1)) # desorption
                for ind,neigh in enumerate(sys.neighbour_key[new_site,:]):
                    if lattice[neigh,1] == 0:
                        t_hop = sys._t_gen(sys._FRM_site_prop,time,sys.rng.random(),(ind+2,lattice,new_site))
                        sys._FRM_insert(t_hop,ID=(new_site,ind+2)) # hops
        # Neighbour sites, only changes hops
        direction_key = {0:1,1:0,2:3,3:2} # +x<->-x, -y<->+y hops
        for ind,neigh in enumerate(sys.neighbour_key[site,:]):
            # remove disabled neighbour events
            sys._FRM_remove((neigh,direction_key[ind]+2))
            
            # add enabled neighbour events if lattice available
            if lattice[site,1] == 0 and lattice[neigh,1] == 1: # target available and exists an adatom to hop
                if sys._FRM_site_prop(time,direction_key[ind]+2,lattice,neigh)>0:
                    t_hop = sys._t_gen(sys._FRM_site_prop,time,sys.rng.random(),(direction_key[ind]+2,lattice,neigh))
                    sys._FRM_insert(t_hop,ID=(neigh,direction_key[ind]+2)) # neighbours hops
                else:
                    sys.rxn_log.append(f'2. Null rxn: t={time} at site={site} rxn={direction_key[ind]+2}')
        if new_site != site:
            for ind,neigh in enumerate(sys.neighbour_key[new_site,:]):
                # removed disabled neighbour events
                sys._FRM_remove((neigh,direction_key[ind]+2))

                # add enabled neighbour events if lattice available
                if lattice[new_site,1] == 0 and lattice[neigh,1] == 1: # target available and exists an adatom to hop
                    t_hop = sys._t_gen(sys._FRM_site_prop,time,sys.rng.random(),(direction_key[ind]+2,lattice,neigh))
                    sys._FRM_insert(t_hop,ID=(neigh,direction_key[ind]+2)) # neighbours hops
        return
    
    ######################
    ### KMC algortihms ###
    ######################

    def run_DM(sys,return_n_steps=False):
        """Runs a kinetic Monte Carlo simulation on the defined lattice \n
        Uses the Direct method \n
        Returns a (4*runs) column dataframe of time, temp, coverage and desorption rate \n
        dataframe labels of the form (e.g. run X): \n
        'timeX','tempX','thetaX','rateX'
        """
        data = {}
        lat_initial = sys.lat.copy()
        for run in range(sys.runs):
            #Initialise
            lat = lat_initial.copy()
            c = sys._DM_gen_c_array(lat)
            t,n,site,new_site,count,old_count,plot_ind=0.0,0,0,0,0,0,0
            adatoms = np.sum(lat[:,1])
            times = np.array([np.nan]*(sys.t_points))
            thetas,rates,temps = times.copy(),times.copy(),times.copy()
            while t<sys.t_max and n<sys.n_max:
                # Local occ change
                c = sys._DM_c_change(lat,c,site,new_site)
                # Generate next time
                new_t = sys._t_gen(sys._DM_total_prop,t,sys.rng.random(),(c,None),False)
                # Save state
                theta_save = adatoms/len(lat[:,1])
                rate_save = (count-old_count)/(new_t-t) if (new_t-t) != 0 else 0
                next_save = (t-t%sys.t_step + sys.t_step) if t!=0 else 0
                while next_save<new_t and plot_ind<sys.t_points:
                    save_state = True
                    # save values of interest
                    thetas[plot_ind] = theta_save
                    rates[plot_ind] = rate_save
                    times[plot_ind] = next_save
                    temps[plot_ind] = sys.T(next_save)
                    next_save += sys.t_step # next time to save
                    plot_ind += 1 # next grid point
                if save_state: old_count = count; save_state=False # reset counts
                # Advance system time
                t = new_t
                # Global prop gen
                a_acc = sys._DM_get_prop_array(c,t)
                if a_acc[-1] == 0: print('Reactions complete (total_propensity = 0)'); break
                # Choose reaction
                mu_index = np.searchsorted(a_acc,a_acc[-1]*sys.rng.random(),side='left') # binary search
                rxn_index = mu_index % len(sys.E_a[0,:])
                site = mu_index//len(sys.E_a[0,:])
                # Advance system state
                lat,new_site,count,adatoms = sys._rxn_step(lat,site,rxn_index,count,adatoms)
                n += 1
            if return_n_steps: print(f'run{run}: n={n}, t={t}')
            # save run data
            run_label = [f'time{run}',f'temp{run}',f'theta{run}',f'rate{run}']
            run_data = {
                run_label[0]:times,
                run_label[1]:temps,
                run_label[2]:thetas,
                run_label[3]:rates
            }
            data = data | run_data
        return pd.DataFrame(data)
    
    def run_FRM(sys,return_n_steps=False):
        """Runs a kinetic Monte Carlo simulation on the defined lattice \n
        Uses the First reaction method \n
        Returns a 4*runs column dataframe of time, temp, coverage and desorption rate \n
        dataframe labels of the form (e.g. run X): \n
        'timeX','tempX','thetaX','rateX'
        """
        data = {}
        lat_initial = sys.lat.copy()
        for run in range(sys.runs):
            lat = lat_initial.copy()
            t,n,count,old_count,plot_ind=0.0,0,0,0,0
            adatoms = np.sum(lat)
            times = np.array([np.nan]*(sys.t_points))
            thetas,rates,temps = times.copy(),times.copy(),times.copy()
            # Initialise data structure
            sys._FRM_generate_queue()
            while t<sys.t_max and n<sys.n_max:
                # Choose reaction and time
                if len(sys.FRM_sortlist)==0:print('Reactions complete (reaction queue empty)'); break
                new_t,index_tup = sys.FRM_sortlist[0]
                # Save state
                theta_save = adatoms/lat.size
                rate_save = (count-old_count)/(new_t-t) if (new_t-t) != 0 else 0
                next_save = (t-t%sys.t_step + sys.t_step) if t!=0 else 0
                while next_save<new_t and plot_ind<sys.t_points:
                    save_state = True
                    # save values of interest
                    thetas[plot_ind] = theta_save
                    rates[plot_ind] = rate_save
                    times[plot_ind] = next_save
                    temps[plot_ind] = sys.T(next_save)
                    next_save += sys.t_step # next time to save
                    plot_ind += 1 # next grid point
                if save_state: old_count = count; save_state=False # reset counts
                site,rxn = index_tup
                t = new_t
                # Advance state and update queue
                lat,new_site,count,adatoms = sys._rxn_step(lat,site,rxn,count,adatoms)
                sys._FRM_update(t,site,new_site,lat)
                n += 1
            if return_n_steps: print(f'run{run}: n={n}, t={t}')
            # save run data
            run_label = [f'time{run}',f'temp{run}',f'theta{run}',f'rate{run}']
            run_data = {
                run_label[0]:times,
                run_label[1]:temps,
                run_label[2]:thetas,
                run_label[3]:rates
            }
            data = data | run_data
        return pd.DataFrame(data)
    
    ##########################
    ### Benchmarking funcs ###
    ##########################

    def run_DM_no_data(sys,return_n_steps=False):
        """Runs a kinetic Monte Carlo simulation on the defined lattice \n
        Uses the Direct method \n
        Returns a (4*runs) column dataframe of time, temp, coverage and desorption rate \n
        dataframe labels of the form (e.g. run X): \n
        'timeX','tempX','thetaX','rateX'
        """
        lat_initial = sys.lat.copy()
        for run in range(sys.runs):
            #Initialise
            lat = lat_initial.copy()
            c = sys._DM_gen_c_array(lat)
            t,n,site,new_site,count=0.0,0,0,0,0
            adatoms = np.sum(lat[:,1])
            while t<sys.t_max and n<sys.n_max:
                # Local occ change
                c = sys._DM_c_change(lat,c,site,new_site)
                # Generate next time
                new_t = sys._t_gen(sys._DM_total_prop,t,sys.rng.random(),(c,None),False)
                # Advance system time
                t = new_t
                # Global prop gen
                a_acc = sys._DM_get_prop_array(c,t)
                if a_acc[-1] == 0: print('Reactions complete (total_propensity = 0)'); break
                # Choose reaction
                mu_index = np.searchsorted(a_acc,a_acc[-1]*sys.rng.random(),side='left') # binary search
                rxn_index = mu_index % len(sys.E_a[0,:])
                site = mu_index//len(sys.E_a[0,:])
                # Advance system state
                lat,new_site,count,adatoms = sys._rxn_step(lat,site,rxn_index,count,adatoms)
                n += 1
            if return_n_steps: print(f'run{run}: n={n}, t={t}')
        return
    
    def run_FRM_no_data(sys,return_n_steps=False):
        """Runs a kinetic Monte Carlo simulation on the defined lattice \n
        Uses the First reaction method \n
        Returns a 4*runs column dataframe of time, temp, coverage and desorption rate \n
        dataframe labels of the form (e.g. run X): \n
        'timeX','tempX','thetaX','rateX'
        """
        lat_initial = sys.lat.copy()
        for run in range(sys.runs):
            lat = lat_initial.copy()
            t,n,count=0.0,0,0
            adatoms = np.sum(lat)
            # Initialise data structure
            sys._FRM_generate_queue()
            while t<sys.t_max and n<sys.n_max:
                # Choose reaction and time
                if len(sys.FRM_sortlist)==0:print('Reactions complete (reaction queue empty)'); break
                new_t,index_tup = sys.FRM_sortlist[0]
                site,rxn = index_tup
                t = new_t
                # Advance state and update queue
                lat,new_site,count,adatoms = sys._rxn_step(lat,site,rxn,count,adatoms)
                sys._FRM_update(t,site,new_site,lat)
                n += 1
            if return_n_steps: print(f'run{run}: n={n}, t={t}')
        return

    ##################
    ### Data funcs ###
    ##################
    
    def get_avg(sys,data):
        """Calculates the averages of each column in a KMC data output \n
        Make sure the data comes from this system
        """
        df = pd.DataFrame(data)
        labels = np.empty((4,sys.runs),dtype=object)
        for i in range(sys.runs):
            labels[0,i] = f'time{i}'
            labels[1,i] = f'temp{i}'
            labels[2,i] = f'theta{i}'
            labels[3,i] = f'rate{i}'
        df['time avg'] = df[labels[0,:]].mean(axis=1)
        df['temp avg'] = df[labels[1,:]].mean(axis=1)
        df['theta avg'] = df[labels[2,:]].mean(axis=1)
        df['rate avg'] = df[labels[3,:]].mean(axis=1)
        return df
    
    def view_sq_lat(sys,site_labels=None,adatom_labels=None):
        """Visualise the initial lattice for the systems with sqaure neighbours
        default labels are alphabetical, uppercase for sites and lowercase for adatoms
        note vew may look uneven is label names are different character lengths
        """
        char_size = 1
        if site_labels == None:
            site_labels = {index:letter for index,letter in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}
        if adatom_labels == None:
            adatom_labels = {index:letter for index,letter in enumerate('_abcdefghijklmnopqrstuvwxyz')}
        char_size += max(len(adatom_labels[s]) for s in adatom_labels.keys()) + max(len(site_labels[s]) for s in site_labels.keys())
        built_lat = np.empty((sys.lat_dimensions),dtype=f'U{char_size}')
        for site,site_type in enumerate(sys.lat[:,0]):
            built_lat[sys._get_coords(site)] = site_labels[site_type] + f'*{adatom_labels[sys.lat[site,1]]}'
        for row in range(len(built_lat[:,0])):
            print(built_lat[row,:])