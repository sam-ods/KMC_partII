import numpy as np
import pandas as pd
from sortedcontainers.sortedlist import SortedList
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from scipy.constants import Boltzmann as k_B

## ## ######################## ## ##
## ## ######################## ## ##
## ## ### Simulation setup ### ## ##
## ## ######################## ## ##
## ## ######################## ## ##
class SimParams:
    def __init__(self,t_max:float,n_max:int,points_to_plot:int,Lattice_dimensions:tuple,runs:int=1,rng_seed=None):
        self.t_max = t_max
        self.n_max = n_max
        self.t_points = points_to_plot+1
        self.t_step = t_max/points_to_plot
        self.runs = runs
        self.lat = np.zeros(Lattice_dimensions,dtype=int)
        self.lat_occ = 'Empty'
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)

    def what_params(self):
        y,x=self.lat.shape
        out = f'{self.runs} run(s) on {y}x{x} {self.lat_occ} lattice\n' 
        out += f'Stop conditions: t={self.t_max},n={self.n_max}\n'
        out += f'Plotting interval: t_step={self.t_step} ({self.t_points} steps)\n'
        out += f'Generator seed: {self.rng_seed}'
        return print(out)
    
    def set_lat_occ(self,method:str,num_species:int,fill_species:int=1,lat_template=np.empty((1,1))):
        _,cols = np.shape(self.lat)
        if method == 'random':
            for site in range(self.lat.size):
                self.lat[int(site // cols), int(site % cols)] = self.rng.integers(0,num_species,endpoint=True)
        if method == 'saturated':
            if fill_species>num_species: raise ValueError('Invalid fill species given')
            self.lat = np.full((self.lat.shape),fill_value=fill_species,dtype=int)
            method += f' with species {fill_species}'
        if method == 'empty':
            self.lat = np.zeros((np.shape(self.lat)),dtype=int)
        if method == 'custom':
            if self.lat.shape != lat_template.shape: raise ValueError(f'Lattice suppled is wrong dimensions, should be: {self.lat.shape}')
            self.lat = lat_template
        self.lat_occ = method
        return 

    def what_lat_occ(self,see_lattice=False):
        if see_lattice: print(f'Initial lattice:\n{self.lat}')
        return print(f'Lattice occupancy: {self.lat_occ}')

    def to_KMC(self):
        param_dict = {
            't_max':self.t_max,
            'n_max':self.n_max,
            't_step':self.t_step,
            'runs':self.runs,
            'lattice':self.lat,
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
    def __init__(sys,sim_param:dict,E_a:np.ndarray,Pre_exp_function:callable,Temp_function:callable):
        sys.rng = np.random.default_rng(sim_param['generator'])
        sys.t_max,sys.n_max,sys.t_step,sys.t_points = sim_param['t_max'],sim_param['n_max'],sim_param['t_step'],sim_param['t_points']
        sys.runs = sim_param['runs']
        sys.lat = sim_param['lattice']
        sys.E_a = E_a
        sys.A = Pre_exp_function
        sys.T = Temp_function
        _,sys.x_max = np.shape(sys.lat)
        sys.FRM_sortlist = SortedList(key=lambda tup:tup[0]) # sort list based on first tuple entry (time)
        sys.FRM_id_key = {} # ID -> time
        sys.FRM_site_keys = {site:[] for site in range(sys.lat.size)} # site -> IDs
        sys.log = []
        sys.int_log = []

        key = np.empty((sys.lat.size,4),dtype=int) # 4 neighbours in square lattice
        rows,cols = sys.lat.shape
        for site_index in range(sys.lat.size):
            row,col = sys._get_coords(site_index)
            key[site_index,:] = [
                sys._get_index((row,(col+1)%cols)), # +x
                sys._get_index((row,col-1)) if col!=0 else sys._get_index((row,cols-1)), # -x
                sys._get_index(((row-1),col)) if row!=0 else sys._get_index((rows-1,col)), # +y
                sys._get_index(((row+1)%rows,col)) # -y
            ]
        sys.neighbour_key = key # key contains neighbour site indices

    def what_params(params,see_lattice=False):
        print(f'Reaction channels = {len(params.E_a)}')
        print(f'Simulation: t_max={params.t_max}, n_max={params.n_max}, grid_points={params.t_points}, runs={params.runs}')
        print(f'Kinetic parameters:\n    E_a = {np.around(params.E_a,23)}\n    T(0) = {params.T(0)}\n    A(T(0)) = {params.A(params.T(0))}')
        if see_lattice: print(f'Initial lattice:\n{params.lat}')
    
    def change_params(sys,**params_to_change):
        for keyword in params_to_change.keys():
            setattr(sys,keyword,params_to_change[keyword])
    
    def _get_index(grid,location:tuple):
        row,col = location
        _,cols = np.shape(grid.lat)
        return int(col + row*cols)
    
    def _get_coords(grid,index:int):
        _,cols = np.shape(grid.lat)
        return int(index // cols), int(index % cols)
    
    #####################################
    ### FRM_data structure definition ###
    #####################################

    def _FRM_insert(tree,time:float,ID:tuple):
        if not np.isfinite(time):
            tree.log.append(f'Null rxn: t={time} ID={ID}')
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
    
    def _t_gen(self,prop_func:callable,time:float,random_number:float,other_args:tuple):
        """Generates a new absolute time from a random time step by solving: \n
        $int_{t}^{t+delta_t}(a0(t,other_args)) + ln(r) == 0$ \n
        Uses newton root-finding method with x0 = -ln(r)/a0(t) \n
        If newton fails resorts to brentq method \n
        Relative tolerance: 10**-6
        """
        # Setup intial guess
        a0_t = prop_func(time,*other_args)
        if a0_t<=0: raise ValueError(f'Negative or zero propensity:\na0(t)={a0_t},t={time},r={random_number}\nother={other_args}') 
        guess = -np.log(random_number)/a0_t
        rel_tol = 10**-6
        max_tau = 10**4 * self.t_max
        if guess > max_tau: return np.inf
        # Generate A0(t)
        t_lim = min(time+2*guess,self.t_max)
        sol_prop = solve_ivp(
            fun=lambda tt, y : np.array([prop_func(tt,*other_args)],dtype=float),
            t_span=(time,t_lim),
            y0=[0.0], # int from time -> time+time_step is zero when time_step=0
            method='Radau',
            dense_output=True,
            rtol=rel_tol
        )
        # Define functions
        def f(time_step:float):
            return float(sol_prop.sol(time+time_step)[0] + np.log(random_number))
        def fprime(time_step:float):
            return prop_func(time+time_step,*other_args)
        # Newton method
        try:
            x0 = max(guess,10**-12)
            sol = root_scalar(f,method='newton',x0=x0,fprime=fprime,rtol=rel_tol)
            if sol.converged:
                return time+sol.root
        except Exception:
            pass
        # If Newton method fails use brentq backup
        newt_attempt = sol.root
        tau_lo = 0 if random_number > 0 else -10**-12 
        tau_hi = min(guess,max_tau)
        if f(tau_hi)<0:
            loop_count = 0
            while f(tau_hi)<0:
                tau_hi*=2
                loop_count += 1
                if loop_count>60: return np.inf
        sol = root_scalar(f,method='brentq',bracket=[tau_lo,tau_hi],rtol=rel_tol)
        if not sol.converged: raise RuntimeError(f'Both root finding methods failed, check prop_func behaviour (t={time})')
        if sol.root > max_tau:
            self.int_log.append(f'Newton method failed, max timestep reached -> returned np,inf')
            return np.inf
        else:
            self.int_log.append(f'Newton failed: step={newt_attempt}, Brentq converged: step={sol.root}')
            return time+sol.root # absolute time of next reaction
    
    def _rxn_step(sys,lattice:np.ndarray,site:int,rxn_ind:int,event_count:int,adatom_count:int):
        """Updates the lattice according to the chosen reaction \n
        returns the updated lattice
        """
        rxn_key = {
            0 : (1,0,site), # adsorption
            1 : (0,0,site), # desorption
            2 : (0,1,sys.neighbour_key[site,0]), # hop +x
            3 : (0,1,sys.neighbour_key[site,1]), # hop -x
            4 : (0,1,sys.neighbour_key[site,2]), # hop +y
            5 : (0,1,sys.neighbour_key[site,3]) # hop -y
        }
        species,new_species,new_site = rxn_key[rxn_ind]
        lattice[sys._get_coords(site)] = species
        if new_site != site:
            lattice[sys._get_coords(new_site)] = new_species
        # Counters
        if rxn_ind == 0: adatom_count += 1
        if rxn_ind == 1: adatom_count -= 1 ; event_count += 1
        return lattice,new_site,event_count,adatom_count

    def _k(sys,time:float)->np.ndarray:
        return np.multiply(sys.A(sys.T(time)),np.exp(-sys.E_a/(k_B*sys.T(time))))
    
    ## DM funcs ##
    def _k_array(sys,time):
        return np.full((sys.lat.size,len(sys.E_a)),sys._k(time),dtype=float)

    def _DM_total_prop(sys,time:float,c_tot:np.ndarray,dummy=None)->float:
        return np.sum(np.multiply(sys._k(time),c_tot))
        
    def _DM_site_c(
            sys,
            lattice:np.ndarray,
            site:tuple,
        )->np.ndarray:
        if int(lattice[sys._get_coords(site)]) == 0:
            c = np.array([1,0,0,0,0,0],dtype=int)
        elif int(lattice[sys._get_coords(site)]) == 1:
            c = np.array([0,1,0,0,0,0],dtype=int)
            neighs = sys.neighbour_key[site,:]
            for ind,neighbour in enumerate(neighs):
                if int(lattice[sys._get_coords(neighbour)]) == 0:
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
    
    def _DM_total_c_change(sys,c_tot:np.ndarray,lat_f:np.ndarray,site:int,new_site:int):
        c_initial = np.zeros((6),dtype=int)
        c_final = c_initial.copy()
        # reconstruct initial lattice
        lat_i = lat_f.copy()
        if new_site != site:
            lat_i[sys._get_coords(site)] = 1 # retrieving these coordiantes wrong?
            lat_i[sys._get_coords(new_site)] = 0
        elif new_site == site:
            lat_i[sys._get_coords(site)] = 1 - int(lat_f[sys._get_coords(site)])
        # sweep site and neighbours
        sites_to_change = set()
        sites_to_change.add(site)
        sites_to_change.add(new_site)
        for site_n in sys.neighbour_key[site,:]: sites_to_change.add(site_n)
        if new_site != site:
            for site_n in sys.neighbour_key[new_site,:]: sites_to_change.add(site_n)
        for site_c in sites_to_change:
            c_initial += sys._DM_site_c(lat_i,site_c)
            c_final += sys._DM_site_c(lat_f,site_c)
        delta_c = np.asarray((c_final-c_initial),dtype=int)
        c_tot += delta_c
        return c_tot
    
    def _DM_gen_c_array(sys,lattice):
        c = np.empty((sys.lat.size,len(sys.E_a)),dtype=int)
        for site in range(sys.lat.size):
            c[site,:] = sys._DM_site_c(lattice,site)
        return c
    
    def _DM_get_prop_array(sys,c_array,a_array,time):
        a_array = np.multiply(c_array,sys._k_array(time))
        a_acc = np.cumsum(a_array)
        return a_array,a_acc

    ## FRM funcs ##
    def _FRM_site_prop(
            sys,
            time:float,
            rxn:int,
            lattice:np.ndarray,
            site:int,
        )->float:
        if int(lattice[sys._get_coords(site)]) == 0: # adsorption
            c = 1
        elif int(lattice[sys._get_coords(site)]) == 1:
            if rxn == 1: # desorption
                c = 1
            elif rxn > 1:
                neigh_rxn = rxn - 2
                c = 1 - int(lattice[sys._get_coords(sys.neighbour_key[site,neigh_rxn])])
        k = sys._k(time)
        return  k[rxn]*c
    
    def _FRM_generate_queue(sys):
        for site_index in range(sys.lat.size):
            if sys.lat[sys._get_coords(site_index)] == 0:
                t_k = sys._t_gen(sys._FRM_site_prop,0,sys.rng.random(),(0,sys.lat,site_index)) # adsorption
                sys._FRM_insert(t_k,(site_index,0))
            elif sys.lat[sys._get_coords(site_index)] == 1:
                t_k = sys._t_gen(sys._FRM_site_prop,0,sys.rng.random(),(1,sys.lat,site_index)) # desorption
                sys._FRM_insert(t_k,(site_index,1))
                for ind,neigh in enumerate(sys.neighbour_key[site_index,:]):
                    if sys.lat[sys._get_coords(neigh)] == 0:
                        t_k = sys._t_gen(sys._FRM_site_prop,0,sys.rng.random(),(ind+2,sys.lat,site_index))
                        sys._FRM_insert(t_k,(site_index,ind+2))
        return
    
    def _FRM_update(sys,time:float,site:int,new_site:int,lattice:np.ndarray):
        # This site
        to_remove = sys.FRM_site_keys[site].copy()
        for ID in to_remove:
            sys._FRM_remove(ID)
        
        lat_at_site = lattice[sys._get_coords(site)]
        if lat_at_site == 0:
            t_ads = sys._t_gen(sys._FRM_site_prop,time,sys.rng.random(),(0,lattice,site))
            sys._FRM_insert(t_ads,ID=(site,0)) # adsorption
        elif lat_at_site == 1:
            t_des = sys._t_gen(sys._FRM_site_prop,time,sys.rng.random(),(1,lattice,site))
            sys._FRM_insert(t_des,ID=(site,1)) # desorption
            for ind,neigh in enumerate(sys.neighbour_key[site,:]):
                if lattice[sys._get_coords(neigh)] == 0:
                    t_hop = sys._t_gen(sys._FRM_site_prop,time,sys.rng.random(),(ind+2,lattice,site))
                    sys._FRM_insert(t_hop,ID=(site,ind+2)) # hops
        # New_site
        if new_site != site:
            to_remove = sys.FRM_site_keys[new_site].copy() #[ID for ID in list(sys.FRM_id_key.keys()) if ID[0] == new_site]
            for ID in to_remove:
                sys._FRM_remove(ID)

            lat_at_new_site = lattice[sys._get_coords(new_site)]
            if lat_at_new_site == 0:
                t_ads = sys._t_gen(sys._FRM_site_prop,time,sys.rng.random(),(0,lattice,new_site))
                sys._FRM_insert(t_ads,ID=(new_site,0)) # adsorption
            elif lat_at_new_site == 1:
                t_des = sys._t_gen(sys._FRM_site_prop,time,sys.rng.random(),(1,lattice,new_site))
                sys._FRM_insert(t_des,ID=(new_site,1)) # desorption
                for ind,neigh in enumerate(sys.neighbour_key[new_site,:]):
                    if lattice[sys._get_coords(neigh)] == 0:
                        t_hop = sys._t_gen(sys._FRM_site_prop,time,sys.rng.random(),(ind+2,lattice,new_site))
                        sys._FRM_insert(t_hop,ID=(new_site,ind+2)) # hops
        # Neighbour sites, only changes hops
        direction_key = {0:1,1:0,2:3,3:2} # +x<->-x, -y<->+y hops
        for ind,neigh in enumerate(sys.neighbour_key[site,:]):
            # remove disabled neighbour events
            sys._FRM_remove((neigh,direction_key[ind]+2))
            
            # add enabled neighbour events if lattice available
            if lattice[sys._get_coords(site)] == 0 and lattice[sys._get_coords(neigh)] == 1: # target available and exists an adatom to hop
                if sys._FRM_site_prop(time,direction_key[ind]+2,lattice,neigh)>0:
                    t_hop = sys._t_gen(sys._FRM_site_prop,time,sys.rng.random(),(direction_key[ind]+2,lattice,neigh))
                    sys._FRM_insert(t_hop,ID=(neigh,direction_key[ind]+2)) # neighbours hops
                else:
                    sys.log.append(f'Null rxn: t={time} at site={site} rxn={direction_key[ind]+2}')
        if new_site != site:
            for ind,neigh in enumerate(sys.neighbour_key[new_site,:]):
                # removed disabled neighbour events
                sys._FRM_remove((neigh,direction_key[ind]+2))

                # add enabled neighbour events if lattice available
                if lattice[sys._get_coords(new_site)] == 0 and lattice[sys._get_coords(neigh)] == 1: # target available and exists an adatom to hop
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
        k = sys._k(0)
        a = np.empty((lat_initial.size,len(k)),dtype=float)
        c_tot = np.empty((len(k)),dtype=int)
        for run in range(sys.runs):
            #Initialise
            lat = lat_initial.copy()
            c_tot.fill(0)
            for site in range(lat_initial.size): c_tot += sys._DM_site_c(lat_initial,site)
            c = sys._DM_gen_c_array(lat)
            a,a_acc = sys._DM_get_prop_array(c,a,0)
            t,n,site,new_site,count,old_count,plot_ind=0.0,0,0,0,0,0,0
            adatoms = np.sum(lat)
            times = np.array([np.nan]*(sys.t_points))
            thetas,rates,temps = times.copy(),times.copy(),times.copy()
            while t<sys.t_max and n<sys.n_max:
                # Local occ change
                c = sys._DM_c_change(lat,c,site,new_site)
                # Generate next time
                new_t = sys._t_gen(sys._DM_total_prop,t,sys.rng.random(),(c_tot,None))
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
                # Advance system time
                t = new_t
                # Global prop gen
                a,a_acc = sys._DM_get_prop_array(c,a,t)
                if a_acc[-1] == 0: print('Reactions complete (total_propensity = 0)'); break
                # Choose reaction
                mu_index = np.searchsorted(a_acc,a_acc[-1]*sys.rng.random(),side='left') # binary search
                rxn_index = mu_index % len(k)
                site = mu_index//len(k)
                # Advance system state
                lat,new_site,count,adatoms = sys._rxn_step(lat,site,rxn_index,count,adatoms)
                c_tot = sys._DM_total_c_change(c_tot,lat,site,new_site)
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
    
    def run_DM_no_data(sys,return_n_steps=False):
        """Runs a kinetic Monte Carlo simulation on the defined lattice \n
        Uses the Direct method \n
        Returns a (4*runs) column dataframe of time, temp, coverage and desorption rate \n
        dataframe labels of the form (e.g. run X): \n
        'timeX','tempX','thetaX','rateX'
        """
        lat_initial = sys.lat.copy()
        k = sys._k(0)
        a = np.empty((lat_initial.size,len(k)),dtype=float)
        c_tot = np.empty((len(k)),dtype=int)
        for run in range(sys.runs):
            #Initialise
            lat = lat_initial.copy()
            c_tot.fill(0)
            for site in range(lat_initial.size): c_tot += sys._DM_site_c(lat_initial,site)
            c = sys._DM_gen_c_array(lat)
            a,a_acc = sys._DM_get_prop_array(c,a,0)
            t,n,site,new_site,count=0.0,0,0,0,0
            adatoms = np.sum(lat)
            while t<sys.t_max and n<sys.n_max:
                # Local occ change
                c = sys._DM_c_change(lat,c,site,new_site)
                # Generate next time
                new_t = sys._t_gen(sys._DM_total_prop,t,sys.rng.random(),(c_tot,None))
                # Advance system time
                t = new_t
                # Global prop gen
                a,a_acc = sys._DM_get_prop_array(c,a,t)
                if a_acc[-1] == 0: print('Reactions complete (total_propensity = 0)'); break
                # Choose reaction
                mu_index = np.searchsorted(a_acc,a_acc[-1]*sys.rng.random(),side='left') # binary search
                rxn_index = mu_index % len(k)
                site = mu_index//len(k)
                # Advance system state
                lat,new_site,count,adatoms = sys._rxn_step(lat,site,rxn_index,count,adatoms)
                c_tot = sys._DM_total_c_change(c_tot,lat,site,new_site)
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