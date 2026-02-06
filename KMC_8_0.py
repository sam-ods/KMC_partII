import numpy as np
from sortedcontainers.sortedlist import SortedList
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from scipy.special import lambertw
from scipy.constants import Boltzmann as k_B

## ## ##################### ## ##
## ## ##################### ## ##
## ## ### KMC execution ### ## ##
## ## ##################### ## ##
## ## ##################### ## ##
class KMC:
    def __init__(sys,sim_param:dict,E_a:np.ndarray,Pre_exp:np.ndarray,Temp_function:callable,J_NNs_arr:np.ndarray,w_arr:np.ndarray):
        """E_a and pre-exponetial required as specific shape (n_site_types,2+n_site_types) \n
        T-dependent pre-exponentials not supported (yet?) \n
        [ads0,des0,diff00,diff01,diff02,...] \n
        [ads1,des1,diff10,diff11,diff12,...] \n
        [ads2,des2,diff20,diff21,diff22,...] \n
        [...] \n
        Nearest neighbour (2-body) interactions should be passed as a (n_species+1,n_species+1,2) with J[:,:,0] as direct NNs and J[:,:,1] as diagonal NNs \n
        Double counting of interactions corrected internally \n
        J_ij == J_ji but will need to access for i->j and j->i interactions \n
        Note J[0,:] and J[:,0] must all be zero to represent absence of neighbour \n
        For each site type the 2d array is: \n
        [0, 0  , 0  , 0  ,...]
        [0,J_11,J_12,J_13,...] \n
        [0,J_21,J_22,J_31,...] \n
        [0,J_31,J_32,J_33,...] \n
        [...] \n 
        w should be a (n_site_types,n_rxns) array like E_a \n

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
        w_arr = np.atleast_2d(w_arr)
        n_site_types = {'ideal':1,'SAA':2,'stepped':3} # SAA
        expect_shape = (n_site_types[sys.lat_type],2+n_site_types[sys.lat_type])
        if np.shape(E_a) != expect_shape:
            raise IndexError(f'Actvation energies input wrong, should be {expect_shape} but E_a input is {np.shape(E_a)}')
        if np.shape(E_a) != np.shape(Pre_exp):
            raise IndexError(f'Pre-exp and E_a array dimensions dont match: {np.shape(Pre_exp)} and {np.shape(E_a)}')
        # build base E_a,Pre_exp array
        sys.E_a = np.empty((len(sys.lat[:,0]),6),dtype=float) # only 6 process: [ads,des,diff+x,diff-x,diff+y,diff-y]
        sys.A = np.empty((len(sys.lat[:,0]),6),dtype=float) # only 6 process: [ads,des,diff+x,diff-x,diff+y,diff-y]
        for site,site_type in enumerate(sys.lat[:,0]):
            sys.E_a[site,0:2] = E_a[site_type,0:2]
            sys.A[site,0:2] = Pre_exp[site_type,0:2]
            for neigh_ind,neigh in enumerate(sys.neighbour_key[site,:]):
                sys.E_a[site,2+neigh_ind] = E_a[site_type,2+sys.lat[neigh,0]]
                sys.A[site,2+neigh_ind] = Pre_exp[site_type,2+sys.lat[neigh,0]]
        # build NN coupling matrix and TS factor
        sys.w_BEP = w_arr
        sys.J_BEP_dir = J_NNs_arr[:,:,0]
        sys.J_BEP_diag = J_NNs_arr[:,:,1]
        # build base BEP contribution to E_a, will be updated each step
        sys.E_BEP = np.empty((len(sys.lat[:,0]),6),dtype=float)
        for site in range(len(sys.lat[:,0])):
            sys._lateral_interactions_update(sys.lat,site,site)
        print('Setup done!')

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
            elif entry[0] == '2':
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

    ########################################
    ## BEP lateral interactions functions ##
    ########################################

    def _lateral_interactions_update(sys,lattice:np.ndarray,site:int,new_site:int):
        lat_f = sys.lat.copy()
        # reconstruct initial lattice here
        lat_i = lat_f.copy()
        if new_site != site:
            lat_i[site,1] = 1
            lat_i[new_site,1] = 0
        elif new_site == site:
            lat_i[site] = 1 - int(lat_f[site,1]) # <--------- relies on binary lattice
        # update sites
        sites_to_update = sys._get_neigh_set(site)
        sites_to_update = sites_to_update[0].union(sites_to_update[1])
        sites_to_update.add(site)
        if new_site != site: sites_to_update.union(set(sys.neighbour_key[new_site,:]))
        for s in sites_to_update:
            sys.E_BEP[s,:] = sys.w_BEP[lattice[s,0],:]*(sys._lateral_int(lat_f,s)-sys._lateral_int(lat_i,s))# w is how product/reactant like TS is

    def _lateral_int(sys,lattice:np.ndarray,site:int):
        NN_dir,NN_diag = sys._get_neigh_set(site)
        F_NN_dir = np.zeros((len(sys.E_BEP[0,:])),dtype=float)
        F_NN_diag = F_NN_dir.copy()
        for s_dir,s_diag in zip(NN_dir,NN_diag):
            if lattice[s_dir,1] != 0:
                F_NN_dir += 0.5*sys.J_BEP_dir[lattice[site,1],lattice[s_dir,1]] # 0.5 since we are using the 2 body interaction energy
            if lattice[s_diag,1] != 0:
                F_NN_diag += 0.5*sys.J_BEP_diag[lattice[site,1],lattice[s_diag,1]]
        return F_NN_dir+F_NN_diag

    def _get_neigh_set(self,site:int): # Only square lattices although triangular simple to adapt
        NN_dir = set() # dont add site or new_site to avoid self interaction
        NN_diag = set()
        diag_key = {0:3,1:2,2:0,3:1} # clockwise relating each neighbours neighbour to sites diagonal neghbour
        for index,direct_neigh in enumerate(self.neighbour_key[site,:]):
            NN_dir.add(direct_neigh)
            NN_diag.add(self.neighbour_key[direct_neigh,diag_key[index]])
        return NN_dir,NN_diag

    #####################
    ### KMC functions ###
    #####################
    
    def _t_gen(self,prop_func:callable,time:float,random_number:float,other_args:tuple,**kwargs):
        """Generates a new absolute time from a random time step by solving: \n
        $int_{t}^{t+delta_t}(a0(t,other_args)) + ln(r) == 0$ \n
        Uses newton root-finding method with x0 = -ln(r)/a0(t) \n
        Uses intial guess method according to method kwarg: \n
            1. method='DM' 2. method='FRM' 3. method='TI' (default if no kwarg) \n
        If newton fails resorts to brentq method \n
        Relative tolerance: 10**-6
        """
        try:
            if kwargs['method'] == 'FRM':
                # Setup improved FRM initial guess
                rxn,_,site = other_args
                guess = time+self._FRM_improved_guess(time,random_number,self.E_a[site,rxn],self.A[site,rxn])
            elif kwargs['method'] == 'DM':
                # Setup improved FRM initial guess
                c,_ = other_args
                guess = time+self._DM_improved_guess(time,random_number,c)
            elif kwargs['method'] == 'TI':
                a0_t = prop_func(time,*other_args)
                if a0_t<=0: raise ValueError(f'Negative or zero propensity:\na0(t)={a0_t},t={time},r={random_number}\nother={other_args}') 
                guess = time-np.log(random_number)/a0_t
        except KeyError:
            # Setup intial guess (naive)
            a0_t = prop_func(time,*other_args)
            if a0_t<=0: raise ValueError(f'Negative or zero propensity:\na0(t)={a0_t},t={time},r={random_number}\nother={other_args}') 
            guess = time-np.log(random_number)/a0_t
        
        rel_tol = 10**-6
        max_tau = 5 * self.t_max
        if guess > max_tau:
            try:
                guess_method = kwargs['method']
            except KeyError:
                guess_method = 'TI'
            self.int_log.append(f'1. Guess greater than max time: Method={guess_method}')
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
    
    def _FRM_improved_guess(self,time:float,random_number:float,E_a:float,Pre_exp:float):
        """Michail's improved intial guess for Newton root-finding in FRM
        only applies to linear temperature ramps and time-independent Pre-exponential factors
        """
        sim_temp = self.T(time)
        beta = self.T(1)-self.T(0)

        C = Pre_exp * np.exp(-E_a/(k_B*sim_temp)) * (k_B*sim_temp**2)/E_a + beta*np.log(1/random_number)
        temp_guess = (E_a/k_B) / (2*lambertw(1/2*np.sqrt((Pre_exp*E_a)/(C*k_B))))
        if np.imag(temp_guess) != 0: return ValueError('Complex valued initial guess!')
        # lambertw returns complex types but k=0 branch is real valued for all z>-1/e so can safely ignore imaginary part
        return ((temp_guess - sim_temp)/(beta)).real

    def _DM_improved_guess(self,time:float,random_number:float,c_arr:np.ndarray):
        """My improved intial guess for Newton root-finding in DM
        only applies to linear temperature ramps and time-independent Pre-exponential factors
        """
        sim_temp = self.T(time)
        beta = self.T(1)-self.T(0)

        all_E_a = (self.E_a + self.E_BEP)
        nz_rs,nz_cs = np.nonzero(c_arr)
        E_a_arr = np.empty(len(nz_rs))
        Pre_exp_arr = E_a_arr.copy()
        for ind,r,c in zip(range(len(E_a_arr)),nz_rs,nz_cs):
            E_a_arr[ind] = all_E_a[r,c]
            Pre_exp_arr[ind] = self.A[r,c]
        
        arg_E_a_min = np.argmin(E_a_arr)
        A_min = Pre_exp_arr[arg_E_a_min]
        E_a_min = E_a_arr[arg_E_a_min]
        
        B_fac , H_fac = 0 , 0
        for ind,E_a,Pre_exp in zip(range(len(E_a_arr.flat)),E_a_arr,Pre_exp_arr):
            B_fac += (k_B*sim_temp**2)/E_a * Pre_exp*np.exp(-E_a/(k_B*sim_temp))
            if ind == arg_E_a_min:
                H_sep = Pre_exp*k_B*sim_temp**2/E_a * np.exp(-E_a/(k_B*sim_temp))
            else:
                H_fac += Pre_exp*k_B*sim_temp**2/E_a * np.exp(-E_a/(k_B*sim_temp))
        C = (1/(1+H_fac/H_sep)) * E_a_min/(k_B*A_min) * ( beta*np.log(1/random_number) + B_fac )
    
        temp_guess = (E_a_min/k_B) / (2*lambertw(1/2*np.sqrt((E_a_min/k_B)**2/C)))
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
        return np.multiply(sys.A[site,rxn],np.exp(-(sys.E_a[site,rxn]+sys.E_BEP[site,rxn])/(k_B*sys.T(time))))
    
    ## DM funcs ##
    def _k_array(sys,time):
        return np.multiply(sys.A,np.exp(-(sys.E_a+sys.E_BEP)/(k_B*sys.T(time))))

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
    
    def _FRM_generate_queue(sys,guess='FRM'):
        for site in range(len(sys.lat[:,0])):
            if sys.lat[site,1] == 0:
                t_k = sys._t_gen(sys._FRM_site_prop,0,sys.rng.random(),(0,sys.lat,site),method=guess) # adsorption
                sys._FRM_insert(t_k,(site,0))
            elif sys.lat[site,1] == 1:
                t_k = sys._t_gen(sys._FRM_site_prop,0,sys.rng.random(),(1,sys.lat,site),method=guess) # desorption
                sys._FRM_insert(t_k,(site,1))
                for ind,neigh in enumerate(sys.neighbour_key[site,:]):
                    if sys.lat[neigh,1] == 0:
                        t_k = sys._t_gen(sys._FRM_site_prop,0,sys.rng.random(),(ind+2,sys.lat,site),method=guess)
                        sys._FRM_insert(t_k,(site,ind+2))
        return
    
    def _FRM_update(sys,time:float,site:int,new_site:int,lattice:np.ndarray,guess='FRM'):
        # This site
        to_remove = sys.FRM_site_keys[site].copy()
        for ID in to_remove:
            sys._FRM_remove(ID)
        
        lat_at_site = lattice[site,1]
        if lat_at_site == 0:
            t_ads = sys._t_gen(sys._FRM_site_prop,time,sys.rng.random(),(0,lattice,site),method=guess)
            sys._FRM_insert(t_ads,ID=(site,0)) # adsorption
        elif lat_at_site == 1:
            t_des = sys._t_gen(sys._FRM_site_prop,time,sys.rng.random(),(1,lattice,site),method=guess)
            sys._FRM_insert(t_des,ID=(site,1)) # desorption
            for ind,neigh in enumerate(sys.neighbour_key[site,:]):
                if lattice[neigh,1] == 0:
                    t_hop = sys._t_gen(sys._FRM_site_prop,time,sys.rng.random(),(ind+2,lattice,site),method=guess)
                    sys._FRM_insert(t_hop,ID=(site,ind+2)) # hops
        # New_site
        if new_site != site:
            to_remove = sys.FRM_site_keys[new_site].copy() #[ID for ID in list(sys.FRM_id_key.keys()) if ID[0] == new_site]
            for ID in to_remove:
                sys._FRM_remove(ID)

            lat_at_new_site = lattice[new_site,1]
            if lat_at_new_site == 0:
                t_ads = sys._t_gen(sys._FRM_site_prop,time,sys.rng.random(),(0,lattice,new_site),method=guess)
                sys._FRM_insert(t_ads,ID=(new_site,0)) # adsorption
            elif lat_at_new_site == 1:
                t_des = sys._t_gen(sys._FRM_site_prop,time,sys.rng.random(),(1,lattice,new_site),method=guess)
                sys._FRM_insert(t_des,ID=(new_site,1)) # desorption
                for ind,neigh in enumerate(sys.neighbour_key[new_site,:]):
                    if lattice[neigh,1] == 0:
                        t_hop = sys._t_gen(sys._FRM_site_prop,time,sys.rng.random(),(ind+2,lattice,new_site),method=guess)
                        sys._FRM_insert(t_hop,ID=(new_site,ind+2)) # hops
        # Neighbour sites, only changes hops
        direction_key = {0:1,1:0,2:3,3:2} # +x<->-x, -y<->+y hops
        for ind,neigh in enumerate(sys.neighbour_key[site,:]):
            # remove disabled neighbour events
            sys._FRM_remove((neigh,direction_key[ind]+2))
            
            # add enabled neighbour events if lattice available
            if lattice[site,1] == 0 and lattice[neigh,1] == 1: # target available and exists an adatom to hop
                if sys._FRM_site_prop(time,direction_key[ind]+2,lattice,neigh)>0:
                    t_hop = sys._t_gen(sys._FRM_site_prop,time,sys.rng.random(),(direction_key[ind]+2,lattice,neigh),method=guess)
                    sys._FRM_insert(t_hop,ID=(neigh,direction_key[ind]+2)) # neighbours hops
                else:
                    sys.rxn_log.append(f'2. Null rxn: t={time} at site={site} rxn={direction_key[ind]+2}')
        if new_site != site:
            for ind,neigh in enumerate(sys.neighbour_key[new_site,:]):
                # removed disabled neighbour events
                sys._FRM_remove((neigh,direction_key[ind]+2))

                # add enabled neighbour events if lattice available
                if lattice[new_site,1] == 0 and lattice[neigh,1] == 1: # target available and exists an adatom to hop
                    t_hop = sys._t_gen(sys._FRM_site_prop,time,sys.rng.random(),(direction_key[ind]+2,lattice,neigh),method=guess)
                    sys._FRM_insert(t_hop,ID=(neigh,direction_key[ind]+2)) # neighbours hops
        return
    
    ######################
    ### KMC algortihms ###
    ######################

    def run_DM(sys,guess='TI',report=False):
        """Runs a kinetic Monte Carlo simulation on the defined lattice \n
        Uses the Direct method \n
        Returns a dict of time, temp, coverage and desorption rate \n
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
                # Generate next time
                new_t = sys._t_gen(sys._DM_total_prop,t,sys.rng.random(),(c,None),method=guess)
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
                # Local occ and lateral interactions change
                c = sys._DM_c_change(lat,c,site,new_site)
                sys._lateral_interactions_update(lat,site,new_site)
                n += 1
            if report: print(f'run{run}: n={n}, t={t}')
            # save run data
            run_label = [f'time{run}',f'temp{run}',f'theta{run}',f'rate{run}']
            run_data = {
                run_label[0]:times,
                run_label[1]:temps,
                run_label[2]:thetas,
                run_label[3]:rates
            }
            data = data | run_data
        return data
    
    def run_FRM(sys,guess='FRM',report=False):
        """Runs a kinetic Monte Carlo simulation on the defined lattice \n
        Uses the First reaction method \n
        Returns a dict of time, temp, coverage and desorption rate \n
        dataframe labels of the form (e.g. run X): \n
        'timeX','tempX','thetaX','rateX'
        """
        data = {}
        lat_initial = sys.lat.copy()
        for run in range(sys.runs):
            lat = lat_initial.copy()
            t,n,count,old_count,plot_ind=0.0,0,0,0,0
            adatoms = np.sum(lat[:,1])
            times = np.array([np.nan]*(sys.t_points))
            thetas,rates,temps = times.copy(),times.copy(),times.copy()
            # Initialise data structure
            sys._FRM_generate_queue(guess)
            while t<sys.t_max and n<sys.n_max:
                # Choose reaction and time
                if len(sys.FRM_sortlist)==0:print('Reactions complete (reaction queue empty)'); break
                new_t,index_tup = sys.FRM_sortlist[0]
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
                site,rxn = index_tup
                t = new_t
                # Advance state and update queue + lateral interactions
                lat,new_site,count,adatoms = sys._rxn_step(lat,site,rxn,count,adatoms)
                sys._FRM_update(t,site,new_site,lat,guess)
                sys._lateral_interactions_update(lat,site,new_site)
                n += 1
            if report: print(f'run{run}: n={n}, t={t}')
            # save run data
            run_label = [f'time{run}',f'temp{run}',f'theta{run}',f'rate{run}']
            run_data = {
                run_label[0]:times,
                run_label[1]:temps,
                run_label[2]:thetas,
                run_label[3]:rates
            }
            data = data | run_data
        return data
    
    ##########################
    ### Benchmarking funcs ###
    ##########################

    def run_DM_no_data(sys,guess='TI'):
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
                # Generate next time
                new_t = sys._t_gen(sys._DM_total_prop,t,sys.rng.random(),(c,None),method=guess)
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
                # Local occ and lateral interactions change
                c = sys._DM_c_change(lat,c,site,new_site)
                sys._lateral_interactions_update(lat,site,new_site)
                n += 1
        return
    
    def run_FRM_no_data(sys,guess='FRM'):
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
            sys._FRM_generate_queue(guess)
            while t<sys.t_max and n<sys.n_max:
                # Choose reaction and time
                if len(sys.FRM_sortlist)==0:print('Reactions complete (reaction queue empty)'); break
                new_t,index_tup = sys.FRM_sortlist[0]
                site,rxn = index_tup
                t = new_t
                # Advance state and update queue + lateral interactions
                lat,new_site,count,adatoms = sys._rxn_step(lat,site,rxn,count,adatoms)
                sys._FRM_update(t,site,new_site,lat,guess)
                sys._lateral_interactions_update(lat,site,new_site)
                n += 1
        return

    ##################
    ### Data funcs ###
    ##################
    
    def get_avg(sys,data):
        """Calculates the averages of each column in a KMC data output \n
        Make sure the data comes from this system
        """
        labels = np.empty((4,sys.runs),dtype=object)
        for i in range(sys.runs):
            labels[0,i] = f'time{i}'
            labels[1,i] = f'temp{i}'
            labels[2,i] = f'theta{i}'
            labels[3,i] = f'rate{i}'
        data['time avg'] = data[labels[0,:]].mean(axis=1)
        data['temp avg'] = data[labels[1,:]].mean(axis=1)
        data['theta avg'] = data[labels[2,:]].mean(axis=1)
        data['rate avg'] = data[labels[3,:]].mean(axis=1)
        return data
    
    def view_sq_lat(sys,site_labels=None,adatom_labels=None,lattice=None):
        """Visualise the initial lattice for the systems with sqaure neighbours
        default labels are alphabetical, uppercase for sites and lowercase for adatoms
        note view may look uneven is label names are different character lengths
        """
        if lattice != None: lat = lattice
        else: lat = sys.lat
        char_size = 1
        if site_labels == None:
            site_labels = {index:letter for index,letter in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}
        if adatom_labels == None:
            adatom_labels = {index:letter for index,letter in enumerate('_abcdefghijklmnopqrstuvwxyz')}
        char_size += max(len(adatom_labels[s]) for s in adatom_labels.keys()) + max(len(site_labels[s]) for s in site_labels.keys())
        built_lat = np.empty((sys.lat_dimensions),dtype=f'U{char_size}')
        for site,site_type in enumerate(lat[:,0]):
            built_lat[sys._get_coords(site)] = site_labels[site_type] + f'*{adatom_labels[lat[site,1]]}'
        for row in range(len(built_lat[:,0])):
            print(built_lat[row,:])

