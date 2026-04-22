import numpy as np
import time
import copy
from sortedcontainers.sortedlist import SortedList
from scipy.integrate import fixed_quad
from scipy.optimize import root_scalar
from scipy.special import lambertw
from scipy.constants import Boltzmann as k_B

## ## ##################### ## ##
## ## ##################### ## ##
## ## ### KMC execution ### ## ##
## ## ##################### ## ##
## ## ##################### ## ##
class KMC:
    def __init__(
            sys,sim_param:dict,
            E_a:np.ndarray,
            Pre_exp:np.ndarray,
            Temp_function:callable,
            J_arr:np.ndarray,
            w_arr:np.ndarray
            ):
        """KMC for a single-atom alloy TPD process"""
        out_i = r"""
______________ ___________ ___________
\              \    ___   \     __    \
 \____     _____\   \__\   \    \ \    \
      \    \     \     _____\    \ \    \
       \    \     \    \     \    \_\    |
        \    \     \    \     \         /
         \____\     \  __\     \_______/ """
        print(out_i)
        out1 = 'Time-dependent Kinetic Monte Carlo for surface catalysis'
        out2 = 'Script written by Sam Oades for MChem part II project'
        # Simulation parameters
        sys.rng = np.random.default_rng(sim_param['generator'])
        sys.t_max,sys.n_max,sys.t_step,sys.t_points = sim_param['t_max'],sim_param['n_max'],sim_param['t_step'],sim_param['t_points']
        sys.runs = sim_param['runs']
        if max(sim_param['lattice'][:,1]) > 1:
            print('Module designed for one species but lattice contains many:')
            sys.what_coverages(lattice=sim_param['lattice'][:,1])
            raise ValueError('Species beyond index 1 exist, will result in KeyErrors')
        else:
            sys.lat = sim_param['lattice']
        # Get lattice info
        sys.lat_type,sys.sys_type,sys.lat_dimensions = sim_param['lattice_info']
        sys.neighbour_key = sim_param['neighbours']
        # Error Logging
        sys.rxn_log = []
        # Check correct lattice passed
        if not (sys.lat_type.lower() == 'triangular' and sys.sys_type.upper() == 'SAA'): raise AttributeError('This module is built for a TPD simulation on a PtCu (111) SAA. Make sure the correct lattice setup supplied')
        out3 = f'Intialising {sys.lat_type} lattice system on a {sys.lat_dimensions[0]}x{sys.lat_dimensions[1]} supercell'
        # Numerical method parameters
        sys.order_guass = 5 # Default in scipy = 5
        sys.rel_tol,sys.abs_tol = 1e-6,1e-9
        sys.brentq_bracket_max = 60
        sys.switch_lim = 5
        # sys dimensions
        sys.n_sites = len(sys.lat[:,0])
        sys.n_neighs = len(sys.neighbour_key[0,:])
        num_site_types = 3
        # process counter
        sys.counter = np.zeros((num_site_types,3),dtype=int)
        sys.n_sites_per_type = np.zeros((num_site_types),dtype=int)
        ##
        ## Kinetic parameters
        ##
        sys.species_rxns = {
            0:{0},
            1:{1,2,3,4,5,6,7}
        }
        sys.T = Temp_function
        # check array dimensions
        expect_shape = (num_site_types,2+num_site_types)
        if np.shape(E_a) != expect_shape:
            raise IndexError(f'Actvation energies input wrong, should be {expect_shape} but E_a input is {np.shape(E_a)}')
        if np.shape(E_a) != np.shape(Pre_exp):
            raise IndexError(f'Pre-exp and E_a array dimensions dont match: {np.shape(Pre_exp)} and {np.shape(E_a)}')
        if np.shape(J_arr) != (num_site_types,num_site_types,2,2):
            raise IndexError('J_arr should be (n_site_types,n_species+1,n_species+1)')
        if np.shape(w_arr) != expect_shape:
            raise IndexError('w_arr should be (n_site_types,n_processes)')
        ##
        ## build arrays
        ##
        # site independent arrays
        sys.w_BEP = np.empty((num_site_types,num_site_types,2+sys.n_neighs),dtype=float)
        # w array
        for site_type1 in range(num_site_types):
            # site == new_site rxns
            sys.w_BEP[0:3,site_type1,0:2] = w_arr[0:3,0:2]
            for site_type2 in range(num_site_types):
                # site != new_site rxns
                sys.w_BEP[site_type1,site_type2,2:2+sys.n_neighs] = [w_arr[site_type1,2+site_type2]]*sys.n_neighs
        # site specific arrays
        sys.E_a = np.empty((sys.n_sites,2+sys.n_neighs),dtype=float)
        sys.A = np.empty((sys.n_sites,2+sys.n_neighs),dtype=float)
        sys.J_BEP = J_arr
        sys.E_BEP = np.empty((sys.n_sites,2+sys.n_neighs),dtype=float)
        sys.n_proc = len(sys.E_a[0,:])
        for site in range(sys.n_sites):
            # Counters
            sys.counter[sys.lat[site,0],0] += sys.lat[site,1]
            sys.n_sites_per_type[sys.lat[site,0]] += 1
            # Base A and E_a arrays
            sys.E_a[site,0:2] = E_a[sys.lat[site,0],0:2]
            sys.A[site,0:2] = Pre_exp[sys.lat[site,0],0:2]
            for neigh_ind,neigh in enumerate(sys.neighbour_key[site,:]):
                rxn = 2+sys.lat[neigh,0] # row = initial species and col = final species
                sys.E_a[site,2+neigh_ind] = E_a[sys.lat[site,0],rxn]
                sys.A[site,2+neigh_ind] = Pre_exp[sys.lat[site,0],rxn]
            # lateral interactions
            sys.E_BEP = sys._lateral_interactions_update(sys.E_BEP,sys.lat,site,site)
        out4 = f'Kinetic parameters saved in {np.shape(sys.E_a)[0]}x{np.shape(sys.E_a)[1]} array'
        # fancy message
        length = max([len(out1)+4,len(out2)+4,len(out3)+4,len(out4)+4])
        space1,remain1,space2,remain2 = (length - len(out1)+4)//2, (length - len(out1)+4)%2, (length - len(out2)+4)//2, (length - len(out2)+4)%2
        space3,remain3,space4,remain4 = (length - len(out3)+4)//2, (length - len(out3)+4)%2, (length - len(out4)+4)//2, (length - len(out4)+4)%2
        breaks = max([len(out1)+4+2*space1+remain1,len(out2)+4+2*space2+remain2,len(out3)+4+2*space3+remain3,len(out4)+4+2*space4+remain4])
        print('-'*breaks+'\n| '+space1*' '+out1+space1*' '+remain1*' '+' |\n| '+space2*' '+out2+space2*' '+remain2*' '+' |')
        print('-'*breaks+'\n| '+space3*' '+out3+space3*' '+remain3*' '+' |\n| '+space4*' '+out4+space4*' '+remain4*' '+' |')
        print('-'*breaks)

    ################################
    ##  System specific functions ##
    ################################

    ##
    ## make sure empty sets passed as set() since {} is a dict !
    ##

    def _get_rxn_key(sys,site):
        ## This seems to complex to make adaptable, going to hardcode PtCu SAA C-H activation
        neighs = sys.neighbour_key[site,:]
        rxn_key = {
            0 : (1,0,site), # adsorption
            1 : (0,0,site) # desorption
        }
        neigh_rxns = {(i+2) : (0,1,neighs[i]) for i in range(sys.n_neighs)} # hops
        rxn_key.update(neigh_rxns)
        return rxn_key

    def _get_dependency_key(sys,site):
        neighs = sys.neighbour_key[site,:]
        # rxn -> tuple( new_site , set(required neighbours) )
        dependency_key = { 
            0:(site,set()),
            1:(site,set())
        }
        neigh_deps = {(i+2) : (neighs[i],{0}) for i in range(sys.n_neighs)}
        dependency_key.update(neigh_deps)
        return dependency_key

    def _rxn_step(sys,lattice:np.ndarray,site:int,rxn_ind:int,counts:np.ndarray)->tuple[np.ndarray,int,int,np.ndarray]:
        """Updates the lattice according to the chosen reaction \n
        returns the updated lattice
        """
        rxn_key = sys._get_rxn_key(site)
        species,new_species,new_site = rxn_key[rxn_ind]
        lattice[site,1] = species
        if new_site != site:
            lattice[new_site,1] = new_species
        if type(counts) == np.ndarray:
            if rxn_ind == 0: # ads
                counts[lattice[site,0],0] += 1 
            elif rxn_ind == 1:# des
                counts[lattice[site,0],0] -= 1
                counts[lattice[site,0],1] += 1 
            else: # diff
                counts[lattice[site,0],0] -= 1
                counts[lattice[new_site,0],0] += 1 
        return lattice,new_site,counts

    ########################################
    ## BEP lateral interactions functions ##
    ########################################

    def _lateral_interactions_update(sys,E_BEP:np.ndarray,lattice:np.ndarray,site:int,new_site:int):
        # update sites
        sites_to_update = sys._get_neigh_set(site)
        sites_to_update.add(site)
        if new_site != site: sites_to_update.union(set(sys.neighbour_key[new_site,:]))
        for s in sites_to_update:
            E_BEP[s,:] = 0.0
            lat_int_i = sys._lateral_int(lattice,s)
            for rxn in sys.species_rxns[lattice[s,1]]:
                lat_f,s_f,_ = sys._rxn_step(lattice.copy(),s,rxn,None)
                E_BEP[s,rxn] = sys.w_BEP[lattice[s,0],lattice[s_f,0],rxn]*(sys._lateral_int(lat_f,s)-lat_int_i)
        return E_BEP

    def _lateral_int(sys,lattice:np.ndarray,site:int):
        NN = sys._get_neigh_set(site)
        F_NN = 0.0
        for s in NN:
            F_NN += 0.5*sys.J_BEP[lattice[site,0],lattice[s,0],lattice[site,1],lattice[s,1]] # 0.5 since we are using the 2 body interaction energy
        return F_NN

    #####################################
    ### FRM data structure definition ###
    #####################################

    def _FRM_insert(tree,ID:tuple,sortlist:SortedList,site_keys:dict):
        if not np.isfinite(ID[0]):
            tree.rxn_log.append(f'1. Null rxn: ID={ID}')
        else:
            sortlist.add(ID)
            site_keys[ID[1]].append(ID)
        return sortlist,site_keys
    
    def _FRM_remove(tree,ID:tuple,sortlist:SortedList,site_keys:dict):
        sortlist.remove(ID)
        site_keys[ID[1]].remove(ID)
        return sortlist,site_keys

    #####################
    ### KMC functions ###
    #####################
    
    def _t_gen(self,prop_func:callable,time:float,random_number:float,other_args:tuple,method:str='TI'):
        """Generates a new absolute time from a random time step by solving: \n
        $int_{t}^{t+delta_t}(a0(t,other_args)) + ln(r) == 0$ \n
        Uses newton root-finding method with x0 = -ln(r)/a0(t) \n
        Uses intial guess method according to method kwarg: \n
            1. method='DM' 2. method='FRM' 3. method='TI' (default if no kwarg) \n
        If newton fails resorts to brentq method \n
        Relative tolerance: 10**-6
        """
        order_guass = int(self.order_guass)
        rel_tol,self.abs_tol = self.rel_tol,self.abs_tol

        if method == 'FRM':
            # Setup improved FRM initial guess
            rxn,_,site,E_BEP = other_args
            guess = time+self._FRM_improved_guess(time,random_number,(self.E_a[site,rxn]+E_BEP[site,rxn]),self.A[site,rxn])
        elif method == 'DM':
            # Setup improved FRM initial guess
            c,E_BEP = other_args
            guess = time+self._DM_improved_guess(time,random_number,c,E_BEP)
        elif method == 'TI':
            a0_t = prop_func(time,*other_args)
            if a0_t<=0: raise ValueError(f'Negative or zero propensity:\na0(t)={a0_t},t={time},r={random_number}\nother={other_args}') 
            guess = time-np.log(random_number)/a0_t
        else:
            raise ValueError('Unrecognised guess type:',method)
        if guess == None:
            # Setup intial guess (naive)
            a0_t = prop_func(time,*other_args)
            if a0_t<=0: raise ValueError(f'Negative or zero propensity:\na0(t)={a0_t},t={time},r={random_number}\nother={other_args}') 
            guess = time-np.log(random_number)/a0_t
        
        max_tau = 10**2 * self.t_max
        if guess > max_tau: return np.inf # is this needed?
        # Define functions
        def f(new_time:float):
            int_sol,_ = fixed_quad(prop_func,time,new_time,args=other_args,n=order_guass)
            return float(int_sol + np.log(random_number))
        def fprime(new_time:float):
            return float(prop_func(new_time,*other_args))
        # Newton method
        x0 = max(guess,10**-12)
        sol = root_scalar(f,method='newton',x0=x0,fprime=fprime,rtol=rel_tol,xtol=self.abs_tol)
        if sol.converged: return sol.root
        print('need brentq:\n',sol)
        if sol.root > max_tau: print('inf step'); return np.inf
        # If Newton method fails use brentq backup
        tau_lo = 0 if random_number > 0 else -10**-12  # can make this more refined? or keep robust?
        tau_hi = min(guess,max_tau)
        if f(tau_hi)<0:
            loop_count = 0
            while f(tau_hi)<0:
                tau_hi*=2
                loop_count += 1
                if loop_count>self.brentq_bracket_max: return np.inf
        sol = root_scalar(f,method='brentq',bracket=[tau_lo,tau_hi],rtol=rel_tol)
        if not sol.converged: raise RuntimeError(f'Both root finding methods failed, check prop_func behaviour (t={time})')
        return sol.root # absolute time of next reaction
    
    def _check_allowed(sys,site:int,rxn:int,lattice:np.ndarray):
        """Checks if a site has a predefined neighbour arrangement
        i.e. if a reaction is possible"""
        # rxn -> tuple( new_site , set(required neighbours) )
        dependency_key = sys._get_dependency_key(site)
        neigh_species = set()
        new_site,req_neighs = dependency_key[rxn]
        if new_site != site: # checks a specific direction
            neigh_species.add(lattice[new_site,1])
            if req_neighs.issubset(neigh_species):
                return True
            else:
                return False
        else: # checks all neighbours
            for s_n in sys.neighbour_key[site,:]: neigh_species.add(lattice[s_n,1])
            if req_neighs.issubset(neigh_species):
                return True
            else:
                return False

    ## DM funcs ##
    def _DM_total_prop(sys,time,c_arr:np.ndarray,E_BEP:np.ndarray)->float:
        Am,Eam,Ebm = sys.A[c_arr],sys.E_a[c_arr],E_BEP[c_arr]
        tmp = np.empty(np.shape(Eam),dtype=np.float64)
        np.add(Eam,Ebm,out=tmp)
        if np.asarray(time).ndim == 0:
            np.exp((-tmp/(k_B*sys.T(time))),out=tmp)
            np.multiply(Am,tmp,out=tmp)
            return np.sum(tmp,axis=0)
        else:
            ans = np.empty((len(time),len(Am)),dtype=np.float64)
            np.exp((-tmp[None,:]/(k_B*sys.T(time)[:,None])),out=ans)
            np.multiply(Am[None,:],ans,out=ans)
            return  np.sum(ans,axis=1)
        
    def _DM_get_prop_array(sys,c_array:np.ndarray,E_BEP:np.ndarray,time:float):
        Am,Eam,Ebm = sys.A[c_array],sys.E_a[c_array],E_BEP[c_array]
        tmp = np.empty(np.shape(Eam),dtype=np.float64)
        np.add(Eam,Ebm,out=tmp)
        np.exp((-tmp/(k_B*sys.T(time))),out=tmp)
        np.multiply(Am,tmp,out=tmp)
        return np.cumsum(tmp)

    def _DM_site_c(
            sys,
            lattice:np.ndarray,
            site:tuple,
        )->np.ndarray:
        c = np.zeros(sys.n_proc,dtype=bool)
        for rxn in sys.species_rxns[lattice[site,1]]:
            c[rxn] = sys._check_allowed(site,rxn,lattice)
        return c
        
    def _DM_c_change(
            sys,
            lattice:np.ndarray,
            c_array:np.ndarray,
            c_count:int,
            site:int,
            new_site:int
        ):
        c_i,c_f = 0,0
        to_update = set(sys.neighbour_key[site,:])
        to_update.add(site)
        if new_site != site: to_update.update(sys.neighbour_key[new_site,:])
        for s in to_update:
            c_i += np.sum(np.asarray(c_array[s,:],dtype=int))
            c_array[s,:] = sys._DM_site_c(lattice,s)
            c_f += np.sum(np.asarray(c_array[s,:],dtype=int))
        c_count += (c_f-c_i)
        return c_array,c_count
    
    def _DM_gen_c_array(sys,lattice):
        c = np.empty((np.shape(sys.E_a)),dtype=bool)
        for site in range(sys.n_sites):
            c[site,:] = sys._DM_site_c(lattice,site)
        c_count = np.sum(np.asarray(c,dtype=int))
        return c,c_count
    
    def _DM_improved_guess(self,time:float,random_number:float,c_arr:np.ndarray,E_BEP:np.ndarray):# speed this up using matmul opts
        """My improved intial guess for Newton root-finding in DM
        only applies to linear temperature ramps and time-independent Pre-exponential factors
        """
        sim_temp = self.T(time)
        beta = self.T(1)-self.T(0)

        Am,Eam,Ebm = self.A[c_arr],self.E_a[c_arr],E_BEP[c_arr]
        tmp1 = np.empty(np.shape(Am),dtype=np.float64)
        tmp2 = tmp1.copy()
        np.add(Eam,Ebm,out=tmp1) # tmp1 is E_acts

        arg_min = np.argmin(tmp1)
        Ea_min,A_min = tmp1[arg_min],Am[arg_min]
        Bmin_cur = A_min*k_B*sim_temp**2/Ea_min * np.exp(-Ea_min/(k_B*sim_temp))

        np.divide(Am,tmp1,out=tmp2) # tmp2 is Pre_exps/E_acts
        np.exp(-tmp1/(k_B*sim_temp),out=tmp1) # tmp1 is Boltzmann factors of E_acts
        np.multiply(tmp2*k_B*sim_temp**2,tmp1,out=tmp1) # tmp1 is B factor array
        Btot_cur = np.sum(tmp1)

        C = (Ea_min*Bmin_cur)/(A_min*k_B) * (1+np.log(1/random_number)*beta/Btot_cur)
        temp_guess = (Ea_min/k_B) / (2*lambertw(1/2*np.sqrt((Ea_min/k_B)**2/C)))
        if temp_guess.imag != 0: raise ValueError('Complex valued initial guess!')
        # lambertw returns complex types but k=0 branch is real valued for all z>-1/e so can safely ignore imaginary part
        return ((temp_guess - sim_temp)/(beta)).real

    ## FRM funcs ##
    def _k(sys,site:int,rxn:int,time:float,E_BEP:np.ndarray)->np.ndarray:
        return sys.A[site,rxn]*np.exp(-(sys.E_a[site,rxn]+E_BEP[site,rxn])/(k_B*sys.T(time)))
    
    def _FRM_generate_queue(sys,lattice:np.ndarray,E_BEP:np.ndarray,guess_method:str):
        sortlist = SortedList(key=lambda tup:tup[0]) # sort list based on first tuple entry (time)
        site_keys = {site:[] for site in range(sys.n_sites)} # site -> IDs
        for site in range(sys.n_sites):
            for rxn in sys.species_rxns[lattice[site,1]]:
                if sys._check_allowed(site,rxn,lattice):
                    rxn_time = sys._t_gen(sys._FRM_site_prop,0,sys.rng.random(),(rxn,lattice,site,E_BEP),method=guess_method)
                    sortlist,site_keys = sys._FRM_insert((rxn_time,site,rxn),sortlist,site_keys)
        return sortlist,site_keys

    def _FRM_site_prop( # adapt for many species
            sys,
            time:float,
            rxn:int,
            lattice:np.ndarray,
            site:int,
            E_BEP:np.ndarray
        )->float:
        c = sys._check_allowed(site,rxn,lattice)
        k = sys._k(site,rxn,time,E_BEP)
        return k*c
    
    def _FRM_update(sys,sortlist:SortedList,site_keys:dict,time:float,site:int,new_site:int,lattice:np.ndarray,E_BEP:np.ndarray,guess_method:str):
        # update lateral interactions
        E_BEP = sys._lateral_interactions_update(E_BEP,lattice,site,new_site)
        # site to update
        to_update = sys._get_neigh_set(site)
        to_update.add(site)
        if new_site != site: to_update.update(sys._get_neigh_set(new_site))
        for s in to_update:
            # remove old reactions
            to_remove = site_keys[s].copy()
            for ID in to_remove:
                sortlist,site_keys = sys._FRM_remove(ID,sortlist,site_keys)
            # add new reactions
            for rxn in sys.species_rxns[lattice[s,1]]:
                if sys._check_allowed(s,rxn,lattice):
                    rxn_time = sys._t_gen(sys._FRM_site_prop,time,sys.rng.random(),(rxn,lattice,s,E_BEP),method=guess_method)
                    sortlist,site_keys = sys._FRM_insert((rxn_time,s,rxn),sortlist,site_keys)
        return sortlist,site_keys,E_BEP

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

    ######################
    ### KMC algortihms ###
    ######################

    def run_DM(sys,guess:str='TI',report:bool=False):
        """Runs a kinetic Monte Carlo simulation on the defined lattice \n
        Uses the Direct method \n
        Returns a dict of time, temp, coverage and desorption rate \n
        dataframe labels of the form (e.g. run X): \n
        'timeX','tempX','thetaX','rateX'
        """
        print(f'Starting DM with {guess} guess scheme for {sys.runs} runs ...')
        data = {}
        # Guess switching
        switch = False
        if guess == 'switch':
            print('Guess swicth-scheme ON')
            guess,switch,switch_limit = 'DM',True,sys.switch_lim
        for run in range(sys.runs):
            #Initialise
            lat = sys.lat.copy()
            E_BEP = sys.E_BEP.copy()
            c,c_count = sys._DM_gen_c_array(lat)
            t,n,site,new_site,plot_ind=0.0,0,0,0,0
            count = sys.counter.copy()
            times = np.full((sys.t_points),fill_value=np.nan)
            thetas = np.full((3,sys.t_points),fill_value=np.nan)
            temps,rates,pops = times.copy(),thetas.copy(),thetas.copy()
            while t<sys.t_max and n<sys.n_max:
                if switch and n == switch_limit: guess = 'TI' # Swicth guess type after i-th step
                if c_count == 0: print('Reactions complete (c array empty)'); break
                # Generate next time
                new_t = sys._t_gen(sys._DM_total_prop,t,sys.rng.random(),other_args=(c,E_BEP),method=guess)
                # Save state
                count,plot_ind,times,temps,pops,thetas,rates = sys._save_state(t,new_t,count,plot_ind,times,temps,pops,thetas,rates)
                # Advance system time
                t = new_t
                # Global prop gen
                a_acc = sys._DM_get_prop_array(c,E_BEP,t)
                # Choose reaction
                mu_index = np.searchsorted(a_acc,a_acc[-1]*sys.rng.random(),side='left') # binary search
                active_rxns = np.nonzero(c)
                site = active_rxns[0][mu_index]
                rxn_index = active_rxns[1][mu_index]
                # Advance system state
                lat,new_site,count = sys._rxn_step(lat,site,rxn_index,count)
                # Local occ and lateral interactions change
                c,c_count = sys._DM_c_change(lat,c,c_count,site,new_site)
                E_BEP = sys._lateral_interactions_update(E_BEP,lat,site,new_site)
                n += 1
            if report: print(f'run{run}: n={n}, t={t}')
            if switch: guess = 'DM' # switch back to improved guess for next run
            # save run data
            run_label = [f'time{run}',f'temp{run}',f'theta{run}',f'rate{run}',f'pops{run}']
            run_data = {
                run_label[0]:times,
                run_label[1]:temps,
                run_label[2]:thetas,
                run_label[3]:rates,
                run_label[4]:pops
            }
            data.update(run_data)
        print('DM runs complete')
        return data
    
    def run_FRM(sys,guess:str='FRM',report:bool=False):
        """Runs a kinetic Monte Carlo simulation on the defined lattice \n
        Uses the First reaction method \n
        Returns a dict of time, temp, coverage and desorption rate \n
        dataframe labels of the form (e.g. run X): \n
        'timeX','tempX','thetaX','rateX'
        """
        print(f'Starting FRM with {guess} guess scheme for {sys.runs} runs ...')
        data = {}
        for run in range(sys.runs):
            lat = sys.lat.copy()
            E_BEP = sys.E_BEP.copy()
            t,n,plot_ind=0.0,0,0
            count = sys.counter.copy()
            times = np.full((sys.t_points),fill_value=np.nan)
            thetas = np.full((3,sys.t_points),fill_value=np.nan)
            temps,rates,pops = times.copy(),thetas.copy(),thetas.copy()
            # Initialise data structure
            queue,queue_IDs = sys._FRM_generate_queue(lat,E_BEP,guess)
            while t<sys.t_max and n<sys.n_max:
                # Choose reaction and time
                if len(queue)==0:print('Reactions complete (reaction queue empty)'); break
                new_t,site,rxn = queue[0]
                # Save state
                count,plot_ind,times,temps,pops,thetas,rates = sys._save_state(t,new_t,count,plot_ind,times,temps,pops,thetas,rates)
                # Advance state and update queue + lateral interactions
                t = new_t
                lat,new_site,count = sys._rxn_step(lat,site,rxn,count)
                queue,queue_IDs,E_BEP = sys._FRM_update(queue,queue_IDs,t,site,new_site,lat,E_BEP,guess)
                n += 1
            if report: print(f'run{run}: n={n}, t={t}')
            # save run data
            run_label = [f'time{run}',f'temp{run}',f'theta{run}',f'rate{run}',f'pops{run}']
            run_data = {
                run_label[0]:times,
                run_label[1]:temps,
                run_label[2]:thetas,
                run_label[3]:rates,
                run_label[4]:pops
            }
            data.update(run_data)
        print('FRM runs complete')
        return data
    
    ##########################
    ### Benchmarking funcs ###
    ##########################

    def run_DM_benchmark(sys,guess:str='TI'):
        """Runs a kinetic Monte Carlo simulation on the defined lattice \n
        Uses the Direct method \n
        Returns a (4*runs) column dataframe of time, temp, coverage and desorption rate \n
        dataframe labels of the form (e.g. run X): \n
        'timeX','tempX','thetaX','rateX'
        """
        print(f'Starting DM with {guess} guess scheme for {sys.runs} runs ...')
        print('Note: this is for benchmarking - I\'m not saving any data!')
        guess_scheme = guess
        bench_CPU,bench_wall,n_steps = [],[],[]
        # Guess switching
        switch = False
        if guess == 'switch':
            print('Guess swicth-scheme ON')
            guess,switch,switch_limit = 'DM',True,sys.switch_lim
        for run in range(sys.runs):
            #Initialise
            lat = sys.lat.copy()
            E_BEP = sys.E_BEP.copy()
            c,c_count = sys._DM_gen_c_array(lat)
            t,n=0.0,0
            counts = sys.counter.copy()
            s_CPU = time.process_time()
            s_wall = time.time()
            while t<sys.t_max and n<sys.n_max:
                if switch and n == switch_limit: guess = 'TI' # Swicth guess type after i-th step
                if c_count == 0: print('Reactions complete (c array empty)'); break
                # Generate next time
                new_t = sys._t_gen(sys._DM_total_prop,t,sys.rng.random(),other_args=(c,E_BEP),method=guess)
                # Advance system time
                t = new_t
                # Global prop gen
                a_acc = sys._DM_get_prop_array(c,E_BEP,t)
                # Choose reaction
                mu_index = np.searchsorted(a_acc,a_acc[-1]*sys.rng.random(),side='left') # binary search
                active_rxns = np.nonzero(c)
                site = active_rxns[0][mu_index]
                rxn_index = active_rxns[1][mu_index]
                # Advance system state
                lat,new_site,counts = sys._rxn_step(lat,site,rxn_index,counts)
                # Local occ and lateral interactions change
                c,c_count = sys._DM_c_change(lat,c,c_count,site,new_site)
                E_BEP = sys._lateral_interactions_update(E_BEP,lat,site,new_site)
                n += 1
            if switch: guess = 'DM' # switch back to improved guess for next run
            e_CPU = time.process_time()
            e_wall = time.time()
            bench_CPU.append(e_CPU-s_CPU)
            bench_wall.append(e_wall-s_wall)
            n_steps.append(n)
        print('DM runs complete')
        return {'CPU':bench_CPU,'wall':bench_wall,'steps':n_steps,'guess':guess_scheme}
    
    def run_FRM_benchmark(sys,guess:str='FRM'):
        """Runs a kinetic Monte Carlo simulation on the defined lattice \n
        Uses the First reaction method \n
        Returns a 4*runs column dataframe of time, temp, coverage and desorption rate \n
        dataframe labels of the form (e.g. run X): \n
        'timeX','tempX','thetaX','rateX'
        """
        print(f'Starting FRM with {guess} guess scheme for {sys.runs} runs ...')
        print('Note: this is for benchmarking - I\'m not saving any data!')
        bench_CPU,bench_wall,n_steps = [],[],[]
        for run in range(sys.runs):
            lat = sys.lat.copy()
            E_BEP = sys.E_BEP.copy()
            t,n=0.0,0
            counts = sys.counter.copy()
            # Initialise data structure
            queue,queue_IDs = sys._FRM_generate_queue(lat,E_BEP,guess)
            s_CPU = time.process_time()
            s_wall = time.time()
            while t<sys.t_max and n<sys.n_max:
                # Choose reaction and time
                if len(queue)==0:print('Reactions complete (reaction queue empty)'); break
                new_t,site,rxn = queue[0]
                t = new_t
                # Advance state and update queue + lateral interactions
                lat,new_site,counts = sys._rxn_step(lat,site,rxn,counts)
                queue,queue_IDs,E_BEP = sys._FRM_update(queue,queue_IDs,t,site,new_site,lat,E_BEP,guess)
                n += 1
            e_CPU = time.process_time()
            e_wall = time.time()
            bench_CPU.append(e_CPU-s_CPU)
            bench_wall.append(e_wall-s_wall)
            n_steps.append(n)
        print('FRM runs complete')
        return {'CPU':bench_CPU,'wall':bench_wall,'steps':n_steps,'guess':guess}
    
    #################
    ## Single loop ##
    #################

    def SL_DM(sys,lat:np.ndarray,E_BEP:np.ndarray,guess:str,n_reps:int=100):
        """Single loop benchmark\n
        Output in ns"""
        c,c_count = sys._DM_gen_c_array(lat)
        counts = sys.counter.copy()
        s_WALL = time.perf_counter_ns()
        s_CPU = time.process_time_ns()
        for i in range(n_reps):
            if c_count == 0: return np.nan
            new_t = sys._t_gen(sys._DM_total_prop,0,sys.rng.random(),other_args=(c,E_BEP),method=guess)
            a_acc = sys._DM_get_prop_array(c,E_BEP,new_t)
            mu_index = np.searchsorted(a_acc,a_acc[-1]*sys.rng.random(),side='left')
            active_rxns = np.nonzero(c)
            site = active_rxns[0][mu_index]
            rxn_index = active_rxns[1][mu_index]
            new_lat,new_site,counts = sys._rxn_step(lat.copy(),site,rxn_index,counts)
            _ = sys._DM_c_change(new_lat,c.copy(),c_count.copy(),site,new_site)
            _ = sys._lateral_interactions_update(E_BEP.copy(),new_lat,site,new_site)
        e_CPU = time.process_time_ns()
        e_WALL = time.perf_counter_ns()
        return {'CPU':(e_CPU-s_CPU)/n_reps, 'wall':(e_WALL-s_WALL)/n_reps,'runs':n_reps}
    
    def SL_FRM(sys,lat:np.ndarray,E_BEP:np.ndarray,guess:str,n_reps:int=100):
        """Single loop benchmark\n
        Output in ns"""
        counts = sys.counter.copy()
        queue,queue_IDs = sys._FRM_generate_queue(lat,E_BEP,guess)
        print(f'FRM-{guess} system initialised with {len(queue)} reactions')
        s_WALL = time.time_ns()
        s_CPU = time.process_time_ns()
        for i in range(n_reps):
            if len(queue)==0:print('Reactions complete (reaction queue empty)'); return np.nan
            new_t,site,rxn = queue[0]
            t = new_t
            new_lat,new_site,counts = sys._rxn_step(lat.copy(),site,rxn,counts)
            _ = sys._FRM_update(queue.copy(),copy.deepcopy(queue_IDs),t,site,new_site,new_lat,E_BEP.copy(),guess)
        e_CPU = time.process_time_ns()
        e_WALL = time.time_ns()
        return {'CPU':(e_CPU-s_CPU)/n_reps, 'wall':(e_WALL-s_WALL)/n_reps,'runs':n_reps}
    
    ######################################################
    ## BEP lateral interactions functions up to 2nd NNs ##
    ######################################################

    def _FRM_update_2NNs(sys,sortlist:SortedList,site_keys:dict,time:float,site:int,new_site:int,E_BEP:np.ndarray,lattice:np.ndarray,guess_method:str):
        # update lateral interactions
        E_BEP = sys._lateral_interactions_update_2NNs(E_BEP,lattice,site,new_site)
        # site to update
        to_update = set(sys.neighbour_key[site,:])
        to_update.update(sys.NNs2nd(site))
        to_update.add(site)
        if new_site != site:
            to_update.update(sys.NNs2nd(new_site))
        for s in to_update:
            # remove old reactions
            to_remove = site_keys[s].copy()
            for ID in to_remove:
                sortlist,site_keys = sys._FRM_remove(ID,sortlist,site_keys)
            # add new reactions
            for rxn in sys.species_rxns[lattice[s,1]]:
                if sys._check_allowed(s,rxn,lattice):
                    rxn_time = sys._t_gen(sys._FRM_site_prop,time,sys.rng.random(),(rxn,lattice,s,E_BEP),method=guess_method)
                    sortlist,site_keys = sys._FRM_insert((rxn_time,s,rxn),sortlist,site_keys)
        return sortlist,site_keys,E_BEP
    
    def _lateral_interactions_update_2NNs(sys,E_BEP:np.ndarray,lattice:np.ndarray,site:int,new_site:int):
        # update sites
        to_update = set(sys.neighbour_key[site,:])
        to_update.update(sys.NNs2nd(site))
        to_update.add(site)
        if new_site != site:
            to_update.update(sys.NNs2nd(new_site))
        for s in to_update:
            E_BEP[s,:] = 0.0
            lat_int_i = sys._lateral_int_2NNS(lattice,s)
            for rxn in sys.species_rxns[lattice[s,1]]:
                if sys._check_allowed(s,rxn,lattice): # only updated allowed reactions?
                    lat_f,s_f,_ = sys._rxn_step(lattice.copy(),s,rxn,None)
                    E_BEP[s,rxn] = sys.w_BEP[lattice[s,0],lattice[s_f,0],rxn]*(sys._lateral_int_2NNS(lat_f,s)-lat_int_i)
        return E_BEP

    def _lateral_int_2NNS(sys,lattice:np.ndarray,site:int):
        NN = set(sys.neighbour_key[site,:])
        NNs2 = sys.NNs2nd(site)
        F_NN = 0.0
        # 0.5 since we are using the 2 body interaction energy
        for s in NN: # 1st NN interactions
            F_NN += 0.5*sys.J_BEP[lattice[site,0],lattice[s,0],lattice[site,1],lattice[s,1]]
        for s in NNs2: # 2nd NN interactions
            F_NN += 0.5*sys.J_BEP2[lattice[site,0],lattice[s,0],lattice[site,1],lattice[s,1]]
        return float(F_NN)

    def run_DM_2NNs(sys,J_2NNs:np.ndarray,guess:str='TI',report:bool=False):
        """Runs a kinetic Monte Carlo simulation on the defined lattice \n
        Uses the Direct method \n
        Returns a dict of time, temp, coverage and desorption rate \n
        dataframe labels of the form (e.g. run X): \n
        'timeX','tempX','thetaX','rateX'
        """
        print(f'Starting DM with {guess} guess scheme for {sys.runs} runs ...')
        print('Note: this is for benchmarking - I\'m not saving any data!')
        bench_CPU,bench_wall,n_steps = [],[],[]
        data = {}
        if np.shape(J_2NNs) != np.shape(sys.J_BEP):
            raise ValueError('Wrong shape of 2nd NN lateral interactions, make sure this matches the 1st NNs array!')
        else:
            sys.J_BEP2 = J_2NNs
        E_BEP_initial = np.empty((np.shape(sys.E_BEP)),dtype=float)
        for site in range(sys.n_sites):
            E_BEP_initial = sys._lateral_interactions_update_2NNs(E_BEP_initial,sys.lat,site,site)
        # Guess switching
        switch = False
        if guess == 'switch':
            print('Guess swicth-scheme ON')
            guess,switch,switch_limit = 'DM',True,sys.switch_lim
        for run in range(sys.runs):
            #Initialise
            lat = sys.lat.copy()
            E_BEP = E_BEP_initial.copy() # gen this new w/ new lat ints
            c,c_count = sys._DM_gen_c_array(lat)
            t,n,site,new_site,plot_ind=0.0,0,0,0,0
            count = sys.counter.copy()
            times = np.full((sys.t_points),fill_value=np.nan)
            thetas = np.full((3,sys.t_points),fill_value=np.nan)
            temps,rates,pops = times.copy(),thetas.copy(),thetas.copy()
            s_wall = time.time()
            s_CPU = time.process_time()
            while t<sys.t_max and n<sys.n_max:
                if switch and n == switch_limit: guess = 'TI' # Swicth guess type after i-th step
                if c_count == 0: print('Reactions complete (c array empty)'); break
                # Generate next time
                new_t = sys._t_gen(sys._DM_total_prop,t,sys.rng.random(),other_args=(c,E_BEP),method=guess)
                # Save state
                count,plot_ind,times,temps,pops,thetas,rates = sys._save_state(t,new_t,count,plot_ind,times,temps,pops,thetas,rates)
                # Advance system time
                t = new_t
                # Global prop gen
                a_acc = sys._DM_get_prop_array(c,E_BEP,t)
                # Choose reaction
                mu_index = np.searchsorted(a_acc,a_acc[-1]*sys.rng.random(),side='left') # binary search
                active_rxns = np.nonzero(c)
                site = active_rxns[0][mu_index]
                rxn_index = active_rxns[1][mu_index]
                # Advance system state
                lat,new_site,count = sys._rxn_step(lat,site,rxn_index,count)
                # Local occ and lateral interactions change
                c,c_count = sys._DM_c_change(lat,c,c_count,site,new_site)
                E_BEP = sys._lateral_interactions_update_2NNs(E_BEP,lat,site,new_site)
                n += 1
            if report: print(f'run{run}: n={n}, t={t}')
            if switch: guess = 'DM' # swicth back to improved guess for next run
            # save run data
            run_label = [f'time{run}',f'temp{run}',f'theta{run}',f'rate{run}',f'pops{run}']
            run_data = {
                run_label[0]:times,
                run_label[1]:temps,
                run_label[2]:thetas,
                run_label[3]:rates,
                run_label[4]:pops
            }
            data.update(run_data)
            e_wall = time.time()
            e_CPU = time.process_time()
            bench_CPU.append(e_CPU-s_CPU)
            bench_wall.append(e_wall-s_wall)
            n_steps.append(n)
        print('DM runs complete')
        return {'CPU':bench_CPU,'wall':bench_wall,'steps':n_steps,'guess':guess}
    
    def run_FRM_2NNs(sys,J_2NNs:np.ndarray,guess:str='FRM',report:bool=False):
        """Runs a kinetic Monte Carlo simulation on the defined lattice \n
        Uses the First reaction method \n
        Returns a dict of time, temp, coverage and desorption rate \n
        dataframe labels of the form (e.g. run X): \n
        'timeX','tempX','thetaX','rateX'
        """
        print(f'Starting FRM with {guess} guess scheme for {sys.runs} runs ...')
        print('Note: this is for benchmarking - I\'m not saving any data!')
        bench_CPU,bench_wall,n_steps = [],[],[]
        data = {}
        if np.shape(J_2NNs) != np.shape(sys.J_BEP):
            raise ValueError('Wrong shape of 2nd NN lateral interactions, make sure this matches the 1st NNs array!')
        else:
            sys.J_BEP2 = J_2NNs
        E_BEP_initial = np.empty((np.shape(sys.E_BEP)),dtype=float)
        for site in range(sys.n_sites):
            E_BEP_initial = sys._lateral_interactions_update_2NNs(E_BEP_initial,sys.lat,site,site)
        for run in range(sys.runs):
            lat = sys.lat.copy()
            E_BEP = E_BEP_initial.copy() # gen this new w/ new lat ints
            t,n,plot_ind=0.0,0,0
            count = sys.counter.copy()
            times = np.full((sys.t_points),fill_value=np.nan)
            thetas = np.full((3,sys.t_points),fill_value=np.nan)
            temps,pops,rates = times.copy(),thetas.copy(),thetas.copy()
            # Initialise data structure
            queue,queue_IDs = sys._FRM_generate_queue(lat,E_BEP,guess)
            s_wall = time.time()
            s_CPU = time.process_time()
            while t<sys.t_max and n<sys.n_max:
                # Choose reaction and time
                if len(queue)==0:print('Reactions complete (reaction queue empty)'); break
                new_t,site,rxn = queue[0]
                # Save state
                count,plot_ind,times,temps,pops,thetas,rates = sys._save_state(t,new_t,count,plot_ind,times,temps,pops,thetas,rates)
                # Advance state and update queue + lateral interactions
                t = new_t
                lat,new_site,count = sys._rxn_step(lat,site,rxn,count)
                queue,queue_IDs,E_BEP = sys._FRM_update_2NNs(queue,queue_IDs,t,site,new_site,E_BEP,lat,guess)
                n += 1
            if report: print(f'run{run}: n={n}, t={t}')
            # save run data
            run_label = [f'time{run}',f'temp{run}',f'theta{run}',f'rate{run}',f'pops{run}']
            run_data = {
                run_label[0]:times,
                run_label[1]:temps,
                run_label[2]:thetas,
                run_label[3]:rates,
                run_label[4]:pops
            }
            data.update(run_data)
            e_wall = time.time()
            e_CPU = time.process_time()
            bench_CPU.append(e_CPU-s_CPU)
            bench_wall.append(e_wall-s_wall)
            n_steps.append(n)
        print('FRM runs complete')
        return {'CPU':bench_CPU,'wall':bench_wall,'steps':n_steps,'guess':guess}
    
    ##################
    ### Data funcs ###
    ##################
    def _save_state(
            sys,
            t:float,
            new_t:float,
            counter:np.ndarray,
            plot_ind:int,
            times:np.ndarray,
            temps:np.ndarray,
            pops:np.ndarray,
            thetas:np.ndarray,
            rates:np.ndarray
        ):
        next_save = (t-t%sys.t_step + sys.t_step) if t!=0 else 0
        if new_t > next_save:
            pops_save = counter[0:3,0]
            theta_save = pops_save/sys.n_sites_per_type
            if plot_ind != 0: last_save = times[(plot_ind-1)]
            else: last_save = 0
            rate_save = (counter[0:3,1]-counter[0:3,2])/(new_t-last_save)
            while next_save<new_t and plot_ind<sys.t_points:
                # save values of interest
                pops[0:3,plot_ind] = pops_save[:]
                thetas[0:3,plot_ind] = theta_save[:]
                rates[0:3,plot_ind] = rate_save
                times[plot_ind] = next_save
                temps[plot_ind] = sys.T(next_save)
                next_save += sys.t_step # next time to save
                plot_ind += 1 # next grid point
            counter[0:3,2] = counter[0:3,1] # reset counts
        return counter,plot_ind,times,temps,pops,thetas,rates
        
    def get_avg(sys,data):
        """Calculates the averages of each column in a KMC data output \n
        Make sure the data comes from this system
        """
        labels = np.empty((5,sys.runs),dtype=object)
        for i in range(sys.runs):
            labels[0,i] = f'time{i}'
            labels[1,i] = f'temp{i}'
            labels[2,i] = f'theta{i}'
            labels[3,i] = f'rate{i}'
            labels[4,i] = f'pops{i}'
        data['time avg'] = data[labels[0,:]].mean(axis=1)
        data['temp avg'] = data[labels[1,:]].mean(axis=1)
        data['theta avg'] = data[labels[2,:]].mean(axis=1) # is this still the right axis?
        data['rate avg'] = data[labels[3,:]].mean(axis=1)
        data['pops avg'] = data[labels[3,:]].mean(axis=1)
        return data
    
    def view_sq_lat(sys,site_labels=None,adatom_labels=None,lattice=None): # needs updated for helical boundaries
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

    ##########################
    ## sys info / utilities ##
    ##########################
    
    def NNs2nd(sys,site:int):
        s_ns = sys.neighbour_key[site,:]
        NNs2 = [
            sys.neighbour_key[s_ns[0],0],sys.neighbour_key[s_ns[0],1],
            sys.neighbour_key[s_ns[1],1],sys.neighbour_key[s_ns[1],2],
            sys.neighbour_key[s_ns[2],2],sys.neighbour_key[s_ns[2],3],
            sys.neighbour_key[s_ns[3],3],sys.neighbour_key[s_ns[3],4],
            sys.neighbour_key[s_ns[4],4],sys.neighbour_key[s_ns[4],5],
            sys.neighbour_key[s_ns[5],5],sys.neighbour_key[s_ns[5],0]
        ]
        return set(NNs2)
    
    def what_coverages(self,lattice=None):
        if type(lattice) != np.ndarray: lattice = self.lat
        max_id = max(lattice[:,1])+1
        counts = np.zeros((max_id),dtype=int)
        for site in lattice[:,1]:
            counts[site] += 1
        counts = counts/self.n_sites
        thetas = {i:float(counts[i]) for i in range(max_id)}
        print('{Species : fractional coverage} key is:')
        print(thetas)
    
    def what_params(params,see_lattice=False):
        print(f'Reaction channels = {params.n_proc}')
        print(f'Simulation: t_max={params.t_max}, n_max={params.n_max}, grid_points={params.t_points}, runs={params.runs}')
        if see_lattice: print(f'Initial lattice:\n{params.lat}')
    
    def change_params(sys,**params_to_change):
        for keyword in params_to_change.keys():
            setattr(sys,keyword,params_to_change[keyword])
    
    def _get_neigh_set(self,site:int):
        NN = set() # dont add site or new_site to avoid self interaction when used for lateral interactions
        for neigh in self.neighbour_key[site,:]:
            NN.add(neigh)
        return NN
    
    def _get_index(grid,location:tuple):
        row,col = location
        _,cols = np.shape(grid.lat)
        return int(col + row*cols)
    
    def _get_coords(grid,index:int):
        _,cols = grid.lat_dimensions
        return int(index // cols), int(index % cols)
