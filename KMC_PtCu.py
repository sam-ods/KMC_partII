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
        """KMC for a PtCu C-H activation system"""
        out_i = r"""
_________        _________
\   ___  \       \   _____\
 \  \__\  \   __  \  \     __    __
  \   _____\__\  \_\  \    \  \  \  \
   \  \     \__   __\  \    \  \  \  \
    \  \       \  \__\  \____\  \__\  \
     \__\       \ ____\_______\________\ """
        print(out_i)
        out1 = 'Time-dependent Kinetic Monte Carlo for surface catalysis'
        out2 = 'Script written by Sam Oades for MChem part II project'
        # Simulation parameters
        sys.rng = np.random.default_rng(sim_param['generator'])
        sys.t_max,sys.n_max,sys.t_step,sys.t_points = sim_param['t_max'],sim_param['n_max'],sim_param['t_step'],sim_param['t_points']
        sys.runs = sim_param['runs']
        sys.lat = sim_param['lattice']
        # Get lattice info
        sys.lat_type,sys.sys_type,sys.lat_dimensions = sim_param['lattice_info']
        sys.neighbour_key = sim_param['neighbours']
        # Error Logging
        sys.rxn_log= []
        # Check correct lattice passed
        if not (sys.lat_type.lower() == 'triangular' and sys.sys_type.upper() == 'SAA'): raise AttributeError('This module is built for C-H activation with surface O on a PtCu (111) SAA. Make sure the correct lattice setup supplied')
        out3 = f'Intialising {sys.lat_type} lattice system on a {sys.lat_dimensions[0]}x{sys.lat_dimensions[1]} supercell'
        # Numerical method parameters
        sys.order_guass = 5 # Default in scipy = 5
        sys.rel_tol,sys.abs_tol = 1e-6,1e-9
        sys.brentq_bracket_max = 60
        sys.switch_lim = 5
        ##
        ## system info
        ##
        sys.n_sites = len(sys.lat[:,0])
        sys.n_neighs = len(sys.neighbour_key[0,:])
        # array sizes
        num_species = 10
        num_site_types = 3
        n_channel = []
        for species in range(num_species):
            n_channel.append(len(sys._get_rxn_key(0,species)))
        n_rxns = max(n_channel)
        # build allowed_rxns key
        sys.species_rxns = {i:set() for i in range(num_species)}
        for species in range(1,num_species):
            # no allowed reactions for species 0
            sys.species_rxns[species].update(sys._get_rxn_key(0,species).keys())
        ##
        ## Kinetic parameters
        ##
        sys.T = Temp_function
        # check array dimensions
        expect_shape = (num_site_types,num_site_types,num_species,7) # CH3 max rxns == diff, 3 H loss, 3 H gain
        if np.shape(E_a) != expect_shape:
            raise IndexError(f'Activation energies input wrong, should be {expect_shape} but E_a input is {np.shape(E_a)}')
        if np.shape(E_a) != np.shape(Pre_exp):
            raise IndexError(f'Pre-exp and E_a array dimensions dont match: {np.shape(Pre_exp)} and {np.shape(E_a)}')
        if np.shape(w_arr) != expect_shape: 
            raise IndexError(f'w_arr wrong shape! should be {expect_shape} but is {np.shape(w_arr)}')
        if np.shape(J_arr) == (num_site_types,num_site_types,num_species,num_species):
            sys.J_BEP = J_arr 
        else:
            raise IndexError(f'J_arr wrong shape! should be {(num_site_types,num_site_types,num_species,num_species)} but is {np.shape(J_arr)}')
        # build refernce E_a,Pre_exp arrays and w_BEP array
        sys.Ea_ref = np.empty((num_site_types,num_site_types,num_species,n_rxns),dtype=float)
        sys.A_ref = np.empty((num_site_types,num_site_types,num_species,n_rxns),dtype=float)
        sys.w_BEP = np.empty((num_site_types,num_site_types,num_species,n_rxns),dtype=float)
        for species in range(num_species):
            i = 0
            for rxn in range(len(E_a[0,0,0,:])):
                # downscaling for double counted channels (assocative desorptions)
                if species == 5 and rxn == 2 : ds = 0.5
                elif species == 7 and rxn == 1 : ds = 0.5
                else: ds = 1
                # OH2 and CO desoprtion have no neighbour dependence
                if (species == 8 and rxn == 3) or (species == 9 and rxn == 3):
                    for site_type1 in range(num_site_types):
                        for site_type2 in range(num_site_types):
                            sys.Ea_ref[site_type1,site_type2,species,i] = E_a[site_type1,site_type2,species,rxn]
                            sys.A_ref[site_type1,site_type2,species,i] = Pre_exp[site_type1,site_type2,species,rxn]
                            sys.w_BEP[site_type1,site_type2,species,i] = w_arr[site_type1,site_type2,species,rxn]
                    i += 1
                # all other rxns have neighbour dependent reactions so split directions
                else:
                    for site_type1 in range(num_site_types):
                        for site_type2 in range(num_site_types):
                            sys.Ea_ref[site_type1,site_type2,species,i:i+sys.n_neighs] = [E_a[site_type1,site_type2,species,rxn]]*sys.n_neighs
                            sys.A_ref[site_type1,site_type2,species,i:i+sys.n_neighs] = [ds*Pre_exp[site_type1,site_type2,species,rxn]]*sys.n_neighs
                            sys.w_BEP[site_type1,site_type2,species,i:i+sys.n_neighs] = [w_arr[site_type1,site_type2,species,rxn]]*sys.n_neighs
                    i += sys.n_neighs
        # build initial E_a and A arrays
        sys.E_a = np.empty((sys.n_sites,n_rxns),dtype=float) 
        sys.A = np.empty((sys.n_sites,n_rxns),dtype=float) 
        for site in range(sys.n_sites):
            sys.E_a,sys.A = sys._kinetic_param_update(sys.lat,sys.E_a,sys.A,site,site)
        out4 = f'Kinetic parameters saved in {np.shape(sys.E_a)[0]}x{np.shape(sys.E_a)[1]} array'
        sys.n_proc = len(sys.E_a[0,:])
        # build base BEP contribution to E_a, will be updated each step
        sys.E_BEP = np.zeros((sys.n_sites,n_rxns),dtype=float)
        for site in range(sys.n_sites):
            sys.E_BEP = sys._lateral_interactions_update(sys.E_BEP.copy(),sys.lat,site,site)
        # build base adatom counter
        sys.counter = np.zeros((num_site_types,14),dtype=int) # rows = site types, columns = CH4 des + species + O2 des + H2 des + OH2 des + CO des
        for species in range(1,10):
            for site_type,atom in zip(sys.lat[:,0],sys.lat[:,1]):
                if atom == species:
                    sys.counter[site_type,species] += 1
        
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

    # make sure empty sets passed as set() since {} is a dict !

    def _get_rxn_key(sys,site:int,species:int):
        #                0  1   2  3  4 5 6  7 8   (9)
        # species 0-8 =  * CH3 CH2 CH C O OH H CO (OH2?)
        # forms should be [diff,H-loss,H-gain,desorption]
        neighs = sys.neighbour_key[site,:]
        n_neighs = sys.n_neighs
        if species == 1: # CH3
            key_CH3 = {i : (0,1,neighs[i]) for i in range(n_neighs)} # diff
            key_CH3.update({i+n_neighs : (0,0,neighs[i]) for i in range(n_neighs)}) # CH4 formation + * (+des)
            key_CH3.update({i+2*n_neighs : (0,5,neighs[i]) for i in range(n_neighs)}) # CH4 formation + OH (+des)
            key_CH3.update({i+3*n_neighs : (0,6,neighs[i]) for i in range(n_neighs)}) # CH4 formation + OH2 (+des)
            key_CH3.update({i+4*n_neighs : (2,7,neighs[i]) for i in range(n_neighs)}) # C-H loss + *
            key_CH3.update({i+5*n_neighs : (2,6,neighs[i]) for i in range(n_neighs)}) # C-H loss + O
            key_CH3.update({i+6*n_neighs : (2,9,neighs[i]) for i in range(n_neighs)}) # C-H loss + OH
            return key_CH3
        elif species == 2: # CH2
            key_CH2 = {i : (0,2,neighs[i]) for i in range(n_neighs)} # diff
            key_CH2.update({i+n_neighs : (1,0,neighs[i]) for i in range(n_neighs)}) # C-H gain + *
            key_CH2.update({i+2*n_neighs : (1,5,neighs[i]) for i in range(n_neighs)}) # C-H gain + OH
            key_CH2.update({i+3*n_neighs : (1,6,neighs[i]) for i in range(n_neighs)}) # C-H gain + OH2
            key_CH2.update({i+4*n_neighs : (3,7,neighs[i]) for i in range(n_neighs)}) # C-H loss + *
            key_CH2.update({i+5*n_neighs : (3,6,neighs[i]) for i in range(n_neighs)}) # C-H loss + O
            key_CH2.update({i+6*n_neighs : (3,9,neighs[i]) for i in range(n_neighs)}) # C-H loss + OH
            return key_CH2
        elif species == 3: # CH
            key_CH = {i : (0,3,neighs[i]) for i in range(n_neighs)} # diff
            key_CH.update({i+n_neighs : (2,0,neighs[i]) for i in range(n_neighs)}) # C-H gain + *
            key_CH.update({i+2*n_neighs : (2,5,neighs[i]) for i in range(n_neighs)}) # C-H gain + OH
            key_CH.update({i+3*n_neighs : (2,6,neighs[i]) for i in range(n_neighs)}) # C-H gain + OH2
            key_CH.update({i+4*n_neighs : (4,7,neighs[i]) for i in range(n_neighs)}) # C-H loss + *
            key_CH.update({i+5*n_neighs : (4,6,neighs[i]) for i in range(n_neighs)}) # C-H loss + O
            key_CH.update({i+6*n_neighs : (4,9,neighs[i]) for i in range(n_neighs)}) # C-H loss + OH
            return key_CH
        elif species == 4: # C
            key_C = {i : (0,4,neighs[i]) for i in range(n_neighs)} # diff
            key_C.update({i+n_neighs : (3,0,neighs[i]) for i in range(n_neighs)}) # C-H gain + *
            key_C.update({i+2*n_neighs : (3,5,neighs[i]) for i in range(n_neighs)}) # C-H gain + OH
            key_C.update({i+3*n_neighs : (3,6,neighs[i]) for i in range(n_neighs)}) # C-H gain + OH2
            key_C.update({i+4*n_neighs : (8,0,neighs[i]) for i in range(n_neighs)}) # C+O->CO
            return key_C
        elif species == 5: # O
            key_O = {i : (0,5,neighs[i]) for i in range(n_neighs)} # diff
            key_O.update({i+n_neighs : (6,0,neighs[i]) for i in range(n_neighs)}) # O-H gain
            key_O.update({i+2*n_neighs : (0,0,neighs[i]) for i in range(n_neighs)}) # associative desorption
            return key_O
        elif species == 6: # OH
            key_OH = {i : (0,6,neighs[i]) for i in range(n_neighs)} # diff
            key_OH.update({i+n_neighs : (9,0,neighs[i]) for i in range(n_neighs)}) # O-H gain
            key_OH.update({i+2*n_neighs : (5,7,neighs[i]) for i in range(n_neighs)}) # O-H loss
            return key_OH
        elif species == 7: # H
            key_H = {i : (0,7,neighs[i]) for i in range(n_neighs)} # diff
            key_H.update({i+n_neighs : (0,0,neighs[i]) for i in range(n_neighs)}) # associative desorption
            return key_H
        elif species == 8: # CO
            key_CO = {i : (0,8,neighs[i]) for i in range(n_neighs)} # diff
            key_CO.update({i+n_neighs : (4,5,neighs[i]) for i in range(n_neighs)}) # dissociation
            key_CO.update({2*n_neighs:(0,0,site)}) # desorption
            return key_CO
        elif species == 9: # OH2 < ----------------- DOES THIS SPECIES STAY ON SURFACE UNDER VACUUM
            key_OH2 = {i : (0,9,neighs[i]) for i in range(n_neighs)} # diff
            key_OH2.update({i+n_neighs : (6,7,neighs[i]) for i in range(n_neighs)}) # O-H loss
            key_OH2.update({2*n_neighs:(0,0,site)}) # desorption
            return key_OH2
        else: return {}
    
    def _get_dependency_key(sys,site:int,species:int):
        # rxn -> tuple( new_site , set(required neighbours) )
        #                0  1   2  3  4 5 6  7 8   (9)
        # species 0-8 =  * CH3 CH2 CH C O OH H CO (OH2?)
        # forms should be [diff,H-gain,H-loss,desorption]
        neighs = sys.neighbour_key[site,:]
        n_neighs = sys.n_neighs
        if species == 1: # CH3
            key_CH3 = {i : (neighs[i],{0}) for i in range(n_neighs)} # diff
            key_CH3.update({i+n_neighs : (neighs[i],{7}) for i in range(n_neighs)}) # CH4 formation + * (+des)
            key_CH3.update({i+2*n_neighs : (neighs[i],{6}) for i in range(n_neighs)}) # CH4 formation + OH (+des)
            key_CH3.update({i+3*n_neighs : (neighs[i],{9}) for i in range(n_neighs)}) # CH4 formation + OH2 (+des)
            key_CH3.update({i+4*n_neighs : (neighs[i],{0}) for i in range(n_neighs)}) # C-H loss + *
            key_CH3.update({i+5*n_neighs : (neighs[i],{5}) for i in range(n_neighs)}) # C-H loss + O
            key_CH3.update({i+6*n_neighs : (neighs[i],{6}) for i in range(n_neighs)}) # C-H loss + OH
            return key_CH3
        elif species == 2: # CH2
            key_CH2 = {i : (neighs[i],{0}) for i in range(n_neighs)} # diff
            key_CH2.update({i+n_neighs : (neighs[i],{7}) for i in range(n_neighs)}) # C-H gain + *
            key_CH2.update({i+2*n_neighs : (neighs[i],{6}) for i in range(n_neighs)}) # C-H gain + OH
            key_CH2.update({i+3*n_neighs : (neighs[i],{9}) for i in range(n_neighs)}) # C-H gain + OH2
            key_CH2.update({i+4*n_neighs : (neighs[i],{0}) for i in range(n_neighs)}) # C-H loss + *
            key_CH2.update({i+5*n_neighs : (neighs[i],{5}) for i in range(n_neighs)}) # C-H loss + O
            key_CH2.update({i+6*n_neighs : (neighs[i],{6}) for i in range(n_neighs)}) # C-H loss + OH
            return key_CH2
        elif species == 3: # CH
            key_CH = {i : (neighs[i],{0}) for i in range(n_neighs)} # diff
            key_CH.update({i+n_neighs : (neighs[i],{7}) for i in range(n_neighs)}) # C-H gain + *
            key_CH.update({i+2*n_neighs : (neighs[i],{6}) for i in range(n_neighs)}) # C-H gain + OH
            key_CH.update({i+3*n_neighs : (neighs[i],{9}) for i in range(n_neighs)}) # C-H gain + OH2
            key_CH.update({i+4*n_neighs : (neighs[i],{0}) for i in range(n_neighs)}) # C-H loss + *
            key_CH.update({i+5*n_neighs : (neighs[i],{5}) for i in range(n_neighs)}) # C-H loss + O
            key_CH.update({i+6*n_neighs : (neighs[i],{6}) for i in range(n_neighs)}) # C-H loss + OH
            return key_CH
        elif species == 4: # C
            key_C = {i : (neighs[i],{0}) for i in range(n_neighs)} # diff
            key_C.update({i+n_neighs : (neighs[i],{7}) for i in range(n_neighs)}) # C-H gain + *
            key_C.update({i+2*n_neighs : (neighs[i],{6}) for i in range(n_neighs)}) # C-H gain + OH
            key_C.update({i+3*n_neighs : (neighs[i],{9}) for i in range(n_neighs)}) # C-H gain + OH2
            key_C.update({i+4*n_neighs : (neighs[i],{5}) for i in range(n_neighs)}) # C+O->CO
            return key_C
        elif species == 5: # O
            key_O = {i : (neighs[i],{0}) for i in range(n_neighs)} # diff
            key_O.update({i+n_neighs : (neighs[i],{7}) for i in range(n_neighs)}) # O-H gain
            key_O.update({i+2*n_neighs : (neighs[i],{5}) for i in range(n_neighs)}) # associative desorption
            return key_O
        elif species == 6: # OH
            key_OH = {i : (neighs[i],{0}) for i in range(n_neighs)} # dff
            key_OH.update({i+n_neighs : (neighs[i],{7}) for i in range(n_neighs)}) # O-H gain
            key_OH.update({i+2*n_neighs : (neighs[i],{0}) for i in range(n_neighs)}) # O-H loss
            return key_OH
        elif species == 7: # H
            key_H = {i : (neighs[i],{0}) for i in range(n_neighs)} # diff
            key_H.update({i+n_neighs : (neighs[i],{7}) for i in range(n_neighs)}) # associative desorption
            return key_H
        elif species == 8: # CO
            key_CO = {i : (neighs[i],{0}) for i in range(n_neighs)} # diff
            key_CO.update({i+n_neighs : (neighs[i],{0}) for i in range(n_neighs)}) # dissociation
            key_CO.update({2*n_neighs:(site,set())}) # desorption
            return key_CO
        elif species == 9: # OH2 < ----------------- DOES THIS SPECIES STAY ON SURFACE UNDER VACUUM
            key_OH2 = {i : (neighs[i],{0}) for i in range(n_neighs)} # diff
            key_OH2.update({i+n_neighs : (neighs[i],{0}) for i in range(n_neighs)}) # O-H loss
            key_OH2.update({2*n_neighs:(site,set())}) # desorption
            return key_OH2
        else: return set()

    def _rxn_step(sys,lattice:np.ndarray,site:int,rxn_ind:int,counts:np.ndarray)->tuple[np.ndarray,int,np.ndarray]:
        """Updates the lattice according to the chosen reaction \n
        returns the updated lattice
        """
        old_species = lattice[site,1]
        rxn_key = sys._get_rxn_key(site,lattice[site,1])
        species,new_species,new_site = rxn_key[rxn_ind]
        lattice[site,1] = species
        if new_site != site:
            lattice[new_site,1] = new_species
        #                0  1   2  3  4 5 6  7 8   (9)
        # species 0-8 =  * CH3 CH2 CH C O OH H CO (OH2?)
        if type(counts) == np.ndarray:
            if {old_species}.issubset(set([1,2,3,4])):
                if 6<=rxn_ind<=23:
                    counts[lattice[site,0],old_species-1] += 1; counts[lattice[site,0],old_species] -= 1 # H gain
                    if rxn_ind<=11: counts[lattice[site,0],7] -= 1 # + *
                    elif 11<rxn_ind<=17: counts[lattice[site,0],5] += 1; counts[lattice[site,0],6] -= 1 # + OH
                    else: counts[lattice[site,0],6] += 1; counts[lattice[site,0],9] -= 1 # + OH2
            if {old_species}.issubset(set([1,2,3])):
                if 24<=rxn_ind<=41:
                    counts[lattice[site,0],old_species+1] += 1; counts[lattice[site,0],old_species] -= 1 # H loss
                    if rxn_ind<=29: counts[lattice[site,0],7] += 1 # *
                    elif 39<rxn_ind<=35: counts[lattice[site,0],5] -= 1; counts[lattice[site,0],6] += 1 # + O
                    else: counts[lattice[site,0],6] -= 1; counts[lattice[site,0],9] += 1 # + OH
            if old_species == 4 and rxn_ind>23: counts[lattice[site,0],4] -= 1; counts[lattice[site,0],5] -= 1; counts[lattice[site,0],8] += 1 # CO formation
            if old_species == 5 and 6<=rxn_ind<=17:
                if rxn_ind<12: counts[lattice[site,0],5] -= 1; counts[lattice[site,0],7] -= 1; counts[lattice[site,0],6] += 1 # O-H gain
                if rxn_ind>=12: counts[lattice[site,0],5] -= 2; counts[lattice[site,0],10] += 1 # O2 des
            if species == 6:
                if rxn_ind<12: counts[lattice[site,0],6] -= 1; counts[lattice[site,0],7] -= 1; counts[lattice[site,0],9] += 1 # O-H gain
                if rxn_ind>=12: counts[lattice[site,0],6] -= 1; counts[lattice[site,0],5] += 1; counts[lattice[site,0],7] += 1 # O-H loss
            if species == 7 and rxn_ind>=6: counts[lattice[site,0],7] -= 2; counts[lattice[site,0],11] += 1 # H2 des
            if species == 8:
                if 6<=rxn_ind<=11: counts[lattice[site,0],4] += 1; counts[lattice[site,0],5] += 1; counts[lattice[site,0],8] -= 1 # CO dssociations
                if rxn_ind == 12: counts[lattice[site,0],13] += 1 # CO des
            if species == 9:
                if 6<=rxn_ind<=11: counts[lattice[site,0],9] -= 1; counts[lattice[site,0],6] += 1; counts[lattice[site,0],7] += 1 # O-H loss
                if rxn_ind == 12: counts[lattice[site,0],12] += 1 # OH2 des
        return lattice,new_site,counts

    ########################################
    ## BEP lateral interactions functions ##
    ########################################
    
    def _lateral_interactions_update(sys,E_BEP:np.ndarray,lattice:np.ndarray,site:int,new_site:int):
        # update sites
        sites_to_update = set(sys.neighbour_key[site,:])
        sites_to_update.add(site)
        if new_site != site: sites_to_update.update(set(sys.neighbour_key[new_site,:]))
        for s in sites_to_update:
            E_BEP[s,:] = 0.0
            lat_int_i = sys._lateral_int(lattice,s)
            for rxn in sys.species_rxns[lattice[s,1]]:
                if sys._check_allowed(s,rxn,lattice): # only updated allowed reactions?
                    lat_f,s_f,_ = sys._rxn_step(lattice.copy(),s,rxn,None)
                    E_BEP[s,rxn] = sys.w_BEP[lattice[s,0],lattice[s_f,0],lattice[s,1],rxn]*(sys._lateral_int(lat_f,s)-lat_int_i)
        return E_BEP

    def _lateral_int(sys,lattice:np.ndarray,site:int):
        NN = set(sys.neighbour_key[site,:])
        F_NN = 0
        for s in NN:
            F_NN += 0.5*sys.J_BEP[lattice[site,0],lattice[s,0],lattice[site,1],lattice[s,1]] # 0.5 since we are using the 2 body interaction energy
        return float(F_NN)

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
    
    def _t_gen(self,prop_func:callable,time:float,random_number:float,other_args:tuple,**kwargs):
        """Generates a new absolute time from a random time step by solving: \n
        $int_{t}^{t+delta_t}(a0(t,other_args)) + ln(r) == 0$ \n
        Uses newton root-finding method with x0 = -ln(r)/a0(t) \n
        Uses intial guess method according to method kwarg: \n
            1. method='DM' 2. method='FRM' 3. method='TI' (default if no kwarg) \n
        If newton fails resorts to brentq method \n
        Relative tolerance: 10**-6
        """
        order_guass = int(self.order_guass)
        rel_tol,abs_tol = self.rel_tol,self.abs_tol
        try:
            if kwargs['method'] == 'FRM':
                # Setup improved FRM initial guess
                rxn,_,site,E_a,A,E_BEP = other_args
                guess = time+self._FRM_improved_guess(time,random_number,E_a[site,rxn]+E_BEP[site,rxn],A[site,rxn])
            elif kwargs['method'] == 'DM':
                # Setup improved FRM initial guess
                c,E_a,A,E_BEP = other_args
                guess = time+self._DM_improved_guess(time,random_number,c,E_a,A,E_BEP)
            elif kwargs['method'] == 'TI':
                a0_t = prop_func(time,*other_args)
                if a0_t<=0: raise ValueError(f'Negative or zero propensity:\na0(t)={a0_t},t={time},r={random_number}\nother={other_args}') 
                guess = time-np.log(random_number)/a0_t
        except KeyError:
            # Setup intial guess (naive)
            a0_t = prop_func(time,*other_args)
            if a0_t<=0: raise ValueError(f'Negative or zero propensity:\na0(t)={a0_t},t={time},r={random_number}\nother={other_args}') 
            guess = time-np.log(random_number)/a0_t
        max_tau = 10**2 * self.t_max
        if guess > max_tau: return np.inf # is this needed?
        # Define functions
        def f(new_time:float):
            int_sol,_ = fixed_quad(prop_func,time,new_time,args=other_args,n=order_guass) #  <-------------- TRY SOLVING IN LOG SPACE
            return float(int_sol) + np.log(random_number)
        def fprime(new_time:float):
            return prop_func(new_time,*other_args)
        # Newton method
        x0 = max(guess,10**-12)
        sol = root_scalar(f,method='newton',x0=x0,fprime=fprime,rtol=rel_tol,xtol=abs_tol)
        if sol.converged: return sol.root
        print('need brentq:\n',sol)
        if sol.root > max_tau: print('inf step'); return np.inf
        # If Newton method fails use brentq backup
        tau_lo = 0 if random_number > 0 else -10**-12 
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
        neigh_species = set()
        new_site,req_neighs = sys._get_dependency_key(site,lattice[site,1])[rxn]
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

    def _kinetic_param_update(sys,lattice:np.ndarray,E_a:np.ndarray,A:np.ndarray,site:int,new_site:int):
        site_type,site_species = lattice[site,0],lattice[site,1]
        for rxn in sys.species_rxns[site_species]:
            if (site_species == 8 and rxn == 2*sys.n_neighs) or (site_species == 9 and rxn == 2*sys.n_neighs):
                # CO and OH2 des both have 2x sets of neigh rxns before them in array
                s_f = site
            else:
                s_f = sys.neighbour_key[site,rxn%sys.n_neighs]
            E_a[site,rxn] = sys.Ea_ref[site_type,lattice[s_f,0],site_species,rxn]
            A[site,rxn] = sys.A_ref[site_type,lattice[s_f,0],site_species,rxn]
        if new_site != site:
            for rxn in sys.species_rxns[lattice[new_site,1]]:
                if (site_species == 8 and rxn == 2*sys.n_neighs) or (site_species == 9 and rxn == 2*sys.n_neighs):
                    s_f = site
                else:
                    s_f = sys.neighbour_key[site,rxn%sys.n_neighs]
                E_a[new_site,rxn] = sys.Ea_ref[lattice[new_site,0],lattice[s_f,0],lattice[new_site,1],rxn]
                A[new_site,rxn] = sys.A_ref[lattice[new_site,0],lattice[s_f,0],lattice[new_site,1],rxn]
        return E_a,A

    ## DM funcs ##
    def _DM_total_prop(sys,time,c_arr:np.ndarray,E_a:np.ndarray,A:np.ndarray,E_BEP:np.ndarray)->float:
        Am,Eam,Ebm = A[c_arr],E_a[c_arr],E_BEP[c_arr]
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
        
    def _DM_get_prop_array(sys,c_array:np.ndarray,E_a:np.ndarray,A:np.ndarray,E_BEP:np.ndarray,time:float):
        ans = np.zeros(np.shape(E_a),dtype=np.float64)
        np.add(E_a,E_BEP,out=ans,where=c_array)
        np.exp((-ans/(k_B*sys.T(time))),out=ans,where=c_array)
        np.multiply(A,ans,out=ans,where=c_array)
        return np.cumsum(ans)
    
    def _DM_site_c(
            sys,
            lattice:np.ndarray,
            site:int,
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
    
    def _DM_gen_c_array(sys,lattice:np.ndarray):
        c = np.empty((np.shape(sys.E_a)),dtype=bool)
        for site in range(sys.n_sites):
            c[site,:] = sys._DM_site_c(lattice,site)
        c_count = np.sum(np.asarray(c,dtype=int))
        return c,c_count
    
    def _DM_improved_guess(self,time:float,random_number:float,c_arr:np.ndarray,E_a:np.ndarray,A:np.ndarray,E_BEP:np.ndarray):# speed this up using matmul opts
        """My improved intial guess for Newton root-finding in DM
        only applies to linear temperature ramps and time-independent Pre-exponential factors
        """
        sim_temp = self.T(time)
        beta = self.T(1)-self.T(0)

        Am,Eam,Ebm = A[c_arr],E_a[c_arr],E_BEP[c_arr]
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
    def _k(sys,site:int,rxn:int,E_a:np.ndarray,A:np.ndarray,E_BEP:np.ndarray,time:float)->np.ndarray:
        return A[site,rxn]*np.exp(-(E_a[site,rxn]+E_BEP[site,rxn])/(k_B*sys.T(time)))
    
    def _FRM_generate_queue(sys,lattice:np.ndarray,E_a:np.ndarray,A:np.ndarray,E_BEP:np.ndarray,guess_method:str):
        sortlist =  SortedList(key=lambda tup:tup[0]) # sort list based on first tuple entry (time)
        site_keys = {site:[] for site in range(sys.n_sites)} # site -> IDs
        for site in range(sys.n_sites):
            for rxn in sys.species_rxns[lattice[site,1]]:
                if sys._check_allowed(site,rxn,lattice):
                    rxn_time = sys._t_gen(sys._FRM_site_prop,0,sys.rng.random(),(rxn,lattice,site,E_a,A,E_BEP),method=guess_method)
                    sortlist,site_keys = sys._FRM_insert((rxn_time,site,rxn),sortlist,site_keys)
        return sortlist,site_keys

    def _FRM_site_prop( # adapt for many species
            sys,
            time:float,
            rxn:int,
            lattice:np.ndarray,
            site:int,
            E_a:np.ndarray,
            A:np.ndarray,
            E_BEP:np.ndarray
        )->float:
        if sys._check_allowed(site,rxn,lattice):
            return sys._k(site,rxn,E_a,A,E_BEP,time)
        else:
            return 0
    
    def _FRM_update(sys,sortlist:SortedList,site_keys:dict,time:float,site:int,new_site:int,E_a:np.ndarray,A:np.ndarray,E_BEP:np.ndarray,lattice:np.ndarray,guess_method:str):
        # update lateral interactions
        E_BEP = sys._lateral_interactions_update(E_BEP,lattice,site,new_site)
        # site to update
        to_update = set(sys.neighbour_key[site,:])
        to_update.add(site)
        if new_site != site:
            to_update.update(sys.neighbour_key[new_site,:])
        for s in to_update:
            # remove old reactions
            to_remove = site_keys[s].copy()
            for ID in to_remove:
                sortlist,site_keys = sys._FRM_remove(ID,sortlist,site_keys)
            # add new reactions
            for rxn in sys.species_rxns[lattice[s,1]]:
                if sys._check_allowed(s,rxn,lattice):
                    rxn_time = sys._t_gen(sys._FRM_site_prop,time,sys.rng.random(),(rxn,lattice,s,E_a,A,E_BEP),method=guess_method)
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
            E_a,A,E_BEP = sys.E_a.copy(),sys.A.copy(),sys.E_BEP.copy()
            c,c_count = sys._DM_gen_c_array(lat)
            t,n,site,new_site,plot_ind=0.0,0,0,0,0
            count = sys.counter.copy()
            times = np.full((sys.t_points),fill_value=np.nan)
            temps = times.copy()
            pop_dict = {}
            for i in range(14):
                for s in range(3):
                    pop_dict[(s,i)] = times.copy()
            while t<sys.t_max and n<sys.n_max:
                if switch and n == switch_limit: guess = 'TI' # Swicth guess type after i-th step
                if c_count == 0: print('Reactions complete (c array empty)'); break
                # Generate next time
                new_t = sys._t_gen(sys._DM_total_prop,t,sys.rng.random(),other_args=(c,E_a,A,E_BEP),method=guess)
                # Save state
                pop_dict,plot_ind = sys._save_state(t,new_t,count,plot_ind,times,temps,pop_dict)
                # Advance system time
                t = new_t
                # Global prop gen
                a_acc = sys._DM_get_prop_array(c,E_a,A,E_BEP,t)
                # Choose reaction
                mu_index = np.searchsorted(a_acc,a_acc[-1]*sys.rng.random(),side='left') # binary search
                rxn_index = mu_index % sys.n_proc
                site = mu_index // sys.n_proc
                # Advance system state
                lat,new_site,count = sys._rxn_step(lat,site,rxn_index,count)
                # Local occ and lateral interactions change
                c,c_count = sys._DM_c_change(lat,c,c_count,site,new_site)
                E_a,A = sys._kinetic_param_update(lat,E_a,A,site,new_site)
                E_BEP = sys._lateral_interactions_update(E_BEP,lat,site,new_site)
                n += 1
            if report: print(f'run{run}: n={n}, t={t}')
            if switch: guess = 'DM' # swicth back to improved guess for next run
            # save run data
            run_label = [f'time{run}',f'temp{run}',f'pops{run}']
            run_data = {
                run_label[0]:times,
                run_label[1]:temps,
                run_label[2]:pop_dict
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
            E_a,A,E_BEP = sys.E_a.copy(),sys.A.copy(),sys.E_BEP.copy()
            t,n,plot_ind=0.0,0,0
            count = sys.counter.copy()
            times = np.full((sys.t_points),fill_value=np.nan)
            temps = times.copy()
            pop_dict = {}
            for i in range(14): pop_dict[(0,i)] = times.copy(); pop_dict[(1,i)] = times.copy(); pop_dict[(2,i)] = times.copy()
            # Initialise data structure
            queue,queue_IDs = sys._FRM_generate_queue(lat,E_a,A,E_BEP,guess)
            while t<sys.t_max and n<sys.n_max:
                # Choose reaction and time
                if len(queue)==0: print('Reactions complete (reaction queue empty)'); break
                new_t,site,rxn = queue[0]
                if lat[site,1] == 0: raise ValueError('Selected empty site!')
                # Save state
                pop_dict,plot_ind = sys._save_state(t,new_t,count,plot_ind,times,temps,pop_dict)
                t = new_t
                # Advance state and update queue + lateral interactions
                lat,new_site,count = sys._rxn_step(lat,site,rxn,count)
                E_a,A = sys._kinetic_param_update(lat,E_a,A,site,new_site)
                queue,queue_IDs,E_BEP = sys._FRM_update(queue,queue_IDs,t,site,new_site,E_a,A,E_BEP,lat,guess)
                n += 1
            if report: print(f'run{run}: n={n}, t={t}')
            # save run data
            run_label = [f'time{run}',f'temp{run}',f'pops{run}']
            run_data = {
                run_label[0]:times,
                run_label[1]:temps,
                run_label[2]:pop_dict
            }
            data.update(run_data)
        print(f'FRM runs complete')
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
        # bench_data
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
            E_a,A,E_BEP = sys.E_a.copy(),sys.A.copy(),sys.E_BEP.copy()
            c,c_count = sys._DM_gen_c_array(lat)
            t,n=0.0,0
            count = sys.counter.copy()
            s_wall = time.time()
            s_CPU = time.process_time()
            while t<sys.t_max and n<sys.n_max:
                if switch and n == switch_limit: guess = 'TI' # Swicth guess type after i-th step
                if c_count == 0: print('Reactions complete (c array empty)'); break
                # Generate next time
                new_t = sys._t_gen(sys._DM_total_prop,t,sys.rng.random(),other_args=(c,E_a,A,E_BEP),method=guess)
                # Advance system time
                t = new_t
                # Global prop gen
                a_acc = sys._DM_get_prop_array(c,E_a,A,E_BEP,t)
                # Choose reaction
                mu_index = np.searchsorted(a_acc,a_acc[-1]*sys.rng.random(),side='left') # binary search
                rxn_index = mu_index % sys.n_proc
                site = mu_index // sys.n_proc
                # Advance system state
                lat,new_site,count = sys._rxn_step(lat,site,rxn_index,count)
                # Local occ and lateral interactions change
                c,c_count = sys._DM_c_change(lat,c,c_count,site,new_site)
                E_a,A = sys._kinetic_param_update(lat,E_a,A,site,new_site)
                E_BEP = sys._lateral_interactions_update(E_BEP,lat,site,new_site)
                n += 1
            if switch: guess = 'DM' # switch back to improved guess for next run
            e_wall = time.time()
            e_CPU = time.process_time()
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
        # bench_data
        bench_CPU,bench_wall,n_steps = [],[],[]
        for run in range(sys.runs):
            lat = sys.lat.copy()
            E_a,A,E_BEP = sys.E_a.copy(),sys.A.copy(),sys.E_BEP.copy()
            t,n=0.0,0
            count = sys.counter.copy()
            # Initialise data structure
            queue,queue_IDs = sys._FRM_generate_queue(lat,E_a,A,E_BEP,guess)
            s_wall = time.time()
            s_CPU = time.process_time()
            while t<sys.t_max and n<sys.n_max:
                # Choose reaction and time
                if len(queue)==0:print('Reactions complete (reaction queue empty)'); break
                new_t,site,rxn = queue[0]
                t = new_t
                # Advance state and update queue + lateral interactions
                lat,new_site,count = sys._rxn_step(lat,site,rxn,count)
                E_a,A = sys._kinetic_param_update(lat,E_a,A,site,new_site)
                queue,queue_IDs,E_BEP = sys._FRM_update(queue,queue_IDs,t,site,new_site,E_a,A,E_BEP,lat,guess)
                n += 1
            e_CPU = time.process_time()
            e_wall = time.time()
            bench_CPU.append(e_CPU-s_CPU)
            bench_wall.append(e_wall-s_wall)
            n_steps.append(n)
        print('FRM runs complete')
        return {'CPU':bench_CPU,'wall':bench_wall,'steps':n_steps,'guess':guess}

    ##################
    ## Single loops ##
    ##################

    def SL_DM(sys,lat:np.ndarray,E_a:np.ndarray,A:np.ndarray,E_BEP:np.ndarray,guess:str,n_reps:int=100):
        """Single loop benchmark\n
        Output in ns"""
        c,c_count = sys._DM_gen_c_array(lat)
        counts = sys.counter.copy()
        s_WALL = time.perf_counter_ns()
        s_CPU = time.process_time_ns()
        for i in range(n_reps):
            if c_count == 0: return np.nan
            new_t = sys._t_gen(sys._DM_total_prop,0,sys.rng.random(),other_args=(c,E_a,A,E_BEP),method=guess)
            a_acc = sys._DM_get_prop_array(c,E_a,A,E_BEP,new_t)
            mu_index = np.searchsorted(a_acc,a_acc[-1]*sys.rng.random(),side='left')
            rxn_index = mu_index % sys.n_proc
            site = mu_index // sys.n_proc
            new_lat,new_site,counts = sys._rxn_step(lat.copy(),site,rxn_index,counts)
            _ = sys._DM_c_change(lat,c.copy(),c_count.copy(),site,new_site)
            _,_ = sys._kinetic_param_update(new_lat,E_a.copy(),A.copy(),site,new_site)
            _ = sys._lateral_interactions_update(E_BEP.copy(),new_lat,site,new_site)
        e_CPU = time.process_time_ns()
        e_WALL = time.perf_counter_ns()
        return {'CPU':(e_CPU-s_CPU)/n_reps, 'wall':(e_WALL-s_WALL)/n_reps,'runs':n_reps}
    
    def SL_FRM(sys,lat:np.ndarray,E_a:np.ndarray,A:np.ndarray,E_BEP:np.ndarray,guess:str,n_reps:int=100):
        """Single loop benchmark\n
        Output in ns"""
        queue,queue_IDs = sys._FRM_generate_queue(lat,E_a,A,E_BEP,guess)
        counts = sys.counter.copy()
        s_WALL = time.perf_counter_ns()
        s_CPU = time.process_time_ns()
        for i in range(n_reps):
            if len(queue)==0:print('Reactions complete (reaction queue empty)'); return np.nan
            new_t,site,rxn = queue[0]
            t = new_t
            new_lat,new_site,counts = sys._rxn_step(lat.copy(),site,rxn,counts)
            E_a_new,A_new = sys._kinetic_param_update(new_lat,E_a.copy(),A.copy(),site,new_site)
            _ = sys._FRM_update(queue.copy(),copy.deepcopy(queue_IDs),t,site,new_site,E_a_new,A_new,E_BEP.copy(),new_lat,guess)
        e_CPU = time.process_time_ns()
        e_WALL = time.perf_counter_ns()
        return {'CPU':(e_CPU-s_CPU)/n_reps, 'wall':(e_WALL-s_WALL)/n_reps,'runs':n_reps}
    
    ######################################################
    ## BEP lateral interactions functions up to 2nd NNs ##
    ######################################################

    def _FRM_update_2NNs(sys,sortlist:SortedList,site_keys:dict,time:float,site:int,new_site:int,E_a:np.ndarray,A:np.ndarray,E_BEP:np.ndarray,lattice:np.ndarray,guess_method:str):
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
                    rxn_time = sys._t_gen(sys._FRM_site_prop,time,sys.rng.random(),(rxn,lattice,s,E_a,A,E_BEP),method=guess_method)
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
                    E_BEP[s,rxn] = sys.w_BEP[lattice[s,0],lattice[s_f,0],lattice[s,1],rxn]*(sys._lateral_int_2NNS(lat_f,s)-lat_int_i)
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
            E_a,A,E_BEP = sys.E_a.copy(),sys.A.copy(),E_BEP_initial.copy()
            c,c_count = sys._DM_gen_c_array(lat)
            t,n,site,new_site,plot_ind=0.0,0,0,0,0
            count = sys.counter.copy()
            times = np.full((sys.t_points),fill_value=np.nan)
            temps = times.copy()
            pop_dict = {}
            for i in range(14): pop_dict[(0,i)] = times.copy(); pop_dict[(1,i)] = times.copy(); pop_dict[(2,i)] = times.copy()
            while t<sys.t_max and n<sys.n_max:
                if switch and n == switch_limit: guess = 'TI' # Swicth guess type after i-th step
                if c_count == 0: print('Reactions complete (c array empty)'); break
                # Generate next time
                new_t = sys._t_gen(sys._DM_total_prop,t,sys.rng.random(),other_args=(c,E_a,A,E_BEP),method=guess)
                # Save state
                pop_dict,plot_ind = sys._save_state(t,new_t,count,plot_ind,times,temps,pop_dict)
                # Advance system time
                t = new_t
                # Global prop gen
                a_acc = sys._DM_get_prop_array(c,E_a,A,E_BEP,t)
                if a_acc[-1] == 0: print('Reactions complete (total_propensity = 0)'); break
                # Choose reaction
                mu_index = np.searchsorted(a_acc,a_acc[-1]*sys.rng.random(),side='left') # binary search
                rxn_index = mu_index % sys.n_proc
                site = mu_index // sys.n_proc
                # Advance system state
                lat,new_site,count = sys._rxn_step(lat,site,rxn_index,count)
                # Local occ and lateral interactions change
                c,c_count = sys._DM_c_change(lat,c,c_count,site,new_site)
                E_a,A = sys._kinetic_param_update(lat,E_a,A,site,new_site)
                E_BEP = sys._lateral_interactions_update_2NNs(E_BEP,lat,site,new_site)
                n += 1
            if report: print(f'run{run}: n={n}, t={t}')
            if switch: guess = 'DM' # swicth back to improved guess for next run
            # save run data
            run_label = [f'time{run}',f'temp{run}',f'pops{run}']
            run_data = {
                run_label[0]:times,
                run_label[1]:temps,
                run_label[2]:pop_dict
            }
            data.update(run_data)
        print('DM runs complete')
        return data
    
    def run_FRM_2NNs(sys,J_2NNs:np.ndarray,guess:str='FRM',report:bool=False):
        """Runs a kinetic Monte Carlo simulation on the defined lattice \n
        Uses the First reaction method \n
        Returns a dict of time, temp, coverage and desorption rate \n
        dataframe labels of the form (e.g. run X): \n
        'timeX','tempX','thetaX','rateX'
        """
        print(f'Starting FRM with {guess} guess scheme for {sys.runs} runs ...')
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
            E_a,A,E_BEP = sys.E_a.copy(),sys.A.copy(),E_BEP_initial.copy()
            t,n,plot_ind=0.0,0,0
            count = sys.counter.copy()
            times = np.full((sys.t_points),fill_value=np.nan)
            temps = times.copy()
            pop_dict = {}
            for i in range(14): pop_dict[(0,i)] = times.copy(); pop_dict[(1,i)] = times.copy(); pop_dict[(2,i)] = times.copy()
            # Initialise data structure
            queue,queue_IDs = sys._FRM_generate_queue(lat,E_a,A,E_BEP,guess)
            while t<sys.t_max and n<sys.n_max:
                # Choose reaction and time
                if len(queue)==0: print('Reactions complete (reaction queue empty)'); break
                new_t,site,rxn = queue[0]
                # Save state
                pop_dict,plot_ind = sys._save_state(t,new_t,count,plot_ind,times,temps,pop_dict)
                t = new_t
                # Advance state and update queue + lateral interactions
                lat,new_site,count = sys._rxn_step(lat,site,rxn,count)
                E_a,A = sys._kinetic_param_update(lat,E_a,A,site,new_site)
                queue,queue_IDs,E_BEP = sys._FRM_update_2NNs(queue,queue_IDs,t,site,new_site,E_a,A,E_BEP,lat,guess)
                n += 1
            if report: print(f'run{run}: n={n}, t={t}')
            # save run data
            run_label = [f'time{run}',f'temp{run}',f'pops{run}']
            run_data = {
                run_label[0]:times,
                run_label[1]:temps,
                run_label[2]:pop_dict
            }
            data.update(run_data)
        print(f'FRM runs complete')
        return data
    
    ##################
    ### Data funcs ###
    ##################
    
    def _save_state(sys,t:float,new_t:float,counter:np.ndarray,plot_ind:int,times:np.ndarray,temps:np.ndarray,pop_dict:dict):
        next_save = (t-t%sys.t_step + sys.t_step) if t!=0 else 0
        while next_save<new_t and plot_ind<sys.t_points:
            # save values of interest
            times[plot_ind] = next_save
            temps[plot_ind] = sys.T(next_save)
            for i in range(14): # all species + desoprtion counters
                for s in range(3):
                    pop_dict[(s,i)][plot_ind] = counter[s,i]
            next_save += sys.t_step # next time to save
            plot_ind += 1 # next grid point
        return pop_dict,plot_ind

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
    
    def change_params(sys,**params_to_change):
        for keyword in params_to_change.keys():
            setattr(sys,keyword,params_to_change[keyword])
    
    def _get_index(grid,location:tuple):
        row,col = location
        _,cols = np.shape(grid.lat)
        return int(col + row*cols)
    
    def _get_coords(grid,index:int):
        _,cols = grid.lat_dimensions
        return int(index // cols), int(index % cols)