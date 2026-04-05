import numpy as np
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
        """E_a and pre-exponetial required as specific shape (n_site_types,2+n_site_types) \n
        T-dependent pre-exponentials not supported (yet?) \n
        [ads0,des0,diff00,diff01,diff02,...] \n
        [ads1,des1,diff10,diff11,diff12,...] \n
        [ads2,des2,diff20,diff21,diff22,...] \n
        [...] \n
        Nearest neighbour (2-body) interactions should be passed as a (n_species+1,n_species+1) \n
        Diagonal NNs in square lattices also need diagonal interactions in the same format as the J_diag kwarg \n
        Double counting of interactions corrected internally \n
        J_ij == J_ji but will need to access for i->j and j->i interactions \n
        Note J[0,:] and J[:,0] must all be zero to represent absence of neighbour \n
        For each site type the 2d array is: \n
        [0, 0  , 0  , 0  ,...] \n
        [0,J_11,J_12,J_13,...] \n
        [0,J_21,J_22,J_31,...] \n
        [0,J_31,J_32,J_33,...] \n
        [...] \n 
        w should be a (n_site_types,n_rxns) array like E_a \n
        """
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
        # Initialise FRM queue objects
        sys.FRM_sortlist = SortedList(key=lambda tup:tup[0]) # sort list based on first tuple entry (time)
        sys.FRM_site_keys = {site:[] for site in range(len(sys.lat[:,0]))} # site -> IDs
        # Error Logging
        sys.rxn_log,sys.int_log = [],[]
        # Check correct lattice passed
        if not (sys.lat_type.lower() == 'triangular' and sys.sys_type.upper() == 'SAA'): raise AttributeError('This module is built for C-H activation with surface O on a PtCu (111) SAA. Make sure the correct lattice setup supplied')
        out3 = f'Intialising {sys.lat_type} lattice system on a {sys.lat_dimensions[0]}x{sys.lat_dimensions[1]} supercell'
        # Numerical method parameters
        sys.order_guass = 5 # Default in scipy = 5
        sys.rel_tol = 10**-6
        sys.brentq_bracket_max = 60

        ##
        ## Kinetic parameters
        ##
        sys.T = Temp_function

        # work out array size
        neighs = len(sys.neighbour_key[0,:])
        num_species = 10
        n_channel = []
        for species in range(num_species):
            n_channel.append(len(sys._get_rxn_key(0,species)))
        n_rxns = max(n_channel)

        # check array dimensions
        expect_shape = (2,num_species,7) # CH3 max rxns == diff,3 H loss, 3 H gain
        E_a = np.atleast_2d(E_a)
        Pre_exp = np.atleast_2d(Pre_exp)
        w_arr = np.atleast_2d(w_arr)
        J_arr = np.atleast_2d(J_arr)
        if np.shape(E_a) != expect_shape:
            raise IndexError(f'Activation energies input wrong, should be {expect_shape} but E_a input is {np.shape(E_a)}')
        if np.shape(E_a) != np.shape(Pre_exp):
            raise IndexError(f'Pre-exp and E_a array dimensions dont match: {np.shape(Pre_exp)} and {np.shape(E_a)}')
        if np.shape(w_arr) != expect_shape: 
            raise IndexError(f'w_arr wrong shape! should be {expect_shape} but is {np.shape(w_arr)}')
        if np.shape(J_arr) == (num_species,num_species):
            sys.J_BEP = J_arr 
        else:
            raise IndexError(f'J_arr wrong shape! should be {(num_species,num_species)} but is {np.shape(J_arr)}')
        
        # build base E_a,Pre_exp array
        sys.Ea_ref = np.empty((2,num_species,n_rxns),dtype=float)
        sys.A_ref = np.empty((2,num_species,n_rxns),dtype=float)
        sys.w_BEP = np.empty((2,num_species,n_rxns),dtype=float)
        for species in range(num_species):
            i = 0
            for rxn in range(len(E_a[0,0,:])):
                # downscaling for double counted channels (assocative desorptions)
                if species == 5 and rxn == 2 : ds = 0.5
                elif species == 7 and rxn == 1 : ds = 0.5
                else: ds = 1
                # OH2 and CO desoprtion have no neighbour dependence
                if (species == 8 and rxn == 3) or (species == 9 and rxn == 3):
                    sys.Ea_ref[0,species,i] = E_a[0,species,rxn]
                    sys.Ea_ref[1,species,i] = E_a[1,species,rxn]
                    sys.A_ref[0,species,i] = Pre_exp[0,species,rxn]
                    sys.A_ref[1,species,i] = Pre_exp[1,species,rxn]
                    sys.w_BEP[0,species,i] = w_arr[0,species,rxn]
                    sys.w_BEP[1,species,i] = w_arr[1,species,rxn]
                    i += 1
                # all other rxns have neighbour dependent reactions so split directions
                else:
                    sys.Ea_ref[0,species,i:i+neighs] = [E_a[0,species,rxn]]*neighs
                    sys.Ea_ref[1,species,i:i+neighs] = [E_a[1,species,rxn]]*neighs
                    sys.A_ref[0,species,i:i+neighs] = [ds*Pre_exp[0,species,rxn]]*neighs
                    sys.A_ref[1,species,i:i+neighs] = [ds*Pre_exp[1,species,rxn]]*neighs
                    sys.w_BEP[0,species,i:i+neighs] = [w_arr[0,species,rxn]]*neighs
                    sys.w_BEP[1,species,i:i+neighs] = [w_arr[1,species,rxn]]*neighs
                    i += neighs
        sys.E_a = np.empty((len(sys.lat[:,0]),n_rxns),dtype=float) 
        sys.A = np.empty((len(sys.lat[:,0]),n_rxns),dtype=float) 
        for site,site_info in enumerate(zip(sys.lat[:,0],sys.lat[:,1])):
            sys.E_a[site,:] = sys.Ea_ref[site_info[0],site_info[1],:]
            sys.A[site,:] = sys.A_ref[site_info[0],site_info[1],:]
        out4 = f'Kinetic parameters saved in {np.shape(sys.E_a)[0]}x{np.shape(sys.E_a)[1]} array'

        # build allowed_rxns key
        sys.species_rxns = {i:set() for i in range(num_species)}
        for species in range(1,num_species):
            # no allowed reactions for species 0
            sys.species_rxns[species].update(sys._get_rxn_key(0,species).keys())
        
        # build base BEP contribution to E_a, will be updated each step
        sys.E_BEP = np.zeros((len(sys.lat[:,0]),n_rxns),dtype=float)
        for site in range(len(sys.lat[:,0])):
            sys.E_BEP = sys._lateral_interactions_update(sys.E_BEP.copy(),sys.lat,site,site)

        # build base adatom counter
        sys.counter = np.zeros((2,14),dtype=int) # rows = site types, columns = CH4 des + species + O2 des + H2 des + OH2 des + CO des
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
        n_neighs = len(neighs)
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
        n_neighs = len(neighs)
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

    def _rxn_step(sys,lattice:np.ndarray,site:int,rxn_ind:int,counts:np.ndarray)->tuple[np.ndarray,int,int,np.ndarray]:
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
            for rxn in sys.species_rxns[lattice[s,1]]:
                if bool(sys._check_allowed(s,rxn,lattice)): # only updated allowed reactions?
                    lat_i = lattice.copy()
                    lat_f,_,_ = sys._rxn_step(lat_i.copy(),s,rxn,None)
                    E_BEP[s,rxn] = sys.w_BEP[lattice[s,0],lattice[s,1],rxn]*(sys._lateral_int(lat_f,s)-sys._lateral_int(lat_i,s))
        return E_BEP

    def _lateral_int(sys,lattice:np.ndarray,site:int):
        NN = set(sys.neighbour_key[site,:])
        F_NN = 0
        for s in NN:
            F_NN += 0.5*sys.J_BEP[lattice[site,1],lattice[s,1]] # 0.5 since we are using the 2 body interaction energy
        return float(F_NN)

    #####################################
    ### FRM data structure definition ###
    #####################################

    def _FRM_insert(tree,ID:tuple):
        if not np.isfinite(ID[0]):
            tree.rxn_log.append(f'1. Null rxn: ID={ID}')
            return
        else:
            tree.FRM_sortlist.add(ID)
            tree.FRM_site_keys[ID[1]].append(ID)
    
    def _FRM_remove(tree,ID:tuple):
        try:
            tree.FRM_sortlist.remove(ID)
            tree.FRM_site_keys[ID[1]].remove(ID)
        except ValueError:
            pass

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
        rel_tol = self.rel_tol
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
        if guess > max_tau:
            try:
                guess_method = kwargs['method']
            except KeyError:
                guess_method = 'TI'
            self.int_log.append(f'1. Guess greater than max time: Method={guess_method}')
            return np.inf # is this needed?
        # Define functions
        def f(new_time:float):
            int_sol,_ = fixed_quad(prop_func,time,new_time,args=other_args,n=order_guass) #  <-------------- TRY SOLVING IN LOG SPACE
            return float(int_sol) + np.log(random_number)
        def fprime(new_time:float):
            return prop_func(new_time,*other_args)
        # Newton method
        x0 = max(guess,10**-12)
        sol = root_scalar(f,method='newton',x0=x0,fprime=fprime,rtol=rel_tol)
        if sol.converged: return sol.root
        print('need brentq:\n',sol)
        if sol.root > max_tau: print('inf step'); return np.inf
        # If Newton method fails use brentq backup
        newt_attempt = sol.root,sol.flag
        tau_lo = 0 if random_number > 0 else -10**-12 
        tau_hi = min(guess,max_tau)
        if f(tau_hi)<0:
            loop_count = 0
            while f(tau_hi)<0:
                tau_hi*=2
                loop_count += 1
                if loop_count>self.brentq_bracket_max: self.int_log.append('2. Failed to find appropriate bracket') ; return np.inf
        sol = root_scalar(f,method='brentq',bracket=[tau_lo,tau_hi],rtol=rel_tol)
        if not sol.converged: raise RuntimeError(f'Both root finding methods failed, check prop_func behaviour (t={time})')
        self.int_log.append(f'3. Newton failed: step={newt_attempt[0]}, Brentq converged: step={sol.root}, flag: {newt_attempt[1]}')
        return sol.root # absolute time of next reaction
    
    def _check_allowed(sys,site:int,rxn:int,lattice:np.ndarray):
        """Checks if a site has a predefined neighbour arrangement
        i.e. if a reaction is possible"""
        # rxn -> tuple( new_site , set(required neighbours) )
        neigh_species = set()
        new_site,req_neighs = sys._get_dependency_key(site,lattice[site,1])[rxn]
        if site == new_site: # checks all neighbours
            for s_n in sys.neighbour_key[site,:]: neigh_species.add(lattice[s_n,1])
            if req_neighs.issubset(neigh_species):
                return 1
            else:
                return 0
        else: # checks a specific direction
            neigh_species.add(lattice[new_site,1])
            if req_neighs.issubset(neigh_species):
                return 1
            else:
                return 0

    def _kinetic_param_update(sys,lattice:np.ndarray,E_a:np.ndarray,A:np.ndarray,site:int,new_site:int):
        E_a[site,:] = sys.Ea_ref[lattice[site,0],lattice[site,1]]
        A[site,:] = sys.A_ref[lattice[site,0],lattice[site,1]]
        if new_site != site:
            E_a[new_site,:] = sys.Ea_ref[lattice[new_site,0],lattice[new_site,1]]
            A[new_site,:] = sys.A_ref[lattice[new_site,0],lattice[new_site,1]]
        return E_a,A

    ## DM funcs ##
    def _k_array(sys,time,E_a:np.ndarray,A:np.ndarray,E_BEP:np.ndarray):
        if np.asarray(time).ndim == 0:
            return A*np.exp(-(E_a+E_BEP)/(k_B*sys.T(time)))
        else:
            return A[None,...]*np.exp(-(E_a[None,...]+E_BEP[None,...])/(k_B*sys.T(time)[:,None,None]))

    def _DM_total_prop(sys,time,c_arr:np.ndarray,E_a:np.ndarray,A:np.ndarray,E_BEP:np.ndarray)->float:
        if np.asarray(time).ndim == 0:
            ax=(0,1)
        else:
            ax=(1,2)
        return np.sum(sys._k_array(time,E_a,A,E_BEP)*c_arr,axis=ax)
    
    def _DM_site_c(
            sys,
            lattice:np.ndarray,
            site:int,
        )->np.ndarray:
        c = np.zeros(len(sys.E_a[0,:]),dtype=int)
        for rxn in sys.species_rxns[lattice[site,1]]:
            c[rxn] = sys._check_allowed(site,rxn,lattice)
        return c
        
    def _DM_c_change(
            sys,
            lattice:np.ndarray,
            c_array:np.ndarray,
            site:int,
            new_site:int
        ):
        # Origin
        sites_to_update = set(sys.neighbour_key[site,:])
        sites_to_update.add(site)
        if new_site != site: sites_to_update.update(sys.neighbour_key[new_site,:])
        for s in sites_to_update:
            c_array[s,:] = sys._DM_site_c(lattice,s)
        return c_array
    
    def _DM_gen_c_array(sys,lattice:np.ndarray):
        c = np.empty((np.shape(sys.E_a)),dtype=int)
        for site in range(len(lattice[:,0])):
            c[site,:] = sys._DM_site_c(lattice,site)
        return c
    
    def _DM_get_prop_array(sys,c_array:np.ndarray,E_a:np.ndarray,A:np.ndarray,E_BEP:np.ndarray,time:float):
        return np.cumsum(c_array*sys._k_array(time,E_a,A,E_BEP))
    
    def _DM_improved_guess(self,time:float,random_number:float,c_arr:np.ndarray,E_a:np.ndarray,A:np.ndarray,E_BEP:np.ndarray):
        """My improved intial guess for Newton root-finding in DM
        only applies to linear temperature ramps and time-independent Pre-exponential factors
        """
        sim_temp = self.T(time)
        beta = self.T(1)-self.T(0)

        all_E_a = (E_a + E_BEP)
        nz_rs,nz_cs = np.nonzero(c_arr)
        E_a_arr = np.empty(len(nz_rs))
        Pre_exp_arr = E_a_arr.copy()
        for ind,r,c in zip(range(len(E_a_arr)),nz_rs,nz_cs):
            E_a_arr[ind] = all_E_a[r,c]
            Pre_exp_arr[ind] = A[r,c]
        
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
        if temp_guess.imag != 0: raise ValueError('Complex valued initial guess!')
        # lambertw returns complex types but k=0 branch is real valued for all z>-1/e so can safely ignore imaginary part
        return ((temp_guess - sim_temp)/(beta)).real

    ## FRM funcs ##
    def _k(sys,site:int,rxn:int,E_a:np.ndarray,A:np.ndarray,E_BEP:np.ndarray,time:float)->np.ndarray:
        return A[site,rxn]*np.exp(-(E_a[site,rxn]+E_BEP[site,rxn])/(k_B*sys.T(time)))
    
    def _FRM_generate_queue(sys,lattice:np.ndarray,E_a:np.ndarray,A:np.ndarray,E_BEP:np.ndarray,guess_method:str):
        for site in range(len(lattice[:,0])):
            site_rxns = sys.species_rxns[lattice[site,1]]
            for rxn in site_rxns:
                if bool(sys._check_allowed(site,rxn,lattice)):
                    rxn_time = sys._t_gen(sys._FRM_site_prop,0,sys.rng.random(),(rxn,lattice,site,E_a,A,E_BEP),method=guess_method)
                    sys._FRM_insert((rxn_time,site,rxn))

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
        c = sys._check_allowed(site,rxn,lattice)
        k = sys._k(site,rxn,E_a,A,E_BEP,time)
        return  k*c
    
    def _FRM_update(sys,time:float,site:int,new_site:int,E_a:np.ndarray,A:np.ndarray,E_BEP:np.ndarray,lattice:np.ndarray,guess_method:str):
        # update lateral interactions
        E_BEP = sys._lateral_interactions_update(E_BEP,lattice,site,new_site)
        # site to update
        to_update = set(sys.neighbour_key[site,:])
        to_update.add(site)
        if new_site != site: to_update.update(sys.neighbour_key[new_site])
        for s in to_update:
            # remove old reactions
            to_remove = sys.FRM_site_keys[s].copy()
            for ID in to_remove:
                sys._FRM_remove(ID)
            # add new reactions
            site_rxns = sys.species_rxns[lattice[s,1]]
            for rxn in site_rxns:
                if bool(sys._check_allowed(s,rxn,lattice)):
                    rxn_time = sys._t_gen(sys._FRM_site_prop,time,sys.rng.random(),(rxn,lattice,s,E_a,A,E_BEP),method=guess_method)
                    sys._FRM_insert((rxn_time,s,rxn))
        return E_BEP

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

    def run_DM(sys,guess='TI',report=False):
        """Runs a kinetic Monte Carlo simulation on the defined lattice \n
        Uses the Direct method \n
        Returns a dict of time, temp, coverage and desorption rate \n
        dataframe labels of the form (e.g. run X): \n
        'timeX','tempX','thetaX','rateX'
        """
        data = {}
        for run in range(sys.runs):
            #Initialise
            lat = sys.lat.copy()
            E_a,A,E_BEP = sys.E_a.copy(),sys.A.copy(),sys.E_BEP.copy()
            c = sys._DM_gen_c_array(lat)
            t,n,site,new_site,plot_ind=0.0,0,0,0,0
            counter = sys.counter.copy()
            times = np.array([np.nan]*(sys.t_points))
            temps = times.copy()
            pop_dict = {}
            for i in range(14): pop_dict[(0,i)] = times.copy(); pop_dict[(1,i)] = times.copy()
            while t<sys.t_max and n<sys.n_max:
                # Generate next time
                new_t = sys._t_gen(sys._DM_total_prop,t,sys.rng.random(),other_args=(c,E_a,A,E_BEP),method=guess)
                # Save state
                next_save = (t-t%sys.t_step + sys.t_step) if t!=0 else 0
                while next_save<new_t and plot_ind<sys.t_points:
                    # save values of interest
                    times[plot_ind] = next_save
                    temps[plot_ind] = sys.T(next_save)
                    for i in range(14):
                        pop_dict[(0,i)][plot_ind] = counter[0,i]
                        pop_dict[(1,i)][plot_ind] = counter[1,i]
                    next_save += sys.t_step # next time to save
                    plot_ind += 1 # next grid point
                # Advance system time
                t = new_t
                # Global prop gen
                a_acc = sys._DM_get_prop_array(c,E_a,A,E_BEP,t)
                if a_acc[-1] == 0: print('Reactions complete (total_propensity = 0)'); break
                # Choose reaction
                mu_index = np.searchsorted(a_acc,a_acc[-1]*sys.rng.random(),side='left') # binary search
                rxn_index = mu_index % len(sys.E_a[0,:])
                site = mu_index // len(sys.E_a[0,:])
                # Advance system state
                lat,new_site,counter = sys._rxn_step(lat,site,rxn_index,counter)
                # Local occ and lateral interactions change
                c = sys._DM_c_change(lat,c,site,new_site)
                E_a,A = sys._kinetic_param_update(lat,E_a,A,site,new_site)
                E_BEP = sys._lateral_interactions_update(E_BEP,lat,site,new_site)
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
        return data
    
    def run_FRM(sys,guess='FRM',report=False):
        """Runs a kinetic Monte Carlo simulation on the defined lattice \n
        Uses the First reaction method \n
        Returns a dict of time, temp, coverage and desorption rate \n
        dataframe labels of the form (e.g. run X): \n
        'timeX','tempX','thetaX','rateX'
        """
        data = {}
        for run in range(sys.runs):
            lat = sys.lat.copy()
            E_a,A,E_BEP = sys.E_a.copy(),sys.A.copy(),sys.E_BEP.copy()
            t,n,plot_ind=0.0,0,0
            counter = sys.counter.copy()
            times = np.array([np.nan]*(sys.t_points))
            temps = times.copy()
            pop_dict = {}
            for i in range(14): pop_dict[(0,i)] = times.copy(); pop_dict[(1,i)] = times.copy()
            # Initialise data structure
            sys._FRM_generate_queue(lat,E_a,A,E_BEP,guess)
            while t<sys.t_max and n<sys.n_max:
                # Choose reaction and time
                if len(sys.FRM_sortlist)==0:print('Reactions complete (reaction queue empty)'); break
                new_t,site,rxn = sys.FRM_sortlist[0]
                # Save state
                next_save = (t-t%sys.t_step + sys.t_step) if t!=0 else 0
                while next_save<new_t and plot_ind<sys.t_points:
                    # save values of interest
                    times[plot_ind] = next_save
                    temps[plot_ind] = sys.T(next_save)
                    for i in range(14): # all species + desoprtion counters
                        pop_dict[(0,i)][plot_ind] = counter[0,i]
                        pop_dict[(1,i)][plot_ind] = counter[1,i]
                    next_save += sys.t_step # next time to save
                    plot_ind += 1 # next grid point
                t = new_t
                # Advance state and update queue + lateral interactions
                lat,new_site,counter = sys._rxn_step(lat,site,rxn,counter)
                E_a,A = sys._kinetic_param_update(lat,E_a,A,site,new_site)
                E_BEP = sys._FRM_update(t,site,new_site,E_a,A,E_BEP,lat,guess)
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
        for run in range(sys.runs):
            #Initialise
            lat = sys.lat.copy()
            E_a,A,E_BEP = sys.E_a.copy(),sys.A.copy(),sys.E_BEP.copy()
            c = sys._DM_gen_c_array(lat)
            t,n=0.0,0
            while t<sys.t_max and n<sys.n_max:
                # Generate next time
                new_t = sys._t_gen(sys._DM_total_prop,t,sys.rng.random(),other_args=(c,E_a,A,E_BEP),method=guess)
                # Advance system time
                t = new_t
                # Global prop gen
                a_acc = sys._DM_get_prop_array(c,E_a,A,E_BEP,t)
                if a_acc[-1] == 0: print('Reactions complete (total_propensity = 0)'); break
                # Choose reaction
                mu_index = np.searchsorted(a_acc,a_acc[-1]*sys.rng.random(),side='left') # binary search
                rxn_index = mu_index % len(sys.E_a[0,:])
                site = mu_index//len(sys.E_a[0,:])
                # Advance system state
                lat,new_site,_ = sys._rxn_step(lat,site,rxn_index,sys.counter.copy())
                # Local occ and lateral interactions change
                c = sys._DM_c_change(lat,c,site,new_site)
                E_a,A = sys._kinetic_param_update(lat,E_a,A,site,new_site)
                E_BEP = sys._lateral_interactions_update(E_BEP,lat,site,new_site)
                n += 1
        return
    
    def run_FRM_no_data(sys,guess='FRM'):
        """Runs a kinetic Monte Carlo simulation on the defined lattice \n
        Uses the First reaction method \n
        Returns a 4*runs column dataframe of time, temp, coverage and desorption rate \n
        dataframe labels of the form (e.g. run X): \n
        'timeX','tempX','thetaX','rateX'
        """
        for run in range(sys.runs):
            lat = sys.lat.copy()
            E_a,A,E_BEP = sys.E_a.copy(),sys.A.copy(),sys.E_BEP.copy()
            t,n=0.0,0
            # Initialise data structure
            sys._FRM_generate_queue(lat,E_a,A,E_BEP,guess)
            while t<sys.t_max and n<sys.n_max:
                # Choose reaction and time
                if len(sys.FRM_sortlist)==0:print('Reactions complete (reaction queue empty)'); break
                new_t,site,rxn = sys.FRM_sortlist[0]
                t = new_t
                # Advance state and update queue + lateral interactions
                lat,new_site,_ = sys._rxn_step(lat,site,rxn,sys.counter.copy())
                E_a,A = sys._kinetic_param_update(lat,E_a,A,site,new_site)
                E_BEP = sys._FRM_update(t,site,new_site,E_a,A,E_BEP,lat,guess)
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

    def what_params(params,see_lattice=False):
        print(f'Reaction channels = {len(params.E_a[0,:])}')
        print(f'Simulation: t_max={params.t_max}, n_max={params.n_max}, grid_points={params.t_points}, runs={params.runs}')
        if see_lattice: print(f'Initial lattice:\n{params.lat}')

    def what_coverages(self,lattice=None):
        if type(lattice) != np.ndarray: lattice = self.lat
        max_id = max(lattice[:,1])+1
        counts = np.zeros((max_id),dtype=int)
        for site in lattice[:,1]:
            counts[site] += 1
        counts = counts/len(lattice[:,1])
        thetas = {i:counts[i] for i in range(max_id)}
        print('{Species : fractional coverage} key is ...')
        print(thetas)
    
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