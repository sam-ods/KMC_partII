## Inputs
These simulations expect kinetic parameter inputs as numpy arrays.
These will be calculated via DFT for investigations of reaction networks, but for model testing and benchmarking any sensible value can be used.
Reference arrays of this size for each system are supplied in this folder.
Expected dimesions and corresponding reactions for SAA systems are as follows:
#### TPD
- E_a,A,w = (3,5) rows = Cu and Pt site and columns for (ads,des,self - site 0/1/2 diff)
- J = (3,3,2,2), 1st ind for species_i site type, 2nd for species_j, others for species_i and species_j
#### PtCu
- E_a,A,w = (3,3,10,7) 1st ind for initial site type, 2nd for final site type, 3rd for species, 4th for rxn
- forms of rxn should be (diff,H-gain,H-loss,desorption) (w/ H,OH,OH2 or *,O,OH for H gain/loss of C species)
- J = (3,3,10,10), 1st ind for initial site type, 2nd for final site type, others for species_i and species_j
  Misc notes for PtCu:
  - H gain/loss are rxns of the C/O species not H
  - bimolecular H/O desorption rates are corrected by a factor of 0.5 since these rxns are double counted
  - for C rxns are: (diff, H-gain w/ H,OH,OH2 ,C+O coupling)
  - for CO rxns are: (diff, C-O breaking, desoprtion)
e.g. for CH3 (diff,H-gain+*,H-gain+OH,H-gain+OH2,H-loss+*,H-loss+O,H-loss+OH)
#### Some simplifications
- Since PtCu is monatomic Pt on Cu, we know Pt - bulk Cu neighbours never exist so these parameters can be set to zero
- can approximate boundary Cu as bulk and ignore directionality by repeating entries in the 1st/2nd indices in some cases
