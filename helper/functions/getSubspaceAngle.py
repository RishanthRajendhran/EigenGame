from helper.imports.mainImports import *
from helper.imports.functionImports import *

#Function to return the subspace angle 
# between current player positions (current eigenvalues) 
# and expected final player positions (i.e. expected eigenvalues)
#Inputs
#   V   - Numpy array of Current player positions
#   EVs - Numpy array of Expected final player positions
#   i   - Current player
#Outputs
#   Returns a scalar angular measure
def getSubspaceAngle(V, EVs):
    return np.sum(subspace_angles(V[:,:i+1], EVs[:, :i+1]))