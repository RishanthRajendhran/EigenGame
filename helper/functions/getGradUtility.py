from helper.imports.mainImports import *
from helper.imports.functionImports import *

#Function to return the penalty
#Inputs
#   X       - Numpy array of the dataset
#   reward  - Reward term
#   penalty - Penalty term
#Outputs
#   Returns a vector gradient of dimension X.shape[1]  x 1
def getGradUtility(X, reward, penalty):
    return 2*np.dot(X.T,(reward - penalty))