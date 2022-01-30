from helper.imports.mainImports import *
from helper.imports.functionImports import *
import helper.config.mainConfig as config
import helper.config.gradientAscentConfig as gaConfig
import helper.config.miscellaneousConfig as mlConfig

#Function to return the penalty
#Inputs
#   X   - Numpy array of the dataset
#   V   - Numpy array whose columns should eventually represent the eigenbectors of X
#   i   - Index of eign vector in consideration
#Outputs
#   Returns a vector penalty of dimension X.shape[0] x 1
def getPenalty(X, V, i):
    M  = np.dot(X.T, X)
    penalty = np.zeros((X.shape[0], 1))
    for j in range(config.k):
        condition = j < i
        if "-symmetric" in sys.argv:
            condition = j != i
        if condition:
            dotProd = (np.dot(np.dot(X, V[:,i]), np.dot(X, V[:,j]))/np.dot(np.dot(X, V[:,j]), np.dot(X, V[:,j])))*np.dot(X,V[:,j]).reshape(-1,1)
            if mlConfig.hasConverged[j] > gaConfig.T/100:
                penalty += gaConfig.extraPenaltyCoefficient*dotProd
            else:
                penalty += gaConfig.penaltyCoefficient*dotProd
    return penalty.reshape(-1, 1)

    