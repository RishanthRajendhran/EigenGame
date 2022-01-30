from helper.imports.mainImports import *
from helper.imports.functionImports import *

#Function to find eigenvectors of a given X using Numpy
#Inputs 
#   X   - Numpy array of the dataset
#Outputs
#   Returns all the EigenVectors of V as a matrix of dimensions X.shape[1] x X.shape[1] 
#   in the order of decreasing EigenValues
def getEigenVectors(X):
    E, V = np.linalg.eig(np.dot(X.T, X))
    return V[:,np.argsort(E)[::-1]]