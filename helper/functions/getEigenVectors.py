from helper.imports.mainImports import *
from helper.imports.functionImports import *

#Function to find eigenvectors of a given X using Numpy
#Inputs 
#   X   - Numpy array of the dataset
#Outputs
#   Returns all the eigenvectors of V as a matrix of dimensions X.shape[1] x X.shape[1]
def getEigenVectors(X):
    return np.linalg.eig(np.dot(X.T, X))[1]