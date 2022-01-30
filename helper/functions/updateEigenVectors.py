from helper.imports.mainImports import *
from helper.imports.functionImports import *
import helper.config.gradientAscentConfig as gaConfig
import helper.config.miscellaneousConfig as mlConfig

#Function to update eigenvectors 
#Inputs 
#   X       - Numpy array of the dataset
#   V       - Numpy array whose columns should eventually represent the eigenbectors of X
#   i       - Index of eign vector in consideration
#   reward  - Reward term
#   penalty - Penalty term
#   alpha   - step size of updates
#Outputs
#   Returns updated numpy array of eigenvectors 
def updateEigenVectors(X, V, i, reward, penalty, alpha=gaConfig.learningRate):
    gradV = getGradUtility(X, reward, penalty)
    gradV = gradV - np.dot(gradV.T, V[:,i])*V[:,i].reshape(gradV.shape[0],1)
    oldVi = V[:,i].copy()
    V[:,i] = V[:,i] + alpha*gradV.reshape(gradV.shape[0],)
    V[:,i] = V[:,i]/np.sqrt(np.sum(V[:,i]**2))
    mlConfig.hasConverged[i] += -1 + 2*np.all(np.isclose(V[:,i], oldVi))
    if "-checkVectors" in sys.argv:
        V = checkVectors(V, i, oldVi)
    return V