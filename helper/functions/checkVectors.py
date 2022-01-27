from helper.imports.mainImports import *
from helper.imports.functionImports import *
import helper.config.thresholdConfig as thConfig

#Function to check is current player is close to previous players 
#If it is the case, reinitialise current player and return the new set of player positions
#Inputs 
#   V           - Numpy array whose columns should eventually represent the eigenbectors
#   curPlayer   - Index of eign vector in consideration
#   oldPos      - Position of current player before last update
#Outputs
#   Returns updated numpy array of eigenvectors 
def checkVectors(V, curPlayer, oldPos):
    for i in range(V.shape[1]):
        if "-symmetric" in sys.argv:
            if i != curPlayer and 0 <= getAngle(V[:,i], V[:, curPlayer]) <= thConfig.tolerance and 180-thConfig.tolerance <= getAngle(V[:,i], V[:, curPlayer]) <= 180:
                V[:,curPlayer] = -oldPos
                break
        else:
            if i < curPlayer and 0 <= getAngle(V[:,i], V[:, curPlayer]) <= thConfig.tolerance and 180-thConfig.tolerance <= getAngle(V[:,i], V[:, curPlayer]) <= 180:
                V[:,curPlayer] = -oldPos
                break
    return V