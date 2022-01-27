from helper.imports.mainImports import *
from helper.imports.functionImports import *

#Function to get the angle (in degrees) between two vectors u and v 
#Inputs 
#   u       - Numpy column array
#   v       - Numpy column array
#Outputs
#   Returns angle between the vectors in degree measure 
def getAngle(u, v):
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    return np.rad2deg(np.arccos(np.clip(np.dot(u,v),-1.0,1.0)))