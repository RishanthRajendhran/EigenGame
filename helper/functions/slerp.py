from helper.imports.mainImports import *

#Function to return the spherical linear interpolation given two points on a unit sphere
#Further reading: https://en.wikipedia.org/wiki/Slerp
#Inputs
#   p1   - First point on the unit sphere
#   p2   - Second point on the unit sphere
#Outputs
#   Returns the slerp
def slerp(p1, p2):
    t = np.linspace(0, 1, 30)
    omega = np.arccos( min(1.0 ,p1.dot(p2)) )
    sin_omega = max(0.0001, np.sin(omega))
    t = t[:, np.newaxis]
    return ( np.sin( (1-t)*omega )*p1 + np.sin( t*omega )*p2 )/sin_omega
