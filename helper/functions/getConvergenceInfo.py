from helper.imports.mainImports import *
import helper.imports.functionImports as fns
import helper.config.mainConfig as config
import helper.config.gradientAscentConfig as gaConfig
import helper.config.thresholdConfig as thConfig

#Function to find the time taken by the last game to converge 
#and distance measure at convergence
#In case of no convergence, returns None
#Inputs
#   Nothing
#Outputs
#   Returns time taken to converge/None and the distance 
#   measure at that convergence 
def getConvergenceInfo():
    X = np.load(f"X_{config.xDim}.npy")
    Vs = np.load(f"Vs_{config.variant}_{gaConfig.ascentVariant}.npy")
    iterTimes = np.load(f"iterTimes_{config.variant}_{gaConfig.ascentVariant}.npy")
    EVs = np.around(fns.getEigenVectors(X),decimals=3)
    EVs = fns.rearrange(EVs, Vs[-1])
    convergenceTime = None
    config.stopIteration = len(Vs)
    for i in range(len(Vs)):
        V = Vs[i]
        distanceMeasure = np.around(fns.getDistance(V,EVs), decimals=3)
        if distanceMeasure <= thConfig.distanceTolerance:
            convergenceTime = iterTimes[i]
            config.stopIteration = i
            break 
    return convergenceTime, distanceMeasure
    