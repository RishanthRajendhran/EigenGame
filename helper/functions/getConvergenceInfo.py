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

    config.stopIterations = [config.L]*config.k
    convergenceTime = 0
    if "a" in config.variant:
        modifiedVs = []
        modifiedTs = []
        config.stopIterations = [config.L]*config.k
        for pos in range(config.k):
            newVs = []
            newTs = []
            for i in range(pos*config.L, (pos+1)*config.L+1, 25):
                if i >= len(Vs):
                    break
                V = Vs[i][:,pos].copy()
                iT = iterTimes[i].copy() - iterTimes[pos*config.L]
                newVs.append(V.copy())
                newTs.append(iT.copy())
                modifiedVs.append(Vs[i].copy())
                modifiedTs.append(iT.copy())
                distanceMeasure = np.around(fns.getDistance(V,EVs[:,pos]), decimals=3)
                if distanceMeasure <= thConfig.distanceTolerance/config.k:
                    convergenceTime = max(convergenceTime, iterTimes[i])
                    config.stopIterations[pos] = i + 1
                    break 
            newVs = np.array(newVs)
            newTs = np.array(newTs)
            np.save(f"Vs{pos}_{config.variant}_{gaConfig.ascentVariant}.npy", newVs)
            np.save(f"iterTimes{pos}_{config.variant}_{gaConfig.ascentVariant}.npy", newTs)
        modifiedVs = np.array(modifiedVs)
        modifiedTs = np.array(modifiedTs)
        np.save(f"Vs_modified_{config.variant}_{gaConfig.ascentVariant}.npy", modifiedVs)
        np.save(f"iterTimes_modified_{config.variant}_{gaConfig.ascentVariant}.npy", modifiedTs)
        config.stopIteration = np.max(config.stopIterations)
    elif "b" in config.variant:
        modifiedVs = Vs.copy()
        modifiedTs = iterTimes.copy()
        config.stopIterations = [len(Vs)]*config.k
        for pos in range(config.k):
            newVs = []
            newTs = []
            skippedIterationCount = 0
            for i in range(pos, len(Vs), config.k):
                if i >= len(Vs):
                    break
                if skippedIterationCount%25 != 0:
                    skippedIterationCount += 1
                    continue 
                skippedIterationCount += 1
                V = Vs[i][:,pos].copy()
                iT = iterTimes[i].copy()
                newVs.append(V.copy())
                newTs.append(iT)
                distanceMeasure = np.around(fns.getDistance(V,EVs[:,pos]), decimals=3)
                if distanceMeasure <= thConfig.distanceTolerance/config.k:
                    convergenceTime = max(convergenceTime, iterTimes[i])
                    config.stopIterations[pos] = i + 1
                    break 
            newVs = np.array(newVs)
            newTs = np.array(newTs)
            np.save(f"Vs{pos}_{config.variant}_{gaConfig.ascentVariant}.npy", newVs)
            np.save(f"iterTimes{pos}_{config.variant}_{gaConfig.ascentVariant}.npy", newTs)
        config.stopIteration = np.max(config.stopIterations)
        if config.stopIteration < len(Vs):
            modifiedVs = Vs[:config.stopIteration].copy()
            modifiedTs = iterTimes[:config.stopIteration].copy()
        np.save(f"Vs_modified_{config.variant}_{gaConfig.ascentVariant}.npy", modifiedVs)
        np.save(f"iterTimes_modified_{config.variant}_{gaConfig.ascentVariant}.npy", modifiedTs)
    elif "c" in config.variant:
        config.stopIterations = [len(Vs)]*config.k
        for pos in range(config.k):
            newVs = []
            newTs = []
            lastStopTime = 0
            breakOut = False
            for i in range(pos*config.L, len(Vs), config.k*config.L):
                if breakOut:
                    break
                for j in range(0, config.L, 25):
                    V = Vs[i+j][:, pos].copy()
                    iT = iterTimes[i+j] - iterTimes[i] + lastStopTime
                    newVs.append(V.copy())
                    newTs.append(iT)
                    distanceMeasure = np.around(fns.getDistance(V,EVs[:,pos]), decimals=3)
                    if distanceMeasure <= thConfig.distanceTolerance/config.k:
                        convergenceTime = max(convergenceTime, iterTimes[i+j])
                        config.stopIterations[pos] = i + j + 1
                        breakOut = True
                        break
                lastStopTime = iterTimes[i+config.L]
            newVs = np.array(newVs)
            newTs = np.array(newTs)
            np.save(f"Vs{pos}_{config.variant}_{gaConfig.ascentVariant}.npy", newVs)
            np.save(f"iterTimes{pos}_{config.variant}_{gaConfig.ascentVariant}.npy", newTs)
        skipIt = [False]*config.k 
        modifiedVs = []
        modifiedTs = []
        lastStopTime = 0
        config.stopIteration = np.max(config.stopIterations)
        for i in range(0, min(config.stopIteration, len(Vs)), config.k*config.L):
            for pos in range(config.k):
                if skipIt[pos]:
                    lastStopTime = iterTimes[i + (pos+1)*config.L] 
                    continue
                for j in range(config.L):
                    modifiedVs.append(Vs[i + pos*config.L + j])
                    modifiedTs.append(iterTimes[i + pos*config.L + j] - (iterTimes[i + pos*config.L] - lastStopTime))
                    if i + pos*config.L + j > config.stopIterations[pos]:
                        skipIt[pos] = True
                        lastStopTime = iterTimes[i + pos*config.L + j] 
                        break 
                if not skipIt[pos]:
                    lastStopTime = iterTimes[i + (pos+1)*config.L] 

        modifiedVs = np.array(modifiedVs)
        modifiedTs = np.array(modifiedTs)
        np.save(f"Vs_modified_{config.variant}_{gaConfig.ascentVariant}.npy", modifiedVs)
        np.save(f"iterTimes_modified_{config.variant}_{gaConfig.ascentVariant}.npy", modifiedTs)
    if convergenceTime == 0:
        convergenceTime = None
    return convergenceTime, distanceMeasure
    