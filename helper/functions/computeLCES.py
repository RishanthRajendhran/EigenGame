from helper.imports.mainImports import *
from helper.imports.functionImports import *
import helper.config.mainConfig as config
import helper.config.gradientAscentConfig as gaConfig
import helper.config.thresholdConfig as thConfig

#Function to return the length of the longest correct eigenvectors streak (LCES)
#Inputs
#   X   -   Input data matrix
#Outputs
#   Returns an array, of length equal to number of iterations of the eigengame played,
#   containing the LCES at every iteration of the game. This array also gets saved as 
#   "LCES_<config.variant>.npy" in the current working directory for purpose of future analysis. 
def computeLCES(X):
    Vs = np.load(f"Vs_{config.variant}_{gaConfig.ascentVariant}.npy")
    EVs = np.around(getEigenVectors(X),decimals=3)
    EVs = rearrange(EVs, Vs[-1])
    print("EigenVectors obtained through EigenGame:")
    print(np.around(Vs[-1],decimals=3))
    print("\nEigenVectors obtained through numpy:")
    print(np.around(EVs,decimals=3))
    E = rearrange(EVs, Vs[-1])
    streakCounts = []
    for t in range(Vs.shape[0]):
        curStreak = 0
        V = Vs[t]
        for i in range(E.shape[1]):
            if getAngle(E[:,i], V[:,i]) <= thConfig.angularThreshold or getAngle(E[:,i], -V[:,i]) <= thConfig.angularThreshold:
                curStreak += 1
            else:
                break 
        streakCounts.append(curStreak)
    if "-continueEigenGame" in sys.argv:
        LCES_old = np.load(f"LCES_{config.variant}_{gaConfig.ascentVariant}.npy")
        np.append(LCES_old, np.array(streakCounts), 0)
        np.save(f"LCES_{config.variant}_{gaConfig.ascentVariant}.npy", LCES_old)  
        iterTimes = np.load(f"./iterTimes_{config.variant}_{gaConfig.ascentVariant}.npy")
        plt.plot(iterTimes, LCES_old)
        plt.xlabel("Time elapsed (s)")
        plt.ylabel("LCES")
        plt.title(f"Variant {config.variant} ({gaConfig.ascentVariant}): lr = {gaConfig.learningRate}, xDim = {config.xDim}, k = {config.k},L = {config.L}, T = {gaConfig.numIterations}")
        if "-savePlots" in sys.argv or "-saveMode" in sys.argv:
            plt.savefig(f"./LCES/LCES_{config.variant}_{gaConfig.ascentVariant}")
        if "-saveMode" not in sys.argv:
            plt.show()
        return LCES_old 
    else:
        np.save(f"LCES_{config.variant}_{gaConfig.ascentVariant}.npy", streakCounts) 
        iterTimes = np.load(f"./iterTimes_{config.variant}_{gaConfig.ascentVariant}.npy")
        plt.plot(iterTimes, streakCounts)   
        plt.xlabel("Time elapsed (s)")
        plt.ylabel("LCES")
        plt.title(f"Variant {config.variant} ({gaConfig.ascentVariant}): lr = {gaConfig.learningRate}, xDim = {config.xDim}, k = {config.k},L = {config.L}, T = {gaConfig.numIterations}")
        if "-savePlots" in sys.argv or "-saveMode" in sys.argv:
            plt.savefig(f"./LCES/LCES_{config.variant}_{gaConfig.ascentVariant}")
        if "-saveMode" not in sys.argv:
            plt.show()
        return streakCounts