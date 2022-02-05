from helper.imports.mainImports import *
import helper.imports.functionImports as fns
import helper.config.mainConfig as config
import helper.config.gradientAscentConfig as gaConfig
import helper.config.thresholdConfig as thConfig

#Function to plot sum of euclidean distance between each player and the corresponsing 
#EigenVector against number of iterations and time elapsed
#Inputs
#   X   -   Input data matrix
#Outputs
#   Shows the plots to the user
#   Returns nothing 
def analyseResults(X):
    Vs = np.load(f"Vs_{config.variant}_{gaConfig.ascentVariant}.npy")
    iterTimes = np.load(f"iterTimes_{config.variant}_{gaConfig.ascentVariant}.npy")

    #The following imports should be avoided as these files do not quite capture the 
    #ways in which the different variants progress, in the sense that these files
    #might not show the redundant steps made by the players after they have converged
    #which is an artifact of the variant (a/b/c) and such plots might not be useful
    #to make comparisons between variants
    # Vs = np.load(f"Vs_modified_{config.variant}_{gaConfig.ascentVariant}.npy")
    # iterTimes = np.load(f"iterTimes_modified_{config.variant}_{gaConfig.ascentVariant}.npy")
    
    EVs = np.around(fns.getEigenVectors(X),decimals=3)
    EVs = fns.rearrange(EVs, Vs[-1])
    print("EigenVectors obtained through EigenGame:")
    diffs = []
    for V in Vs:
        diffs.append(fns.getDistance(V,EVs))
        if "-debug" in sys.argv:
            print(np.around(V,decimals=3))
    if "-debug" not in sys.argv:
        print(np.around(Vs[-1],decimals=3))
    print("\nEigenVectors obtained through numpy:")
    print(np.around(EVs,decimals=3))

    plt.plot(diffs[:min(config.stopIteration, len(diffs))])
    plt.xlabel("Iterations")
    plt.ylabel("Distance")
    plt.title(f"Variant {config.variant} ({gaConfig.ascentVariant}): lr = {gaConfig.learningRate}, xDim = {config.xDim}, k = {config.k},L = {config.L}, T = {gaConfig.numIterations}")
    if "-savePlots" in sys.argv or "-saveMode" in sys.argv:
        plt.savefig(f"{fns.getLocation('./plots')}distanceVSiterations_{config.variant}_{gaConfig.ascentVariant}")
    if "-saveMode" not in sys.argv:
        plt.show()
    plt.clf()
    plt.plot(iterTimes[:min(config.stopIteration, len(diffs))], diffs[:min(config.stopIteration, len(diffs))])
    plt.xlabel("Time elapsed (s)")
    plt.ylabel("Distance")
    plt.title(f"Variant {config.variant} ({gaConfig.ascentVariant}): lr = {gaConfig.learningRate}, xDim = {config.xDim}, k = {config.k},L = {config.L}, T = {gaConfig.numIterations}")
    if "-savePlots" in sys.argv or "-saveMode" in sys.argv:
        plt.savefig(f"{fns.getLocation('./plots')}distanceVStimeElapsed_{config.variant}_{gaConfig.ascentVariant}")
    if "-saveMode" not in sys.argv:
        plt.show()
    plt.clf()