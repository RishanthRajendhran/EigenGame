from helper.imports.mainImports import *
from helper.imports.functionImports import *
import helper.config.mainConfig as config
import helper.config.gradientAscentConfig as gaConfig

#UNDER CONSTRUCTION
#Function to plot the angle between the subspace spanned by
#the players and the subspace spanned by the expected final
#positions of the players (i.e. subspace spanned by the 
#EigenVectors) against number of iterations and time elapsed
#Inputs
#   X   -   Input data matrix
#Outputs
#   Shows the plots to the user
#   Returns nothing 
def analyseSubspaceAngles(X):
    Vs = np.load(f"Vs_{config.variant}_{gaConfig.ascentVariant}.npy")
    iterTimes = np.load(f"iterTimes_{config.variant}_{gaConfig.ascentVariant}.npy")
    EVs = np.around(getEigenVectors(X),decimals=3)
    EVs = rearrange(EVs, Vs[-1])
    print("EigenVectors obtained through EigenGame:")
    diffs = []
    for V in Vs:
        diffs.append(getDistance(V,EVs))
        if "-debug" in sys.argv:
            print(np.around(V,decimals=3))
    if "-debug" not in sys.argv:
        print(np.around(Vs[-1],decimals=3))
    print("\nEigenVectors obtained through numpy:")
    print(np.around(EVs,decimals=3))
    EVs = rearrange(EVs, Vs[-1])

    angles = []
    for t in range(len(Vs)):
        # print(Vs[t])
        angle = np.sum((subspace_angles(Vs[t][:,:2], EVs[:,1:])))
        # if angle < 10**-5:
        #     angle = 0
        angles.append(angle)
        # if t>20:
        #     continue
        # print(Vs[t][:,:2])
        # print(EVs[:,1:])
        # print(np.rad2deg(subspace_angles(Vs[t], EVs)))
    np.save(f"./subspaceAngles_{config.variant}_{gaConfig.ascentVariant}.npy",angles)
    pltTitle = f"Variant {config.variant} ({gaConfig.ascentVariant}): lr = {gaConfig.learningRate}, xDim = {config.xDim}, k = {config.k},L = {config.L}, T = {gaConfig.numIterations}"
    plt.xlabel("Iterations")
    plt.ylabel("Subspace Angle between obtained EV and expected EV")
    plt.title(pltTitle)
    plt.plot(np.arange(len(angles)), angles)
    if "-savePlots" in sys.argv or "-saveMode" in sys.argv:
        plt.savefig(f"./plots/subspaceAnglesVSiterations_{config.variant}_{gaConfig.ascentVariant}")
    if "-saveMode" not in sys.argv:
        plt.show()

    plt.xlabel("Total Time Elapsed")
    plt.ylabel("Angle between obtained EV and expected EV")
    plt.title(pltTitle)
    plt.plot(iterTimes, angles)
    if "-savePlots" in sys.argv or "-saveMode" in sys.argv:
        plt.savefig(f"./plots/subspaceAnglesVStotalTimeElapsed_{config.variant}_{gaConfig.ascentVariant}")
    if "-saveMode" not in sys.argv:
        plt.show()



