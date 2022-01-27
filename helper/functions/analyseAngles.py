from helper.imports.mainImports import *
from helper.imports.functionImports import *
import helper.config.mainConfig as config
import helper.config.gradientAscentConfig as gaConfig

#Function to plot cosine of the angle between each player and the corresponsing 
#EigenVector against number of iterations and time elapsed
#Inputs
#   X   -   Input data matrix
#Outputs
#   Shows the plots to the user
#   Returns nothing 
def analyseAngles(X):
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
    for col in range(Vs[0].shape[1]):
        angle = []
        for t in range(len(Vs)):
            curV = Vs[t][:,col]
            angle.append((np.dot(np.transpose(curV),EVs[:,col])/(np.linalg.norm(curV)*np.linalg.norm(EVs[:,col]))))
        angles.append(angle)
    angles = np.array(angles)
    np.save(f"./angles_{config.variant}_{gaConfig.ascentVariant}.npy",angles)
    pltTitle = f"Variant {config.variant} ({gaConfig.ascentVariant}): lr = {gaConfig.learningRate}, xDim = {config.xDim}, k = {config.k},L = {config.L}, T = {gaConfig.numIterations}"
    for i in range(len(angles)):
        plt.xlabel("Iterations")
        plt.ylabel("Cosine of the angle between obtained EV and expected EV")
        plt.title(pltTitle)
        plt.plot(np.arange(len(angles[i])), angles[i], color="C"+str(i))
        if "-analyseAnglesTogether" not in sys.argv:
            if "-savePlots" in sys.argv or "-saveMode" in sys.argv:
                plt.savefig(f"./plots/anglesVSiterations{i}_{config.variant}_{gaConfig.ascentVariant}")
            if "-saveMode" not in sys.argv:
                plt.show()
    if "-analyseAnglesTogether" in sys.argv:
        if "-savePlots" in sys.argv or "-saveMode" in sys.argv:
            plt.savefig(f"./plots/anglesVSiterations_{config.variant}_{gaConfig.ascentVariant}")
        if "-saveMode" not in sys.argv:
            plt.show()
    
    for i in range(len(angles)):
        plt.xlabel("Total Time Elapsed")
        plt.ylabel("Cosine of the angle between obtained EV and expected EV")
        plt.title(pltTitle)
        plt.plot(iterTimes, angles[i], color="C"+str(i))
        if "-analyseAnglesTogether" not in sys.argv:
            if "-savePlots" in sys.argv or "-saveMode" in sys.argv:
                plt.savefig(f"./plots/anglesVStotalTimeElapsed{i}_{config.variant}_{gaConfig.ascentVariant}")
            if "-saveMode" not in sys.argv:
                plt.show()
    if "-analyseAnglesTogether" in sys.argv:
        if "-savePlots" in sys.argv or "-saveMode" in sys.argv:
            plt.savefig(f"./plots/anglesVStotalTimeElapsed_{config.variant}_{gaConfig.ascentVariant}")
        if "-saveMode" not in sys.argv:
            plt.show()
