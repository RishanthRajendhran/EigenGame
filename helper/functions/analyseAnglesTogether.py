from helper.imports.mainImports import *
import helper.imports.functionImports as fns
import helper.config.mainConfig as config
import helper.config.gradientAscentConfig as gaConfig

#Function to plot cosine of the angle between each player and the corresponsing 
#EigenVector against number of iterations and time elapsed all in a single plot
#Inputs
#   X   -   Input data matrix
#Outputs
#   Shows the plots to the user
#   Returns nothing 
def analyseAnglesTogether(X):
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
    if "-postGameAnalysis" not in sys.argv:
        print("EigenVectors obtained through EigenGame:")
    diffs = []
    for V in Vs:
        diffs.append(fns.getDistance(V,EVs))
        if "-debug" in sys.argv and "-postGameAnalysis" not in sys.argv:
            print(np.around(V,decimals=3))
    if "-debug" not in sys.argv and "-postGameAnalysis" not in sys.argv:
        print(np.around(Vs[-1],decimals=3))
    if "-postGameAnalysis" not in sys.argv: 
        print("\nEigenVectors obtained through numpy:")
        print(np.around(EVs,decimals=3))

    angles = []
    for col in range(Vs[0].shape[1]):
        angle = []
        for t in range(min(config.stopIteration, len(Vs))):
            curV = Vs[t][:,col]
            angle.append((np.dot(np.transpose(curV),EVs[:,col])/(np.linalg.norm(curV)*np.linalg.norm(EVs[:,col]))))
        angles.append(angle)
    np.save(f"{fns.getLocation('./angles')}angles_{config.variant}_{gaConfig.ascentVariant}.npy",angles)
    pltTitle = f"Variant {config.variant} ({gaConfig.ascentVariant}): lr = {gaConfig.learningRate}, xDim = {config.xDim}, k = {config.k},L = {config.L}, T = {gaConfig.numIterations}"
    for i in range(len(angles)):
        plt.xlabel("Iterations")
        plt.ylabel("Cosine of the angle between obtained EV and expected EV")
        plt.title(pltTitle)
        plt.plot(np.arange(len(angles[i])), angles[i], color="C"+str(i))
    if "-savePlots" in sys.argv or "-saveMode" in sys.argv:
        plt.savefig(f"{fns.getLocation('./plots')}anglesVSiterations_{config.variant}_{gaConfig.ascentVariant}")
    if "-saveMode" not in sys.argv:
        plt.show()
        plt.clf()
    
    for i in range(len(angles)):
        plt.xlabel("Time Elapsed")
        plt.ylabel("Cosine of the angle between obtained EV and expected EV")
        plt.title(pltTitle)
        plt.plot(iterTimes[:len(angles[i])], angles[i], color="C"+str(i))
    if "-savePlots" in sys.argv or "-saveMode" in sys.argv:
        plt.savefig(f"{fns.getLocation('./plots')}anglesVStimeElapsed_{config.variant}_{gaConfig.ascentVariant}")
    if "-saveMode" not in sys.argv:
        plt.show()
        plt.clf()
