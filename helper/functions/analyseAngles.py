from helper.imports.mainImports import *
import helper.imports.functionImports as fns
import helper.config.mainConfig as config
import helper.config.gradientAscentConfig as gaConfig

#Function to plot cosine of the angle between each player and the corresponsing 
#EigenVector against number of iterations and time elapsed on per-player basis
#Inputs
#   X   -   Input data matrix
#Outputs
#   Shows the plots to the user
#   Returns nothing 
def analyseAngles(X):
    Vs = np.load(f"./Vs/Vs_{config.xDim}_modified_{config.variant}_{gaConfig.ascentVariant}.npy")
    iterTimes = np.load(f"./iterTimes/iterTimes_{config.xDim}_modified_{config.variant}_{gaConfig.ascentVariant}.npy")
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
    elapsedTimes = []
    for col in range(Vs[0].shape[1]):
        Vs = np.load(f"./Vs/Vs_{config.xDim}_{col}_{config.variant}_{gaConfig.ascentVariant}.npy")
        iterTimes = np.load(f"./iterTimes/iterTimes_{config.xDim}_{col}_{config.variant}_{gaConfig.ascentVariant}.npy")
        angle = []
        elapsedTime = []
        for t in range(min(config.stopIterations[col], len(Vs))):
            curV = Vs[t]
            angle.append((np.dot(np.transpose(curV),EVs[:,col])/(np.linalg.norm(curV)*np.linalg.norm(EVs[:,col]))))
            elapsedTime.append(iterTimes[t])
        angles.append(angle)
        elapsedTimes.append(elapsedTime)
    angles = np.array(angles, dtype=object)
    elapsedTimes = np.array(elapsedTimes, dtype=object)
    np.save(f"{fns.getLocation('./angles')}angles_{config.variant}_{gaConfig.ascentVariant}.npy",angles)
    pltTitle = f"Variant {config.variant} ({gaConfig.ascentVariant}): lr = {gaConfig.learningRate}, xDim = {config.xDim}, k = {config.k},L = {config.L}, T = {gaConfig.numIterations}"
    for i in range(len(angles)):
        plt.xlabel("Iterations")
        plt.ylabel("Cosine of the angle between obtained EV and expected EV")
        plt.title(pltTitle)
        plt.plot(25*np.arange(len(angles[i])), angles[i], color="C"+str(i))
        if "-savePlots" in sys.argv or "-saveMode" in sys.argv:
            plt.savefig(f"{fns.getLocation('./plots')}anglesVSiterations{i}_{config.variant}_{gaConfig.ascentVariant}")
        if "-saveMode" not in sys.argv:
            plt.show()
        plt.clf()
    
    for i in range(len(angles)):
        plt.xlabel("Time Elapsed")
        plt.ylabel("Cosine of the angle between obtained EV and expected EV")
        plt.title(pltTitle)
        plt.plot(elapsedTimes[i], angles[i], color="C"+str(i))
        if "-savePlots" in sys.argv or "-saveMode" in sys.argv:
            plt.savefig(f'{fns.getLocation("./plots")}anglesVStimeElapsed{i}_{config.variant}_{gaConfig.ascentVariant}')
        if "-saveMode" not in sys.argv:
            plt.show()
        plt.clf()
