from helper.imports.mainImports import *
from helper.imports.functionImports import *
import helper.config.mainConfig as config
import helper.config.gradientAscentConfig as gaConfig
import helper.config.thresholdConfig as thConfig
import helper.config.miscellaneousConfig as mlConfig

if "-symmetric" in sys.argv:
    config.variant = "2"    #Symmetric
else:
    config.variant = "1"    #Asymmetric

if "-variantC" in sys.argv:
    config.L = gaConfig.numStepsPerIteration
    config.variant += "c"
elif "-variantB" in sys.argv:
    config.L = 1
    config.variant += "b"
else:
    config.L = gaConfig.T
    config.variant += "a"

if "-momentum" in sys.argv:
         gaConfig.ascentVariant = "momentum"
elif "-nesterov" in sys.argv:
    gaConfig.ascentVariant = "nesterov"
elif "-adagrad" in sys.argv:
    gaConfig.ascentVariant = "adagrad"
elif "-rmsprop" in sys.argv:
    gaConfig.ascentVariant = "rmsprop"
elif "-adam" in sys.argv:
    gaConfig.ascentVariant = "adam"
else:
    gaConfig.ascentVariant = "vanilla"

if "-continueEigenGame" not in sys.argv:
    if "-repeatedEVtest" in sys.argv:
        X = [[-5,-6,3],[3,4,-3],[0,0,-2]]
        X = np.array(X)
    elif "-repeatedEVtest2" in sys.argv:
        X = np.load("./repeatedEV_X.npy")
    elif "-generateX" in sys.argv or not (os.path.exists(f"X_{config.xDim}.npy") and os.path.isfile(f"X_{config.xDim}.npy")):
        X = np.random.rand(config.xDim[0], config.xDim[1])
        np.save(f"X_{config.xDim}.npy",X)
elif not (os.path.exists(f"X_{config.xDim}.npy") and os.path.isfile(f"X_{config.xDim}.npy")):
    print("Last game's dataset not found!\nStarting new game with new dataset...")

#Load dataset X from f"X_{config.xDim}.npy"
if "-repeatedEVtest" in sys.argv:
    X = [[-5,-6,3],[3,4,-3],[0,0,-2]]
    X = np.array(X)
    config.xDim = X.shape
elif "-repeatedEVtest2" in sys.argv:
    X = np.load("./repeatedEV_X.npy")
    config.xDim = X.shape
else:
    X = np.load(f"X_{config.xDim}.npy")
    
if "-printX" in sys.argv:
    print(X)
if config.k > X.shape[1]:
    config.k = X.shape[1]

mlConfig.hasConverged = [0]*X.shape[1]

if ("-analyseResults" not in sys.argv 
        and "-visualiseResults" not in sys.argv 
        and "-visualiseResultsTogether" not in sys.argv 
        and "-analyseAngles" not in sys.argv 
        and "-analyseAnglesTogether" not in sys.argv 
        and "-analyseSubspaceAngles" not in sys.argv
        and "-computeLCES" not in sys.argv
    ) or "-playEigenGame" in sys.argv or "-continueEigenGame" in sys.argv:

    if "-symmetric" in sys.argv:
        print(f"Playing the symmetric penalty EigenGame (variant {config.variant[-1]}, {gaConfig.ascentVariant})...")
    else:
        print(f"Playing the asymmetric EigenGame (variant {config.variant[-1]}, {gaConfig.ascentVariant})...")
    startGame = time.time()
    V = playEigenGame(X, gaConfig.numIterations, config.k)
    stopGame = time.time()
    print(f"Time taken: {stopGame-startGame}s")
    EVs = getEigenVectors(X)
    EVs = rearrange(EVs, V)
    print("EigenVectors obtained through EigenGame:")
    print(np.around(V,decimals=3))
    print("\nEigenVectors obtained through numpy:")
    print(np.around(EVs,decimals=3))
    print(f"Learning Rate : {gaConfig.learningRate}")
    print(f"Distance measure: {np.around(getDistance(V,EVs), decimals=3)}")

    #Finding time taken for convergence
    Vs = np.load(f"Vs_{config.variant}_{gaConfig.ascentVariant}.npy")
    iterTimes = np.load(f"iterTimes_{config.variant}_{gaConfig.ascentVariant}.npy")
    EVs = np.around(getEigenVectors(X),decimals=3)
    EVs = rearrange(EVs, Vs[-1])
    convergenceTime = None
    for i in range(len(Vs)):
        V = Vs[i]
        distanceMeasure = np.around(getDistance(V,EVs), decimals=3)
        if distanceMeasure <= thConfig.distanceTolerance:
            convergenceTime = iterTimes[i]
            break 
    if convergenceTime == None:
        print("EigenGame did not converge as per expectation!")
    else:
        print(f"Time taken to converge as per expectation: {convergenceTime} s")

if "-computeLCES" in sys.argv:
    LCES = computeLCES(X)
    print(f"Sum of LCES at the end of {len(LCES)} iterations: {sum(LCES)}")

if "-analyseResults" in sys.argv:
    analyseResults(X)

if "-visualiseResults" in sys.argv and "-3D" in sys.argv:
    visualiseResults(X)

if "-visualiseResultsTogether" in sys.argv and "-3D" in sys.argv:
    visualiseResultsTogether(X)

if "-analyseAngles" in sys.argv or "-analyseAnglesTogether" in sys.argv:
    analyseAngles(X)
    
#UNDER CONSTRUCTION
if "-analyseSubspaceAngles" in sys.argv:
    analyseSubspaceAngles(X)
    