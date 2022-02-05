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

gaConfig.numIterations = gaConfig.T//config.L

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
        and "-postGameAnalysis" not in sys.argv
        and "-visualiseTrajectory" not in sys.argv
        and "-visualiseTrajectoryTogether" not in sys.argv
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

    convergenceTime, distanceMeasure = getConvergenceInfo()

    if convergenceTime == None:
        print("EigenGame did not converge as per expectation!")
    else:
        print(f"Time taken to converge as per expectation: {convergenceTime} s")
        toDocument = {
            "timeStamp" : datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S'),
            "xDim": config.xDim,
            "variant": config.variant,
            "ascentVariant": gaConfig.ascentVariant,
            "learningRate": gaConfig.learningRate,
            "T": gaConfig.numIterations,
            "L": config.L,
            "convergenceTime": convergenceTime,
            "distanceMeasure": distanceMeasure
        }
        with open(f'history_{config.xDim}.txt', 'a') as convert_file:
            convert_file.write(json.dumps(toDocument, indent=4))
        with open('history.txt', 'a') as convert_file:
            convert_file.write(json.dumps(toDocument, indent=4))
else:
    convergenceTime, distanceMeasure = getConvergenceInfo()

if "-analyseResults" in sys.argv or "-postGameAnalysis" in sys.argv:
    if "-postGameAnalysis" in sys.argv:
        print("-analyseResults")
    analyseResults(X)

if "-analyseAngles" in sys.argv or "-postGameAnalysis" in sys.argv:
    if "-postGameAnalysis" in sys.argv:
        print("-analyseAngles")
    analyseAngles(X)

if "-analyseAnglesTogether" in sys.argv or "-postGameAnalysis" in sys.argv:
    if "-postGameAnalysis" in sys.argv:
        print("-analyseAnglesTogether")
    analyseAnglesTogether(X)

if "-computeLCES" in sys.argv or "-postGameAnalysis" in sys.argv:
    if "-postGameAnalysis" in sys.argv:
        print("-computeLCES")
    LCES = computeLCES(X)
    print(f"Sum of LCES at the end of {len(LCES)} iterations: {sum(LCES)} (Max possible: {config.k*len(LCES)})")

if "-visualiseResults" in sys.argv or "-postGameAnalysis" in sys.argv:
    if "-postGameAnalysis" in sys.argv:
        print("-visualiseResults")
    visualiseResults(X)

if "-visualiseResultsTogether" in sys.argv or "-postGameAnalysis" in sys.argv:
    if "-postGameAnalysis" in sys.argv:
        print("-visualiseResultsTogether")
    visualiseResultsTogether(X)

if "-visualiseTrajectory" in sys.argv or "-postGameAnalysis" in sys.argv:
    if "-postGameAnalysis" in sys.argv:
        print("-visualiseTrajectory")
    visualiseTrajectory(X)

if "-visualiseTrajectoryTogether" in sys.argv or "-postGameAnalysis" in sys.argv:
    if "-postGameAnalysis" in sys.argv:
        print("-visualiseTrajectoryTogether")
    visualiseTrajectoryTogether(X)

#UNDER CONSTRUCTION
if "-analyseSubspaceAngles" in sys.argv:
    if "-postGameAnalysis" in sys.argv:
        print("-analyseSubspaceAngles")
    analyseSubspaceAngles(X)
    