from helper.imports.mainImports import *
from helper.imports.functionImports import *
import helper.config.mainConfig as config
import helper.config.gradientAscentConfig as gaConfig

#Function to get the location of the folder to save plots/visualisations to
#Location returned is relative to the corresponding plots/visualisation file
#Inputs 
#   curPath   - Path to the plots/visualisation folder as a string
#Outputs
#   Returns required location as a string
def getLocation(curPath):
    result = curPath + "/"
    if not os.path.exists(result):
        os.mkdir(result)
    folderLoc = str(config.xDim[0]) + "x" + str(config.xDim[1])
    result += folderLoc + "/"
    if not os.path.exists(result):
        os.mkdir(result)
        np.save(result + "X_" + str(config.xDim) + ".npy", np.load(f"X_{config.xDim}.npy"))
    variantLoc = config.variant
    result += variantLoc + "/"
    if not os.path.exists(result):
        os.mkdir(result)
    gaVariantLoc = gaConfig.ascentVariant
    result += gaVariantLoc + "/"
    if not os.path.exists(result):
        os.mkdir(result)
    return result
    