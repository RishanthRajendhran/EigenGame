from helper.imports.mainImports import *
import helper.imports.functionImports as fns
import helper.config.mainConfig as config
import helper.config.gradientAscentConfig as gaConfig

#Function to visualise each player in 3D with the expected final position
#of that player (i.e. corresponding EigenVector) as reference
#Inputs
#   X   -   Input data matrix
#Outputs
#   Shows the visualisations to the user
#   Returns nothing 
def visualiseResults(X):
    Vs = np.load(f"./Vs/Vs_{config.xDim}_{config.variant}_{gaConfig.ascentVariant}.npy")
    if Vs[-1].shape[0] != 3:
        print("Only 3D visualisations allowed!")
        return
    visualisationSpeed = 1         #Default speed : highSpeed
    if "-mediumSpeed" in sys.argv:
        visualisationSpeed = 500 
    elif "-lowSpeed" in sys.argv or (("-saveMode" in sys.argv or "-saveVisualisations" in sys.argv) and "-highSpeed" not in sys.argv):
        visualisationSpeed = 1000

    EVs = np.around(fns.getEigenVectors(X),decimals=3)
    EVs = fns.rearrange(EVs, Vs[-1])
    for pos in range(Vs[-1].shape[1]):
        Vs = np.load(f"./Vs/Vs_{config.xDim}_{pos}_{config.variant}_{gaConfig.ascentVariant}.npy")
        V = []
        minX, minY, minZ = 0, 0, 0
        maxX, maxY, maxZ = 0, 0, 0
        for i in range(len(Vs)):
            v = Vs[i]
            V.append(v)
            if i:
                minX = min(minX, v[0])
                minY = min(minY, v[1])
                minZ = min(minX, v[2])
                maxX = max(maxX, v[0])
                maxY = max(maxY, v[1])
                maxZ = max(maxZ, v[2])
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        plt.title(f"Iteration 0\nVariant {config.variant} ({gaConfig.ascentVariant}): lr = {gaConfig.learningRate}, xDim = {config.xDim}, k = {config.k},L = {config.L}, T = {gaConfig.numIterations}")
        fig.text(.5, .05, "\n" + "Obtained eigenvectors (blue): " + str(np.around(Vs[-1],decimals=3)) + "\n" + "Expected eigenvector (red): " + str(np.around(EVs[:,pos],decimals=3)), ha='center')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        quiverFinal = ax.quiver(0, 0, 0, EVs[:,pos][0], EVs[:,pos][1], EVs[:,pos][2], color="r")
        quiver = ax.quiver(0, 0, 0, V[0][0], V[0][1], V[0][2])
        ax.set_xlim(minX-0.1, maxX+0.1)
        ax.set_ylim(minY-0.1, maxY+0.1)
        ax.set_zlim(minZ-0.1, maxZ+0.1)

        def update(i):
            plt.title(f"Iteration {i*25}\nVariant {config.variant} ({gaConfig.ascentVariant}): lr = {gaConfig.learningRate}, xDim = {config.xDim}, k = {config.k},L = {config.L}, T = {gaConfig.numIterations}")
            nonlocal quiver 
            quiver.remove()
            quiver = ax.quiver(0, 0, 0, V[i][0], V[i][1], V[i][2])
        ani = FuncAnimation(fig, update, frames=np.arange(len(V)), interval=visualisationSpeed, repeat=False)
        if "-saveMode" not in sys.argv:
            plt.show()
        if "-saveVisualisations" in sys.argv or "-saveMode" in sys.argv:
            print("Saving visualisation. Might take a while...")
            ani.save(f'{fns.getLocation("./visualisations")}eigenVector{pos}_{config.variant}_{gaConfig.ascentVariant}.mp4')
            print("Visualisation saved successfully!")
    plt.clf()