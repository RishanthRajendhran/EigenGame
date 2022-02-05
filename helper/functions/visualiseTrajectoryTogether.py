from helper.imports.mainImports import *
import helper.imports.functionImports as fns
import helper.config.mainConfig as config
import helper.config.gradientAscentConfig as gaConfig

#Function to visualise the trajectory of all players in 3D on a unit sphere 
# with the expected final position of that player (i.e. corresponding 
# EigenVector) as reference in a single visualisation
#Inputs
#   X   -   Input data matrix
#Outputs
#   Shows the visualisations to the user
#   Returns nothing 
def visualiseTrajectoryTogether(X):
    Vs = np.load(f"Vs_modified_{config.variant}_{gaConfig.ascentVariant}.npy")
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

    # minX, minY, minZ = 0, 0, 0
    # maxX, maxY, maxZ = 0, 0, 0
    for pos in range(Vs[-1].shape[1]):
        for i in range(len(Vs)):
            v = Vs[i][:,pos]
            # if i:
            #     minX = min(minX, v[0])
            #     minY = min(minY, v[1])
            #     minZ = min(minX, v[2])
            #     maxX = max(maxX, v[0])
            #     maxY = max(maxY, v[1])
            #     maxZ = max(maxZ, v[2])

    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))

    # draw sphere
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    # alpha controls opacity
    ax.plot_surface(x, y, z, color="lightgray", alpha=0.3)

    plt.title(f"Variant {config.variant} ({gaConfig.ascentVariant}): lr = {gaConfig.learningRate}, xDim = {config.xDim}, k = {config.k},L = {config.L}, T = {gaConfig.numIterations}")
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    quiverFinals = []
    quivers = []
    for z in range(Vs[-1].shape[1]):
        quiverFinals.append(ax.quiver(0, 0, 0, EVs[:,z][0], EVs[:,z][1], EVs[:,z][2], color="r"))
        quivers.append(ax.quiver(0, 0, 0, Vs[0][:,z][0], Vs[0][:,z][1], Vs[0][:,z][2],color=str(z/100000)))
        a1 = Vs[0][:,z]
        a1 /= np.linalg.norm(a1)
        b1 = EVs[:,z]
        b1 /= np.linalg.norm(b1)
        arc1 = fns.slerp(a1, b1)
        plt.plot(arc1[:,0], arc1[:,1], arc1[:,2], color="r")

    # ax.set_xlim(minX-0.1, maxX+0.1)
    # ax.set_ylim(minY-0.1, maxY+0.1)
    # ax.set_zlim(minZ-0.1, maxZ+0.1)
    def update(i):
        nonlocal quivers
        for quiver in quivers:
            quiver.remove()
        quivers = []
        for z in range(Vs[-1].shape[1]):
            quivers.append(ax.quiver(0, 0, 0, Vs[i][:,z][0], Vs[i][:,z][1], Vs[i][:,z][2],color="C"+str(z)))
            if i!=0:
                a = Vs[i-1][:,z]
                a /= np.linalg.norm(a)
                b = Vs[i][:,z]
                b /= np.linalg.norm(b)
                arc = fns.slerp(a, b)
                plt.plot(arc[:,0], arc[:,1], arc[:,2], color="C"+str(z))
    ani = FuncAnimation(fig, update, frames=np.arange(len(Vs)), interval=visualisationSpeed, repeat=False)
    if "-saveMode" not in sys.argv:
        plt.show()
    if "-saveVisualisations" in sys.argv or "-saveMode" in sys.argv:
        print("Saving visualisation. Might take a while...")
        ani.save(f'{fns.getLocation("./visualisations")}eigenVectorsTrajectory_{config.variant}_{gaConfig.ascentVariant}.mp4')
        print("Visualisation saved successfully!")
    plt.clf()