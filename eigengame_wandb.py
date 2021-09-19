import numpy as np
import matplotlib.pyplot as plt
import time
import os.path
import wandb

hyperparameter_defaults = dict(
    xDim = (3, 7),
    numIterations = 10000,
    k = 3,
    learningRate = 1,
    flags = ["-playEigenGame", "-modified", "-repeatedEVtest"],
)

run = wandb.init(project="eigengame", entity="rishanthrajendhran",config=hyperparameter_defaults)
config = wandb.config
wandb.run.name = f"{config.xDim}_{config.numIterations}_{config.k}_{config.learningRate}"
wandb.run.save(wandb.run.name)

xDim = config.xDim
numIterations = config.numIterations
k = config.k
learningRate = config.learningRate

#Function to return the distance measure 
# between current player positions (current eigenvalues) 
# and expected final player positions (i.e. expected eigenvalues)
#Inputs
#   V   - Current player positions
#   EVs - Expected final player positions
#Outputs
#   Returns a scalar distance measure
def getDistance(V, EVs):
    return np.sqrt(np.sum((V - EVs) ** 2))      #Euclidean distance

#Function to rearrange columns of matrix A based on their closest matching column index in matrix B
#Inputs 
#   A   - Matrix whose columns have to be rearranged
#   B   - Matrix which is the standard based on which rearrangements have to be done
#Outputs
#   Returns rearranged A matrix
def rearrange(A, B):
    newA = A.copy()
    for i in range(A.shape[1]):
        a = A[:,i]
        minDist = np.inf 
        minCol = i 
        isNeg = False
        for j in range(B.shape[1]):
            b = np.around(B[:,j],decimals=3).copy()
            dist = getDistance(a, b)
            distNeg = getDistance(a, -b)
            if dist < distNeg and dist < minDist:
                minDist = dist 
                minCol = j
                isNeg = False
            elif distNeg < minDist:
                minDist = distNeg 
                minCol = j
                isNeg = True
        newA[:,minCol] = (not isNeg)*A[:,i].copy() - isNeg*A[:,i].copy()
    return newA

#Function to return the reward
#Inputs
#   X   - Numpy array of the dataset
#   V   - Numpy array whose columns should eventually represent the eigenbectors of X
#   i   - Index of eign vector in consideration
#Outputs
#   Returns a vector reward of dimension X.shape[0] x 1
def getReward(X, V, i):
    return (np.dot(X, V[:,i])).reshape(-1, 1)

#Function to return the penalty
#Inputs
#   X   - Numpy array of the dataset
#   V   - Numpy array whose columns should eventually represent the eigenbectors of X
#   i   - Index of eign vector in consideration
#Outputs
#   Returns a vector penalty of dimension X.shape[0] x 1
def getPenalty(X, V, i):
    M  = np.dot(X.T, X)
    penalty = np.zeros((X.shape[0], 1))
    for j in range(X.shape[1]):
        condition = j < i
        if "-modified" in config.flags:
            condition = j != i
        if condition:
            penalty += 10*(np.dot(np.dot(X, V[:,i]), np.dot(X, V[:,j]))/np.dot(np.dot(X, V[:,j]), np.dot(X, V[:,j])))*np.dot(X,V[:,j]).reshape(-1,1)
    return penalty.reshape(-1, 1)

#Function to return the penalty
#Inputs
#   X       - Numpy array of the dataset
#   reward  - Reward term
#   penalty - Penalty term
#Outputs
#   Returns a vector gradient of dimension X.shape[1]  x 1
def getGradUtility(X, reward, penalty):
    return 2*np.dot(X.T,(reward - penalty))

#Function to update eigenvectors 
#Inputs 
#   X       - Numpy array of the dataset
#   V       - Numpy array whose columns should eventually represent the eigenbectors of X
#   i       - Index of eign vector in consideration
#   reward  - Reward term
#   penalty - Penalty term
#   alpha   - step size of updates
#Outputs
#   Returns updated numpy array of eigenvectors 
def updateEigenVectors(X, V, i, reward, penalty, alpha=learningRate):
    gradV = getGradUtility(X, reward, penalty)
    gradV = gradV - np.dot(gradV.T, V[:,i])*V[:,i].reshape(gradV.shape[0],1)
    V[:,i] = V[:,i] + alpha*gradV.reshape(gradV.shape[0],)
    V[:,i] = V[:,i]/np.sqrt(np.sum(V[:,i]**2))
    return V

#Function to find eigenvectors of a given X using Numpy
#Inputs 
#   X   - Numpy array of the dataset
#   k   - Number of eigenvectors to find
#Outputs
#   Returns k eigenvectors of V as a matrix of dimensions X.shape[1] x k
def getEigenVectorsK(X, k=1):
    return np.linalg.eig(np.dot(X.T, X))[1][:,:k]

#Function to find eigenvectors of a given X
#Inputs 
#   X   - Numpy array of the dataset
#   T   - Number of iterations for each player 
#   k   - Number of eigenvectors to find
#Outputs
#   Returns k eigenvectors of V as a matrix of dimensions X.shape[1] x k
def playEigenGame(X, T, k):
    X = np.array(X)
    V = None
    if "-continueEigenGame" in config.flags:
        print("Continuing the last game...")
        if "-modified" in config.flags:
            if os.path.exists("Vs_modified.npy") and os.path.isfile("Vs_modified.npy"):
                V = np.load("Vs_modified.npy")[-1]
            else:
                print("Last game not found!\nStarting new game...")
                V = np.ones((X.shape[1],k))
        else: 
            if os.path.exists("Vs.npy") and os.path.isfile("Vs.npy"):
                V = np.load("Vs.npy")[-1]
            else:
                print("Last game not found!\nStarting new game...")
                V = np.ones((X.shape[1],k))
    if "-continueEigenGame" not in config.flags or V.shape != (X.shape[1], k):
        V = np.ones((X.shape[1],k))
    Vs = [V.copy()]
    iterTimes = [0]      #Array to store time taken for every iteration
    iterTimesSum = 0    #Variable to keep track of total time elapsed
    if "-debug" in config.flags:
        print(f"Learning Rate : {learningRate}")
    for t in range(T):
        startIter = time.time()
        for i in range(k):
            reward = getReward(X, V, i)
            penalty = getPenalty(X, V, i)
            V = updateEigenVectors(X, V, i, reward, penalty)
        Vs.append(V.copy())
        stopIter = time.time()
        timeIter = stopIter - startIter
        iterTimesSum += timeIter
        iterTimes.append(iterTimesSum)
        if "-debug" in config.flags and not t%100:
            print(f"{t}/{T} => total time elapsed : {np.around(iterTimesSum,decimals=3)}s")
    Vs = np.array(Vs)
    iterTimes = np.array(iterTimes)
    if "-continueEigenGame" in config.flags:
        if "-modified" in config.flags:
            oldVs = np.load("Vs_modified.npy")
            oldIterTimes = np.load("iterTimes_modified.npy")
            Vs = np.append(oldVs, Vs.copy(),0)
            iterTimes = iterTimes + np.sum(oldIterTimes)
            iterTimes = np.append(oldIterTimes, iterTimes.copy(),0)
        else:
            oldVs = np.load("Vs.npy")
            oldIterTimes = np.load("iterTimes.npy")
            Vs = np.append(oldVs, Vs.copy(),0)
            iterTimes = iterTimes + np.sum(oldIterTimes)
            iterTimes = np.append(oldIterTimes, iterTimes.copy(),0)
    if "-modified" in config.flags:
        np.save("Vs_modified.npy",Vs)
        np.save("iterTimes_modified.npy",iterTimes)
    else:
        np.save("Vs.npy",Vs)
        np.save("iterTimes.npy",iterTimes)
    return V

#-------------------------------------------------------------------------------------------------------------------------

if "-continueEigenGame" not in config.flags:
    if "-repeatedEVtest" in config.flags:
        X = [[-5,-6,3],[3,4,-3],[0,0,-2]]
        X = np.array(X)
    elif "-generateX" in config.flags or not (os.path.exists("./X.npy") and os.path.isfile("./X.npy")):
        X = np.random.rand(xDim[0], xDim[1])
        np.save("./X.npy",X)
elif not (os.path.exists("./X.npy") and os.path.isfile("./X.npy")):
    print("Last game's dataset not found!\nStarting new game with new dataset...")

#Load dataset X from "./X.npy"
if "-repeatedEVtest" not in config.flags:
    X = np.load("./X.npy")
else:
    X = [[-5,-6,3],[3,4,-3],[0,0,-2]]
    X = np.array(X)
    xDim = X.shape

if "-printX" in config.flags:
    print(X)

if ("-analyseResults" not in config.flags and "-visualiseResults" not in config.flags) or "-playEigenGame" in config.flags:
    if "-modified" in config.flags:
        print("Playing the Modified EigenGame...")
    else:
        print("Playing the EigenGame...")
    V = playEigenGame(X, numIterations, k)
    EVs = getEigenVectorsK(X, k)
    V = rearrange(V, EVs)
    print("EigenVectors obtained through EigenGame:")
    print(np.around(V,decimals=3))
    print("\nEigenVectors obtained through numpy:")
    print(np.around(EVs,decimals=3))
    print(f"Distance measure: {np.around(getDistance(V,EVs), decimals=3)}")
    metrics = {
        'distance_measure': np.around(getDistance(V,EVs), decimals=3),
    }
    wandb.log(metrics)
    run.finish()

if "-analyseResults" in config.flags:
    if "-modified" in config.flags:
        Vs = np.load("Vs_modified.npy")
        iterTimes = np.load("iterTimes_modified.npy")
    else:
        Vs = np.load("Vs.npy")
        iterTimes = np.load("iterTimes.npy")
    EVs = np.around(getEigenVectorsK(X, k),decimals=3)
    EVs = rearrange(EVs, Vs[-1])
    diffs = []
    print("EigenVectors obtained through EigenGame:")
    for V in Vs:
        diffs.append(getDistance(V,EVs))
        if "-debug" in config.flags:
            print(np.around(V,decimals=3))
    if "debug" not in config.flags:
        print(np.around(Vs[-1],decimals=3))
    print("\nEigenVectors obtained through numpy:")
    print(np.around(EVs,decimals=3))
    plt.plot(diffs)
    plt.xlabel("Iterations")
    plt.ylabel("Distance")
    if "-modified" in config.flags:
            plt.title(f"Variant 2.b: Learning rate = {learningRate}, xDim = {xDim}, k = {k}, T = {numIterations}")
    else:
        plt.title(f"Variant 1.b: Learning rate = {learningRate}, xDim = {xDim}, k = {k}, T = {numIterations}")
    plt.show()
    plt.plot(iterTimes, diffs)
    plt.xlabel("Time elapsed (s)")
    plt.ylabel("Distance")
    if "-modified" in config.flags:
            plt.title(f"Variant 2.b: Learning rate = {learningRate}, xDim = {xDim}, k = {k}, T = {numIterations}")
    else:
        plt.title(f"Variant 1.b: Learning rate = {learningRate}, xDim = {xDim}, k = {k}, T = {numIterations}")
    plt.show()

if "-visualiseResults" in config.flags and "-3D" in config.flags:
    if "-modified" in config.flags:
        Vs = np.load("./Vs_modified.npy")
    else:
        Vs = np.load("./Vs.npy")
    if Vs[-1].shape[0] != 3:
        print("Only 3D visualisations allowed!")
        exit(0)

    visualisationSpeed = 1         #Default speed : highSpeed
    if "-mediumSpeed" in config.flags:
        visualisationSpeed = 500 
    elif "-lowSpeed" in config.flags:
        visualisationSpeed = 1000

    EVs = np.around(getEigenVectorsK(X, k),decimals=3)
    EVs = rearrange(EVs, Vs[-1])
    for pos in range(Vs[-1].shape[1]):
        V = []
        minX, minY, minZ = 0, 0, 0
        maxX, maxY, maxZ = 0, 0, 0
        for i in range(len(Vs)):
            v = Vs[i][:,pos]
            V.append(v)
            if i:
                minX = min(minX, v[0])
                minY = min(minY, v[1])
                minZ = min(minX, v[2])
                maxX = max(maxX, v[0])
                maxY = max(maxY, v[1])
                maxZ = max(maxZ, v[2])
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        if "-modified" in config.flags:
            plt.title(f"Variant 2.b: Learning rate = {learningRate}, xDim = {xDim}, k = {k}, T = {numIterations}")
        else:
            plt.title(f"Variant 1.b: Learning rate = {learningRate}, xDim = {xDim}, k = {k}, T = {numIterations}")
        fig.text(.5, .05, "\n" + "Obtained eigenvectors (blue): " + str(np.around(Vs[-1][:,pos],decimals=3)) + "\n" + "Expected eigenvector (red): " + str(np.around(EVs[:,pos],decimals=3)), ha='center')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        quiverFinal = ax.quiver(0, 0, 0, V[-1][0], V[-1][1], V[-1][2], color="r")
        quiver = ax.quiver(0, 0, 0, V[0][0], V[0][1], V[0][2])
        ax.set_xlim(minX-0.1, maxX+0.1)
        ax.set_ylim(minY-0.1, maxY+0.1)
        ax.set_zlim(minZ-0.1, maxZ+0.1)
        def update(i):
            global quiver 
            quiver.remove()
            quiver = ax.quiver(0, 0, 0, V[i][0], V[i][1], V[i][2])
        ani = FuncAnimation(fig, update, frames=np.arange(len(V)), interval=visualisationSpeed)
        plt.show()
        if "-saveVisualisations" in config.flags:
            print("Saving visualisation. Might take a while...")
            if "-modified" in config.flags:
                ani.save(f'eigenVector{pos}_modified.mp4')
            else:
                ani.save(f'eigenVector{pos}.mp4')
            print("Visualisation saved successfully!")

if "-visualiseResultsTogether" in config.flags and "-3D" in config.flags:
    if "-modified" in config.flags:
        Vs = np.load("./Vs_modified.npy")
    else:
        Vs = np.load("./Vs.npy")
    if Vs[-1].shape[0] != 3:
        print("Only 3D visualisations allowed!")
        exit(0)

    visualisationSpeed = 1         #Default speed : highSpeed
    if "-mediumSpeed" in config.flags:
        visualisationSpeed = 500 
    elif "-lowSpeed" in config.flags:
        visualisationSpeed = 1000

    EVs = np.around(getEigenVectorsK(X, k),decimals=3)
    EVs = rearrange(EVs, Vs[-1])

    minX, minY, minZ = 0, 0, 0
    maxX, maxY, maxZ = 0, 0, 0
    for pos in range(Vs[-1].shape[1]):
        for i in range(len(Vs)):
            v = Vs[i][:,pos]
            if i:
                minX = min(minX, v[0])
                minY = min(minY, v[1])
                minZ = min(minX, v[2])
                maxX = max(maxX, v[0])
                maxY = max(maxY, v[1])
                maxZ = max(maxZ, v[2])

    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    if "-modified" in config.flags:
        plt.title(f"Variant 2.b: Learning rate = {learningRate}, xDim = {xDim}, k = {k}, T = {numIterations}")
    else:
        plt.title(f"Variant 1.b: Learning rate = {learningRate}, xDim = {xDim}, k = {k}, T = {numIterations}")
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    quiverFinals = []
    quivers = []
    for z in range(Vs[-1].shape[1]):
        quiverFinals.append(ax.quiver(0, 0, 0, EVs[:,z][0], EVs[:,z][1], EVs[:,z][2], color="r"))
        quivers.append(ax.quiver(0, 0, 0, Vs[0][:,z][0], Vs[0][:,z][1], Vs[0][:,z][2],color=str(z/100000)))
    ax.set_xlim(minX-0.1, maxX+0.1)
    ax.set_ylim(minY-0.1, maxY+0.1)
    ax.set_zlim(minZ-0.1, maxZ+0.1)
    def update(i):
        global quivers
        for quiver in quivers:
            quiver.remove()
        quivers = []
        for z in range(Vs[-1].shape[1]):
            quivers.append(ax.quiver(0, 0, 0, Vs[i][:,z][0], Vs[i][:,z][1], Vs[i][:,z][2],color="C"+str(z)))
    ani = FuncAnimation(fig, update, frames=np.arange(len(Vs)), interval=visualisationSpeed)
    plt.show()
    if "-saveVisualisations" in config.flags:
        print("Saving visualisation. Might take a while...")
        if "-modified" in config.flags:
            ani.save(f'eigenVectors_modified.mp4')
        else:
            ani.save(f'eigenVectors.mp4')
        print("Visualisation saved successfully!")

if "-analyseAngles" in config.flags or "-analyseAnglesTogether" in config.flags:
    if "-modified" in config.flags:
        Vs = np.load("./Vs_modified.npy")
        iterTimes = np.load("./iterTimes_modified.npy")
    else:
        Vs = np.load("./Vs.npy")
        iterTimes = np.load("./iterTimes.npy")
    EVs = np.linalg.eig(np.dot(X.T,X))[1]
    EVs = rearrange(EVs, Vs[-1])
    angles = []
    for col in range(Vs[0].shape[1]):
        angle = []
        for t in range(len(Vs)):
            curV = Vs[t][:,col]
            angle.append((np.dot(np.transpose(curV),EVs[:,col])/(np.linalg.norm(curV)*np.linalg.norm(EVs[:,col]))))
        angles.append(angle)
    angles = np.array(angles)
    if "-modified" in config.flags:
        np.save("./angles_modified.npy",angles)
    else: 
        np.save("./angles.npy",angles)
    if "-modified" in config.flags:
        pltTitle = f"Variant 2.b: Learning rate = {learningRate}, xDim = {xDim}, k = {k}, T = {numIterations}"
    else:
        pltTitle = f"Variant 1.b: Learning rate = {learningRate}, xDim = {xDim}, k = {k}, T = {numIterations}"
    plt.xlabel("Iterations")
    plt.ylabel("Angle between obtainer EV and expected EV")
    for i in range(len(angles)):
        plt.title(pltTitle)
        plt.plot(np.arange(len(angles[i])), angles[i], color="C"+str(i))
        if "-analyseAnglesTogether" not in config.flags:
            if "-savePlots" in config.flags:
                if "-modified" in config.flags:
                    plt.savefig(f"./anglesVSiterations{i}_modified")
                else:
                    plt.savefig(f"./anglesVSiterations{i}")
            plt.show()
    if "-analyseAnglesTogether" in config.flags:
        if "-savePlots" in config.flags:
            if "-modified" in config.flags:
                plt.savefig(f"./anglesVSiterations_modified")
            else:
                plt.savefig(f"./anglesVSiterations")
        plt.show()
    
    plt.xlabel("Total Time Elapsed")
    plt.ylabel("Angle between obtainer EV and expected EV")
    for i in range(len(angles)):
        plt.title(pltTitle)
        plt.plot(iterTimes, angles[i], color="C"+str(i))
        if "-analyseAnglesTogether" not in config.flags:
            if "-savePlots" in config.flags:
                if "-modified" in config.flags:
                    plt.savefig(f"./anglesVSiterations{i}_modified")
                else:
                    plt.savefig(f"./anglesVSiterations{i}")
            plt.show()
    if "-analyseAnglesTogether" in config.flags:
        if "-savePlots" in config.flags:
            if "-modified" in config.flags:
                plt.savefig(f"./anglesVSiterations_modified")
            else:
                plt.savefig(f"./anglesVSiterations")
        plt.show()
