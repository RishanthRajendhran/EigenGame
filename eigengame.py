import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import os.path
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.linalg import subspace_angles

xDim = (10000, 3)
numStepsPerIteration = 100
T = 1100

if "-variantA" in sys.argv:
    L = T
elif "-variantB" in sys.argv:
    L = 1
elif "-variantC" in sys.argv:
    L = numStepsPerIteration

numIterations = T//L
k = 3
learningRate = 0.001
tolerance = 10

#Function to return the subspace angle 
# between current player positions (current eigenvalues) 
# and expected final player positions (i.e. expected eigenvalues)
#Inputs
#   V   - Numpy array of Current player positions
#   EVs - Numpy array of Expected final player positions
#   i   - Current player
#Outputs
#   Returns a scalar angular measure
def getDistance(V, EVs):
    return np.sum(subspace_angles(V[:,:i+1], EVs[:, :i+1]))      

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
    newA = B.copy()
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
    for j in range(k):
        condition = j < i
        if "-modified" in sys.argv:
            condition = j != i
        if condition:
            dotProd = (np.dot(np.dot(X, V[:,i]), np.dot(X, V[:,j]))/np.dot(np.dot(X, V[:,j]), np.dot(X, V[:,j])))*np.dot(X,V[:,j]).reshape(-1,1)
            penalty += 10*dotProd
            penalty += 1*abs(dotProd)
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

#Function to get the angle (in degrees) between two vectors u and v 
#Inputs 
#   u       - Numpy column array
#   v       - Numpy column array
#Outputs
#   Returns angle between the vectors in degree measure 
def getAngle(u, v):
    return np.rad2deg(np.arccos(np.dot(u.T,v)/(np.linalg.norm(u)*np.linalg.norm(v))))

#Function to check is current player is close to previous players 
#If it is the case, reinitialise current player and return the new set of player positions
#Inputs 
#   V           - Numpy array whose columns should eventually represent the eigenbectors
#   curPlayer   - Index of eign vector in consideration
#   oldPos      - Position of current player before last update
#Outputs
#   Returns updated numpy array of eigenvectors 
def checkVectors(V, curPlayer, oldPos):
    for i in range(V.shape[1]):
        if "-modified" in sys.argv:
            if i != curPlayer and 0 <= getAngle(V[:,i], V[:, curPlayer]) <= tolerance and 180-tolerance <= getAngle(V[:,i], V[:, curPlayer]) <= 180:
                V[:,curPlayer] = -oldPos
                break
        else:
            if i < curPlayer and 0 <= getAngle(V[:,i], V[:, curPlayer]) <= tolerance and 180-tolerance <= getAngle(V[:,i], V[:, curPlayer]) <= 180:
                print("Hey")
                V[:,curPlayer] = -oldPos
                break
    return V
 
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
    oldVi = V[:,i].copy()
    V[:,i] = V[:,i] + alpha*gradV.reshape(gradV.shape[0],)
    V[:,i] = V[:,i]/np.sqrt(np.sum(V[:,i]**2))
    # V = checkVectors(V, i, oldVi)
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
    if "-continueEigenGame" in sys.argv:
        print("Continuing the last game...")
        if "-modified" in sys.argv:
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
                # V = np.ones((X.shape[1],k))
                V = np.random.rand(X.shape[0],k)
    if "-continueEigenGame" not in sys.argv or V.shape != (X.shape[1], k):
        # V = np.ones((X.shape[1],k))
        # V = np.eye(X.shape[1],k)
        V = np.random.rand(X.shape[1],k)
    Vs = [V.copy()]
    iterTimes = [0]      #Array to store time taken for every iteration
    iterTimesSum = 0    #Variable to keep track of total time elapsed
    for t in range(T):
        startIter = time.time()
        for i in range(k):
            for ti in range(L):
                reward = getReward(X, V, i)
                penalty = getPenalty(X, V, i)
                V = updateEigenVectors(X, V, i, reward, penalty)
            Vs.append(V.copy())
            stopIter = time.time()
            timeIter = stopIter - startIter
            iterTimesSum += timeIter
            iterTimes.append(iterTimesSum)
            if "-debug" in sys.argv and not t%100:
                print(f"{t}/{T} => total time elapsed : {np.around(iterTimesSum,decimals=3)}s")
    Vs = np.array(Vs)
    iterTimes = np.array(iterTimes)
    if "-continueEigenGame" in sys.argv:
        if "-modified" in sys.argv:
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
    if "-modified" in sys.argv:
        np.save("Vs_modified.npy",Vs)
        np.save("iterTimes_modified.npy",iterTimes)
    else:
        np.save("Vs.npy",Vs)
        np.save("iterTimes.npy",iterTimes)
    return V

#-------------------------------------------------------------------------------------------------------------------------

if "-continueEigenGame" not in sys.argv:
    if "-repeatedEVtest" in sys.argv:
        X = [[-5,-6,3],[3,4,-3],[0,0,-2]]
        X = np.array(X)
    elif "-repeatedEVtest2" in sys.argv:
        X = np.load("./repeatedEV_X.npy")
    elif "-generateX" in sys.argv or not (os.path.exists("./X.npy") and os.path.isfile("./X.npy")):
        X = np.random.rand(xDim[0], xDim[1])
        np.save("./X.npy",X)
elif not (os.path.exists("./X.npy") and os.path.isfile("./X.npy")):
    print("Last game's dataset not found!\nStarting new game with new dataset...")

#Load dataset X from "./X.npy"
if "-repeatedEVtest" in sys.argv:
    X = [[-5,-6,3],[3,4,-3],[0,0,-2]]
    X = np.array(X)
    xDim = X.shape
elif "-repeatedEVtest2" in sys.argv:
    X = np.load("./repeatedEV_X.npy")
    xDim = X.shape
else:
    X = np.load("./X.npy")
    
if "-printX" in sys.argv:
    print(X)

if ("-analyseResults" not in sys.argv and "-visualiseResults" not in sys.argv and "-visualiseResultsTogether" not in sys.argv and "-analyseAngles" not in sys.argv and "-analyseAnglesTogether" not in sys.argv) or "-playEigenGame" in sys.argv:
    if "-modified" in sys.argv:
        print("Playing the Modified EigenGame...")
    else:
        print("Playing the EigenGame...")
    V = playEigenGame(X, numIterations, k)
    EVs = getEigenVectorsK(X, X.shape[1])
    V = rearrange(V, EVs)
    EVs = EVs[:,:k].copy()
    print("EigenVectors obtained through EigenGame:")
    print(np.around(V,decimals=3))
    print("\nEigenVectors obtained through numpy:")
    print(np.around(EVs,decimals=3))
    print(f"Learning Rate : {learningRate}")
    print(f"Distance measure: {np.around(getDistance(V,EVs), decimals=3)}")

if "-analyseResults" in sys.argv:
    if "-modified" in sys.argv:
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
        if "-debug" in sys.argv:
            print(np.around(V,decimals=3))
    if "debug" not in sys.argv:
        print(np.around(Vs[-1],decimals=3))
    print("\nEigenVectors obtained through numpy:")
    print(np.around(EVs,decimals=3))
    plt.plot(diffs)
    plt.xlabel("Iterations")
    plt.ylabel("Distance")
    if "-modified" in sys.argv:
            plt.title(f"Variant 2.b: Learning rate = {learningRate}, xDim = {xDim}, k = {k},L = {L}, T = {numIterations}")
    else:
        plt.title(f"Variant 1.b: Learning rate = {learningRate}, xDim = {xDim}, k = {k},L = {L}, T = {numIterations}")
    if "-savePlots" in sys.argv:
        if "-modified" in sys.argv:
            plt.savefig("./plots/distanceVSiterations_modified")
        else:
            plt.savefig("./plots/distanceVSiterations")
    plt.show()
    plt.plot(iterTimes, diffs)
    plt.xlabel("Time elapsed (s)")
    plt.ylabel("Distance")
    if "-modified" in sys.argv:
            plt.title(f"Variant 2.b: Learning rate = {learningRate}, xDim = {xDim}, k = {k},L = {L}, T = {numIterations}")
    else:
        plt.title(f"Variant 1.b: Learning rate = {learningRate}, xDim = {xDim}, k = {k},L = {L}, T = {numIterations}")
    if "-savePlots" in sys.argv:
        if "-modified" in sys.argv:
            plt.savefig("./plots/distanceVStotalTimeElapsed_modified")
        else:
            plt.savefig("./plots/distanceVStotalTimeElapsed")
    plt.show()

if "-visualiseResults" in sys.argv and "-3D" in sys.argv:
    if "-modified" in sys.argv:
        Vs = np.load("./Vs_modified.npy")
    else:
        Vs = np.load("./Vs.npy")
    if Vs[-1].shape[0] != 3:
        print("Only 3D visualisations allowed!")
        exit(0)

    visualisationSpeed = 1         #Default speed : highSpeed
    if "-mediumSpeed" in sys.argv:
        visualisationSpeed = 500 
    elif "-lowSpeed" in sys.argv:
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
        if "-modified" in sys.argv:
            plt.title(f"Variant 2.b: Learning rate = {learningRate}, xDim = {xDim}, k = {k},L = {L}, T = {numIterations}")
        else:
            plt.title(f"Variant 1.b: Learning rate = {learningRate}, xDim = {xDim}, k = {k},L = {L}, T = {numIterations}")
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
        if "-saveMode" not in sys.argv:
            plt.show()
        if "-saveVisualisations" in sys.argv:
            print("Saving visualisation. Might take a while...")
            if "-modified" in sys.argv:
                ani.save(f'./visualisations/eigenVector{pos}_modified.mp4')
            else:
                ani.save(f'./visualisations/eigenVector{pos}.mp4')
            print("Visualisation saved successfully!")

if "-visualiseResultsTogether" in sys.argv and "-3D" in sys.argv:
    if "-modified" in sys.argv:
        Vs = np.load("./Vs_modified.npy")
    else:
        Vs = np.load("./Vs.npy")
    # if Vs[-1].shape[0] != 3:
    #     print("Only 3D visualisations allowed!")
    #     exit(0)

    visualisationSpeed = 1         #Default speed : highSpeed
    if "-mediumSpeed" in sys.argv:
        visualisationSpeed = 500 
    elif "-lowSpeed" in sys.argv:
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
    if "-modified" in sys.argv:
        plt.title(f"Variant 2.b: Learning rate = {learningRate}, xDim = {xDim}, k = {k},L = {L}, T = {numIterations}")
    else:
        plt.title(f"Variant 1.b: Learning rate = {learningRate}, xDim = {xDim}, k = {k},L = {L}, T = {numIterations}")
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
    if "-saveMode" not in sys.argv:
        plt.show()
    if "-saveVisualisations" in sys.argv:
        print("Saving visualisation. Might take a while...")
        if "-modified" in sys.argv:
            ani.save(f'./visualisations/eigenVectors_modified.mp4')
        else:
            ani.save(f'./visualisations/eigenVectors.mp4')
        print("Visualisation saved successfully!")

if "-analyseAngles" in sys.argv or "-analyseAnglesTogether" in sys.argv:
    if "-modified" in sys.argv:
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
    if "-modified" in sys.argv:
        np.save("./angles_modified.npy",angles)
    else: 
        np.save("./angles.npy",angles)
    if "-modified" in sys.argv:
        pltTitle = f"Variant 2.b: Learning rate = {learningRate}, xDim = {xDim}, k = {k},L = {L}, T = {numIterations}"
    else:
        pltTitle = f"Variant 1.b: Learning rate = {learningRate}, xDim = {xDim}, k = {k},L = {L}, T = {numIterations}"
    for i in range(len(angles)):
        plt.xlabel("Iterations")
        plt.ylabel("Angle between obtained EV and expected EV")
        plt.title(pltTitle)
        plt.plot(np.arange(len(angles[i])), angles[i], color="C"+str(i))
        if "-analyseAnglesTogether" not in sys.argv:
            if "-savePlots" in sys.argv:
                if "-modified" in sys.argv:
                    plt.savefig(f"./plots/anglesVSiterations{i}_modified")
                else:
                    plt.savefig(f"./plots/anglesVSiterations{i}")
            plt.show()
    if "-analyseAnglesTogether" in sys.argv:
        if "-savePlots" in sys.argv:
            if "-modified" in sys.argv:
                plt.savefig(f"./plots/anglesVSiterations_modified")
            else:
                plt.savefig(f"./plots/anglesVSiterations")
        plt.show()
    
    for i in range(len(angles)):
        plt.xlabel("Total Time Elapsed")
        plt.ylabel("Angle between obtained EV and expected EV")
        plt.title(pltTitle)
        plt.plot(iterTimes, angles[i], color="C"+str(i))
        if "-analyseAnglesTogether" not in sys.argv:
            if "-savePlots" in sys.argv:
                if "-modified" in sys.argv:
                    plt.savefig(f"./plots/anglesVStotalTimeElapsed{i}_modified")
                else:
                    plt.savefig(f"./plots/anglesVStotalTimeElapsed{i}")
            plt.show()
    if "-analyseAnglesTogether" in sys.argv:
        if "-savePlots" in sys.argv:
            if "-modified" in sys.argv:
                plt.savefig(f"./plots/anglesVStotalTimeElapsed_modified")
            else:
                plt.savefig(f"./plots/anglesVStotalTimeElapsed")
        plt.show()

if "-analyseSubspaceAngles" in sys.argv:
    if "-modified" in sys.argv:
        Vs = np.load("./Vs_modified.npy")
        iterTimes = np.load("./iterTimes_modified.npy")
    else:
        Vs = np.load("./Vs.npy")
        iterTimes = np.load("./iterTimes.npy")
    EVs = np.linalg.eig(np.dot(X.T,X))[1]
    EVs = rearrange(EVs, Vs[-1])
    angles = []
    for t in range(len(Vs)):
        # print(Vs[t])
        angle = np.sum((subspace_angles(Vs[t][:,:2], EVs[:,1:])))
        # if angle < 10**-5:
        #     angle = 0
        angles.append(angle)
        if t>20:
            continue
        print(Vs[t][:,:2])
        print(EVs[:,1:])
        print(np.rad2deg(subspace_angles(Vs[t], EVs)))
    if "-modified" in sys.argv:
        np.save("./subspaceAngles_modified.npy",angles)
    else: 
        np.save("./subspaceAngles.npy",angles)
    if "-modified" in sys.argv:
        pltTitle = f"Variant 2.b: Learning rate = {learningRate}, xDim = {xDim}, k = {k},L = {L}, T = {numIterations}"
    else:
        pltTitle = f"Variant 1.b: Learning rate = {learningRate}, xDim = {xDim}, k = {k},L = {L}, T = {numIterations}"
    plt.xlabel("Iterations")
    plt.ylabel("Subspace Angle between obtained EV and expected EV")
    plt.title(pltTitle)
    plt.plot(np.arange(len(angles)), angles)
    if "-savePlots" in sys.argv:
        if "-modified" in sys.argv:
            plt.savefig(f"./plots/subspaceAnglesVSiterations_modified")
        else:
            plt.savefig(f"./plots/subspaceAnglesVSiterations")
    plt.show()

    plt.xlabel("Total Time Elapsed")
    plt.ylabel("Angle between obtained EV and expected EV")
    plt.title(pltTitle)
    plt.plot(iterTimes, angles)
    if "-savePlots" in sys.argv:
        if "-modified" in sys.argv:
            plt.savefig(f"./plots/subspaceAnglesVStotalTimeElapsed_modified")
        else:
            plt.savefig(f"./plots/subspaceAnglesVStotalTimeElapsed")
    plt.show()
