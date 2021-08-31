import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import os.path
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

xDim = (3, 7)
numIterations = 10000
k = 3
learningRate = 0.001

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
        if "-modified" in sys.argv:
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
                V = np.ones((X.shape[1],k))
    if "-continueEigenGame" not in sys.argv or V.shape != (X.shape[1], k):
        V = np.ones((X.shape[1],k))
    Vs = [V.copy()]
    iterTimes = [0]      #Array to store time taken for every iteration
    iterTimesSum = 0    #Variable to keep track of total time elapsed
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
    elif "-generateX" in sys.argv or not (os.path.exists("./X.npy") and os.path.isfile("./X.npy")):
        X = np.random.rand(xDim[0], xDim[1])
        np.save("./X.npy",X)
elif not (os.path.exists("./X.npy") and os.path.isfile("./X.npy")):
    print("Last game's dataset not found!\nStarting new game with new dataset...")

#Load dataset X from "./X.npy"
if "-repeatedEVtest" not in sys.argv:
    X = np.load("./X.npy")
else:
    X = [[-5,-6,3],[3,4,-3],[0,0,-2]]
    X = np.array(X)

if "-printX" in sys.argv:
    print(X)

if ("-analyseResults" not in sys.argv and "-visualiseResults" not in sys.argv) or "-playEigenGame" in sys.argv:
    if "-modified" in sys.argv:
        print("Playing the Modified EigenGame...")
    else:
        print("Playing the EigenGame...")
    V = playEigenGame(X, numIterations, k)
    EVs = getEigenVectorsK(X, k)
    EVs[:,0] = -EVs[:,0]
    EVs[:,1] = EVs[:,2]
    V = rearrange(V, EVs)
    print("EigenVectors obtained through EigenGame:")
    print(np.around(V,decimals=3))
    print("\nEigenVectors obtained through numpy:")
    print(np.around(EVs,decimals=3))
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
    # EVs[:,0] = -EVs[:,0]
    # EVs[:,1] = EVs[:,2]
    # EVs = EVs[:,:2]
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
    plt.show()
    plt.plot(iterTimes, diffs)
    plt.xlabel("Time elapsed (s)")
    plt.ylabel("Distance")
    plt.show()

if "-visualiseResults" in sys.argv and "-3D" in sys.argv:
    Vs = np.load("./Vs.npy")
    if Vs[-1].shape[0] != 3:
        print("Only 3D visualisations allowed!")
        exit(0)
    EVs = np.around(getEigenVectorsK(X, k),decimals=3)
    EVs = rearrange(EVs, Vs[-1])
    for pos in range(Vs[-1].shape[1]):
        V = []
        minX, minY, minZ = 0, 0, 0
        maxX, maxY, maxZ = 0, 0, 0
        for i in range(len(Vs)):
            v = Vs[i]
            V.append(v[pos])
            if i:
                minX = min(minX, v[pos][0])
                minY = min(minY, v[pos][1])
                minZ = min(minX, v[pos][2])
                maxX = max(maxX, v[pos][0])
                maxY = max(maxY, v[pos][1])
                maxZ = max(maxZ, v[pos][2])
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        fig.text(.5, .05, "\n" + "Obtained eigenvectors (blue): " + str(np.around(v[pos],decimals=3)) + "\n" + "Expected eigenvector (red): " + str(np.around(EVs[pos],decimals=3)), ha='center')
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
        ani = FuncAnimation(fig, update, frames=np.arange(len(V)), interval=100)
        if "-saveVisualisations" in sys.argv:
            print("Saving animation as video. Might take a while...")
            ani.save(f'eigenVector{pos}.mp4')
        plt.show()
