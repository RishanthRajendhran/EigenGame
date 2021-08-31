import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import os.path
import wandb

hyperparameter_defaults = dict(
    xDim = (100000, 4),
    numIterations = 10000,
    k = 3,
    learningRate = 1,
    type = "modified",
    )

run = wandb.init(project="eigengame", entity="rishanthrajendhran",config=hyperparameter_defaults)
config = wandb.config
wandb.run.name = f"{config.xDim}_{config.numIterations}_{config.k}_{config.learningRate}"
wandb.run.save(wandb.run.name)

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
        # if "-modified" in sys.argv:
        if config.type == "modified":
            condition = j != i
        if condition:
            penalty += (np.dot(np.dot(X, V[:,i]), np.dot(X, V[:,j]))/np.dot(np.dot(X, V[:,j]), np.dot(X, V[:,j])))*np.dot(X,V[:,j]).reshape(-1,1)
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
def updateEigenVectors(X, V, i, reward, penalty, alpha=config.learningRate):
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
        # if "-modified" in sys.argv:
        if config.type == "modified":
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
        # if "-modified" in sys.argv:
        if config.type == "modified":
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
    # if "-modified" in sys.argv:
    if config.type == "modified":
        np.save("Vs_modified.npy",Vs)
        np.save("iterTimes_modified.npy",iterTimes)
    else:
        np.save("Vs.npy",Vs)
        np.save("iterTimes.npy",iterTimes)
    return V

#-------------------------------------------------------------------------------------------------------------------------
def main():
    # if "-continueEigenGame" not in sys.argv:
    #     if "-repeatedEVtest" in sys.argv:
    #         X = [[-5,-6,3],[3,4,-3],[0,0,-2]]
    #         X = np.array(X)
    #     elif "-generateX" in sys.argv or not (os.path.exists("./X.npy") and os.path.isfile("./X.npy")):
    #         X = np.random.rand(config.xDim[0], config.xDim[1])
    #         np.save("./X.npy",X)
    # elif not (os.path.exists("./X.npy") and os.path.isfile("./X.npy")):
    #     print("Last game's dataset not found!\nStarting new game with new dataset...")

    # #Load dataset X from "./X.npy"
    # X = np.load("./X.npy")
    X = [[-5,-6,3],[3,4,-3],[0,0,-2]]
    X = np.array(X)

    if "-printX" in sys.argv:
        print(X)

    if "-analyseResults" not in sys.argv or "-playEigenGame" in sys.argv:
        # if "-modified" in sys.argv:
        if config.type == "modified":
            print("Playing the Modified EigenGame...")
        else:
            print("Playing the EigenGame...")
        V = playEigenGame(X, config.numIterations, config.k)
        EVs = getEigenVectorsK(X, config.k)
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

    if "-analyseResults" in sys.argv:
        # if "-modified" in sys.argv:
        if config.type == "modified":
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
        plt.show()
        plt.plot(iterTimes, diffs)
        plt.xlabel("Time elapsed (s)")
        plt.ylabel("Distance")
        plt.show()

if __name__ == "__main__":
    main()
