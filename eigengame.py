import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import os.path
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.linalg import subspace_angles
from heapq import heapify, heappush, heappop

xDim = (10000, 15)
numStepsPerIteration = 200
T = 1000000
gamma = 0.9
beta = 0.9
eps = 1e-8
beta1 = 0.9
beta2 = 0.999
distanceTolerance = 0.022
ascentVariant = "vanilla"
variant = "1c"
angularThreshold = 0.5 

if "-symmetric" in sys.argv:
    variant = "2"
else:
    variant = "1"

if "-variantC" in sys.argv:
    L = numStepsPerIteration
    variant += "c"
elif "-variantB" in sys.argv:
    L = 1
    variant += "b"
else:
    L = T
    variant += "a"

if "-momentum" in sys.argv:
    ascentVariant = "momentum"
elif "-rmsprop" in sys.argv:
    ascentVariant = "rmsprop"
elif "-adagrad" in sys.argv:
    ascentVariant = "adagrad"

numIterations = T//L
k = 15
learningRate = 5e-6
tolerance = 10

#Function to get the angle (in degrees) between two vectors u and v 
#Inputs 
#   u       - Numpy column array
#   v       - Numpy column array
#Outputs
#   Returns angle between the vectors in degree measure 
def getAngle(u, v):
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    return np.rad2deg(np.arccos(np.clip(np.dot(u,v),-1.0,1.0)))

#Function to return the subspace angle 
# between current player positions (current eigenvalues) 
# and expected final player positions (i.e. expected eigenvalues)
#Inputs
#   V   - Numpy array of Current player positions
#   EVs - Numpy array of Expected final player positions
#   i   - Current player
#Outputs
#   Returns a scalar angular measure
def getSubspaceAngle(V, EVs):
    return np.sum(subspace_angles(V[:,:i+1], EVs[:, :i+1]))  

#Function to return the length of the longest correct eigenvectors streak (LCES)
#Inputs
#   EVs - Numpy array of Expected final player positions
#   Vs   - Numpy array of Current player positions
#Outputs
#   Returns an array, of length equal to number of iterations of the eigengame played,
#   containing the LCES at every iteration of the game. This array also gets saved as 
#   "LCES_<variant>.npy" in the current working directory for purpose of future analysis. 
def computeLongestCorrectEigenvectorsStreak(EVs, Vs):
    E = rearrange(EVs, Vs[-1])
    streakCounts = []
    for t in range(Vs.shape[0]):
        curStreak = 0
        V = Vs[t]
        for i in range(E.shape[1]):
            if getAngle(E[:,i], V[:,i]) <= angularThreshold or getAngle(E[:,i], -V[:,i]) <= angularThreshold:
                curStreak += 1
            else:
                break 
        streakCounts.append(curStreak)
    if "-continueEigenGame" in sys.argv:
        LCES_old = np.load(f"LCES_{variant}_{ascentVariant}.npy")
        np.append(LCES_old, np.array(streakCounts), 0)
        np.save(f"LCES_{variant}_{ascentVariant}.npy", LCES_old)  
        iterTimes = np.load(f"./iterTimes_{variant}_{ascentVariant}.npy")
        plt.plot(iterTimes, LCES_old)
        plt.xlabel("Time elapsed (s)")
        plt.ylabel("LCES")
        plt.title(f"Variant {variant} ({ascentVariant}): lr = {learningRate}, xDim = {xDim}, k = {k},L = {L}, T = {numIterations}")
        plt.savefig(f"./LCES/LCES_{variant}_{ascentVariant}")
        return LCES_old 
    else:
        np.save(f"LCES_{variant}_{ascentVariant}.npy", streakCounts) 
        iterTimes = np.load(f"./iterTimes_{variant}_{ascentVariant}.npy")
        plt.plot(iterTimes, streakCounts)   
        plt.xlabel("Time elapsed (s)")
        plt.ylabel("LCES")
        plt.title(f"Variant {variant} ({ascentVariant}): lr = {learningRate}, xDim = {xDim}, k = {k},L = {L}, T = {numIterations}")
        plt.savefig(f"./LCES/LCES_{variant}_{ascentVariant}")
        return streakCounts

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
    bLen = len(B.shape)
    if bLen == 3:
        B = B[-1]
    elif bLen != 2:
        print("Unexpected size for B in rearrange")
        exit(0)
    toRet = B.copy()
    newA = A.copy()
    for i in range(B.shape[1]):
        b = B[:,i].copy()
        minDist = np.inf 
        minCol = None 
        isNeg = False 
        for j in range(A.shape[1]):
            a = A[:,j].copy()
            dist = getDistance(b, a)
            distNeg = getDistance(b, -a)
            if dist < distNeg and dist < minDist:
                minDist = dist 
                minCol = j 
                isNeg = False 
            elif distNeg < minDist and distNeg < minDist:
                minDist = distNeg 
                minCol = j 
                isNeg = True 
        toRet[:,i] = (not isNeg)*A[:,minCol].copy() - isNeg*A[:,minCol].copy()  
    return toRet
    # toRet = B.copy()
    # heaps = [[]]*B.shape[1]
    # for i in range(B.shape[1]):
    #     heapify(heaps[i])
    #     b = B[:,i].copy()
    #     for j in range(A.shape[1]):
    #         a = A[:,j].copy()
    #         dist = getDistance(b, a)
    #         distNeg = getDistance(b, -a)
    #         isNeg = False 
    #         curMinDist = dist
    #         if distNeg < dist:
    #             isNeg = True 
    #             curMinDist = distNeg
    #         heappush(heaps[i], (curMinDist, j, isNeg))
    # alreadyMatched = [False]*(B.shape[1])
    # for i in range(B.shape[1]):
    #     minEntry = None
    #     while True:
    #         minEntry = heappop(heaps[i])
    #         if not alreadyMatched[minEntry[1]]:
    #             alreadyMatched[minEntry[1]] = True
    #             break
    #     minCol = minEntry[1]
    #     isNeg = minEntry[2]
    #     toRet[:,i] = (not isNeg)*A[:,minCol].copy() - isNeg*A[:,minCol].copy()  
    # return toRet

# def rearrange(A, B):
#     toRet = B.copy()
#     newA = A.copy()
#     for i in range(A.shape[1]):
#         a = A[:,i]
#         minDist = np.inf 
#         minCol = i 
#         isNeg = False
#         for j in range(B.shape[1]):
#             b = np.around(B[:,j],decimals=3).copy()
#             dist = getDistance(a, b)
#             distNeg = getDistance(a, -b)
#             if dist < distNeg and dist < minDist:
#                 minDist = dist 
#                 minCol = j
#                 isNeg = False
#             elif distNeg < minDist:
#                 minDist = distNeg 
#                 minCol = j
#                 isNeg = True
#         newA[:,minCol] = (not isNeg)*A[:,i].copy() - isNeg*A[:,i].copy()
#     for i in range(toRet.shape[1]):
#         toRet[:,i] = newA[:, i]
#     return toRet

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
        if "-symmetric" in sys.argv:
            condition = j != i
        if condition:
            dotProd = (np.dot(np.dot(X, V[:,i]), np.dot(X, V[:,j]))/np.dot(np.dot(X, V[:,j]), np.dot(X, V[:,j])))*np.dot(X,V[:,j]).reshape(-1,1)
            penalty += 10*dotProd
            # penalty += 1*abs(dotProd)
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
        if "-symmetric" in sys.argv:
            if i != curPlayer and 0 <= getAngle(V[:,i], V[:, curPlayer]) <= tolerance and 180-tolerance <= getAngle(V[:,i], V[:, curPlayer]) <= 180:
                V[:,curPlayer] = -oldPos
                break
        else:
            if i < curPlayer and 0 <= getAngle(V[:,i], V[:, curPlayer]) <= tolerance and 180-tolerance <= getAngle(V[:,i], V[:, curPlayer]) <= 180:
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
    if "-checkVectors" in sys.argv:
        V = checkVectors(V, i, oldVi)
    return V

#Function to find eigenvectors of a given X using Numpy
#Inputs 
#   X   - Numpy array of the dataset
#Outputs
#   Returns all the eigenvectors of V as a matrix of dimensions X.shape[1] x X.shape[1]
def getEigenVectors(X):
    return np.linalg.eig(np.dot(X.T, X))[1]

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
        if os.path.exists(f"Vs_{variant}_{ascentVariant}.npy") and os.path.isfile(f"Vs_{variant}_{ascentVariant}.npy"):
            V = np.load(f"Vs_{variant}_{ascentVariant}.npy")[-1]
        else:
            print("Last game not found!\nStarting new game...")
            # V = np.ones((X.shape[1],k))
            V = np.random.rand(X.shape[1],k)
    if "-continueEigenGame" not in sys.argv or V.shape != (X.shape[1], k):
        # V = np.ones((X.shape[1],k))
        # V = np.eye(X.shape[1],k)
        V = np.random.rand(X.shape[1],k)
    Vs = [V.copy()]
    iterTimes = [0]      #Array to store time taken for every iteration
    iterTimesSum = 0    #Variable to keep track of total time elapsed
    if "-momentum" in sys.argv:
        momentum = 0
    elif "-nesterov" in sys.argv:
        momentum = np.zeros(V.shape)
    elif "-adagrad" in sys.argv or "-rmsprop" in sys.argv:
        v = 0
    elif "-adam" in sys.argv:
        m = 0
        v = 0
    
    for t in range(T):
        startIter = time.time()
        for i in range(k):
            for ti in range(L):
                reward = getReward(X, V, i)
                penalty = getPenalty(X, V, i)
                if "-momentum" in sys.argv:
                    gradV = getGradUtility(X, reward, penalty)
                    gradV = gradV - np.dot(gradV.T, V[:,i])*V[:,i].reshape(gradV.shape[0],1)
                    oldVi = V[:,i].copy()
                    momentum = gamma*momentum + learningRate*gradV.reshape(gradV.shape[0],)
                    V[:,i] = V[:,i] + momentum
                    V[:,i] = V[:,i]/np.sqrt(np.sum(V[:,i]**2))
                    if "-checkVectors" in sys.argv:
                        V = checkVectors(V, i, oldVi)
                elif "-nesterov" in sys.argv:
                    reward = getReward(X, V-gamma*momentum, i)
                    penalty = getPenalty(X, V-gamma*momentum, i)
                    gradV = getGradUtility(X, reward, penalty)
                    gradV = gradV - np.dot(gradV.T, V[:,i])*V[:,i].reshape(gradV.shape[0],1)
                    oldVi = V[:,i].copy()
                    momentum = gamma*momentum + learningRate*gradV.reshape(gradV.shape[0],)
                    V[:,i] = V[:,i] + momentum[:,i]
                    V[:,i] = V[:,i]/np.sqrt(np.sum(V[:,i]**2))
                    if "-checkVectors" in sys.argv:
                        V = checkVectors(V, i, oldVi)
                elif "-adagrad" in sys.argv:
                    gradV = getGradUtility(X, reward, penalty)
                    gradV = gradV - np.dot(gradV.T, V[:,i])*V[:,i].reshape(gradV.shape[0],1)
                    oldVi = V[:,i].copy()
                    v = v + gradV**2
                    V[:,i] = V[:,i] + (learningRate/(np.sqrt(np.linalg.norm(v)+eps)))*gradV.reshape(gradV.shape[0],)
                    V[:,i] = V[:,i]/np.sqrt(np.sum(V[:,i]**2))
                    if "-checkVectors" in sys.argv:
                        V = checkVectors(V, i, oldVi)
                elif "-rmsprop" in sys.argv:
                    gradV = getGradUtility(X, reward, penalty)
                    gradV = gradV - np.dot(gradV.T, V[:,i])*V[:,i].reshape(gradV.shape[0],1)
                    oldVi = V[:,i].copy()
                    v = beta*v + (1-beta)*gradV**2
                    V[:,i] = V[:,i] + (learningRate/(np.sqrt(np.linalg.norm(v)+eps)))*gradV.reshape(gradV.shape[0],)
                    V[:,i] = V[:,i]/np.sqrt(np.sum(V[:,i]**2))
                    if "-checkVectors" in sys.argv:
                        V = checkVectors(V, i, oldVi)
                elif "-adam" in sys.argv:
                    gradV = getGradUtility(X, reward, penalty)
                    gradV = gradV - np.dot(gradV.T, V[:,i])*V[:,i].reshape(gradV.shape[0],1)
                    oldVi = V[:,i].copy()
                    m = beta1*m + (1-beta1)*gradV
                    v = beta2*v + (1-beta2)*(gradV**2)
                    if t==0:
                        print(m)
                        print(v)
                        print(beta1)
                        print(beta2)
                        print(m/(1-beta1**(t+1)))
                        print(v/(1-beta2**(t+1)))
                    m /= (1-beta1**(t+1))
                    v /= (1-beta2**(t+1))
                    V[:,i] = V[:,i] + (learningRate/(np.sqrt(np.linalg.norm(v)+eps)))*gradV.reshape(gradV.shape[0],)
                    V[:,i] = V[:,i]/np.sqrt(np.sum(V[:,i]**2))
                    if "-checkVectors" in sys.argv:
                        V = checkVectors(V, i, oldVi)
                else:
                    V_old = V.copy()
                    V = updateEigenVectors(X, V, i, reward, penalty)
                    angleMeasure = (np.dot(np.transpose(V_old[:,i]),V[:,i])/(np.linalg.norm(V_old[:,i])*np.linalg.norm(V[:,i])))
                    # if not t%100 and i == 0:
                    #     print(f"{t} => {angleMeasure}")

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
        oldVs = np.load(f"Vs_{variant}_{ascentVariant}.npy")
        oldIterTimes = np.load(f"iterTimes_{variant}_{ascentVariant}.npy")
        Vs = np.append(oldVs, Vs.copy(),0)
        iterTimes = iterTimes + np.sum(oldIterTimes)
        iterTimes = np.append(oldIterTimes, iterTimes.copy(),0)
    np.save(f"Vs_{variant}_{ascentVariant}.npy",Vs)
    np.save(f"iterTimes_{variant}_{ascentVariant}.npy",iterTimes)
    return V

#-------------------------------------------------------------------------------------------------------------------------
if "-momentum" in sys.argv:
        ascentVariant = "momentum"
elif "-nesterov" in sys.argv:
    ascentVariant = "nesterov"
elif "-adagrad" in sys.argv:
    ascentVariant = "adagrad"
elif "-rmsprop" in sys.argv:
    ascentVariant = "rmsprop"
elif "-adam" in sys.argv:
    ascentVariant = "adam"

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
if k > X.shape[1]:
    k = X.shape[1]

if ("-analyseResults" not in sys.argv and "-visualiseResults" not in sys.argv and "-visualiseResultsTogether" not in sys.argv and "-analyseAngles" not in sys.argv and "-analyseAnglesTogether" not in sys.argv) or "-playEigenGame" in sys.argv:
    if "-symmetric" in sys.argv:
        print(f"Playing the symmetric penalty EigenGame (variant {variant[-1]}, {ascentVariant})...")
    else:
        print(f"Playing the asymmetric EigenGame (variant {variant[-1]}, {ascentVariant})...")
    startGame = time.time()
    V = playEigenGame(X, numIterations, k)
    stopGame = time.time()
    print(f"Time taken: {stopGame-startGame}s")
    EVs = getEigenVectors(X)
    EVs = rearrange(EVs, V)
    print("EigenVectors obtained through EigenGame:")
    print(np.around(V,decimals=3))
    print("\nEigenVectors obtained through numpy:")
    print(np.around(EVs,decimals=3))
    print(f"Learning Rate : {learningRate}")
    print(f"Distance measure: {np.around(getDistance(V,EVs), decimals=3)}")

    #Finding time taken for convergence
    Vs = np.load(f"Vs_{variant}_{ascentVariant}.npy")
    iterTimes = np.load(f"iterTimes_{variant}_{ascentVariant}.npy")
    EVs = np.around(getEigenVectors(X),decimals=3)
    EVs = rearrange(EVs, Vs[-1])
    convergenceTime = None
    for i in range(len(Vs)):
        V = Vs[i]
        distanceMeasure = np.around(getDistance(V,EVs), decimals=3)
        if distanceMeasure <= distanceTolerance:
            convergenceTime = iterTimes[i]
            break 
    if convergenceTime == None:
        print("EigenGame did not converge as per expectation!")
    else:
        print(f"Time taken to converge as per expectation: {convergenceTime} s")

if "-computeLCES" in sys.argv:
    Vs = np.load(f"Vs_{variant}_{ascentVariant}.npy")
    EVs = np.around(getEigenVectors(X),decimals=3)
    EVs = rearrange(EVs, Vs[-1])
    print("EigenVectors obtained through EigenGame:")
    for V in Vs:
        diffs.append(getDistance(V,EVs))
        if "-debug" in sys.argv:
            print(np.around(V,decimals=3))
    if "debug" not in sys.argv:
        print(np.around(Vs[-1],decimals=3))
    print("\nEigenVectors obtained through numpy:")
    print(np.around(EVs,decimals=3))
    LCES = computeLongestCorrectEigenvectorsStreak(EVs, Vs)
    print(f"Sum of LCES at the end of {len(LCES)} iterations: {sum(LCES)}")

if "-analyseResults" in sys.argv:
    Vs = np.load(f"Vs_{variant}_{ascentVariant}.npy")
    iterTimes = np.load(f"iterTimes_{variant}_{ascentVariant}.npy")
    EVs = np.around(getEigenVectors(X),decimals=3)
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
    plt.title(f"Variant {variant} ({ascentVariant}): lr = {learningRate}, xDim = {xDim}, k = {k},L = {L}, T = {numIterations}")
    if "-savePlots" in sys.argv or "-saveMode" in sys.argv:
        plt.savefig(f"./plots/distanceVSiterations_{variant}")
    if "-saveMode" not in sys.argv:
        plt.show()
    plt.plot(iterTimes, diffs)
    plt.xlabel("Time elapsed (s)")
    plt.ylabel("Distance")
    plt.title(f"Variant {variant} ({ascentVariant}): lr = {learningRate}, xDim = {xDim}, k = {k},L = {L}, T = {numIterations}")
    if "-savePlots" in sys.argv or "-saveMode" in sys.argv:
        plt.savefig(f"./plots/distanceVStotalTimeElapsed_{variant}")
    if "-saveMode" not in sys.argv:
        plt.show()

if "-visualiseResults" in sys.argv and "-3D" in sys.argv:
    Vs = np.load(f"./Vs_{variant}_{ascentVariant}.npy")
    if Vs[-1].shape[0] != 3:
        print("Only 3D visualisations allowed!")
        exit(0)

    visualisationSpeed = 1         #Default speed : highSpeed
    if "-mediumSpeed" in sys.argv:
        visualisationSpeed = 500 
    elif "-lowSpeed" in sys.argv:
        visualisationSpeed = 1000

    EVs = np.around(getEigenVectors(X),decimals=3)
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
        plt.title(f"Variant {variant} ({ascentVariant}): lr = {learningRate}, xDim = {xDim}, k = {k},L = {L}, T = {numIterations}")
        fig.text(.5, .05, "\n" + "Obtained eigenvectors (blue): " + str(np.around(Vs[-1][:,pos],decimals=3)) + "\n" + "Expected eigenvector (red): " + str(np.around(EVs[:,pos],decimals=3)), ha='center')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        quiverFinal = ax.quiver(0, 0, 0, V[-1][0], V[-1][1], V[-1][2], color="r")
        quiver = ax.quiver(0, 0, 0, V[0][0], V[0][1], V[0][2])
        ax.set_xlim(minX-0.1, maxX+0.1)
        ax.set_ylim(minY-0.1, maxY+0.1)
        ax.set_zlim(minZ-0.1, maxZ+0.1)

        # r = 1
        # pi = np.pi
        # cos = np.cos
        # sin = np.sin
        # phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
        # x = r*sin(phi)*cos(theta)
        # y = r*sin(phi)*sin(theta)
        # z = r*cos(phi)
        # ax.plot_surface(
        # x, y, z,  rstride=1, cstride=1, color='c', alpha=0.3, linewidth=0)

        def update(i):
            global quiver 
            quiver.remove()
            quiver = ax.quiver(0, 0, 0, V[i][0], V[i][1], V[i][2])
        ani = FuncAnimation(fig, update, frames=np.arange(len(V)), interval=visualisationSpeed)
        if "-saveMode" not in sys.argv:
            plt.show()
        if "-saveVisualisations" in sys.argv or "-saveMode" in sys.argv:
            print("Saving visualisation. Might take a while...")
            ani.save(f'./visualisations/eigenVector{pos}_{variant}.mp4')
            print("Visualisation saved successfully!")

if "-visualiseResultsTogether" in sys.argv and "-3D" in sys.argv:
    Vs = np.load(f"./Vs_{variant}_{ascentVariant}.npy")

    if Vs[-1].shape[0] != 3:
        print("Only 3D visualisations allowed!")
        exit(0)

    visualisationSpeed = 1         #Default speed : highSpeed
    if "-mediumSpeed" in sys.argv:
        visualisationSpeed = 500 
    elif "-lowSpeed" in sys.argv:
        visualisationSpeed = 1000

    EVs = np.around(getEigenVectors(X),decimals=3)
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
    plt.title(f"Variant {variant} ({ascentVariant}): lr = {learningRate}, xDim = {xDim}, k = {k},L = {L}, T = {numIterations}")
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
    if "-saveVisualisations" in sys.argv or "-saveMode" in sys.argv:
        print("Saving visualisation. Might take a while...")
        ani.save(f'./visualisations/eigenVectors_{variant}.mp4')
        print("Visualisation saved successfully!")

if "-analyseAngles" in sys.argv or "-analyseAnglesTogether" in sys.argv:
    Vs = np.load(f"./Vs_{variant}_{ascentVariant}.npy")
    iterTimes = np.load(f"./iterTimes_{variant}_{ascentVariant}.npy")
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
    np.save(f"./angles_{variant}_{ascentVariant}.npy",angles)
    pltTitle = f"Variant {variant} ({ascentVariant}): lr = {learningRate}, xDim = {xDim}, k = {k},L = {L}, T = {numIterations}"
    for i in range(len(angles)):
        plt.xlabel("Iterations")
        plt.ylabel("Angle between obtained EV and expected EV")
        plt.title(pltTitle)
        plt.plot(np.arange(len(angles[i])), angles[i], color="C"+str(i))
        if "-analyseAnglesTogether" not in sys.argv:
            if "-savePlots" in sys.argv or "-saveMode" in sys.argv:
                plt.savefig(f"./plots/anglesVSiterations{i}_{variant}")
            if "-saveMode" not in sys.argv:
                plt.show()
    if "-analyseAnglesTogether" in sys.argv:
        if "-savePlots" in sys.argv or "-saveMode" in sys.argv:
            plt.savefig(f"./plots/anglesVSiterations_{variant}")
        if "-saveMode" not in sys.argv:
            plt.show()
    
    for i in range(len(angles)):
        plt.xlabel("Total Time Elapsed")
        plt.ylabel("Angle between obtained EV and expected EV")
        plt.title(pltTitle)
        plt.plot(iterTimes, angles[i], color="C"+str(i))
        if "-analyseAnglesTogether" not in sys.argv:
            if "-savePlots" in sys.argv or "-saveMode" in sys.argv:
                plt.savefig(f"./plots/anglesVStotalTimeElapsed{i}_{variant}")
            if "-saveMode" not in sys.argv:
                plt.show()
    if "-analyseAnglesTogether" in sys.argv:
        if "-savePlots" in sys.argv or "-saveMode" in sys.argv:
            plt.savefig(f"./plots/anglesVStotalTimeElapsed_{variant}")
        if "-saveMode" not in sys.argv:
            plt.show()

if "-analyseSubspaceAngles" in sys.argv:
    Vs = np.load(f"./Vs_{variant}_{ascentVariant}.npy")
    iterTimes = np.load(f"./iterTimes_{variant}_{ascentVariant}.npy")
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
    np.save(f"./subspaceAngles_{variant}_{ascentVariant}.npy",angles)
    pltTitle = f"Variant {variant} ({ascentVariant}): lr = {learningRate}, xDim = {xDim}, k = {k},L = {L}, T = {numIterations}"
    plt.xlabel("Iterations")
    plt.ylabel("Subspace Angle between obtained EV and expected EV")
    plt.title(pltTitle)
    plt.plot(np.arange(len(angles)), angles)
    if "-savePlots" in sys.argv or "-saveMode" in sys.argv:
        plt.savefig(f"./plots/subspaceAnglesVSiterations_{variant}")
    if "-saveMode" not in sys.argv:
        plt.show()

    plt.xlabel("Total Time Elapsed")
    plt.ylabel("Angle between obtained EV and expected EV")
    plt.title(pltTitle)
    plt.plot(iterTimes, angles)
    if "-savePlots" in sys.argv or "-saveMode" in sys.argv:
        plt.savefig(f"./plots/subspaceAnglesVStotalTimeElapsed_{variant}")
    if "-saveMode" not in sys.argv:
        plt.show()



