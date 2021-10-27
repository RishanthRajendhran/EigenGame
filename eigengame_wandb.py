import numpy as np
import matplotlib.pyplot as plt
import time
import os.path
import wandb

hyperparameter_defaults = dict(
    xDim = (10000, 7),
    numIterations = 35000,
    numStepsPerIteration = 100,
    k = 7,
    learningRate = 1,
    flags = ["-rmsprop"],
    isSymmetric = False,
    variant = "b",
    tolerance = 10,
    distanceTolerance = 0.01
)

run = wandb.init(project="eigengame", entity="rishanthrajendhran",config=hyperparameter_defaults)
config = wandb.config
wandb.run.name = f"{config.xDim}_{config.numIterations}_{config.k}_{config.learningRate}"
wandb.run.save(wandb.run.name)

xDim = config.xDim
numIterations = config.numIterations
k = config.k
learningRate = config.learningRate
numStepsPerIteration = config.numStepsPerIteration
tolerance = config.tolerance
distanceTolerance = config.distanceTolerance
ascentVariant = "vanilla"

gamma = 0.9
beta = 0.9
eps = 1e-8
beta1 = 0.9
beta2 = 0.999

if config.variant == "c":
    L = numStepsPerIteration
elif config.variant == "b":
    L = 1
else:
    L = numIterations

numIterations = numIterations//L

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
    toRet = B.copy()
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
    for i in range(toRet.shape[1]):
        toRet[:,i] = newA[:, i]
    return toRet

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
        if config.isSymmetric:
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
        if config.isSymmetric:
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
    if "-checkVectors" in config.flags:
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
    V = np.random.rand(X.shape[1],k)
    Vs = [V.copy()]
    iterTimes = [0]      #Array to store time taken for every iteration
    iterTimesSum = 0    #Variable to keep track of total time elapsed
    if "-momentum" in config.flags:
        momentum = 0
    elif "-nesterov" in config.flags:
        momentum = np.zeros(V.shape)
    elif "-adagrad" in config.flags or "-rmsprop" in config.flags:
        v = 0
    elif "-adam" in config.flags:
        m = 0
        v = 0
    for t in range(T):
        startIter = time.time()
        for i in range(k):
            for ti in range(L):
                reward = getReward(X, V, i)
                penalty = getPenalty(X, V, i)
                if "-momentum" in config.flags:
                    gradV = getGradUtility(X, reward, penalty)
                    gradV = gradV - np.dot(gradV.T, V[:,i])*V[:,i].reshape(gradV.shape[0],1)
                    oldVi = V[:,i].copy()
                    momentum = gamma*momentum + learningRate*gradV.reshape(gradV.shape[0],)
                    V[:,i] = V[:,i] + momentum
                    V[:,i] = V[:,i]/np.sqrt(np.sum(V[:,i]**2))
                    if "-checkVectors" in config.flags:
                        V = checkVectors(V, i, oldVi)
                elif "-nesterov" in config.flags:
                    reward = getReward(X, V-gamma*momentum, i)
                    penalty = getPenalty(X, V-gamma*momentum, i)
                    gradV = getGradUtility(X, reward, penalty)
                    gradV = gradV - np.dot(gradV.T, V[:,i])*V[:,i].reshape(gradV.shape[0],1)
                    oldVi = V[:,i].copy()
                    momentum = gamma*momentum + learningRate*gradV.reshape(gradV.shape[0],)
                    V[:,i] = V[:,i] + momentum[:,i]
                    V[:,i] = V[:,i]/np.sqrt(np.sum(V[:,i]**2))
                    if "-checkVectors" in config.flags:
                        V = checkVectors(V, i, oldVi)
                elif "-adagrad" in config.flags:
                    gradV = getGradUtility(X, reward, penalty)
                    gradV = gradV - np.dot(gradV.T, V[:,i])*V[:,i].reshape(gradV.shape[0],1)
                    oldVi = V[:,i].copy()
                    v = v + gradV**2
                    V[:,i] = V[:,i] + (learningRate/(np.sqrt(np.linalg.norm(v)+eps)))*gradV.reshape(gradV.shape[0],)
                    V[:,i] = V[:,i]/np.sqrt(np.sum(V[:,i]**2))
                    if "-checkVectors" in config.flags:
                        V = checkVectors(V, i, oldVi)
                elif "-rmsprop" in config.flags:
                    gradV = getGradUtility(X, reward, penalty)
                    gradV = gradV - np.dot(gradV.T, V[:,i])*V[:,i].reshape(gradV.shape[0],1)
                    oldVi = V[:,i].copy()
                    v = beta*v + (1-beta)*gradV**2
                    V[:,i] = V[:,i] + (learningRate/(np.sqrt(np.linalg.norm(v)+eps)))*gradV.reshape(gradV.shape[0],)
                    V[:,i] = V[:,i]/np.sqrt(np.sum(V[:,i]**2))
                    if "-checkVectors" in config.flags:
                        V = checkVectors(V, i, oldVi)
                elif "-adam" in config.flags:
                    gradV = getGradUtility(X, reward, penalty)
                    gradV = gradV - np.dot(gradV.T, V[:,i])*V[:,i].reshape(gradV.shape[0],1)
                    oldVi = V[:,i].copy()
                    m = beta1*m + (1-beta1)*gradV
                    v = beta2*v + (1-beta2)*(gradV**2)
                    m /= (1-beta1**(t+1))
                    v /= (1-beta2**(t+1))
                    V[:,i] = V[:,i] + (learningRate/(np.sqrt(np.linalg.norm(v)+eps)))*gradV.reshape(gradV.shape[0],)
                    V[:,i] = V[:,i]/np.sqrt(np.sum(V[:,i]**2))
                    if "-checkVectors" in config.flags:
                        V = checkVectors(V, i, oldVi)
                else:
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
    if config.isSymmetric:
        np.save(f"Vs_2{config.variant}.npy",Vs)
        np.save(f"iterTimes_2{config.variant}.npy",iterTimes)
    else:
        np.save(f"Vs_1{config.variant}.npy",Vs)
        np.save(f"iterTimes_1{config.variant}.npy",iterTimes)
    return V

#-------------------------------------------------------------------------------------------------------------------------
if "-momentum" in config.flags:
        ascentVariant = "momentum"
elif "-nesterov" in config.flags:
    ascentVariant = "nesterov"
elif "-adagrad" in config.flags:
    ascentVariant = "adagrad"
elif "-rmsprop" in config.flags:
    ascentVariant = "rmsprop"
elif "-adam" in config.flags:
    ascentVariant = "adam"

#Load dataset X from "./X.npy"
if "-repeatedEVtest" in config.flags:
    X = [[-5,-6,3],[3,4,-3],[0,0,-2]]
    X = np.array(X)
    xDim = X.shape
elif "-repeatedEVtest2" in config.flags:
    X = np.load("./repeatedEV_X.npy")
    xDim = X.shape
else:
    X = np.load("./X.npy")
    
if "-printX" in config.flags:
    print(X)

if config.isSymmetric:
    print(f"Playing the symmetric penalty EigenGame (Variant {config.variant}, {ascentVariant})...")
else:
    print(f"Playing the asymmetric EigenGame (Variant {config.variant}, {ascentVariant})...")
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
#Finding timee taken for convergence
curVariant = "1"+config.variant
if config.isSymmetric:
    curVariant = "2"+config.variant
Vs = np.load(f"Vs_{curVariant}.npy")
iterTimes = np.load(f"iterTimes_{curVariant}.npy")
EVs = np.around(getEigenVectors(X),decimals=3)
EVs = rearrange(EVs, Vs[-1])
convergenceTime = float("inf")
for i in range(len(Vs)):
    V = Vs[i]
    distanceMeasure = np.around(getDistance(V,EVs), decimals=3)
    if distanceMeasure <= distanceTolerance:
        convergenceTime = iterTimes[i]
        break 

print(f"Time taken to converge as per expectation: {convergenceTime} s")

metrics = {
    'distance_measure': np.around(getDistance(Vs[-1],EVs), decimals=3),
    'convergence_time': convergenceTime,
}
wandb.log(metrics)
run.finish()