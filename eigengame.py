import sys
import numpy as np
import matplotlib.pyplot as plt

xDim = (100000, 4)
numIterations = 100000
k = 4

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
        if j < i:
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
def updateEigenVectors(X, V, i, reward, penalty, alpha=0.00000075):
    gradV = getGradUtility(X, reward, penalty)
    gradV = gradV - np.dot(gradV.T, V[:,i])*V[:,i].reshape(gradV.shape[0],1)
    V[:,i] = V[:,i] + alpha*gradV.reshape(gradV.shape[0],)
    V[:,i] = V[:,i]/np.sqrt(np.sum(V[:,i]**2))
    return V

#Function to find eigenvectors of a given X
#Inputs 
#   X   - Numpy array of the dataset
#   T   - Number of iterations for each player 
#   k   - Number of eigenvectors to find
#Outputs
#   Returns k eigenvectors of V as a matrix of dimensions X.shape[1] x k
def playEigenGame(X, T, k):
    X = np.array(X)
    V = np.ones((X.shape[1],k))
    Vs = [V.copy()]
    for i in range(k):
        for t in range(T):
            reward = getReward(X, V, i)
            penalty = getPenalty(X, V, i)
            V = updateEigenVectors(X, V, i, reward, penalty)
            Vs.append(V.copy())
    Vs = np.array(Vs)
    np.save("Vs_modified.npy",Vs)
    return V

if "-generateX" in sys.argv:
    X = np.random.rand(xDim[0], xDim[1])
    np.save("./X.npy",X)
X = np.load("./X.npy")
# X = [[-5,-6,3],[3,4,-3],[0,0,-2]]
# X = np.array(X)
# print(X)
if "-analyseResults" in sys.argv:
    Vs_modified = np.load("Vs_modified.npy")
    EVs = np.around(np.linalg.eig(np.dot(X.T, X))[1][:,:k],decimals=3)
    EVs = -EVs
    temp = EVs[:,2].copy()
    EVs[:,2] = EVs[:,3].copy() 
    EVs[:,3] = -temp.copy()
    print(EVs)
    diffs = []
    for V in Vs_modified:
        dist = np.sqrt(np.sum((V - EVs) ** 2))
        diffs.append(dist)
        print(dist)
    plt.plot(diffs)
    plt.show()
    print(Vs_modified.shape)
else:
    print("Playing the EigenGame...")
    V = playEigenGame(X, numIterations, k)
    print("EigenVectors obtained through EigenGame:")
    print(np.around(V,decimals=3))
    print("\nEigenVectors obtained through numpy:")
    print(np.around(np.linalg.eig(np.dot(X.T, X))[1][:,:k],decimals=3))
