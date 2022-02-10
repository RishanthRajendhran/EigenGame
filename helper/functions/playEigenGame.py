from helper.imports.mainImports import *
from helper.imports.functionImports import *
import helper.config.mainConfig as config
import helper.config.gradientAscentConfig as gaConfig
import helper.config.miscellaneousConfig as mlConfig

#Function to find eigenvectors of a given X
#Inputs 
#   X   - Numpy array of the dataset
#   T   - Number of iterations for each player 
#   k   - Number of eigenvectors to find
#Outputs
#   Returns k eigenvectors of V as a matrix of dimensions X.shape[1] x config.k
def playEigenGame(X, T, k = config.k):
    X = np.array(X)
    V = None
    if "-continueEigenGame" in sys.argv:
        print("Continuing the last game...")
        if os.path.exists(f"./Vs/Vs_{config.xDim}_{config.variant}_{gaConfig.ascentVariant}.npy") and os.path.isfile(f"./Vs/Vs_{config.xDim}_{config.variant}_{gaConfig.ascentVariant}.npy"):
            V = np.load(f"./Vs/Vs_{config.xDim}_{config.variant}_{gaConfig.ascentVariant}.npy")[-1]
        else:
            print("Last game not found!\nStarting new game...")
            V = np.full((X.shape[1],config.k), 0.1)
            # V = np.ones((X.shape[1],config.k))
            # V = np.eye(X.shape[1],config.k)
            # V = np.random.rand(X.shape[1],config.k)
            # V = np.load("V_random_initialisation.npy")
            # if V.shape != (X.shape[1],config.k):
            #     V = np.random.rand(X.shape[1],config.k)
            #     np.save("V_random_initialisation.npy",V)
    if "-continueEigenGame" not in sys.argv or V.shape != (X.shape[1], config.k):
        V = np.full((X.shape[1],config.k), 0.1)
        # V = np.linalg.eig(X)[1]
        # V = np.array([
        #     [-0.686, 0.718, -0.113], 
        #     [0.691, 0.597, -0.407], 
        #     [0.225, 0.358, 0.906]
        # ])
        # V = np.ones((X.shape[1],config.k))
        # V = np.eye(X.shape[1],config.k)
        # V = np.load("V_random_initialisation.npy")
        # if V.shape != (X.shape[1],config.k):
        #     V = np.random.rand(X.shape[1],config.k)
        #     np.save("V_random_initialisation.npy",V)
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
        for i in range(config.k):
            for ti in range(config.L):
                startIter = time.time()
                reward = getReward(X, V, i)
                penalty = getPenalty(X, V, i)
                if "-momentum" in sys.argv:
                    gradV = getGradUtility(X, reward, penalty)
                    gradV = gradV - np.dot(gradV.T, V[:,i])*V[:,i].reshape(gradV.shape[0],1)
                    oldVi = V[:,i].copy()
                    momentum = gaConfig.gamma*momentum + gaConfig.learningRate*gradV.reshape(gradV.shape[0],) 
                    V[:,i] = V[:,i] + momentum
                    V[:,i] = V[:,i]/np.sqrt(np.sum(V[:,i]**2))
                    mlConfig.hasConverged[i] += -1 + 2*np.all(np.isclose(V[:,i], oldVi))
                    if "-checkVectors" in sys.argv:
                        V = checkVectors(V, i, oldVi)
                elif "-nesterov" in sys.argv:
                    reward = getReward(X, V-gaConfig.gamma*momentum, i)
                    penalty = getPenalty(X, V-gaConfig.gamma*momentum, i)
                    gradV = getGradUtility(X, reward, penalty)
                    gradV = gradV - np.dot(gradV.T, V[:,i])*V[:,i].reshape(gradV.shape[0],1)
                    oldVi = V[:,i].copy()
                    momentum = gaConfig.gamma*momentum + gaConfig.learningRate*gradV.reshape(gradV.shape[0],)
                    V[:,i] = V[:,i] + momentum[:,i]
                    V[:,i] = V[:,i]/np.sqrt(np.sum(V[:,i]**2))
                    if "-checkVectors" in sys.argv:
                        V = checkVectors(V, i, oldVi)
                elif "-adagrad" in sys.argv:
                    gradV = getGradUtility(X, reward, penalty)
                    gradV = gradV - np.dot(gradV.T, V[:,i])*V[:,i].reshape(gradV.shape[0],1)
                    oldVi = V[:,i].copy()
                    v = v + gradV**2
                    V[:,i] = V[:,i] + (gaConfig.learningRate/(np.sqrt(np.linalg.norm(v)+gaConfig.eps)))*gradV.reshape(gradV.shape[0],)
                    V[:,i] = V[:,i]/np.sqrt(np.sum(V[:,i]**2))
                    if "-checkVectors" in sys.argv:
                        V = checkVectors(V, i, oldVi)
                elif "-rmsprop" in sys.argv:
                    gradV = getGradUtility(X, reward, penalty)
                    gradV = gradV - np.dot(gradV.T, V[:,i])*V[:,i].reshape(gradV.shape[0],1)
                    oldVi = V[:,i].copy()
                    v = gaConfig.beta*v + (1-gaConfig.beta)*gradV**2
                    V[:,i] = V[:,i] + (gaConfig.learningRate/(np.sqrt(np.linalg.norm(v)+gaConfig.eps)))*gradV.reshape(gradV.shape[0],)
                    V[:,i] = V[:,i]/np.sqrt(np.sum(V[:,i]**2))
                    if "-checkVectors" in sys.argv:
                        V = checkVectors(V, i, oldVi)
                elif "-adam" in sys.argv:
                    gradV = getGradUtility(X, reward, penalty)
                    gradV = gradV - np.dot(gradV.T, V[:,i])*V[:,i].reshape(gradV.shape[0],1)
                    oldVi = V[:,i].copy()
                    m = gaConfig.beta1*m + (1-gaConfig.beta1)*gradV
                    v = gaConfig.beta2*v + (1-gaConfig.beta2)*(gradV**2)
                    m /= (1-gaConfig.beta1**(t+1))
                    v /= (1-gaConfig.beta2**(t+1))
                    V[:,i] = V[:,i] + (gaConfig.learningRate/(np.sqrt(np.linalg.norm(v)+gaConfig.eps)))*gradV.reshape(gradV.shape[0],)
                    V[:,i] = V[:,i]/np.sqrt(np.sum(V[:,i]**2))
                    if "-checkVectors" in sys.argv:
                        V = checkVectors(V, i, oldVi)
                else:
                    V_old = V.copy()
                    V = updateEigenVectors(X, V, i, reward, penalty)
                    # angleMeasure = (np.dot(np.transpose(V_old[:,i]),V[:,i])/(np.linalg.norm(V_old[:,i])*np.linalg.norm(V[:,i])))
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
        oldVs = np.load(f"./Vs/Vs_{config.xDim}_{config.variant}_{gaConfig.ascentVariant}.npy")
        oldIterTimes = np.load(f"./iterTimes/iterTimes_{config.xDim}_{config.variant}_{gaConfig.ascentVariant}.npy")
        Vs = np.append(oldVs, Vs.copy(),0)
        iterTimes = iterTimes + oldIterTimes[-1]
        iterTimes = np.append(oldIterTimes, iterTimes.copy(),0)
    np.save(f"./Vs/Vs_{config.xDim}_{config.variant}_{gaConfig.ascentVariant}.npy",Vs)
    np.save(f"./iterTimes/iterTimes_{config.xDim}_{config.variant}_{gaConfig.ascentVariant}.npy",iterTimes)
    return V