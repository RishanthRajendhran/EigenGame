import numpy as np
import matplotlib.pyplot as plt

def getDistance(V, EVs):
    return np.dot(V.T,EVs)/(np.linalg.norm(V)*np.linalg.norm(EVs))      #Euclidean distance

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

X = [[-5,-6,3],[3,4,-3],[0,0,-2]]
X = np.array(X)

if "-modified" in sys.argv:
    Vs = np.load("./Vs_modified.npy")
    iterTimes = np.load("./iterTimes_modified_original.npy")
else:
    Vs = np.load("./Vs.npy")
    iterTimes = np.load("./iterTimes_original.npy")
EVs = np.linalg.eig(np.dot(X.T,X))[1]
EVs = rearrange(EVs, Vs[-1])
angles = []
for col in range(Vs[0].shape[1]):
    angle = []
    for t in range(len(Vs)):
        curV = Vs[t][:,col]
        angle.append(abs(np.dot(np.transpose(curV),EVs[:,col])/(np.linalg.norm(curV)*np.linalg.norm(EVs[:,col]))))
    angles.append(angle)
angles = np.array(angles)
if "-modified" in sys.argv:
    np.save("./angles_modified_original.npy",angles)
else: 
    np.save("./angles_original.npy",angles)

for i in range(len(angles):)
    plt.plot(np.arange(len(angles[i])), angles[i], color="C"+str(i))
plt.show()

def showPlot(i):
    plt.plot(np.arange(len(angles[i])), angles[i])
    plt.show()

# fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
# fig.text(.5, .05, "\n" + "Obtained eigenvectors (blue): " + str(np.around(Vs[-1][:,pos],decimals=3)) + "\n" + "Expected eigenvector (red): " + str(np.around(EVs[:,pos],decimals=3)), ha='center')
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Z-axis')  
# for i in range(min(2000,len(Vs))):
#     for pos in range(Vs[-1].shape[1]):
#         V = Vs[i][:,pos]
#         quiverFinal = ax.quiver(0, 0, 0, V[-1][0], V[-1][1], V[-1][2], color="r")
#         # quiverFinal = ax.quiver(0, 0, 0, V[-1][0], V[-1][1], V[-1][2], color="r")
#         # quiverFinal = ax.quiver(0, 0, 0, V[-1][0], V[-1][1], V[-1][2], color="r")
#         quiver = ax.quiver(0, 0, 0, V[0][0], V[0][1], V[0][2])
#         ax.set_xlim(-1, 1)
#         ax.set_ylim(-1, 1)
#         ax.set_zlim(-1, 1)
#         def update(i):
#             global quiver 
#             quiver.remove()
#             for j in range(3):
#                 quiver = ax.quiver(0, 0, 0, V[j][0], V[j][1], V[j][2])
#         ani = FuncAnimation(fig, update, frames=np.arange(len(V)), interval=100)
#         plt.show()
#         if "-saveVisualisations" in sys.argv:
#             print("Saving visualisation. Might take a while...")
#             if "-modified" in sys.argv:
#                 ani.save(f'eigenVector{pos}_modified.mp4')
#             else:
#                 ani.save(f'eigenVector{pos}.mp4')
#             print("Visualisation saved successfully!")