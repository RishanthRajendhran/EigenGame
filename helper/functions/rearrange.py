from helper.imports.mainImports import *
from helper.imports.functionImports import *

#Erroneous implementation
#Need to implement maximum matching in a bipartite graph
#Function to rearrange columns of matrix A based on their closest matching column index in matrix B
#Inputs 
#   A   - Matrix whose columns have to be rearranged
#   B   - Matrix which is the standard based on which rearrangements have to be done
#Outputs
#   Returns rearranged A matrix
def rearrange(A, B):
    # dists = []
    # for i in range(B.shape[1]):
    #     dist = []
    #     b = B[:,i].copy()
    #     for j in range(A.shape[1]):
    #         a = A[:, j].copy()
    #         dist.append(min(getDistance(a, b), getDistance(-a, b)))
    #     dists.append(dist)
    # dist = np.array(dist)
    # resverseDists = np.zeros(dist.shape)
    # for i in range(dist.shape[0]):
    #     for j in range(dist.shape[1]):
    #         resverseDists[i][j] = dists[j][i]

    bLen = len(B.shape)
    if bLen == 3:
        B = B[-1]
    elif bLen != 2:
        print("Unexpected size for B in rearrange")
        exit(0)
    toRet = B.copy()
    newA = A.copy()
    alreadyAllotted = [0]*A.shape[1]
    for i in range(B.shape[1]):
        b = B[:,i].copy()
        minDist = np.inf 
        minCol = None 
        isNeg = False 
        for j in range(A.shape[1]):
            a = A[:,j].copy()
            dist = getDistance(b, a)
            distNeg = getDistance(b, -a)
            if dist < distNeg and dist < minDist and alreadyAllotted[j] == 0:
                minDist = dist 
                minCol = j 
                isNeg = False 
            elif distNeg < minDist and distNeg < minDist and alreadyAllotted[j] == 0:
                minDist = distNeg 
                minCol = j 
                isNeg = True 
        alreadyAllotted[minCol] = 1
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