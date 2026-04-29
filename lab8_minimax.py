# Asad | BSAI 4C | Roll: 141
# Lab 8: Min-Max Algorithm

import math

def minimax(curDepth, nodeIndex, maxTurn, scores, targetDepth):
    # base case: leaf node reached
    if curDepth == targetDepth:
        return scores[nodeIndex]

    if maxTurn:
        # maximizing player's turn
        return max(
            minimax(curDepth + 1, nodeIndex * 2, False, scores, targetDepth),
            minimax(curDepth + 1, nodeIndex * 2 + 1, False, scores, targetDepth)
        )
    else:
        # minimizing player's turn
        return min(
            minimax(curDepth + 1, nodeIndex * 2, True, scores, targetDepth),
            minimax(curDepth + 1, nodeIndex * 2 + 1, True, scores, targetDepth)
        )

# Scores at leaf nodes
scores = [3, 5, 2, 9, 3, 5, 2, 9]

# depth of tree (log base 2 of number of leaves)
treeDepth = int(math.log(len(scores), 2))

print("The optimal value is: ", end="")
print(minimax(0, 0, True, scores, treeDepth))
