import edist
from edist import uted
# import zss
# from zss import simple_distance, Node
# from treeMatch_partial_match2 import *

def makeAdjacencyList(treeDict,node,result):
    if node is None:
        return result
    else:
        result.append([treeDict[n] for n in node.children])
        # print(node.name, d[node])
        for c in node.children:
            makeAdjacencyList(treeDict, c , result)
    return result

def getEditDistance(tTree, cTree):
    tNodes = [node.name for node in tTree.nodes]
    cNodes = [node.name for node in cTree.nodes]
    tDict = {k: v for v, k in enumerate(tTree.nodes)}
    cDict = {k: v for v, k in enumerate(cTree.nodes)}
    tAdj = makeAdjacencyList(tDict, tTree.root, [])
    cAdj = makeAdjacencyList(cDict, cTree.root, [])
    result = edist.uted.uted(tNodes,tAdj,cNodes, cAdj)
    return result

if __name__ == "__main__":
    newTree = FastTreeMatch()
    # tTree = newTree.makeATree("candidate/madelynnTests/candidate_4.xml")
    tTree = newTree.makeATree("old_target/target_2.xml")
    cTree = newTree.makeATree("candidate/madelynnTests/candidate_8.xml")
    # cTree = newTree.makeATree("candidate/madelynnTests/candidate_6.xml")
    result = getEditDistance(tTree, cTree)
    # print(result)


