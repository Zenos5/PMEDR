import errno
import itertools
import os
import sys
# pip install pptree
import pptree
from statistics import mean
import operator
import TreeBuilder
import NodeBuilder
import TreeParser
import treeEditDistanceUnorder

sys.modules['_elementtree'] = None
import xml.etree.ElementTree as ET

class FastTreeMatch:
    def __init__(self):
        self.numFiles = 0
        self.docNumber = 0
        self.targetTree = TreeBuilder.Tree()
        self.candidateTree = TreeBuilder.Tree()
        self.depth_score_list=[]
        self.longest_submatch={}
        self.temp_depth=0
        self.matchingNodes = []
        self.ranking = []
        #         Tq would be one level. such as q= a, then Ta = {a1, a2, a3 .. an} and all would only have one ancestor or parent
        #         Tq, list of sorted occurrences of q. represented by a triplet, (start,end,level)

    def insertNodeToTree(self, tree, data, parent):
        # insert the current data
        name = data.tag
        if '}' in name:
            name = data.tag.split("}", 1)[1] # ignore anything in the {}
        if name not in ['annotation', 'style', 'error']:
            parentNode = NodeBuilder.Node(name, data.start_line_number, data.end_line_number, parent)
            tree.insert(parentNode, parent)
            childNode = NodeBuilder.Node()
            # if the current tag has value, then insert value as a child
            if data.text and data.text.strip():
                lineNumber = (data.end_line_number - data.start_line_number) // 2 + data.start_line_number
                children = [data.text]
                childNode = NodeBuilder.Node(data.text, lineNumber, lineNumber, parentNode)
                # parentNode.add_child(childNode)
                tree.insert(childNode, parentNode)

            for child in data:
                if data.text and data.text.strip():
                    self.insertNodeToTree(tree, child, childNode)
                else:
                    self.insertNodeToTree(tree, child, parentNode)
        return tree

    def makeATree(self, fileName):
        root = ET.parse(fileName, TreeParser.LineNumberingParser()).getroot()
        tree = TreeBuilder.Tree()
        # data, start, end, parent, children
        tree = self.insertNodeToTree(tree, root, None)
        return tree

    def run(self, targetFile, dataset):
        matchingFile = {}
        self.numFiles = len(dataset)
        try:
            self.targetTree = self.makeATree(targetFile)
            # self.targetTree.root.display()
            # print("targetTree")
            # print(len(self.targetTree.nodes))
        except:
            return None
        scores={}
        for f in dataset:
            try:
                self.candidateTree = self.makeATree(f)
                # self.candidateTree.root.display()
                # print("Candidate_math 8")
                # print(len(self.candidateTree.nodes))

            except:
                continue
            self.depth_score_list=[]
            self.longest_submatch={}
            self.matchingNodes = []
            self.editDistanceScore=0
            self.matchingDepth = 0
            result = self.treeMatch(self.targetTree.root)
            targetNodesNames = [x.name for x in self.targetTree.nodes]
            candidateNodesNames = [x.name for x in self.candidateTree.nodes]
            union= len(targetNodesNames) + len(candidateNodesNames)
            # TODO: final score
            # print(self.matchingNodes)

            if result:
                score = 1
            else:
                editDistance = treeEditDistanceUnorder.getEditDistance(self.targetTree, self.candidateTree)
                self.editDistanceScore = 1- (editDistance/union)
                # score = ( (sum(self.depth_score_list)/len(self.depth_score_list))+ (len(self.matchingNodes)/union) )/2
                score = ( (sum(self.depth_score_list)/len(self.depth_score_list))+ self.editDistanceScore )/2
            scores[f]=score

        return scores
        #     if result:
        #         self.ranking.append((f, score))
        #     else:
        #         matchingFile[f] = score
        # # Result, if the matchingFile is empty means we did not find a matching file
        # ranked_file = sorted(matchingFile.items(), key=lambda x: mean(x[1]), reverse=True)
        # self.ranking.extend(ranked_file)
        # for i in self.ranking:
        #     print(i)
        # with open(r'result_1.txt', 'a') as fp:
        #     fp.write(f"Target File: {targetFile}\n")
        #     # for item in self.ranking:
        #     #     # write each item on a new line
        #     #     fp.write(f"{str(item)}\n")
        #     fp.writelines([str(x)+"\n" for x in self.ranking])
        #     fp.write("\n")
        #     print('Done')

    '''
       Tree match algorithm based off the paper by Yao
       @param q the root node of the sub tree we are trying to match
       '''
    def treeMatch(self, q):
        # Exact or Subtree matching
        tq = self.candidateTree.searchNode(q.name) #(list)Tq are occurrences of the pattern node q in data source.
        for tqi in tq:
            if(self.find(q,tqi)):
                return True
        self.particalTreeMatch(tq, q)
        return False

    def particalTreeMatch(self, tq, q):
        if q.is_leaf():
            if (not q in self.longest_submatch) or (1 > self.longest_submatch[q]):
                if tq:
                    if not q in self.matchingNodes:
                        self.matchingNodes.append(q)
                    self.longest_submatch[q] = 1
                else:
                    self.longest_submatch[q] = 0
            self.depth_score_list.append(self.longest_submatch[q]/q.depth())
            # print(q.name, self.longest_submatch[q], q.depth())
        else:
            for tqi in tq:
                result = self.find_partical(q,tqi, q.depth())
            for child in q.children:
                tq = self.candidateTree.searchNode(child.name)  # (list)Tq are occurrences of the pattern node q in data source.
                self.particalTreeMatch(tq, child)
        return False

    '''
    determines whether the current occurrence Tq→current is a partial solution.
    @param q is the node in the target.xml tree
    @param tq Tq→current is a partial solution means matchings of
    sub-tree patterns rooted by q have been found and encoded
    in the stacks and these matchings are possibly extended to
    final results
    @return If Tq→current is false/ not a partial solution,
    function CleanStack() is called to remove the recoded
    nodes that are its descendants.
    returns true if the list is not empty, the end is bigger than the start, and i = N;
    '''
    def find_partical(self, q, tq, currentDepth):
        if q.is_leaf():
            if not q in self.matchingNodes:
                self.matchingNodes.append(q)
            temp_depth = q.depth() - currentDepth
            if (not q in self.longest_submatch) or (temp_depth > self.longest_submatch[q]):
                self.longest_submatch[q] = temp_depth
            return True
        numOfChildren = len(q.children)
        i = 0 #q_i(i = 0, 1, 2, ... n-1) are q's children

        partialSolution = False
        if q.name =='mi':

            if not q in self.matchingNodes:
                self.matchingNodes.append(q)
                self.matchingNodes.append(q.children[0])

            temp_depth = q.depth()+1 - currentDepth
            value = q.children[0]
            if (not value in self.longest_submatch) or (temp_depth > self.longest_submatch[value]):
                self.longest_submatch[value] = temp_depth
            return True
        tq_i = self.candidateTree.searchInChildren(tq,q.children[i].name)
        while partialSolution or tq_i:
            if (not tq_i) or (tq_i[0].start > tq.end):
                if(partialSolution):
                    i=i+1
                    partialSolution = False
                    if i != numOfChildren:
                        tq_i = self.candidateTree.searchInChildren(tq, q.children[i].name)
                if i==numOfChildren:
                    # self.temp_depth = q.depth() - currentDepth
                    return True
            else:
                if tq_i[0].start >= tq.start:
                    if self.find_partical(q.children[i],tq_i[0],currentDepth):
                        if not q in self.matchingNodes:
                            self.matchingNodes.append(q)
                        partialSolution = True
                tq_i.pop(0)

        temp_depth = q.depth() - currentDepth
        for l in q.get_leaf():
            if (not l in self.longest_submatch) or (temp_depth > self.longest_submatch[l]):
                self.longest_submatch[l] = temp_depth
        if i < numOfChildren-1:
            i=i+1
            tq_i = self.candidateTree.searchInChildren(tq, q.children[i].name)
            if tq_i:
                if self.find_partical(q.children[i], tq_i[0], currentDepth):
                    if not q in self.matchingNodes:
                        self.matchingNodes.append(q)
        return False

    def find(self, q, tq):
        if q.is_leaf():
            self.matchingNodes.append(q)
            return True
        numOfChildren = len(q.children)
        i = 0 #q_i(i = 0, 1, 2, ... n-1) are q's children
        partialSolution = False
        if q.name =='mi':
            self.matchingNodes.append(q)
            return True
        tq_i = self.candidateTree.searchInChildren(tq,q.children[i].name)
        while partialSolution or tq_i:
            if (not tq_i) or (tq_i[0].start > tq.end):
                if(partialSolution):
                    i=i+1
                    partialSolution = False
                if i==numOfChildren:
                    return True
                tq_i = self.candidateTree.searchInChildren(tq, q.children[i].name)
            else:
                if tq_i[0].start >= tq.start:
                    if self.find(q.children[i],tq_i[0]):
                        self.matchingNodes.append(q)
                        partialSolution = True
                tq_i.pop(0)
        return False

