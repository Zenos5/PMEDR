

class Tree:
    def __init__(self):
        self.root = None
        self.level = 0
        self.nodes = []

    def insert(self, node, parent):
        if parent is not None:
            parent.add_child(node)
        else:
            if self.root is None:
                self.root = node
        self.nodes.append(node)

    def get_leaf_num(self):
        count = 0
        for n in self.nodes:
            if n.is_leaf():
                count+=1
        return count

    def getLevel(self):
        dep = 0
        for N in self.nodes:
            temp = N.depth()
            if temp > dep:
                dep = temp
        return dep

    def searchIndex(self, name):
        occurrences = []
        for i, N in enumerate(self.nodes):
            if N.name == name:
                occurrences.append(i)
        return occurrences

    def searchNode(self, name):
        occurrences=[]
        for N in self.nodes:
            if N.name == name:
                occurrences.append(N)
        return occurrences

    def searchInChildren(self,q,name):
        occurrences = []
        for i, N in enumerate(q.children):
            if N.name == name:
                occurrences.append(N)
        return occurrences

    def getNode(self, index):
        return self.nodes[index]

    def root(self):
        return self.root