import pptree

class Node:
    def __init__(self, data='', start=0, end=0, parent=None):
        self.start = start
        self.end = end
        self.parent = parent
        self.children = []
        self.name = data
        self.path_belonged=[]

    def __repr__(self):
        return self.name

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return len(self.children) == 0

    def get_leaf(self):
        result=[]
        if self.children:
            for c in self.children:
                if c.is_leaf():
                    result.extend([c])
                else:
                    result.extend(c.get_leaf())
            return result


    def depth(self):
        if self.is_root():
            return 0
        else:
            return 1 + self.parent.depth()

    def display(self):
        pptree.print_tree(self, 'children', 'name', True)

    def add_child(self, node):
        node.parent = self
        self.children.append(node)