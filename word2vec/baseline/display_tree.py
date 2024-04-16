
import treeMatch_patical_match2

if __name__ == "__main__":
    try:
        tm = treeMatch_patical_match2.FastTreeMatch()
        targetTree = tm.makeATree("target.xml")
        targetTree.root.display()
    except:
        print("doesn't work")
        pass