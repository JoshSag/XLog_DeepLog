import json

class Trie():
    def __init__(self, depth=0):
        """
        A trie object.
        The value of a node is the path from the root.
        """
        self.depth = depth
        self.tree = dict()
    
    def __getitem__(self, key):
        return self.tree.get(key, None)
    
    def keys(self):
        return self.tree.keys()

    def _find(self, branch):
        """
        Returns a pair: the last node on the branch and the remaining subbranch.
        For example, if the branch is [1,2,3,4] and the tree contains the path [1,2] but not the path [1,2,3] - 
        then the return value is the node "1->2", and the remaining subbranch is [3,4].
        """
        assert type(branch) in [list, tuple]
        if len(branch) == 0:
            return self, branch
        elif branch[0] in self.tree.keys():
            return self.tree[branch[0]]._find(branch[1:])
        else:
            return self, branch
    
    def add(self, branch):
        assert type(branch) in [list, tuple]
        obj, subbranch = self._find(branch)
        if len(subbranch) == 0:
            return
        
        obj.tree[subbranch[0]] = __class__(depth = obj.depth+1)
        obj.tree[subbranch[0]].add(subbranch[1:])

    def add_many(self, branches):
        for branch in branches:
            self.add(branch)
    
    def __repr__(self):
        for k in self.tree:
            print("-"*self.depth, k)
            print(self.tree[k],end="")
        return ""
       
    def _my_diverges(self):
        return len(self.tree.keys())
        
            
    def num_leafs(self):
        if len(self.tree.keys()) == 0:
            return 1
        else:
            return sum([child.num_leafs() for child in self.tree.values()])

    def max_diverges(self):
        if len(self.tree.keys()) == 0:
            return 0

        my_diverges = self._my_diverges()
        internal_max_divergres = max([child.max_diverges() for child in self.tree.values()])
        return max(my_diverges, internal_max_divergres)

    def max_depth_with_diverge(self):
        if len(self.tree.keys()) == 0:
            return -1
        depthes_with_diverge = list()
        if len(self.tree.keys()) > 1:
            depthes_with_diverge.append(self.depth)
        inchilds = [child.max_depth_with_diverge() for child in self.tree.values()]
        depthes_with_diverge.extend(inchilds)
        return max(depthes_with_diverge)

def add_ending(workflows):
    return [list(w) + ["$"] for w in workflows]

def calc_g_value(workflows):
    t = Trie()
    t.add_many(add_ending(workflows))
    # print(t)
    return t.max_diverges()

def get_sufixes(w):
    return [w[i:] for i in range(len(w)-1)]

def calc_h_value(workflows):
    t = Trie()
    workflows = add_ending(workflows)
    Wsuffixes = list()
    for w in workflows:
        Wsuffixes.extend(get_sufixes(w))

    t.add_many(Wsuffixes)
    return t.max_depth_with_diverge() + 1


