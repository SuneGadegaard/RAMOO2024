import queue as Q  # Used for storing the branching nodes


class Solution:
    """ Class implementing a solution to the MO-ILP
        Solutions are here understood as both integer feasible and integer infeasible solutions. They may even consist
        of only a vector in outcome space. That is, some "solutions" may not have a preimage, if they e.g. does not
        originate from a feasible solution
    """
    def __init__(self, obj_1_val=0.0, obj_2_val=0.0, x_val=[]):
        self.isIntegerFeasible = False  # Flag indicating if a solution is integer feasible
        self.obj_1 = obj_1_val  # The value of the first objective function
        self.obj_2 = obj_2_val  # The value of the second objective function
        self.x = list(x_val)  # Pre-image of objective vector (obj_1, obj_2) (if one is known)
        self.checkIfIntegerFeasible()  # Check if solution is integer feasible

    def setSolution(self, obj_1_val: float, obj_2_val: float, x_val: list = []):
        """ Method for overwriting/setting the members of the Solution object
            @:param obj_1_val The value of the first objective function
            @:param obj_2_val The value of the second objective function
            @:param x_val List containing the pre-image of the objective vector if one is known. If not, parse []
        """
        self.obj_1 = obj_1_val
        self.obj_2 = obj_2_val
        self.x = list(x_val)
        self.checkIfIntegerFeasible()

    def checkIfIntegerFeasible(self):
        """ Method for checking if the pre-image is 'integer enough' to be declared integer
            In case no pre-image is known, the solution is not integer feasible by definition
        """
        if self.x != []:
            self.isIntegerFeasible = all([x - 0.000001 <= round(x, 0) <= x + 0.000001 for x in self.x])
        else:
            self.isIntegerFeasible = False

    def dominates(self, solution: "Solution"):
        """ Method for checking if this solution dominates the argument Solution object
            @:return True if this solution dominates the Solution object parsed as an argument. False otherwise
        """
        return (self.obj_1 <= solution.obj_1 and self.obj_2 < solution.obj_2) or \
                (self.obj_1 < solution.obj_1 and self.obj_2 <= solution.obj_2)

    def weaklyDominates(self, solution: "Solution"):
        """ Method for checking if this solution weakly dominates the argument Solution object
            @:return True if this solution weakly dominates the Solution object parsed as an argument. False otherwise
        """
        return self.obj_1 <= solution.obj_1 and self.obj_2 <= solution.obj_2


class UpperBoundSet:
    """ Class implementing an upper bound set consisting of Solution objects.
        Every time a new solution is inserted into the upper bound set, the set of local Nadir points is updated as well
    """
    def __init__(self, upperBoundOnObj_1 = 1000000, upperBoundOnObj_2 = 1000000):
        self.UBSet = []
        self.LocalNadirs = [Solution(upperBoundOnObj_1, upperBoundOnObj_2, [])]

    def updateUBSet(self, newSolution: Solution):
        """ Function that updates the upper bound set using a new solution
            In case the new solution is non-dominated by the current upper bound set, the new solution is inserted and
            all dominated solutions are removed from the upper bound set.
            Finally, the set of local Nadir points is updated
        """
        if not self.UBSet:
            self.UBSet.append(Solution(newSolution.obj_1, newSolution.obj_2, list(newSolution.x)))
        if all( [not sol.weaklyDominates(newSolution) for sol in self.UBSet] ):
            hasBeenAdded = False
            for i in range(len(self.UBSet)):
                if newSolution.obj_1 <= self.UBSet[i].obj_1:
                    self.UBSet.insert(i, Solution(newSolution.obj_1, newSolution.obj_2, list(newSolution.x)))
                    hasBeenAdded = True
                    break
            if not hasBeenAdded:
                self.UBSet.append(Solution(newSolution.obj_1, newSolution.obj_2, list(newSolution.x)))
            self.UBSet = [sol for sol in self.UBSet if not newSolution.dominates(sol)]
        self.__updateNadirs()

    def __updateNadirs(self):
        self.LocalNadirs = [Solution(self.UBSet[0].obj_1-0.99, 1000000, [])] \
                           + [Solution(self.UBSet[i + 1].obj_1-0.99, self.UBSet[i].obj_2-0.99, []) for i in range(len(self.UBSet) - 1)] \
                           + [Solution(1000000, self.UBSet[-1].obj_2-0.99, [])]


class BranchingNode:
    """ Class implementing the branching node structure """
    def __init__(self):
        self.depth = 0  # The depth of the node in the branching tree. Useful when searching depth first/breadth first
        self.fixedToZero = []  # List of indices of variables fixed zero
        self.fixedToOne = []  # List of indices of variables fixed one
        self.free = []  # List of indices of variables not fixed yet
        self.bound = []  # Best estimate of the bound produced by the continuous relaxation of the node
        self.priority = 0  # This gives the priority of the node. The node with the lowest priority value is chosen next
        self.index = 0  # Unique identifier for the node
        self.parentLowerBound = []  # The lower bound set of the parent node

    def initializeNode(self, farther: 'node', branchingVar: int, value: int, lowerBoundSet: list = None):
        """
        This method initializes the node-object
        :param farther: a node corresponding to the farther node of the node that should be initialized
        :param branchingVar: index of the variable that is being branched on when creating this node
        :param value: value that the branching variable should be fixed to (must be either 0 (zero) or 1 (one))
        :param lowerBoundSet: a lower bound set for the node. Can be omitted if no lower bound set is known
        """
        self.depth = farther.depth + 1
        self.fixedToOne = list(farther.fixedToOne)
        self.fixedToZero = list(farther.fixedToZero)
        self.free = [i for i in farther.free if i != branchingVar]
        self.parentLowerBound = lowerBoundSet
        if value == 1:
            self.fixedToOne.append(branchingVar)
        elif value == 0:
            self.fixedToZero.append(branchingVar)
        else:
            raise ValueError(f'Branching value should be either zero or one. You provided {value}. Depth: {self.depth}')

    def __lt__(self, other):
        """ Less than comparison for the BranchingNode-class """
        return self.priority < other.priority


class BranchingTree:
    """ Class implementing a branching tree structure for storing branching nodes """
    def __init__(self):
        self.T = Q.PriorityQueue()  # Priority queue used for storing branching nodes

    def getNode(self):
        """
        Method returning a node at the top of the priority queue (sorted based on the lt method of the BranchingNode
        class.
        :return: A branching node (BranhingNode object) stored in the branching node.
        If the queue is empty, None is returned
        """
        if self.T.empty():
            return None
        else:
            return self.T.get()

    def isNotEmpty(self) -> bool:
        return not self.T.empty()

    def addNode(self, node: BranchingNode):
        """
        Method adding a branching node to the branching tree
        :param node: An object of the BnbNode class
        """
        self.T.put(node)
