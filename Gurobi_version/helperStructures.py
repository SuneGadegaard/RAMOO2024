""" This file implements the helper structures/classes used to implement a branch and bound algorith for bi objective
    combinatorial optimisation problems.
    The file implements the following classes
    Solution : A class for storing solutions, including simple methods for dominance comparisons
    UpperBoundSet : A class for storing an upper bound set of feasible solutions.
    BranchingNode : A class for storing information for a branching node in the branch and bound tree
    BranchingTree : A class for storing branching nodes. Implemented using a simple priority queue.
"""

import queue as Q  # Used for storing the branching nodes
import matplotlib.pyplot as plt  # Used for plotting the upper bound set


class Solution:
    """ Class implementing a solution to the MO-ILP
        Solutions are here understood as both integer feasible and integer infeasible solutions. They may even consist
        of only a vector in outcome space. That is, some "solutions" may not have a preimage, if they e.g. does not
        originate from a feasible solution
    """

    def __init__(self, obj_1_val=0.0, obj_2_val=0.0, x_val=[]):
        """ Initializer for the Solution class. Can be called with default values or by specifying the objective vector
            coefficients of the solution (obj_1_val, obj_val_2) along with its corresponding preimage (x)
            @:param obj_1_val floating point value for the first objective's value
            @:param obj_2_val floating point value for the second objective's value
            @param x_val list of floats/integers specifying the preimage of the objective vector (obj_1_val, obj_val_2)
        """
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

    def __init__(self, upperBoundOnObj_1=1000000, upperBoundOnObj_2=1000000):
        """ Initialisation method for the UpperBoundSet class. The parameter upperBoundOnObj_1 and upperBoundOnObj_2
            Specify upper bounds on both objective functions for all non-dominated outcome vectors. In case no values
            are specified, the defualt is simply set to upperBoundOnObj_1 = upperBoundOnObj_2 = 1000000.
            If these values are not large enough to be valid upper bounds on the non-dominated outcome vectors, the
            algorithm may terminate with an incorrect set of non-dominated outcomes.
            @:param upperBoundOnObj_1 specifies an upper bound on the first objective for all non-dominated outcomes
            @:param upperBoundOnObj_2 specifies an upper bound on the first objective for all non-dominated outcomes
        """
        self.UBSet = []  # A list of Solution objects used for storing the upper bound set
        self.dominatedOutcomes = []  # List of Solution objects used for storing dominated solutions found
        self.LocalNadirs = [Solution(upperBoundOnObj_1, upperBoundOnObj_2, [])]  # List of local Nadir points
        # Parameters used for dynamic plotting
        self.displayProgressActive = False  # By default, progress is not plotted
        self.minObj1 = 0  # Min value of the first axis in the plot
        self.minObj2 = 0  # Min value of the first axis in the plot
        self.maxObj1 = upperBoundOnObj_1  # Upper bound on the first object for all non-dominated outcome vectors
        self.maxObj2 = upperBoundOnObj_2  # Upper bound on the second object for all non-dominated outcome vectors

    def saveUpperBoundFigure(self, filename: str):
        """ Method saving the upper bound set figure to a file at the termination of the algorithm.
            @:param filename string specifying the name of the file the figure is saved to
        """
        plt.savefig(filename)

    def updateUBSet(self, newSolution: Solution):
        """ Function that updates the upper bound set using a new solution
            In case the new solution is non-dominated by the current upper bound set, the new solution is inserted and
            all dominated solutions are removed from the upper bound set.
            Finally, the set of local Nadir points is updated

            If the parameter self.displayProgressActive is true, this function also ensure to update the graphic when
            new yet non-dominated solutions are identified.

            @:param newSolution a Solution object that may be added to the upper bound set.
        """
        if self.displayProgressActive:  # If displaying progress, keep track of identified, dominated solutions
            self.dominatedOutcomes = self.dominatedOutcomes + [sol for sol in self.UBSet if newSolution.dominates(sol)]
        if not self.UBSet:  # Upper bound set was empty, so the solution goes in
            self.UBSet.append(Solution(newSolution.obj_1, newSolution.obj_2, list(newSolution.x)))
            self.__updateNadirs()
            if self.displayProgressActive:
                self.displayUBSet()
        elif all(
                [not sol.weaklyDominates(newSolution) for sol in self.UBSet]):  # Check if new solution is not dominated
            hasBeenAdded = False
            for i in range(len(self.UBSet)):  # Find the place to insert
                if newSolution.obj_1 <= self.UBSet[i].obj_1:
                    self.UBSet.insert(i, Solution(newSolution.obj_1, newSolution.obj_2, list(newSolution.x)))
                    hasBeenAdded = True
                    break
            if not hasBeenAdded:
                self.UBSet.append(Solution(newSolution.obj_1, newSolution.obj_2, list(newSolution.x)))
            self.UBSet = [sol for sol in self.UBSet if not newSolution.dominates(sol)]
            self.__updateNadirs()
            if self.displayProgressActive:
                self.displayUBSet()

    def __updateNadirs(self):
        """ Method updating the set of local Nadir points used for bounding nodes in the branch and bound algorithm"""
        self.LocalNadirs = [Solution(self.UBSet[0].obj_1 - 0.99, self.maxObj2, [])] \
                           + [Solution(self.UBSet[i + 1].obj_1 - 0.99, self.UBSet[i].obj_2 - 0.99, []) for i in
                              range(len(self.UBSet) - 1)] \
                           + [Solution(self.maxObj1, self.UBSet[-1].obj_2 - 0.99, [])]

    def displayProgress(self, minObj1: float, minObj2: float):
        """ API method used to activate the progress display of the upper bound set
            @:param minObj1 minimum value on the axis of the first objective
            @:param minObj2 minimum value on the axis of the second objective
        """
        self.displayProgressActive = True
        self.minObj1 = minObj1
        self.minObj2 = minObj2
        plt.ion()  # Turn on interactive mode

    def displayUBSet(self):
        """ Plots the upper bound set and the set of dominated solutions identified during the search.
            The set of Local Nadir points is also plotted along with barrier of the search area
        """
        plt.clf()  # Clear the current figure
        nonDomX = [sol.obj_1 for sol in self.UBSet]
        nonDomY = [sol.obj_2 for sol in self.UBSet]
        plt.scatter(nonDomX, nonDomY, color='blue')

        nadirX = [sol.obj_1 + 0.99 for sol in self.LocalNadirs]
        nadirY = [sol.obj_2 + 0.99 for sol in self.LocalNadirs]
        plt.scatter(nadirX, nadirY, color='black', marker='x')

        xBoundary = [self.LocalNadirs[0].obj_1]
        yBoundary = [self.LocalNadirs[0].obj_2]
        for i, sol in enumerate(self.UBSet):
            xBoundary.append(sol.obj_1)
            yBoundary.append(sol.obj_2)
            if i < len(self.UBSet) - 1:
                xBoundary.append(self.UBSet[i + 1].obj_1)
                yBoundary.append(sol.obj_2)
            else:
                xBoundary.append(self.LocalNadirs[-1].obj_1)
                yBoundary.append(self.LocalNadirs[-1].obj_2)
        plt.plot(xBoundary, yBoundary, color='black', linestyle='dashed')
        if self.dominatedOutcomes != []:  # Check if we have any dominated solutions ready
            xDom = [sol.obj_1 for sol in self.dominatedOutcomes]
            yDom = [sol.obj_2 for sol in self.dominatedOutcomes]
            plt.scatter(xDom, yDom, color='red')

            plt.xlim(self.minObj1 * 0.9, max(max(nonDomX), max(xDom)) * 1.1)
            plt.ylim(self.minObj2 * 0.9, max(max(nonDomY), max(yDom)) * 1.1)
            plt.legend(['Upper bound set', 'Local nadir points', 'Boundary', 'Dominated points'])
        else:
            plt.xlim(min(nonDomX) * 0.9, max(nonDomX) * 1.1)
            plt.ylim(min(nonDomY) * 0.9, max(nonDomY) * 1.1)
            plt.legend(["Upper bound set", 'Local nadir points', 'Boundary'])

        plt.title('Upper bound set  progress')
        plt.pause(0.0001)  # Pause to allow the plot to update


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
    """ Class implementing a branching tree structure for storing branching nodes. The tree is implemted as a simple
        priority queue
    """
    def __init__(self):
        """ Initialization method for the BranchingTree class """
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
        """ Method used for checking if the branching tree is empty
            @:return Boolean. True if the branching tree is empty and False otherwise
        """
        return not self.T.empty()

    def addNode(self, node: BranchingNode):
        """ Method adding a branching node to the branching tree
            :param node: An object of the BranchingNode class
        """
        self.T.put(node)
