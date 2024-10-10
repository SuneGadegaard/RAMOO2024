""" Main file implementing a branch and bound algorithm for bi-objective combinatorial optimisation problems """
import helperStructures as hs
import GRBLowerBoundSets as lbs
import RENS as RNS
import json as js
import time
import random


class BranchAndBound:
    """ Class implementing a branch and bound algorithm for bi-objective combinatorial optimisation (BOCO) problems.
        The BOCO problems must be linear, meaning both objective functions and constraints must be linear. In addition,
        it is assumed that all variables are binary. Hence, problems of the form
        v-min (c1x, c2x) : Ax >= b, x_i in {0,1}
        can be solved using this algorithm.
        At termination, the algorithm provides a minimal complete set of Pareto optimal solutions and their
        corresponding images in R^2.
        NOTE: It is assumed, that the objective function coefficients are integers. No check is performed, but an
        optimal solution is not guaranteed, if floats are used instead of integers.
        """
    def __init__(self):
        # Objects used in the branch and bound algorithm
        self.Tree = hs.BranchingTree()  # A branching tree used to store the branching nodes
        self.UpperBound = None # An upper bound set used to store feasible solutions
        self.LowerBoundSolver = lbs.LowerBoundSolver()  # A solver that generates a lower bound set
        # Instance data
        self.A = None  # The constraint matrix of constraints of the form Ax >= b
        self.b = None  # The right hand side vector of constraints of the form Ax >= b
        self.C = None  # The objective function coefficients of a bi objective problem: min Cx: Ax >= b, x binary
        self.numVars = None  # The number of variables of the problem
        self.numCsts = None  # The number of constraints of the problem
        # Solve statistics
        self.numOfNodesProduced = 0  # Contains the number of branching nodes created by the algorithm
        self.numOfNodesPrunedByInfeasibility = 0  # The number of branching nodes pruned by infeasibility
        self.numOfNodesPrunedByOptimality = 0  # The number of branching nodes pruned by optimality
        self.numOfNodesPrunedByBound = 0  # The number of branching nodes pruned by bound
        self.maxSizeOfTree = 0  # Maximum number of active nodes in the branching tree over the course of the algorithm
        self.maxDepthOfTree = 0  # The maximum depth of the branching tree reached over the course of the algorithm
        self.prunedByParentBound = 0  # Counts the number of nodes pruned by the lower bound of the parent node
        # Settings for the algorithms
        self.branchingStrategy = "mostOftenFractional"  # The branching/variable selection strategy
        self.objectiveBranching = False  # Flag specifying if objective branching is used
        self.nodeSelectionStrategy = "bestBound"  # Node selection strategy in the branch and bound algorithm
        self.probing = False  # Flag specifying if probing is used
        self.dominatedNadirPoints = []  # List of local Nadir points dominated by the lower bound set of a node
        self.superLocalNadirPoints = []
        self.useParentLowerBoundFlag = False  # Set this to True, if the lower bound of parent node should be used
        self.lowerBoundStrategy = 'LPRelaxation'  # Defines the lower bounding strategy
        self.RENSHeuristicsActivated = False
        self.RENSHeuristicMaxDepth = -1  # Maximum depth at which the heuristic is run
        # Flags used to control various options
        self.showUpperBoundProgressActivated = False  # Flag used to control if the upper bound set is displayed

    def readData(self, filename: str):
        """ Method used to read the data for a bi-objective optimization problem of the form min Cx: Ax >= b, x binary
            The data file must be in json format with the following specifications:
            "numVars" integer specifying the number of variables,
            "numCsts" integer specifying the number of constraints,
            "A" list of lists specifying the lhs constraint matrix. A[i][j] is the entry of row i and column j
            "b" list specifying the rhs of the problem,
            "C" list of two lists specifying the objective functions coefficients of the problem.
            @:param filename a string specifying the relative path to the instance file
        """
        with open(filename) as f:
            d = js.load(f)
            self.numVars = d['numVars']
            self.numCsts = d['numCsts']
            self.A = d['A']
            self.b = d['b']
            self.C = d['C']

    def showUpperBoundProgress(self):
        """ Method used to activate a plot of the current upper bound set along with the outcomes found, that are now
            dominated. This slows down computations a bit!"""
        self.showUpperBoundProgressActivated = True

    def __initializeRootNode(self) -> hs.BranchingNode:
        """ Method initializing the root node of the branching tree
            @:return a BranchingNode object specifying the root node
        """
        node = hs.BranchingNode()  # Create a node object
        node.depth = 0  # Set its depth to zero
        node.fixedToZero = []  # Assume no variables are fixed zero
        node.fixedToOne = []  # Assume no variables are fixed one
        node.free = [i for i in range(self.numVars)]  # Assume all variables are free
        node.bound = 0  # Set the node bound to zero
        node.priority = 0  # Set the node priority to zero
        return node

    def __canBePruned(self) -> bool:
        """ Method deciding if the currently selected node can be pruned. It is assumed that the upper bound set object
            has been initialized and that the lower bound set object has been initialized and the computeLowerBoundSet()
            method has been called.
            @:return True if the currently selected node can be pruned and False otherwise
        """
        if self.LowerBoundSolver.isInfeasible:
            # Prune by infeasibility
            self.numOfNodesPrunedByInfeasibility += 1
            return True
        elif self.LowerBoundSolver.lpIsSingleton and self.LowerBoundSolver.lbSet[0].isIntegerFeasible:
            # Prune by optimality
            self.numOfNodesPrunedByOptimality += 1
            return True
        else:  # Test if node can be pruned by bound
            if self.__canBePrunedByBound(self.LowerBoundSolver.lbSet):
                self.numOfNodesPrunedByBound += 1
                return True
            else:
                return False

    def __canBePrunedByBound(self, lowerBoundSet: list) -> bool:
        """ Method deciding if the currently selected node can be pruned by bound. It is assumed that the upper bound
            set object has been initialized and that the lower bound set object has been initialized and the
            computeLowerBoundSet() method has been called.
            @:return True if the currently selected node can be pruned by bound and False otherwise
        """
        if lowerBoundSet == []:  # This is the root node case
            return False
        sizeOfLB = len(lowerBoundSet)
        canBePruned = True  # Assume the node can be pruned
        # For each local Nadir point on the upper bound set, check if it is in L+R^2_>
        for LNsol in self.UpperBound.LocalNadirs:
            # LNsol can only be in L+R^2_> if dominated by the ideal point of the LBset
            if LNsol.obj_1 >= lowerBoundSet[0].obj_1 and LNsol.obj_2 >= lowerBoundSet[-1].obj_2:
                if sizeOfLB == 1:  # If dominated by ideal point, and LBset is a singleton, the node cannot be pruned
                    return False
                for i in range(sizeOfLB - 1):  # Run through the lower bound set
                    if lowerBoundSet[i].dominates(LNsol) or lowerBoundSet[i + 1].dominates(LNsol):
                        if not self.objectiveBranching:  # If not objective branching is activated, return false
                            return False
                        else:  # If objective branching is activated, collect the Nadir point for later branching
                            self.dominatedNadirPoints.append(hs.Solution(LNsol.obj_1, LNsol.obj_2, []))
                            canBePruned = False
                    elif lowerBoundSet[i].obj_1 <= LNsol.obj_1 <= lowerBoundSet[i + 1].obj_1:
                        # Need to check if the line between lbSet[i] and lbSet[i+1] dominates the local Nadir point
                        checkValue = (LNsol.obj_2 - lowerBoundSet[i].obj_2) * (
                                    lowerBoundSet[i + 1].obj_1 - lowerBoundSet[i].obj_1)
                        checkValue -= (LNsol.obj_1 - lowerBoundSet[i].obj_1) * (
                                    lowerBoundSet[i + 1].obj_2 - lowerBoundSet[i].obj_2)
                        if checkValue >= 0:  # If line dominates the Nadir point, node cannot be pruned
                            if not self.objectiveBranching:  # If objective branching is not activated, return false
                                return False
                            else:  # If objective branching is activated, collect the Nadir point for later branching
                                self.dominatedNadirPoints.append(hs.Solution(LNsol.obj_1, LNsol.obj_2, []))
                                canBePruned = False
        return canBePruned

    def __getBranchingVariable(self, node: hs.BranchingNode) -> int:
        """ Returns a branching variable, that the algorithm uses to split a node on
            The options currently implemented are: mostOftenFractional, averageMostFractional, and random
            @:return an integer specifying the index of the branching variable
        """
        if self.branchingStrategy == "mostOftenFractional":
            variableScore = {i: sum(1 for sol in self.LowerBoundSolver.lbSet if 0.00001 <= sol.x[i] <= 0.99999) for i in
                            node.free}
        elif self.branchingStrategy == "averageMostFractional":
            numSolOnLBSet = len(self.LowerBoundSolver.lbSet)
            variableScore = {i: -abs(0.5 - sum(sol.x[i] for sol in self.LowerBoundSolver.lbSet) / numSolOnLBSet) for i
                             in node.free}
        elif self.branchingStrategy == "random":
            variableScore = {i: random.uniform(0, 1) for i in node.free}
        return max(variableScore, key=variableScore.get)

    def __computeSuperLocalNadirPoints(self, LBSet: list):
        self.superLocalNadirPoints = [hs.Solution(sol.obj_1, sol.obj_2, []) for sol in self.dominatedNadirPoints]
        for i in range(len(self.superLocalNadirPoints)-1):
            pointsMerged = False
            for it1, sol1 in enumerate(self.superLocalNadirPoints):
                for it2, sol2 in enumerate(self.superLocalNadirPoints):
                    if it1 < it2:
                        newSuperNadir = hs.Solution(min(sol1.obj_1, sol2.obj_1), min(sol1.obj_2, sol2.obj_2), [])
                        if newSuperNadir.isDominatedBySet(self.LowerBoundSolver.lbSet):
                            self.superLocalNadirPoints.remove(sol1)
                            self.superLocalNadirPoints.remove(sol2)
                            self.superLocalNadirPoints.insert(it1, newSuperNadir)
                            pointsMerged = True
                            break
            if not pointsMerged:
                break

    def setNodeSelectionStrategy(self, strategy: str = "breadthFirst"):
        """ Method for setting the node selection strategy in the branch and bound algorithm
            Options are {'depthFirst', 'breadthFirst', 'bestBound', "random"}
        """
        strategies = {'depthFirst', 'breadthFirst', 'bestBound', "random"}
        if strategy not in strategies:
            self.nodeSelectionStrategy = 'depthFirst'
            print(f'You asked for the node selection strategy "{strategy}". Its is not in the list: {strategies}')
        else:
            self.nodeSelectionStrategy = strategy

    def setBranchingStrategy(self, strategy: str = "mostOftenFractional"):
        """ Method for setting the node branching/variable selection strategy in the branch and bound algorithm
            Options are {"mostOftenFractional", "averageMostFractional", "random"}
        """
        strategies = {"mostOftenFractional", "averageMostFractional", "random"}
        if strategy not in strategies:
            self.branchingStrategy = "mostOftenFractional"
            print(f'You asked for the branching strategy "{strategy}". Its is not in the list: {strategies}')
        else:
            self.branchingStrategy = strategy

    def loadSolutionsFromFile(self, filename: str):
        """ Loads an optimal solution (minimal complete set and its image) from a file. Mimics a strong and fast
            heuristic
            @:param filename: string specifying the relative path to the file containing the optimal solution
        """
        with open(filename) as f:
            d = js.load(f)
            for sol in d['solutions']:
                sol = hs.Solution(sol['obj1'], sol['obj2'], sol['x'])
                self.UpperBound.updateUBSet(sol)

    def setLowerBoundStrategy(self, strategy: str):
        """ Method for controlling the lower bounding strategy.
            Options are {'LPRelaxation', 'IdealPoint'}
        """
        self.LowerBoundSolver.lbSetType = strategy

    def useParentLowerBound(self):
        """ Set method used to control if the lower bound set of the parent node should be inherited by the child nodes
            in the hope that it can be avoided to compute some lower bound sets """
        self.useParentLowerBoundFlag = True

    def useRENSHeuristic(self, maxDepth: int):
        """ Method used to activate the Relaxation Enforced Neighbourhood search heuristic
            The RENS heuristic works as follows: Given a fractional solution (y1,y2,xbar) it solves the problem
            min     || Cx - y||_0
            s.t.:   Ax >= b
                    x[j] = 1, if xbar[j]=1
                    x[j] = 0, if xbar[j]=0
                    x[j] in {0,1}
            That is, it produces an integer feasible solution with minimum L_0-norm distance to the fractional input
            solution. The RENS heuristic solves the above problem for each extreme point solution of the lower bound of
            the node at which it is invoked.
            The RENS heuristic is invoked at all nodes at depth <= maxDepth
            @:param maxDepth: integer specifying how deep in the branching tree the RENS heuristic should be invoked
        """
        self.RENSHeuristicsActivated = True
        self.RENSHeuristicMaxDepth = maxDepth

    def solve(self):
        """ Main method of the class. readData() must be prior to calling the solve method. It runs a bi-objective
            branch and bound algorithm using the settings set prior to calling the "solve()" method. If no settings are
            set, the solve() method runs the branch and bound algorithm with default settings.
        """
        startTime = time.time()
        numTimesDetailsPrinted = 0
        numTimesHeaderPrinted = 0
        maxObj1 = sum(self.C[0][i] for i in range(self.numVars) if self.C[0][i] >= 0)
        maxObj2 = sum(self.C[1][i] for i in range(self.numVars) if self.C[1][i] >= 0)
        self.UpperBound = hs.UpperBoundSet(maxObj1, maxObj2)
        self.Tree.addNode(self.__initializeRootNode())
        self.LowerBoundSolver.setUpLowerBoundSolver(self.C, self.A, self.b)

        columns = ['Nodes', 'Nodes left', 'UB set size', 'Pruned Inf.', 'Pruned Opt.', 'Pruned bound', 'depth']
        while self.Tree.isNotEmpty():
            if 5*self.numVars*numTimesHeaderPrinted <= self.numOfNodesProduced <= 5*self.numVars*(numTimesHeaderPrinted+1):
                print("{: <7} {: >10} {: >14} {: >14} {: >14} {: >14} {: >10}".format(*columns),
                          "\ttime:", round(time.time()-startTime, 2), "seconds")
                numTimesHeaderPrinted +=1

            node = self.Tree.getNode()
            if (not self.useParentLowerBoundFlag) or (not self.__canBePrunedByBound(node.parentLowerBound)):
                self.LowerBoundSolver.updateBounds(node)
                self.LowerBoundSolver.computeLowerBoundSet()

                if node.depth == 0 and self.showUpperBoundProgressActivated:
                    minObj1 = min([sol.obj_1 for sol in self.LowerBoundSolver.lbSet])
                    minObj2 = min([sol.obj_2 for sol in self.LowerBoundSolver.lbSet])
                    self.UpperBound.displayProgress(minObj1, minObj2)
                # Run the Relaxation Enforced Neighbourhood Search heuristic at the top of the tree
                if self.RENSHeuristicsActivated and node.depth <= self.RENSHeuristicMaxDepth:
                    RENSER = RNS.RENS()
                    RENSER.runRENSer(self.LowerBoundSolver.lbSet, self.C, self.A, self.b)
                    for sol in RENSER.feasibleSolutions:
                        self.UpperBound.updateUBSet(sol)
                # Update the upper bound set with feasible solutions found on the LB set
                for sol in self.LowerBoundSolver.integerSolutionsFound:
                    self.UpperBound.updateUBSet(sol)
                # If the node cannot be pruned, a split should be performed
                if not self.__canBePruned():
                    nodes = []
                    branchVar = self.__getBranchingVariable(node)

                    upNode = hs.BranchingNode()
                    upNode.initializeNode(node, branchVar, 1, self.LowerBoundSolver.lbSet)
                    nodes.append(upNode)
                    downNode = hs.BranchingNode()
                    downNode.initializeNode(node, branchVar, 0, self.LowerBoundSolver.lbSet)
                    nodes.append(downNode)
                    for bNode in nodes:
                        self.numOfNodesProduced += 1
                        bNode.index = self.numOfNodesProduced
                        if self.nodeSelectionStrategy == 'depthFirst':
                            bNode.priority = -self.numOfNodesProduced
                        elif self.nodeSelectionStrategy == 'breadthFirst':
                            bNode.priority = bNode.depth
                        elif self.nodeSelectionStrategy == 'bestBound':
                            bNode.priority = self.LowerBoundSolver.equallyWeightedValue
                        elif self.nodeSelectionStrategy == "random":
                            bNode.priority = random.uniform(0, 1)
                        self.Tree.addNode(bNode)
                    self.maxSizeOfTree = max(self.maxSizeOfTree, self.Tree.T.qsize())
                    self.maxDepthOfTree = max(self.maxDepthOfTree, node.depth+1)
                if self.numVars*numTimesDetailsPrinted <= self.numOfNodesProduced <= self.numVars*(numTimesDetailsPrinted+1):
                    row = [self.numOfNodesProduced, self.Tree.T.qsize(), len(self.UpperBound.UBSet), self.numOfNodesPrunedByInfeasibility, self.numOfNodesPrunedByOptimality, self.numOfNodesPrunedByBound, node.depth]
                    print("{: <7} {: >10} {: >14} {: >14} {: >14} {: >14} {: >10}".format(*row))
                    numTimesDetailsPrinted += 1
            else:
                self.prunedByParentBound += 1
                self.numOfNodesPrunedByBound += 1

        print("================================== ")
        print("Statistics")
        print("\t Total computation time   :", round(time.time()-startTime, 2), "seconds")
        print("\t Max depth of search tree :", self.maxDepthOfTree)
        print("\t Max size of active tree  :", self.maxSizeOfTree)
        print("\t Total number of nodes    :", self.numOfNodesProduced)
        print("\t Number of nodes pruned by")
        print("\t\t Infeasibility:", self.numOfNodesPrunedByInfeasibility)
        print("\t\t Optimality   :", self.numOfNodesPrunedByOptimality)
        print("\t\t Bound        :", self.numOfNodesPrunedByBound)
        print("\t\t\t Parent node:", self.prunedByParentBound)
        print("================================== ")
        print("A minimal complete set is given by:")
        for sol in self.UpperBound.UBSet:
            print(sol.obj_1, sol.obj_2, sol.x)
        if self.showUpperBoundProgressActivated:
            print("Do you want to save the figure of the upper bound set (Y/N)?")
            answer = str(input())
            if answer == 'Y' or answer == 'y':
                print('Please enter a name for the file:')
                filename = input()
                self.UpperBound.saveUpperBoundFigure(filename)


if "__main__" == __name__:
    bnb = BranchAndBound()
    bnb.readData('instances/instance_30_10.json')
    bnb.setNodeSelectionStrategy('bestBound')
    bnb.setBranchingStrategy('mostOftenFractional')
    bnb.useParentLowerBound()
    bnb.solve()
