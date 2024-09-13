""" This file implements a lower bound set solver for bi-objective combinatorial optimization (BOCO) problems
    That is, problems of the form min {Cx : Ax>=b, x in {0,1} }
    Two options are available:  'LPRelaxation' that solves the linear programming relaxation of the BOCO problem
                                'IdealPoint' that computes an ideal point for the LP relaxation of the BOCO problem
"""
import math
import pulp as plp
import helperStructures as hs
import matplotlib.pyplot as plt


class LowerBoundSolver:
    def __init__(self):
        self.NadirVar = None
        self.model = None  # The MOILP model stored i  a pulp model object
        self.solver = None  # The solver used to solve the mdoel
        self.numVar = None  # Number of variables in the MOILP model
        self.numCsts = None  # Number of constraints in the MOILP model
        self.lbSet = None  # The lower bound set computed for the node
        self.lbSetType = 'LPRelaxation'  # 0=LP ideal point, 1=Full LP relaxation
        self.x = None  # Decision variables of the problem. Always assumed to be between zero and one
        self.impSlack = None
        self.obj_1 = None  # Decision variable corresponding to the first objective
        self.obj_2 = None  # Decision variable corresponding to the first objective
        self.lpIsSingleton = False  # Flag used to indicate that the LP relaxation consists of single point
        self.isInfeasible = False  # Flag used to indicate that the LP relaxation is infeasible
        self.integerSolutionsFound = []  # List of integer feasible solutions from the LP relaxation
        self.avgObjValue = 0  # Average weighted objective function value over all weighted sums solved
        self.equallyWeightedValue = 0  # Objective function value of the equally weighted scalarization

    def plotLowerBoundSet(self):
        """ Method that plots the lower bound set """
        x = [sol.obj_1 for sol in self.lbSet]
        y = [sol.obj_2 for sol in self.lbSet]
        labels = ["("+str(x[i])+","+str(y[i])+")" for i in range(len(x))]
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        ax.plot(x, y)
        for i, txt in enumerate(labels):
            ax.annotate(txt, (x[i], y[i]))
        plt.show()

    def setLBSetType(self, lbSetType: str):
        """ Method used for setting the lower bound set type that should be computed.
            Options are 'LPRelaxation' and 'IdealPoint'
            @:param lbSetType: string specifying what type of lower bound set that should be computed"""
        self.lbSetType = lbSetType

    def setUpLowerBoundSolver(self, C: list, A: list, b: list):
        """! This function sets up the MO-LP relaxation of the problem
        @:param n integer specifying the number of variables in the problem
        @:param m integer specifying the number of constraints in the problem
        @:param C list of objective function coefficients. C must be of size 2 x n
        @:param A list of constraint coefficients. A must be of size m x n. A[i][j] is the j'th
                coefficient in constraint i
        @:param b list of right hand side coefficients for the constraints. b must be of length m
        """
        if len(A) != len(b):
            print("The constraints matrix and the right hand side vector is not of the correct sizes")
        elif len(A[0]) != len(C[0]):
            print("The constraints matrix and the objective vector is not of the correct sizes")
        else:
            self.numVar = len(A[0])
            self.numCsts = len(A)
            self.model = plp.LpProblem('MOILP', plp.LpMinimize)
            self.x = plp.LpVariable.dict('x', range(self.numVar), 0, 1, plp.LpContinuous)
            self.obj_1 = plp.LpVariable('obj1', lowBound=None, upBound=None)
            self.obj_2 = plp.LpVariable('obj2', lowBound=None, upBound=None)
            self.model.setObjective(self.obj_1 + self.obj_2)
            # Add all constraints to the model
            for i in range(self.numCsts):
                self.model += plp.lpSum(A[i][j] * self.x[j] for j in range(self.numVar)) >= b[i], "cst" + str(i)

            self.model += self.obj_1 == plp.lpSum(C[0][j] * self.x[j] for j in range(self.numVar))
            self.model += self.obj_2 == plp.lpSum(C[1][j] * self.x[j] for j in range(self.numVar))

    def __generateLPIdealPoint(self):
        """ Method for generating the ideal point of the LP relaxation. This is done by solving two single objective
            optimisation problems. In the most likely case that the ideal point has no pre-image, the equally weighted
            convex combination of the two solutions to the single objective optimisation problems is used for branching
        """
        self.isInfeasible = False
        # Set all weight on objective one, to get min f_1
        self.model.objective[self.obj_1] = 1.0
        self.model.objective[self.obj_2] = 0.0
        self.model.solve(plp.PULP_CBC_CMD(msg=0))
        if self.model.status != plp.LpStatusOptimal:
            #  If the model is infeasible, then stop, as no
            self.isInfeasible = True
            return
        # Record the objective function value and the solution
        ideal1 = self.obj_1.value()
        ideal2 = self.obj_2.value()
        sol1 = hs.Solution(ideal1, ideal2, [self.x[i].value() for i in range(self.numVar)])
        # Store the solution in integerSolutionsFound if integer
        if sol1.isIntegerFeasible:
            self.integerSolutionsFound.append(sol1)
        # Set all weight on objective two, to get min f_2
        self.model.objective[self.obj_1] = 0.0
        self.model.objective[self.obj_2] = 1
        self.model.solve(plp.PULP_CBC_CMD(msg=0))

        if math.sqrt(abs(self.obj_1.value() - ideal1)**2 + abs(self.obj_2.value() - ideal2)**2) <= 0.000001:
            # The ideal point is the LP relaxation. Hence, there is a pre-image available
            self.lpIsSingleton = True
            self.lbSet = [hs.Solution(ideal1, ideal2, [self.x[i].value() for i in range(self.numVar)])]
        else:
            # Check if solution is integral
            sol2 = hs.Solution(self.obj_1.value(), self.obj_2.value(), [self.x[i].value() for i in range(self.numVar)])
            if sol2.isIntegerFeasible:
                self.integerSolutionsFound.append(sol2)
            # The ideal point is *not* on the LP-relaxation. Therefor, no pre-image. Use "average" solution to branch
            ideal2 = self.model.objective.value()
            self.lbSet = [hs.Solution(ideal1, ideal2, [(sol1.x[i] + sol2.x[i])/2 for i in range(self.numVar)])]

    def __generateLPRelaxation(self):
        """ Method used for generating the LP relaxation of the problem. The method implements a version of the
            non-inferior set estimation method."""
        self.isInfeasible = False
        objValues = []
        # Find lex-min (f_1,f_2)
        self.model.objective[self.obj_1] = 1.0
        self.model.objective[self.obj_2] = 0.0
        self.model.solve(plp.PULP_CBC_CMD(msg=0))
        objValues.append(self.model.objective.value())
        if self.model.status != plp.LpStatusOptimal:
            #  If the model is infeasible, then stop, as no
            self.isInfeasible = True
            return
        ul = hs.Solution(self.obj_1.value(), self.obj_2.value(), [self.x[i].value() for i in range(self.numVar)])
        if ul.isIntegerFeasible:
            self.integerSolutionsFound.append(hs.Solution(ul.obj_1, ul.obj_2, ul.x))
        # Find lex-min (f_2,f_1)
        self.model.objective[self.obj_1] = 0.0001
        self.model.objective[self.obj_2] = 1.0
        self.model.solve(plp.PULP_CBC_CMD(msg=0))
        objValues.append(self.model.objective.value())
        lr = hs.Solution(self.obj_1.value(), self.obj_2.value(), [self.x[i].value() for i in range(self.numVar)])
        if lr.isIntegerFeasible:
            self.integerSolutionsFound.append(hs.Solution(lr.obj_1, lr.obj_2, lr.x))
        if abs(ul.obj_1-lr.obj_1) <= 0.000001 or abs(ul.obj_2-lr.obj_2) <= 0.000001:
            if ul.dominates(lr):
                self.lbSet = [ul]
            else:
                self.lbSet = [lr]
            self.lpIsSingleton = True
        else:
            self.lbSet = [ul, lr]
            left = ul
            right = lr
            self.equallyWeightedValue = 10E8
            while left != lr:
                if abs(left.obj_1 - right.obj_1) <= 0.000001:  # Check if we have found a solution on vertical line
                    # This may happen, if upper left is not actually the lex-min solution
                    left = self.lbSet[self.lbSet.index(left)+1]
                    right = self.lbSet[self.lbSet.index(left) + 1]

                weight = float(right.obj_2-left.obj_2)/float(left.obj_1 - right.obj_1)
                self.model.objective[self.obj_1] = weight
                self.model.objective[self.obj_2] = 1
                self.model.solve(plp.PULP_CBC_CMD(msg=0))

                if self.model.objective.value() <= right.obj_1*weight + right.obj_2 - 0.0001:
                    if self.obj_1.value() + self.obj_2.value() <= self.equallyWeightedValue:
                        self.equallyWeightedValue = self.obj_1.value() + self.obj_2.value()
                    objValues.append(self.model.objective.value())
                    newSolution = hs.Solution(self.obj_1.value(), self.obj_2.value(),
                                              [self.x[i].value() for i in range(self.numVar)])
                    # Insert after left and before right
                    self.lbSet.insert(self.lbSet.index(right), newSolution)
                    if newSolution.isIntegerFeasible:
                        self.integerSolutionsFound.append(newSolution)
                else:
                    left = right
                if not left == lr:
                    right = self.lbSet[self.lbSet.index(left)+1]
        self.avgObjValue = sum(objValues)/len(objValues)

    def updateBounds(self, node: hs.BranchingNode):
        """ Method used for updating the bounds on variables. The updating is based on the stored information in the
            parsed branching node.
        """
        for i in node.fixedToZero:
            self.x[i].lowBound = 0
            self.x[i].upBound = 0
        for i in node.fixedToOne:
            self.x[i].lowBound = 1
            self.x[i].upBound = 1
        for i in node.free:
            self.x[i].lowBound = 0
            self.x[i].upBound = 1

    def computeLowerBoundSet(self):
        """ Method used for controlling what kind of lower bound set is computed """
        # Reset internal data
        self.lbSet = []
        self.lpIsSingleton = False
        self.integerSolutionsFound = []
        # LB set cases
        if self.lbSetType == 'IdealPoint':
            self.__generateLPIdealPoint()
        elif self.lbSetType == 'LPRelaxation':
            self.__generateLPRelaxation()
        else:
            print("This method is not implemented")
            self.lbSet = [hs.Solution(-10000000, -10000000, [])]


if '__main__' == __name__:
    n = 3
    m = 2
    A = [[4, 12, 13],
         [13, 3, 7]]
    b = [20, 16]
    C = [[6, 3, 5],
         [1, 6, 6]]
    lbSolver = LowerBoundSolver()

    lbSolver.setUpLowerBoundSolver(C, A, b)
    lbSolver.computeLowerBoundSet(1)
    print("Size of lb set:", len(lbSolver.lbSet))
