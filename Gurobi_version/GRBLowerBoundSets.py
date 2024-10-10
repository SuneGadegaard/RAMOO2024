""" This file implements a lower bound set solver for bi-objective combinatorial optimization (BOCO) problems
    That is, problems of the form min {Cx : Ax>=b, x in {0,1} }
    Two options are available:  'LPRelaxation' that solves the linear programming relaxation of the BOCO problem
                                'IdealPoint' that computes an ideal point for the LP relaxation of the BOCO problem
"""
import math
import helperStructures as hs
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB


class LowerBoundSolver:
    """ Class implementing a lower bound set generator for bi-objective combinatorial optimsation problems of the form
        min Cx
        s.t. Ax >= b
              x \in {0,1}
        The class implements two specific lower bound sets: one consisting of a single point, namely the ideal point of
        the LP-relaxation. The other is the extreme supported non-dominated outcomes of the LP relaxation. The class
        makes use of the Gurobi solver, and specifically the gurobipy modelling API. Hence, the gurobi solver and the
        gurobipy package must be installed for the algorithm to run.
    """
    def __init__(self):
        """ Method for initializing a LowerBoundSolver"""
        self.model = None  # The MOILP model stored i  a pulp model object
        self.solver = None  # The solver used to solve the mdoel
        self.numVar = None  # Number of variables in the MOILP model
        self.numCsts = None  # Number of constraints in the MOILP model
        self.lbSet = None  # The lower bound set computed for the node
        self.lbSetType = 'LPRelaxation'  # 0=LP ideal point, 1=Full LP relaxation
        self.x = None  # Decision variables of the problem. Always assumed to be between zero and one
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
        labels = ["("+str(round(x[i],0))+","+str(round(y[i],0))+")" for i in range(len(x))]
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
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.start()

            self.numVar = len(A[0])
            self.numCsts = len(A)

            self.model = gp.Model('MOILP', env=env)
            self.x = self.model.addVars(range(self.numVar), lb=-0, ub=1)
            self.obj_1 = self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
            self.obj_2 = self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
            self.model.setObjective(self.obj_1 + self.obj_2, GRB.MINIMIZE)
            # Add all constraints to the model
            self.model.addConstrs(
                gp.quicksum(A[i][j] * self.x[j] for j in range(self.numVar)) >= b[i] for i in range(self.numCsts)
            )
            self.model.addConstr(self.obj_1 == gp.quicksum(C[0][j] * self.x[j] for j in range(self.numVar)))
            self.model.addConstr(self.obj_2 == gp.quicksum(C[1][j] * self.x[j] for j in range(self.numVar)))

    def __generateLPIdealPoint(self):
        """ Method for generating the ideal point of the LP relaxation. This is done by solving two single objective
            optimisation problems. In the most likely case that the ideal point has no pre-image, the equally weighted
            convex combination of the two solutions to the single objective optimisation problems is used for branching
        """
        self.isInfeasible = False
        # Set all weight on objective one, to get min f_1
        self.obj_1.Obj = 1
        self.obj_2.Obj = 0

        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            #  If the model is infeasible, then stop, as no
            self.isInfeasible = True
            return
        # Record the objective function value and the solution
        ideal1 = self.obj_1.X
        ideal2 = self.obj_2.X
        sol1 = hs.Solution(ideal1, ideal2, [self.x[i].X for i in range(self.numVar)])
        # Store the solution in integerSolutionsFound if integer
        if sol1.isIntegerFeasible:
            self.integerSolutionsFound.append(sol1)
        # Set all weight on objective two, to get min f_2
        self.obj_1.Obj = 0.0
        self.obj_2.Obj = 1
        self.model.optimize()

        if math.sqrt(abs(self.obj_1.value() - ideal1)**2 + abs(self.obj_2.value() - ideal2)**2) <= 0.000001:
            # The ideal point is the LP relaxation. Hence, there is a pre-image available
            self.lpIsSingleton = True
            self.lbSet = [hs.Solution(ideal1, ideal2, [self.x[i].X for i in range(self.numVar)])]
        else:
            # Check if solution is integral
            sol2 = hs.Solution(self.obj_1.X, self.obj_2.X, [self.x[i].X for i in range(self.numVar)])
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
        self.obj_1.Obj = 1.0
        self.obj_2.Obj = 0.0
        self.model.optimize()
        if self.model.status !=GRB.OPTIMAL:
            #  If the model is infeasible, then stop, as no
            self.isInfeasible = True
            return
        objValues.append(self.model.objVal)
        ul = hs.Solution(self.obj_1.X, self.obj_2.X, [self.x[i].X for i in range(self.numVar)])
        if ul.isIntegerFeasible:
            self.integerSolutionsFound.append(hs.Solution(ul.obj_1, ul.obj_2, ul.x))
        # Find lex-min (f_2,f_1)
        self.obj_1.Obj = 0.0001
        self.obj_2.Obj = 1.0
        self.model.optimize()
        objValues.append(self.model.objVal)
        lr = hs.Solution(self.obj_1.X, self.obj_2.X, [self.x[i].X for i in range(self.numVar)])
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
                self.obj_1.Obj = weight
                self.obj_2.Obj = 1
                self.model.optimize()
                if self.model.objVal <= right.obj_1*weight + right.obj_2 - 0.0001:
                    if self.obj_1.X + self.obj_2.X <= self.equallyWeightedValue:
                        self.equallyWeightedValue = self.obj_1.X + self.obj_2.X
                    objValues.append(self.model.objVal)
                    newSolution = hs.Solution(self.obj_1.X, self.obj_2.X, [self.x[i].X for i in range(self.numVar)])
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
            self.x[i].LB = 0
            self.x[i].UB = 0
        for i in node.fixedToOne:
            self.x[i].LB = 1
            self.x[i].UB = 1
        for i in node.free:
            self.x[i].LB = 0
            self.x[i].UB = 1

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
    lbSolver.computeLowerBoundSet()
    for sol in lbSolver.lbSet:
        print(sol.obj_1, sol.obj_2)
    print("Size of lb set:", len(lbSolver.lbSet))
