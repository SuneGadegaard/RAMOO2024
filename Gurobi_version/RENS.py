""" This file contains a class implementing a very simple Relaxation Enforced Neighbourhood Search (RENS) algorithm
    for bi-objective combinatorial optimization (BOCO) problems of the form
    min Cx
    s.t.Ax>=b
         x \in {0,1}
"""

import gurobipy as gp
from gurobipy import GRB
import helperStructures as hs


class RENS:
    """ This class implements a very simple Relaxation Enforced Neighbourhood Search (RENS) algorithm
        for bi-objective combinatorial optimization problems of the form
        min Cx
        s.t.Ax>=b
             x \in {0,1}
        As input it takes an instance of a BOCO problem along with a list of target points of Solutions with fractional
        pre-images. For each target point (f1,f2,xBar) the following single objective ILP is solved
        min |f1-y1| + |f2-y1|
        s.t.Ax >= b
            y = Cx
            x in {0,1}
            x_i = 1 if xBar_i=1
            x_i = 0 if xbar_i=0
        That is, an integer feasible point is found that minimizes the L1 norm distance to the point (f1,f2), under the
        additional constraints that all variables taking integer values in xBar are fixed to their corresponding values.
    """
    def __init__(self):
        """ Initialization methods for the RENS class. Sets everything to None or [] """
        # Data for the BOCO instance
        self.C = None  # Objective function coefficients for the BOCO problem
        self.A = None  # Coefficients of the left hand side constraint matrix for the BOCO problem
        self.b = None  # Right hand side coefficients for the BOCO problem
        self.numVars = None  # Number of variables in the BOCO problem
        self.numCsts = None # Number of constraints in the BOCO problem
        # Variables
        self.model = None  # Variable use to hold the gurobi model used to model the single objective problems
        self.y = None  # Decision variables holding the outcome vector coefficients of a feasible solution
        self.yBar = None  # Variable used to hold the target point
        self.x = None  # Decision variables specifying the preimage of the outcome vector stored in self.y
        self.absDiv = None  # Decision variables used to model the absolute differences in the objective
        self.feasibleSolutions = []  # List of feasible solutions identified

    def runRENSer(self, targetPoints: list, C: list, A: list, b: list, numOfSolutionsToCheck: int = 25):
        """ Main method of the RENS class. For each selected target point a projection problem is solved that finds a
            closest feasible solution (in objective space) to the target point (in L1 norm). Variables are restricted so
            that only variables taking a fractional value for the target point is allowed to vary. All others are fixed
            to their corresponding values.
            If more than numOfSolutionsToCheck target points are supplied, an evenly spread selection of
            numOfSolutionsToCheck points from the list of target points is selected. This is to keep computation times
            down. Default value for numOfSolutionsToCheck is 25.
            @param targetPoints list of Solution objects. Each solution object must have an x-list of the same length as C[0]
            @:param C Objective function coefficients for the BOCO problem. C[i][j] is the j'th variable's coef in obj i
            @:param A Coefficients of the left hand side constraint matrix
            @:param b Coefficients for the right hand sides of the constraints
            @:param numOfSolutionsToCheck integer specifying how many target points should be checked
        """
        self.C = C
        self.A = A
        self.b = b
        self.numVars = len(C[0])
        self.numCsts = len(b)
        self.__buildIPModel()

        if len(targetPoints) <= numOfSolutionsToCheck:
            indices = [i for i in range(len(targetPoints))]
        else:
            indices = [int(round(len(targetPoints) / numOfSolutionsToCheck, 0)) * i for i in range(numOfSolutionsToCheck) if
                       round(len(targetPoints) / numOfSolutionsToCheck, 0) * i <= len(targetPoints) - 1]
            if indices[-1] != len(targetPoints)-1:
                indices.append(len(targetPoints) - 1)
        for i in indices:
            self.__setBounds(targetPoints[i])
            self.__solveModel()

    def __buildIPModel(self):
        """ Method implementing the Gurobi model used to solve the projection problems """
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)  # Suppress the talkative solver's output
        env.start()
        self.model = gp.Model('RENSER', env=env)
        self.x = self.model.addVars(range(self.numVars), lb=-0, ub=1, vtype=GRB.BINARY)
        self.y = self.model.addVars(range(2), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.INTEGER)
        self.yBar = self.model.addVars(range(2), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
        self.absDiv = self.model.addVars(range(2), lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)

        # Add and objective function
        self.model.setObjective(self.absDiv[0] + self.absDiv[1])
        # Ensure that the absolute difference is correctly set
        self.model.addConstr(self.absDiv[0] >= self.y[0] - self.yBar[0])
        self.model.addConstr(self.absDiv[0] >= self.yBar[0] - self.y[0])
        self.model.addConstr(self.absDiv[1] >= self.y[1] - self.yBar[1])
        self.model.addConstr(self.absDiv[1] >= self.yBar[1] - self.y[1])
        # Add all "ordinary" constraints
        self.model.addConstrs(
            gp.quicksum(self.A[i][j] * self.x[j] for j in range(self.numVars)) >= self.b[i] for i in range(self.numCsts)
        )

        # Add definition of the y-variables as the image of the x solution
        self.model.addConstr(self.y[0] == gp.quicksum(self.C[0][j] * self.x[j] for j in range(self.numVars)))
        self.model.addConstr(self.y[1] == gp.quicksum(self.C[1][j] * self.x[j] for j in range(self.numVars)))
        self.model.setParam('TimeLimit', 10)

    def __setBounds(self, sol: hs.Solution):
        """ Method setting the bounds on variables based on the values of the Solution object passed as argument
            @:param sol A solution object that defines the target point
        """
        self.yBar[0].LB = sol.obj_1
        self.yBar[0].UB = sol.obj_1
        self.yBar[1].LB = sol.obj_2
        self.yBar[1].UB = sol.obj_2
        for j in range(self.numVars):
            if 0.001 < sol.x[j] < 0.999:
                self.x[j].LB = 0
                self.x[j].UB = 1
            elif sol.x[j] <= 0.001:
                self.x[j].LB = 0
                self.x[j].UB = 0
            else:
                self.x[j].LB = 1
                self.x[j].UB = 1

    def __solveModel(self):
        """ Method invoking the gurobi solver """
        self.model.optimize()
        if self.model.solCount >= 1:  # Check if solutions we found
            newSol = hs.Solution(self.y[0].X, self.y[1].X, [self.x[j].X for j in range(self.numVars)])
            self.feasibleSolutions.append(newSol)


if '__main__' == __name__:
    A = [[75, 85, 20, 5, 40, 65, 10, 85, 70, 95],
         [85, 85, 90, 10, 60, 25, 45, 80, 40, 55],
         [50, 40, 20, 90, 95, 60, 25, 55, 65, 20]]
    b = [225, 225, 225]
    C = [[45, 20, 79, 4, 10, 6, 34, 50, 38, 23],
         [78, 87, 3, 89, 91, 90, 80, 11, 81, 99]]
    sol1 = hs.Solution(58.97894737, 375.2210526, [0, 1, 0, 0.115789474, 1, 1, 0, 0, 0, 0.978947368])
    sol2 = hs.Solution(186.4146341, 186.4146341, [0.263414634, 0, 1, 0, 0.756097561, 0, 0, 1, 1, 0])
    sol3 = hs.Solution(80.35185185, 229.2314815, [0, 0.861111111, 0, 0.185185185, 1, 0.398148148, 0, 1, 0, 0])
    sol4 = hs.Solution(122.7808731, 205.1559449, [0, 0.337736933, 0.239230327, 0, 0.912693854, 0, 0, 1, 1, 0])
    sol5 = hs.Solution(75.4952251, 243.3042292, [0, 1, 0, 0.204638472, 1, 0.420190996, 0, 0.84311050, 0, 0])
    sols = [sol1, sol2, sol3, sol4, sol5]
    RENSer = RENS()
    RENSer.runRENSer(sols, C, A, b)
    for i, sol in enumerate(RENSer.feasibleSolutions):
        print(f"Target solution ({round(sols[i].obj_1,2)}, {round(sols[i].obj_2,2)})")
        print(f"\tResult : ({sol.obj_1},{sol.obj_2})")
