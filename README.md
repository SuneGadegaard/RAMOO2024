# RAMOO2024
Bi-objective branch and bound algorithm for 0-1 integer linear programming problems. Developed as a "toy" for the RAMOO workshop 2024 in Wuppertal.

In case you find the code useful and use it in an academic context, please include the following reference:

```
@unpublished{GadegaardWuppertal2024,
  author    = {Sune Lauth Gadegaard},
  title     = {Branch and bound for multi-objective optimization: overview and future challenges},
  note      = {Talk presented at the 11th Workshop on Recent Advances in Multi-Objective Optimization (RAMOO 2024) in Wuppertal, Germany},
  year      = {2024},
  month     = {September},
}
```

# The problems tackled
The code solves bi-objective linear 0-1 problems of the form

$$
\begin{align}
\min\ & Cx\\
\text{s.t.}\ & Ax\geq b,\\
&x\in\{0,1\}^n
\end{align}
$$

where $C$ is $2\times n$ matrix, $A$ is an $m\times n$ matrix, $b$ is an $m$-vector, and $x$ is an $n$-vector. The algorithms uses a branch-and-bound approach to compute the set of non-dominated outcome vectors along with one feasible pre-image for each non-dominated vector. A (small) number of settings can be tweaked in order to change the behaviour of the algorithm.

# The two versions of the code
The algorithm is implemented in two versions - one that uses the `PuLP` modelling framework and any supported solver as engine, and then one version (which can be found in the "Gurobi_version" folder) that utilises the `gurobipy` modelling framework as well as the Gurobi solver. 
The latter (the Gurobi version) is orders of magnitude faster than the PuLP + a solver version. This basically boils down to the fact that the PuLP modelling framework writes a `.lp` file which is then solved by the specified solver through terminal commands, whereas the Gurobi version makes use of the Gurobi Python API available through the `gurobipy` package. In addition to this, the Gurobi version also implements a simple primal heuristic, which is run at every node of the branch and bound tree until a specified depth. This also adds to the speed-up of the implementation.

Hence if one has access to the Gurobi solver, it is recommended to use the Gurobi version of the implementation.

In case one wish to use the CPLEX solver, this can be done by altering the `GRBLowerBoundSet.py` and the `RENS.py` files to use the [DOcplex](https://pypi.org/project/docplex/) API to CPLEX. This has not been tested, but it should be relatively easy to transfer from the `gurobipy`syntax to the `DOcplex`syntax.

# Dependencies 
The code makes use of the following libraries

1. [math](https://docs.python.org/3/library/math.html) - used to do mathy things
2. [matplotlib](https://pypi.org/project/matplotlib/) - used for plotting
3. [json](https://docs.python.org/3/library/json.html) - used for reading data/instance files
4. [time](https://docs.python.org/3/library/time.html) - used for timing the algorithm
5. [random](https://docs.python.org/3/library/random.html) - used for random number generation
6. [queue](https://docs.python.org/3/library/queue.html) - used for implementing the branchign tree
7. [PuLP](https://pypi.org/project/PuLP/) - used as AML for building the optimisation problem
8. [gurobipy]([https://pypi.org/project/PuLP/](https://pypi.org/project/gurobipy/)) - used for the "Gurobi_version" implementation which also uses the Gurobi solver as solver engine

In addition to the above packages (that are all part of a standard installation of Python or can be installed using `pip`/`pip3`), <font color="red">**the code makes use of the CBC solver**</font>. On Mac and linux installing CBC is fairly easy using e.g. `$ sudo apt-get install  coinor-cbc coinor-libcbc-dev` on Linux Ubuntu and 

```
$ brew tap coin-or-tools/coinor
$ brew install coin-or-tools/coinor/cbc
```

on a Mac using `homebrew` as package manager. For installing CBC on a Windows machine, please consult the official [COIN-OR github page](https://github.com/coin-or/Cbc).

**In case another solver** (supported by the PuLP modelling framwork) **is installed on the system**, you can alter the code by replacing the lines `self.model.solve(plp.PULP_CBC_CMD(msg=0))` with e.g. `self.model.solve(plp.CPLEX_CMD(msg=0))` if CPLEX should be used or `self.model.solve(plp.GUROBI_CMD(msg=0))` if Gurobi should be used. This should be done in the `LowerBoundSets.py` file. Simply search for the lines to replace.

# Example
First create a new Python project. Next, place the three modules `BiObjectiveBnB.py`, `helperStructures.py`, and `LowerBoundSets.py` in the same folder as your python project. 
The following code solves a Bi objective linear 0-1 program with 10 variables and 3 constraints:

```
import BiObjectiveBnB as BICO_bnb

if "__main__" == __name__:
    bnb = BICO_bnb.BranchAndBound()
    bnb.readData('instance_30_10.json')
    # Options are {'depthFirst', 'breadthFirst', 'bestBound', "random"}
    bnb.setNodeSelectionStrategy('bestBound')
    # Options are {"mostOftenFractional", "averageMostFractional", "random"}
    bnb.setBranchingStrategy('mostOftenFractional')
    # Options are {'LPRelaxation', 'IdealPoint'}
    bnb.setLowerBoundStrategy('LPRelaxation')
    # Un-comment this line if you want to load an optimal solution before solving (mimics a strong heuristic)
    # bnb.loadSolutionsFromFile('instance_10_3_solution.json')
    # Un-comment the following line if you want to inherit the lower bound set from the parent node
    # bnb.useParentLowerBound()
    bnb.solve()
```

The instance is specified in a `.json` file having the format:

```
{
  "numVars" : 10,
  "numCsts" : 3,
  "A" : [[50,	20,	65,	50,	20,	70,	40,	30,	20,	5],
        [75,	50,	75,	10,	60,	65,	25,	75,	45,	35],
        [65,	65,	40,	10,	30,	5,	25,	65,	65,	40]],
  "b" : [200, 300, 250],
  "C" : [[21,	1,	21,	18,	14,	2,	4,	2,	2,	7],
        [24,	22,	13,	16,	15,	19,	3,	12,	10,	11]]
}
```
placed in the same folder as the python script above.
