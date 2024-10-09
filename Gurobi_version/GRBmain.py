import GRBBiObjectiveBnB as BICO_bnb  # Import the branch and bound algorithm based on the gurobipy package

if "__main__" == __name__:
    # Create and object of the bi-objective branch and bond algorithm
    bnb = BICO_bnb.BranchAndBound()
    # Read a data file
    bnb.readData('instances/instance_15_5.json')
    # Options are {'depthFirst', 'breadthFirst', 'bestBound', "random"}
    bnb.setNodeSelectionStrategy('bestBound')
    # Options are {"mostOftenFractional", "averageMostFractional", "random"}
    bnb.setBranchingStrategy('mostOftenFractional')
    # Options are {'LPRelaxation', 'IdealPoint'}
    bnb.setLowerBoundStrategy('LPRelaxation')
    # Activates a plot of the upper bound set - may slow down computations
    bnb.showUpperBoundProgress()
    # Use the RENS heuristic on all nodes at depth <= argument
    bnb.useRENSHeuristic(1)
    # Un-comment this line if you want to load an optimal solution before solving (mimics a strong heuristic)
    # bnb.loadSolutionsFromFile('instance_15_5_solution.json')
    # Un-comment the following line if you want to inherit the lower bound set from the parent node
    bnb.useParentLowerBound()
    bnb.solve()
