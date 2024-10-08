import BiObjectiveBnB as BICO_bnb

if "__main__" == __name__:
    bnb = BICO_bnb.BranchAndBound()
    bnb.readData('instances/instance_10_3.json')
    # Options are {'depthFirst', 'breadthFirst', 'bestBound', "random"}
    bnb.setNodeSelectionStrategy('bestBound')
    # Options are {"mostOftenFractional", "averageMostFractional", "random"}
    bnb.setBranchingStrategy('mostOftenFractional')
    # Options are {'LPRelaxation', 'IdealPoint'}
    bnb.setLowerBoundStrategy('LPRelaxation')
    # Un-comment this line if you want to load an optimal solution before solving (mimics a strong heuristic)
    # bnb.loadSolutionsFromFile('instances/instance_10_3_solution.json')
    # Un-comment the following line if you want to inherit the lower bound set from the parent node
    # bnb.useParentLowerBound()
    bnb.solve()
