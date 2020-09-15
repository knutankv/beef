class Node:
    def __init__(self, label, coordinates, ndofs=None, global_dofs=None):
        self.label = int(label)
        self.coordinates = coordinates
        self.ndofs = ndofs                #number of dofs, normally defined later
        self.global_dofs = global_dofs    #global dofs, normally defined later