import numpy as np

def shearframe_elements(levels, floor_width=1.0, floor_height=1.0):

    element_matrix = np.zeros([levels*3, 3])
    element_matrix[:, 0] = np.arange(1,levels*3+1,1)

    # Left vertical elements
    element_matrix[0:levels, 1] = np.arange(1, levels+1, 1)
    element_matrix[0:levels, 2] = np.arange(1, levels+1, 1) + 1
    
    # Right vertical elements
    element_matrix[levels:2*levels, 1] = np.arange(levels+2, 2*levels+2, 1)
    element_matrix[levels:2*levels, 2] = np.arange(levels+2, 2*levels+2, 1) + 1

    # Horizontal elements
    element_matrix[2*levels:, 1] = np.arange(1, levels+1, 1) + 1
    element_matrix[2*levels:, 2] = np.arange(levels+2, 2*levels+2, 1) + 1

    node_matrix = np.zeros([2+2*levels, 4])
    node_matrix[:, 0] = np.arange(1, 2+2*levels+1,1)

    # x coord
    node_matrix[0:levels+1, 1] = -floor_width/2
    node_matrix[levels+1:, 1] = floor_width/2


    # y coord
    node_matrix[0:levels+1, 2] = np.arange(0, levels+1)*floor_height
    node_matrix[levels+1:, 2] = np.arange(0, levels+1)*floor_height

    return node_matrix, element_matrix
