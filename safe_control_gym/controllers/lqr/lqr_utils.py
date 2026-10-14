"""CartPole utility extracted from safe-control-gym (see LICENSE)."""
import numpy as np

def get_cost_weight_matrix(weights, dim):
    '''Gets weight matrix from input args.

    Args:
        weights (list): A 1D list of weights.
        dim (int): The dimension of the desired cost weight matrix.

    Returns:
        W (ndarray): The cost weight matrix.
    '''

    if len(weights) == dim:
        W = np.diag(weights)
    elif len(weights) == 1:
        if isinstance(weights, list):
            W = np.diag(weights * dim)
        elif isinstance(weights, np.ndarray):
            W = np.diag(weights.item() * np.ones(dim))
        else:
            raise Exception('Wrong type for cost weights.')
    else:
        raise Exception('Wrong dimension for cost weights.')
    return W
