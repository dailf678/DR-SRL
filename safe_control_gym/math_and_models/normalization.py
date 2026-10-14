"""CartPole utility extracted from safe-control-gym (see LICENSE)."""
import numpy as np

def normalize_angle(x):
    '''Wraps input angle to [-pi, pi].'''
    return ((x + np.pi) % (2 * np.pi)) - np.pi
