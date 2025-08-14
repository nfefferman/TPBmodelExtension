# Utility functions for 1-action model of the theory of planned behavior with
# a temporal bias.

from math import isclose
import numpy as np


def isgeq(a, b, tol=1E-20):
    '''Perform an element-wise check if an elements of an array 'a' are greater
    or equal to the elements of an array 'b' while accounting for
    representation errors up to an absolute tolerance 'tol'.
    
    Parameters
    ----------
    a : a scalar or array
       Number (or array of numbers) to be checked for being greater than or
       equal to the number (or array of numbers) 'b'
       
    b : a scalar or array
       If 'b' is an array, it must have the same dimensions as 'a'
       
    tol : float (default=1E-20)
       Tolerance value
    ''' 
    if hasattr(b, "__len__"):
        b_array = b
    else:
        b_array = b*np.ones(a.shape)
        
    comparison = np.logical_or(a>b_array, np.isclose(a,b, atol=tol))
        
    return comparison

def isleq(a, b, tol=1E-20):
    '''Perform an element-wise check if an elements of an array 'a' are less
    or equal to the elements of an array 'b' while accounting for
    representation errors up to an absolute tolerance 'tol'.
    
    Parameters
    ----------
    a : a scalar or array
       Number (or array of numbers) to be checked for being greater than or
       equal to the number (or array of numbers) 'b'
       
    b : a scalar or array
       If 'b' is an array, it must have the same dimensions as 'a'
       
    tol : float (default=1E-20)
       Tolerance value
    ''' 
    if hasattr(b, "__len__"):
        b_array = b
    else:
        b_array = b*np.ones(a.shape)
        
    comparison = np.logical_or(a<b_array, np.isclose(a,b, atol=tol))
        
    return comparison


def random_uniform_force_mean(low=-0.4, high=1., size=10, iteration=0, max_iter=100):
    '''Draw numbers from a uniform distribution with a set mean value.
    
    Parameters:
    -----------
    low : float (default=-0.4)
       lower end of the support of the uniform distribution
       
    high : float (default=1.)
       upper end of the support of the uniform distribution
       
    size : int (default=10)
       number of samples drawn
       
    iteration : int (default=0)
       f

    max_iter : int (default=100)
       f
    
    Returns:
    --------
    sn : 1D array
       Array of contributions of social-norm mechanism to change of intention.
    '''    
    # draw all but 1 number from a uniform distribution
    sequence1 = np.random.uniform(low=low, high=high, size=size-1)
    
    # set last element to achieve desired mean
    last_element = size*(low+(high-low)/2)-np.sum(sequence1)
    sequence2 = np.concatenate([sequence1, [last_element]])
    
    if iteration >= max_iter:
        # give up and return what you currently have
        print('random_uniform_force_mean reached max_iter')
        return sequence2
    
    elif low <= last_element <= high:
        # last element is inside the support of the distribution; success!
        return sequence2
    
    else:
        # last element is outside the support of the distribution; try again
        #print('Run again', iteration, last_element)
        return random_uniform_force_mean(low=low, high=high, size=size, iteration=iteration+1, max_iter=max_iter)