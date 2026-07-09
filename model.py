# Implementation of our 1-action model of the theory of planned behavior with
# a temporal bias.

import random, time
import numpy as np
from scipy.integrate import solve_ivp
from utils import isgeq, random_uniform_force_mean


def social_norms_coefficient(y, A):
    '''Compute betas for perceived-social-norm effect f_S(beta_i). This is 
    currently equivalent to how we compute gammas for behavior control.
    
    Parameters:
    -----------
    y : 1D array
       vector of nudge values
       
    A : 2D array
       adjacency matrix
       
    Returns
    -------
    snc : an array of social-norm coefficients 'beta_i'
    '''
    vis = np.matmul(A, y)
    snc = vis/np.sum(A, axis=1)
    
    return snc


def perceived_control_coefficient(y, A):
    '''Compute gammas for perceived-behavior-control effect f_C(gamma_i). This
    is currently equivalent to how we compute betas for social norms.
    
    Parameters:
    -----------
    y : 1D array
       vector of nudge values
       
    A : 2D array
       adjacency matrix
       
    Returns
    -------
    pcc : an array of perceived-behavior-control coefficients 'gamma_i'
    '''
    vis = np.matmul(A, y)
    pcc = vis/np.sum(A, axis=1)
    
    return pcc


def internal_attitude(y, A, alpha, scale=1, shift=0):
    '''Calculate the change f_A(alpha_i) in intentions x caused by internal 
    attitudes.
    
    Parameters:
    -----------
    y : 1D array
       vector of nudge values
       
    A : 2D array
       adjacency matrix
       
    alpha : 1D array
       array of internal-attitude parameters alpha_i
       
    scale : float (optional; default=1)
       Set 'scale' to a value larger (smaller) than 1 to scale all internal
       attitudes up (down)
    
    shift : float (optional; default=0)
       Shift all internal attitudes the value of 'shift'. If 'shift' is smaller
       than 0, internal attitudes are shifted down. If 'shift' is lager than
       0, internal attitudes are shifted up.
    
    Returns:
    --------
    internal_motivations : 1D array
       Array of contributions of internal-attitude mechanism to change of 
       intentions.
    '''
    internal_motivations = scale*(shift+alpha)
    
    return internal_motivations


def social_norms(y, A, scale=1, shift=-0.5):
    '''Calculate the change f_S(beta_i) in intentions x caused by perceived 
    social norms.
    
    Parameters:
    -----------
    y : 1D array
       nudge values
       
    A : 2D array
       adjacency matrix
       
    scale : float (optional; default=1)
       Set 'scale' to a value larger (smaller) than 1 to scale all perceived 
       social norms up (down)
       
    shift : float (optional; default=-0.5)
       Shift all perceived-social-norm values by the value of 'shift'. If 
       'shift' is smaller than 0, perceived-social-norm values are shifted 
       down. If 'shift' is lager than 0, perceived-social-norm values are 
       shifted up. 
       
       A negative shift is equivalent to a threshold $\tau$ for positive 
       social reinforcement.
    
    Returns:
    --------
    sn : 1D array
       Array of contributions of social-norm mechanism to change of intention.
    '''    
    beta = social_norms_coefficient(y, A)
    sn = scale*(beta+shift)
    
    return sn


def perceived_control(y, A, shift=0, scale=1):
    '''Calculate the change in intentions from perceived control.
    
    Parameters:
    -----------
    y : 1D array
       vector of nudge values
       
    A : 2D array
       adjacency matrix
       
    scale : float (optional; default=1)
       Set 'scale' to a value larger (smaller) than 1 to scale all perceived 
       social norms up (down)
       
    shift : float (optional; default=0)
       Shift all perceived-control values by the value of 'shift'. If 'shift' 
       is smaller than 0, perceived-control values are shifted down. If 'shift'
       is lager than 0, perceived-control values are shifted up. 
    
    Returns:
    --------
    internal_motivations : 1D array
       Array of contributions of perceived-behavioral-control mechanism to
       change of intention.
    '''    
    gamma = perceived_control_coefficient(y, A)
    pc = scale*(gamma+shift)
    
    return pc


def make_system(A, alphas, shiftA=0, scaleA=1, constantA=None, shiftS=-0.5, 
    scaleS=1, constantS=None, shiftC=0, scaleC=1, constantC=None, 
    growth='logistic', action_influence_evolution=None, threshold_x=None, 
    immediacy_parameter=None, primacy_parameter=None, recency_parameter=None):
    '''Make a dynamical system.
    
    Parameters
    ----------
    A : 2D array
       adjacency matrix
       
    alphas : 1D array
       array of internal-attitude parameters
       
    shiftA : float (optional; default=0)
       Shift all internal attitudes the value of 'shift'. If 'shift' is smaller
       than 0, internal attitudes are shifted down. If 'shift' is lager than
       0, internal attitudes are shifted up.  
       
    scaleA : float (optional; default=1)
       Set 'scale' to a value larger (smaller) than 1 to scale all internal
       attitudes up (down).

    constantA : float (default=None)
       If 'constantA' is not None, replace dynamics of internal attitude with a
       constant value in the ODE. Set 'constantA' to 0 to create a system 
       without any effect of internal attitudes.
       
    shiftS : float (optional; default=-0.5)
       Shift all perceived-social-norm values by the value of 'shift'. If 
       'shift' is smaller than 0, perceived-social-norm values are shifted 
       down. If 'shift' is lager than 0, perceived-social-norm values are 
       shifted up. 
       
       A negative shift is equivalent to a threshold $\tau$ for positive 
       social reinforcement.
       
    scaleS : float (optional; default=1)
       Set 'scale' to a value larger (smaller) than 1 to scale all perceived 
       social norms up (down).
       
    constantS : float (default=None)
       If 'constantS' is not None, replace dynamics of perceived social norms
       with a constant value in the ODE. Set 'constantS' to 0 to create a 
       system without any effect of perceived social norms.

    shiftC : float (optional; default=0)
       Shift all perceived-control values by the value of 'shift'. If 'shift' 
       is smaller than 0, perceived-control values are shifted down. If 'shift'
       is lager than 0, perceived-control values are shifted up. 

    scaleC : float (optional; default=1)
       Set 'scale' to a value larger (smaller) than 1 to scale all perceived 
       social norms up (down).
       
    constantC : float (default=None)
       If 'constantC' is not None, replace dynamics of perceived control with
       a constant value in the ODE. Set 'constantC' to 1 to create a system 
       without any effect of perceived social norms.
       
    growth : function or string in ['linear' | 'exponential' | 'logistic' | 'symlogistic']
       (default='logistic')
       Set or select a functional from for the growth of intentions.  If 'g' is
       a function. Intentions grow like $\dot x = g(x)$ (in the absense of 
       other mechanisms).
       
    action_influence_evolution : function (default=None)
       Set functional form for the decay of influence of one's actions on other
       people. When no value is given by the user, (1) implement exponential 
       decay for the action influence evolution using the `recency_parameter`
       or (2) when also no recency parameter is given by the user, use default
       value `action_influence_evolution=lambda x: x*0`. This default value
       corresponds to a system in which action influence does not decay over 
       time. 
       
    threshold_x : float (default=1)
       Threshold parameter for the default stopping condition.

    immediacy_parameter : float in [0,+Inf) (default=None)
       When `action_influence_evolution` is None, use `recency_parameter` to 
       define and action influence evolution with exponential decay via
       `action_influence_evolution=lambda x: x*(-recency_parameter)`.

    recency_parameter : float in [0,+Inf) (default=None)
       Not implemented.
    
    primacy_parameter : float in [0,1] (default=None)
       [DEPRECATED] When an action is taken, the corresponding action influence
       `y` is updated according to 
       `y_new = primacy_parameter*y_old + (1-primacy_parameter)`.
       If no value is given by the user, set `primacy_parameter` to 0.5.
    
    Returns
    -------
    f : function
       This function can be handed over to the integrator.
    '''
    # define growth function
    if growth=='logistic':
        g = lambda x: x*(1-x)
    elif growth=='symlogistic':
        g = lambda x: (x+1)*(1-x)
    elif growth=='exponential':
        g = lambda x: x
    elif growth=='linear':
        g = lambda x: 1.0
    else:
        # assume that a function was passed
        g = growth
        
    # define action influence evolution
    if action_influence_evolution is None:
        if immediacy_parameter is None:
            action_influence_evolution = lambda x: x*0
            immediacy_parameter = 0
        else:
            #action_influence_evolution = lambda x: x*(-immediacy_parameter)
            action_influence_evolution = lambda x: x*(-immediacy_parameter)
            
    # define primacy and immediacy parameter
    if immediacy_parameter is None:
        immediacy_parameter = 0.
    if primacy_parameter is None:
        primacy_parameter = 0.
    
    # define internal-attitudes function
    if constantA is not None:
        # replace function with constant
        ia = lambda y, mat : constantA
    else:
        # do not replace function with constant
        ia = lambda y, mat : internal_attitude(y, mat, alphas, 
            shift=shiftA, scale=scaleA)
        
    # define social-norms function    
    if constantS is not None:
        # replace function with constant
        sn = lambda y, mat : constantS
    else:
        # do not replace function with constant
        sn = lambda y, mat : social_norms(y, mat, 
            shift=shiftS, scale=scaleS)

    # define perceived-behavioral-control function
    if constantC is not None:
        # replace function with constant
        pc = lambda y, mat : constantC
    else:
        # do not replace function with constant
        pc = lambda y, m : perceived_control(y, m, shift=shiftC, scale=scaleC)
        
    # define dynamical system
    x_default = np.zeros(len(A))
    y_default = np.zeros(len(A))
    X_default = np.concatenate([x_default, y_default])
    
    def f(t=0, X=np.copy(X_default)):
        n = len(X)//2
        x, y = X[:n], X[n:]
        dx = ((ia(y, A)+sn(y, A))*pc(y, A))*g(x)
        dy = action_influence_evolution(y)
        dX = np.concatenate([dx, dy])
        return dX
    
    f.threshold_x = threshold_x
    f.primacy_parameter = primacy_parameter
    f.immediacy_parameter = immediacy_parameter
    
    return f
   
    
def integrate_system(system, t_max=10.0, time_step=0.01, X0=None, x0=None, 
    y0=None, stopping_condition=None, threshold_x=None, reset_x=0, 
    max_events=1E3, tol=1E-20):
    '''Integrate the function 'system' until time 't_max'.
    
    Parameters
    ----------
    system : function with numerical attributes
       The system to be integrated.
    
    t_max : float (default=10.0)
       End time of integration.

    time_step : float (default=0.01)
       Step size at which integration should be evaluated.
       
    X0 : 1D array (default=None)
       Initial condition for intentions and nudge values.
       
    x0 : 1D array (default=None)
       Initial condition for intentions.  If x0 is None, initialize 
       with a 1D array of numbers chosen uniformly at random in [0,0.25]. The 
       argument 'x0' is only used when 'X0' is None.
       
    y0 : 1D arrat (default=None)
       Initial condition for nudge values. If y0 is None, initialize 
       with a 1D array of zeros. The argument 'y0' is only used when 'X0' is 
       None. 
       
    stopping_condition : function with attributes 'terminal' and 'direction'
       (default=threshold_x-np.max(x[:len(x)//2]) with 'terminal=True' and 
       'direction=-1')
       Stopping condition for the integrator. Whenever the stopping condition
       is reached, an action occurs. The corresponding intention value is reset
       to 'reset_x' and the corresponding action counter increases by 1.
       
    threshold_x : float (default=None)
       Threshold parameter for the default stopping condition. This option is 
       available to override the threshold parameter given by the system.

    reset_x : float (default=0)
       Whenever the stopping condition is reached, intention values greater 
       than or equal to 1 are set to 'reset_x'.
       
    max_events : integer (default=1E3)
       Number of actions after which the integration should be stopped, even if
       the time 't_max' is not reached. Think of this as an emergency break.

    tol : float (default=1E-20)
       Tolerance value for numerical `is_geq` function in the threshold 
       operation.
       
    Returns
    -------
    t_array : 1D array
       Array of time stamps
    
    X_array : 2D array
       Array of intentions and nudge values 
       
    Y_array : 2D array
       Array of action counts
    '''
    # find n
    output = system()
    n = output.shape[0]//2
    
    # set initial condition
    if X0 is None:
        if x0 is None:
            x0 = np.random.uniform(size=n)*0.25
        
        if y0 is None:
            y0 = np.zeros(n)
            
        X0 = np.concatenate([x0,y0])
        
    # set threshold for x
    if threshold_x is None:
        threshold_x = system.threshold_x
        
    # set default stopping condition
    if stopping_condition is None:
        
        def stopping_condition(t, x): return threshold_x-np.max(x[:len(x)//2])
        stopping_condition.terminal = True
        stopping_condition.direction = -1
            
    # integrate
    t, counter = 0.0, 0
    X = X0 # initial state
    Y = np.zeros(n) # initial count
    
    # initialize lists
    t_list = [np.zeros(1)]
    X_list = [np.zeros((2*n,1 ))]
    X_list[0][:,0] = X0
    Y_list = [np.zeros((n, 1))]
    Y_list[0][:,0] = Y
    
    while (t < t_max and counter < max_events):
        
        #print('counter', counter, t)
        
        # integrate until threshold is reached
        t_eval=np.arange(t,t_max, time_step)
        if t_eval[-1]>t_max:
            t_eval = t_eval[:-1]
        sol = solve_ivp(system, [t, t_max], X, events=stopping_condition, 
                        t_eval=t_eval)
        
        if len(sol.t_events[0]): 
            # collect state (t, x, y) at threshold
            t = sol.t_events[0][0]
            X = sol.y_events[0][0]   
               
            # update state at threshold
            x, y = X[:n], X[n:]
            updates = np.array(isgeq(x,threshold_x, tol=tol), dtype=float)
            pr = system.primacy_parameter
            ymax = 1. #system.immediacy_parameter 
            new_x = x + updates*(reset_x-x)
            new_y = ((1-updates)*y + updates*(pr*y/2 + (1-pr/2)*ymax))
            new_X = np.concatenate([new_x, new_y])
            new_Y = Y + updates
            
        else:
            t = t_max
            new_X = X
            new_Y = Y

        # store solutions
        t_list += [sol.t]
        X_list += [sol.y]
        Y_list += [np.vstack([Y for _ in range(len(sol.t)-1)]+[new_Y]).T]

        X = new_X
        Y = new_Y
        
        # update counter
        counter += 1
        
    # make arrays
    t_array = np.concatenate(t_list)
    X_array = np.concatenate(X_list, axis=1)
    Y_array = np.concatenate(Y_list, axis=1)
    
    return t_array, X_array, Y_array       

def integrate_negative_system(system, t_max=10.0, time_step=0.01, X0=None, x0=None, 
    y0=None, stopping_condition=None, threshold_x=None, reset_x=0, 
    max_events=1E3, tol=1E-20):
    '''Integrate the function 'system' until time 't_max'.
    
    Parameters
    ----------
    system : function with numerical attributes
       The system to be integrated.
    
    t_max : float (default=10.0)
       End time of integration.

    time_step : float (default=0.01)
       Step size at which integration should be evaluated.
       
    X0 : 1D array (default=None)
       Initial condition for intentions and nudge values.
       
    x0 : 1D array (default=None)
       Initial condition for intentions.  If x0 is None, initialize 
       with a 1D array of numbers chosen uniformly at random in [0,0.25]. The 
       argument 'x0' is only used when 'X0' is None.
       
    y0 : 1D arrat (default=None)
       Initial condition for nudge values. If y0 is None, initialize 
       with a 1D array of zeros. The argument 'y0' is only used when 'X0' is 
       None. 
       
    stopping_condition : function with attributes 'terminal' and 'direction'
       (default=threshold_x-np.max(x[:len(x)//2]) with 'terminal=True' and 
       'direction=-1')
       Stopping condition for the integrator. Whenever the stopping condition
       is reached, an action occurs. The corresponding intention value is reset
       to 'reset_x' and the corresponding action counter increases by 1.
       
    threshold_x : float (default=None)
       Threshold parameter for the default stopping condition. This option is 
       available to override the threshold parameter given by the system.

    reset_x : float (default=0)
       Whenever the stopping condition is reached, intention values greater 
       than or equal to 1 are set to 'reset_x'.
       
    max_events : integer (default=1E3)
       Number of actions after which the integration should be stopped, even if
       the time 't_max' is not reached. Think of this as an emergency break.

    tol : float (default=1E-20)
       Tolerance value for numerical `is_geq` function in the threshold 
       operation.
       
    Returns
    -------
    t_array : 1D array
       Array of time stamps
    
    X_array : 2D array
       Array of intentions and nudge values 
       
    Y_array : 2D array
       Array of action counts
    '''
    # find n
    output = system()
    n = output.shape[0]//2
    
    # set initial condition
    if X0 is None:
        if x0 is None:
            x0 = np.random.uniform(size=n)*0.25
        
        if y0 is None:
            y0 = np.zeros(n)
            
        X0 = np.concatenate([x0,y0])
        
    # set threshold for x
    if threshold_x is None:
        threshold_x = system.threshold_x
        
    # set default stopping condition
    if stopping_condition is None:
        
        def stopping_condition(t, x): return min([threshold_x-np.max(x[:len(x)//2]),np.min(x[:len(x)//2])+threshold_x])
        stopping_condition.terminal = True
        stopping_condition.direction = -1
            
    # integrate
    t, counter = 0.0, 0
    X = X0 # initial state
    Y = np.zeros(n) # initial count
    
    # initialize lists
    t_list = [np.zeros(1)]
    X_list = [np.zeros((2*n,1 ))]
    X_list[0][:,0] = X0
    Y_list = [np.zeros((n, 1))]
    Y_list[0][:,0] = Y
    
    while (t < t_max and counter < max_events):
        
        #print('counter', counter, t)
        
        # integrate until threshold is reached
        t_eval=np.arange(t,t_max, time_step)
        if t_eval[-1]>t_max:
            t_eval = t_eval[:-1]
        sol = solve_ivp(system, [t, t_max], X, events=stopping_condition, 
                        t_eval=t_eval)
        
        if len(sol.t_events[0]): 
            # collect state (t, x, y) at threshold
            t = sol.t_events[0][0]
            X = sol.y_events[0][0]   
               
            # update state at threshold
            x, y = X[:n], X[n:]
            plusupdates = np.array(isgeq(x,threshold_x, tol=tol), dtype=float)
            minusupdates = np.array(isgeq(-x,threshold_x,tol=tol),dtype=float)
            pr = system.primacy_parameter
            ymax = 1. #system.immediacy_parameter 
            new_x = x + plusupdates*(reset_x-x)+ minusupdates*(reset_x-x)
            #new_y = ((1-plusupdates-minusupdates)*y + plusupdates*(pr*y/2 + (1-pr/2)*ymax))-minusupdates*(pr*y/2+(1-pr/2)*ymax)
            new_y = (1-plusupdates-minusupdates)*y+plusupdates-minusupdates
            new_X = np.concatenate([new_x, new_y])
            new_Y = Y + plusupdates+minusupdates
            
        else:
            t = t_max
            new_X = X
            new_Y = Y

        # store solutions
        t_list += [sol.t]
        X_list += [sol.y]
        Y_list += [np.vstack([Y for _ in range(len(sol.t)-1)]+[new_Y]).T]

        X = new_X
        Y = new_Y
        
        # update counter
        counter += 1
        
    # make arrays
    t_array = np.concatenate(t_list)
    X_array = np.concatenate(X_list, axis=1)
    Y_array = np.concatenate(Y_list, axis=1)
    
    return t_array, X_array, Y_array      



def integrate_system_and_output_events(system, t_max=10.0, time_step=0.01, X0=None, x0=None, 
    y0=None, stopping_condition=None, threshold_x=None, reset_x=0, 
    max_events=1E3, tol=1E-20):
    '''Integrate the function 'system' until time 't_max' and track the state when threshold is reached.
    
    Parameters
    ----------
    system : function with numerical attributes
       The system to be integrated.
    
    t_max : float (default=10.0)
       End time of integration.

    time_step : float (default=0.01)
       Step size at which integration should be evaluated.
       
    X0 : 1D array (default=None)
       Initial condition for intentions and nudge values.
       
    x0 : 1D array (default=None)
       Initial condition for intentions.  If x0 is None, initialize 
       with a 1D array of numbers chosen uniformly at random in [0,0.25]. The 
       argument 'x0' is only used when 'X0' is None.
       
    y0 : 1D arrat (default=None)
       Initial condition for nudge values. If y0 is None, initialize 
       with a 1D array of zeros. The argument 'y0' is only used when 'X0' is 
       None. 
       
    stopping_condition : function with attributes 'terminal' and 'direction'
       (default=threshold_x-np.max(x[:len(x)//2]) with 'terminal=True' and 
       'direction=-1')
       Stopping condition for the integrator. Whenever the stopping condition
       is reached, an action occurs. The corresponding intention value is reset
       to 'reset_x' and the corresponding action counter increases by 1.
       
    threshold_x : float (default=None)
       Threshold parameter for the default stopping condition. This option is 
       available to override the threshold parameter given by the system.

    reset_x : float (default=0)
       Whenever the stopping condition is reached, intention values greater 
       than or equal to 1 are set to 'reset_x'.
       
    max_events : integer (default=1E3)
       Number of actions after which the integration should be stopped, even if
       the time 't_max' is not reached. Think of this as an emergency break.

    tol : float (default=1E-20)
       Tolerance value for numerical `is_geq` function in the threshold 
       operation.
       
    Returns
    -------
    t_array : 1D array
       Array of time stamps
    
    X_array : 2D array
       Array of intentions and nudge values 
       
    Y_array : 2D array
       Array of action counts

    t_events_array : 1D array
       Array of time stamps at which stopping_condition is triggered

    X_events_array : 2D array
       Array of intentions and nudge values when stopping_condition is triggered
    '''
    # find n
    output = system()
    n = output.shape[0]//2
    
    # set initial condition
    if X0 is None:
        if x0 is None:
            x0 = np.random.uniform(size=n)*0.25
        
        if y0 is None:
            y0 = np.zeros(n)
            
        X0 = np.concatenate([x0,y0])
        
    # set threshold for x
    if threshold_x is None:
        threshold_x = system.threshold_x
        
    # set defauly stopping condition
    if stopping_condition is None:
        
        def stopping_condition(t, x): return threshold_x-np.max(x[:len(x)//2])
        stopping_condition.terminal = True
        stopping_condition.direction = -1
            
    # integrate
    t, counter = 0.0, 0
    X = X0 # initial state
    Y = np.zeros(n) # initial count
    
    # initialize lists
    t_list = [np.zeros(1)]
    X_list = [np.zeros((2*n,1 ))]
    X_list[0][:,0] = X0
    Y_list = [np.zeros((n, 1))]
    Y_list[0][:,0] = Y
    t_events_list = []
    X_events_list = []
    
    while (t < t_max and counter < max_events):
        
        # integrate until threshold is reached
        t_eval=np.arange(t,t_max, time_step)
        if t_eval[-1]>t_max:
            t_eval = t_eval[:-1]
        sol = solve_ivp(system, [t, t_max], X, events=stopping_condition, 
                        t_eval=t_eval)
        # print('t_events', sol.t_events)
        # print('y_events', sol.y_events)
        
        if len(sol.t_events[0]): 
            # collect state (t, x, y) at treshold
            t = sol.t_events[0][0]
            X = sol.y_events[0][0]
               
            # update state at threshold
            x, y = X[:n], X[n:]
            updates = np.array(isgeq(x,threshold_x, tol=tol), dtype=float)
            pr = system.primacy_parameter
            new_x = x + updates*(reset_x-x)
            new_y = ((1-updates)*y + updates*(pr*y + 1-pr))
            new_X = np.concatenate([new_x, new_y])
            new_Y = Y + updates

            # t_events_list += [sol.t_events[0]]
            # X_events_list += [sol.y_events[0]]
            t_events_list += [t]
            X_events_list += [X]
            
        else:
            t = t_max
            new_X = X
            new_Y = Y

        # store solutions
        t_list += [sol.t]
        X_list += [sol.y]
        Y_list += [np.vstack([Y for _ in range(len(sol.t)-1)]+[new_Y]).T]

        X = new_X
        Y = new_Y
        
        # update counter
        counter += 1
        
    # make arrays
    t_array = np.concatenate(t_list)
    X_array = np.concatenate(X_list, axis=1)
    Y_array = np.concatenate(Y_list, axis=1)
    t_events_array = np.array(t_events_list)
    X_events_array = np.array(X_events_list)
    
    return t_array, X_array, Y_array, t_events_array, X_events_array    
    
