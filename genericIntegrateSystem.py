import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import itertools
from pathlib import Path
from joblib import Parallel, delayed

def integrate_system_scipy(t_max, x0, y0, alpha, threshold_x=0.8):
    """
    SciPy RK4
    Handles discontinuous resets
    """
    n = len(x0)
    
    #ODE
    def rhs(t, state):
        # Unpack state
        x = state[:n]
        y = state[n:]
        
        #parameters
        r = 0.86
        mu_c = 0.05
        sigma_a = 1
        sigma_s = 1
        sigma_c = 1
        mu_s = .5
        if n > 1:
            gamma = (np.sum(y) - y) / (n - 1)
        else:
            gamma = np.zeros(n) # if there is only 1 player
        dxdt = (sigma_a * alpha + sigma_s * (gamma - mu_s)) * sigma_c * (gamma + mu_c) * (1 - x) * (1 + x)
        dydt = -r * y 
        return np.concatenate([dxdt, dydt])

    # event
    def hit_threshold(t, state):
        x = state[:n]
        return np.max(x) - threshold_x
    
    # Halt integration when this event returns 0
    hit_threshold.terminal = True 
    hit_threshold.direction = 1 # Only trigger when crossing from below

    # Integration loop
    t_current = 0.0
    state = np.concatenate([x0, y0])
    
    # Arrays to store trajectory
    t_out = []
    state_out = []
    
    events_triggered = 0
    
    while t_current < t_max:
        # Run the solver until t_max
        sol = solve_ivp(
            fun=rhs,
            t_span=(t_current, t_max),
            y0=state,
            method='RK45',      # Runge-Kutta 4
            events=hit_threshold,
            dense_output=True,  
            rtol=1e-5,          
            atol=1e-7           
        )
        
        # store results
        t_out.append(sol.t)
        state_out.append(sol.y)
        
        # Check why solver stopped
        if sol.status == 1: 
            t_current = sol.t[-1]
            state = sol.y[:, -1].copy()
            
            # Extract both x and y
            x_current = state[:n]
            y_current = state[n:] 
            
            # Identify who hit threshold 
            players_to_reset = x_current >= (threshold_x - 1e-8) 
            
            # discontinuous updates
            
            # Reset x for the individual who crossed the threshold
            x_current[players_to_reset] = 0 #back to baseline
            
            # Set y = 1 for same individual
            y_current[players_to_reset] = 1.0
            
            # update arrays
            state[:n] = x_current
            state[n:] = y_current
            
            events_triggered += 1
            
        elif sol.status == 0:
            # 0 means successfully reached t_max without hitting the threshold
            break
            
        else:
            # Status -1 indicates integration failed
            raise RuntimeError(f"Integration failed at t={t_current}: {sol.message}")

    # 4. Stitch segments
    t_final = np.concatenate(t_out)
    state_final = np.concatenate(state_out, axis=1)
    
    x_final = state_final[:n, :]
    y_final = state_final[n:, :]
    
    return t_final, x_final, y_final