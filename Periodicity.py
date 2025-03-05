import random, time, pickle
from datetime import datetime
import numpy as np
import networkx as nx
import pandas as pd
from model import *
import itertools
import math
import matplotlib.pyplot as plt

def detect_periodicity_zero(x, t, rel_std_threshold=0.1, min_events=3):
    """ Detects periodicity by checking when x[i+1] = 0
    
    Parameters:
      x                : 1D numpy array of the time series.
      t                : 1D numpy array of corresponding time points.
      threshold        : Maximum allowed ratio (std/mean) of inter-event intervals.
      min_events       : Minimum number of events required.

    Returns:
      (bool, bool, np.array, np.array, float, float) : A tuple:
                         - True if periodicity is detected
                           False otherwise.
                         - True if solution reaches a limit
                           False otherwise
                         - An array of times at which x acts 
                         - An array of intervals between actions of x
                         - The mean interval length between actions of x
                         - The standard deviation of the interval length between actions of x
                         """
    event_indices = []
    # Look for i+1 where x[i+1] == 0 and x[i] != 0
    for i in range(len(x) - 1):
        # If we want tolerance, use np.isclose()
        if x[i+1] == 0.0 and x[i] != 0.0:
            event_indices.append(i+1)
    limit = max(x[-10:])-min(x[-10:])< 10**(-4)
    event_times = t[event_indices]
    
    # Not enough events to conclude periodicity
    if len(event_times) < min_events:
        return False, limit,  event_times, math.nan, math.nan, math.nan
    
    # Compute intervals between successive events
    intervals = np.diff(event_times)
    mean_interval = np.mean(intervals)
    std_interval = np.std(intervals)
    
    # Relative standard deviation of intervals
    rel_std = std_interval / mean_interval 
    
    # If intervals are close enough, call it periodic
    return (rel_std< rel_std_threshold, limit, event_times, intervals, mean_interval, std_interval)
   


def periodic_sim(alphas, threshold_x, shiftS, r, A, plotting = False, tmax = 3000, time_step = 0.01):
    """ Runs a simulation and collects periodicity information
    
    Parameters:
      alphas           : 1D mumpy array of Internal Attitudes
      threshold_x      : float action threshold.
      shiftS           : float Shift in percieved social norms
      r                : float Immediatcy parameter
      A                : Adjacency matrix 
      plotting         : Bool whether to plot the output or not
      tmax             : int Maximum time of simulation
      time_step        : float interval for each time step

    Returns:
      (np.array, float, float) : A tuple:
                         - An array of intervals between actions of x
                         - The mean interval length between actions of x
                         - The standard deviation of the interval length between actions of x
                         """

    all_results = []

    #for alpha_name, alphas in alpha_sets:
    #    for threshold_x, shiftS in parameter_combinations:
            # Build the system 
    system = make_system(
        A, alphas, shiftS=shiftS, shiftC=0.5, scaleC=0.05,
        growth='symlogistic', threshold_x=threshold_x,
        immediacy_parameter= r
    )

    # Integrate
    t, x, y = integrate_system(system, t_max=tmax, time_step=time_step, max_events=1E4)

    # Find node with lowest alpha
    lowest_alpha_index = np.argmin(alphas)
    x_low = x[lowest_alpha_index, :]

    periodicity_detected, limit_detected, event_times, intervals, interval_means, interval_stds = detect_periodicity_zero(
        x_low, t, rel_std_threshold= .1, min_events=3
    )
        
    #print(f"[{alphas}] threshold_x={threshold_x}, shiftS={shiftS}" f"=> Periodic? {periodicity_detected} (events={len(event_times)})")
        
    if plotting:
        # -------------
        # PLOTTING
        # -------------
        plt.figure(figsize=(16, 12))

        # Time Series with Actions Marked
        plt.subplot(4, 1, 1)
        plt.plot(t, x_low, label=f'Node {lowest_alpha_index} (lowest alpha)')
        plt.scatter(event_times, [0]*len(event_times), color='red', zorder=5, label='x reset to 0')
        plt.xlabel("Time")
        plt.ylabel("x value")
        plt.title(f"Time Series for Node {lowest_alpha_index}\nPeriodicity Detected: {periodicity_detected}")
        plt.legend()

        # Inter-event Intervals


        plt.subplot(4, 1, 2)
        if len(event_times) > 1:
            intervals = np.diff(event_times)
            plt.hist(intervals)
            plt.xlabel("Interval Length")
            plt.ylabel("frequency")
            #plt.plot(intervals, marker='o')
            #plt.xlabel("Event Number")
            #plt.ylabel("Interval")
            plt.title("Inter-event Intervals")
        else:
            plt.text(0.5, 0.5, "Not enough events to compute intervals", 
                     horizontalalignment='center', verticalalignment='center')
            plt.title("Inter-event Intervals")

        plt.subplot(4, 1, 3)
        if len(event_times) > 1:
            intervals = np.diff(event_times)
            plt.plot(intervals, marker='o')
            plt.xlabel("Event Number")
            plt.ylabel("Interval")
            plt.title("Inter-event Intervals")
        else:
            plt.text(0.5, 0.5, "Not enough events to compute intervals", 
                     horizontalalignment='center', verticalalignment='center')
            plt.title("Inter-event Intervals")

        plt.tight_layout()
        plt.show()
        # # -------------

    return [intervals, interval_means, interval_stds, periodicity_detected, limit_detected]
    


""" #

# ----------------------------------------------------
# MAIN SIMULATION CODE
# ----------------------------------------------------

# Good test parameters 
    #[.95, .45], r = .6
    #[.95, .45], r = .2
    #[.95, .45], r = .66
    #[.95, .1], r = .0575
    #[.95, .5, .35], r = .3

# Basic simulation parameters
n = 3
complete_graph = True

# Create adjacency matrix
if complete_graph:
    A = np.ones((n, n)) - np.eye(n)
else:
    A=np.zeros([n,n])
    for i in range(0,n):
        for j in range(i+1,n):
            A[i,j]=np.random.choice([0,1])
            A[j,i]=A[i,j]

print(A)
# Time and step size
tmax = 3000
time_step = 0.01

# Parameter sweeps
threshold_x_values = [0.5]
shiftS_values = [-0.5]
# Add more as needed

parameter_combinations = list(itertools.product(threshold_x_values, shiftS_values))
alpha_sets = [
    ("alpha_set1", np.asarray([.95, .45])) #ADD MORE SETS HERE
]

if ((A.sum(1)== 0).any()):
    print("isolated vertex")
else:  
    ret =periodic_sim(np.array([0.95,0.5,0.8]),0.5,-0.5,0.3,A,True) 
    print(ret)
 """
#n=2
#A= np.ones((n, n)) - np.eye(n)
#print(periodic_sim(np.asarray([0.71428571,0.53]),0.6,-0.55,0.8,A,True))