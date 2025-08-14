# PACKAGES

import random, time, pickle
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
from model import *
from analytics import period, onset

alphabet = ['(' + _ +')' for _ in 'abcdefghijklmnopqrstuvwxyz']
blues = [plt.cm.PuBu(1-val) for val in [0.0, 0.25, 0.50, 0.75]]

plt.rcParams["font.family"] = "Arial"
plt.rcParams['mathtext.fontset'] = "custom"
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

# SET CONFIGURATIONS

# Set social graph
n = 2
complete_graph=True

# Set simulation time and resolution
tmax = 5000
time_step=0.01

# Set model parameters
threshold_x = 0.8

# Set seeds
seed = 0
# seed 0: two slow adopters, but eventually all
# seed 1: one action taker only
# seed 2: no actions taken 

# PROCESS CONFIGURATIONS



def FigFromSim(alpha1, alpha2,threshold_x, immediacy,x1init=0,x2init=0, save = False, file_loc = None, plttitle = "", negative = False ):
    A = np.ones((n,n))-np.eye(n) 

    alphas = np.sort(np.array([alpha1, alpha2]))[::-1]
    x0 = [x1init,x2init]

    system = make_system(A, alphas, shiftS=-0.5, shiftC=0.5, scaleC=0.05,
        growth='symlogistic', threshold_x=threshold_x, 
        immediacy_parameter=immediacy)
    # INTEGRATE
    if(negative):
        t, x, y = integrate_negative_system(system, t_max=tmax, 
                time_step=time_step, max_events=1000,x0=x0)
    else:
        t, x, y = integrate_system(system, t_max=tmax, 
                time_step=time_step, max_events=1000,x0=x0)  
    a = np.copy(alphas)

    if(save):
        pickle.dump([t, x, y, a], open(file_loc, 'wb'))

    xlims = [-25,1750]
    xticks = [0,500,1000,1500]

    
    fig = plt.figure()

    ax1 = plt.gca()
    
    # add column titles
    plt.title(plttitle, fontsize=12)

    # plot simulation results
    for i in range(n):
        plt.plot(t, x[i], color=blues[i], ls='-', 
            label='Individual '+str(i+1))
    plt.ylim([-1.05,1.05])
    plt.xlim(xlims)

    plt.xlabel('time')
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([])
    plt.ylabel(r'Behavioral intention $x_i$')

    return(fig)

def TwoFigsFromSim(alpha1, alpha2,threshold_x, immediacy,x1init=0,x2init=0, save = False, file_loc = None, plttitle = "", negative = False):
    A = np.ones((n,n))-np.eye(n) 

    alphas = np.sort(np.array([alpha1, alpha2]))[::-1]
    x0 = [x1init,x2init]

    system = make_system(A, alphas, shiftS=-0.5, shiftC=0.5, scaleC=0.05,
        growth='symlogistic', threshold_x=threshold_x, 
        immediacy_parameter=immediacy)
    # INTEGRATE
    if(negative):
        t, x, y = integrate_negative_system(system, t_max=tmax, 
                time_step=time_step, max_events=1000,x0=x0)
    else:
        t, x, y = integrate_system(system, t_max=tmax, 
                time_step=time_step, max_events=1000,x0=x0)  
    a = np.copy(alphas)

    if(save):
        pickle.dump([t, x, y, a], open(file_loc, 'wb'))

    xlims = [-25,1750]
    xticks = [0,500,1000,1500]

    
    fig = plt.figure()

    ax1 = plt.subplot(1,2,1)
    
    # add column titles
    plt.title(plttitle, fontsize=12)

    # plot simulation results
    for i in range(n):
        plt.plot(t, x[i], color=blues[i], ls='-', 
            label='Individual '+str(i+1))
    plt.ylim([-1.05,1.05])
    plt.xlim(xlims)

    plt.xlabel('time')
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([])
    plt.ylabel(r'Behavioral intention $x_i$')

    ax2 = plt.subplot(1,2,2) 
    for i in range(n):
        plt.plot(t, x[n+i], color=blues[i], ls='-', 
            label='Individual '+str(i+1))
    plt.ylim([-1.2,1.2])
    plt.xlim(xlims)

    plt.xlabel('')
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([])
    plt.ylabel(r'Nudge effect $y_i$')


    return(fig)


def FigFromData(file_loc, plttitle = ""):

    t, x, y, a = pickle.load(open(file_loc, 'rb'))

    xlims = [-25,1750]
    xticks = [0,500,1000,1500]

    
    fig = plt.figure()

    ax1 = plt.gca()
    
    # add column titles
    plt.title(plttitle, fontsize=12)

    # plot simulation results
    for i in range(n):
        plt.plot(t, x[i], color=blues[i], ls='-', 
            label='Individual '+str(i+1))
    plt.ylim([-1.05,1.05])
    plt.xlim(xlims)

    plt.xlabel('time')
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([])
    plt.ylabel(r'Behavioral intention $x_i$')

    return(fig)



TwoFigsFromSim(alpha1 = 0.1, alpha2 =0.6,
                threshold_x=threshold_x,immediacy= 0.046, 
                plttitle="Test figure",
                negative=True)
plt.show()
