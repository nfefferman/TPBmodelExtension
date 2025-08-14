import random, time, pickle
from datetime import datetime
import numpy as np
import networkx as nx
import pandas as pd
from model import *
from Periodicity import *
import itertools
import math
import matplotlib.pyplot as plt
import argparse



##Read in arguments

#Global variables
job_total = 80
n = 4
complete_graph = True
tmax = 3000
time_step = 0.01
alphan = 20 #must be at least job_total
#thn = 3
#sSn = 3
rn = 25

total = alphan**n*rn

print("total simulations {}".format(total))
print("simulations this run {}".format(int(total/job_total)))


alpha1s = np.linspace(0,1,alphan)
alpha2s = np.linspace(0,1,alphan)
alpha3s = np.linspace(0,1,alphan)
alpha4s = np.linspace(0,1,alphan)
threshold_x_values = 0.45
shiftS_values = -0.55
r_values = np.linspace(0.1,0.9,rn)
# Add more as needed

parser = argparse.ArgumentParser()

parser.add_argument("Run",
                    help = "Job number",
                    type= int)

args= parser.parse_args()


def main(parameter_combinations, n, complete_graph):
    t0=time.time()
    data = []
    for param in parameter_combinations:
        alphas = np.array(param[0])
        threshold_x=param[1]
        shiftS = param[2]
        r=param[3]

        # Create adjacency matrix
        if complete_graph:
            A = np.ones((n, n)) - np.eye(n)
        else:
            A=np.zeros([n,n])
            for i in range(0,n):
                for j in range(i+1,n):
                    A[i,j]=np.random.choice([0,1])
                    A[j,i]=A[i,j]

        ret =periodic_sim(alphas,threshold_x,shiftS,r,A) 
        data.append([alphas,threshold_x,shiftS,r,ret[0],ret[1],ret[2],ret[3],ret[4]])
    t1=time.time()
    print(t1-t0)
    print('Each simulation took {}s'.format((t1-t0)/int(total/job_total)))
    return data

if __name__ == "__main__":
    run = args.Run
    alphas = list(itertools.product(alpha1s,alpha2s, alpha3s,alpha4s))

    i = int(run/job_total*len(alphas))
    j = int((run+1)/job_total*len(alphas))
    parameter_combinations = list(itertools.product(alphas[i:j], 
                                                    [threshold_x_values],
                                                    [shiftS_values],
                                                    r_values))
    #print(parameter_combinations)
    data = main(parameter_combinations=parameter_combinations,
                n=n,
                complete_graph=complete_graph )
    
    df = pd.DataFrame(data, columns=['alphas',
                                     'threshold_x',
                                     'shiftS',
                                     'r',
                                     'intervals',
                                     'intmean',
                                     'intstd',
                                     'periodic',
                                     'limit'])
    csvname = 'DataFiles/Periodicity4Result{}.csv'.format(args.Run)
    df.to_csv(csvname,index=False)
