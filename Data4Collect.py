import pandas as pd
import numpy as np
import os.path

jobrange = range(0,79)

BroadDf = pd.read_csv("./DataFiles/Periodicity4Result0.csv")

for i in jobrange[1:]:
    path = "./DataFiles/Periodicity4Result{}.csv".format(i)
    if os.path.isfile(path):
        BroadDf = pd.concat((BroadDf, pd.read_csv(path)))
        BroadDf=BroadDf[BroadDf['threshold_x'].between(0.4,0.5)]
        BroadDf=BroadDf[BroadDf["shiftS"]==-0.55]
BroadDf.to_csv("./DataFiles/Periodicity4Complete.csv", index = False)