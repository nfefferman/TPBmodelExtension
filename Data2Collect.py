import pandas as pd
import numpy as np
import os.path

jobrange = range(0,40)

BroadDf = pd.read_csv("./DataFiles/Periodicity2Result0.csv")

for i in jobrange[1:]:
    path = "./DataFiles/Periodicity2Result{}.csv".format(i)
    if os.path.isfile(path):
        BroadDf = pd.concat((BroadDf, pd.read_csv(path)))
BroadDf.to_csv("./DataFiles/Periodicity2Complete.csv", index = False)