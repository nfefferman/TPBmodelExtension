# temporally-biased-planned-behavior
Code repository to carry out the analysis of the temporaly biased planned behavior model described in the associated manuscript. This repository includes the code for building and visualizing the model as well as code for the parallelized simulations used to partition the parameter space.

## Builiding the model
### Main script
model.py is the main tool for constructing a model system. Using the auxilary functions in the script to fill in system parameters, the script builds a function with the make_system() method which can be solved by the several integrate different integrating methods ( integrate_system(), integrate_negative_system(), integrate_system_and_output_events()) depending on the desired outputs.

### Model Analysis
analytics.py is a colleciton of methods that take the results of integrate_system() or a similar method and extract a certain metric like period, event times or initial action time. 
The notebooks figure-2.ipynb, figure-3.ipynb, figure-4.ipynb, figure-5.ipynb, and figure-6.ipynb are all examples of using and solving a model system with the methods from model.py and analyzing the output. Each notebook outputs the results of the model analysis as a .pdf figure. 

### Other tools
utils.py is a set of auxilary methods meant to improve performance of the model integration. genericIntegrateSystem.py is a script containing a higher order itegration scheme used rarely when the standard integrate_system() is insufficient. 

## High Performance Computing
To do analysis of the model over a large part of the parameter space, the analysis procedure was parallelized and broken up according to the size of the system being analyzed. For this analysis, the degree of periodicity was measured for each system at a different point in the parameter space. periodicity.py is the main tool used to detect periodicity. PeriodicityXscan.py is the script used to check the periodicity of the system with X player at many places in the parameter space. X=2,3,4. JobPeriodicityXscan.srun.sh is the shell script used to run the parallelized processes on a cluster computer for each X. DataXcollect.py takes the results of each parallel run and combines them into a single csv for each X and DataXCollect.srun.sh is the shell script that directs the cluster computer to run that process. 

Additionally, twoPlayer.py and threePlayer.py are files used to scan a particular two dimensional or three dimensional slice of the parameter space to check the classification which is described in the the manuscript. The output is a PDF image of a heat map (or several heat maps) partitioning the parameter space into parts with qualitativly different dynamics. 

