### ElusivityDCOPs ###

Repository containing the code and results of the work "On the Elusivity of Dynamic Combinatorial Optimisation Problems".

The repository divides the information into three folders:

# Input

This folder contains the required information to construct a dynamic problem based on a static problem. The initial problems (static TSP and KP instances) can be found in the subfolder called "Static_instances". The code makes the stationary problem instancess into dynamic by the files within "Dynamism".

The dynamism files are named as follow:
	(example) "BPn100sH0.1_1.txt".
		- [BP|PP] represents if the dynamism file is based for permutation or binary problems.
		- n<number> represents the unaltered size of the changing problem. 
		- s[H|C|U|K]<number> represents the rotation degree (severity) of the changes. The letters "H", "C", "K" and "U" represent the permutation distance metric used to step the difference between the permutation used to rotate the landscape.
		- \_<number> represents the run index.


# Code
This folder contains the PYTHON code to construct the problems and algorithms. The folder is split into three subfolders: "Algorithms", "Problems" and "Individual".

To run the experiments, the user must locate and execute the wanted algorithm file and pass the imput parameters. See below an example of how to run the code.
```bash
		# Here is an example
		python RIGA.py instance Input/Static_instances/KP/joanA100.kp dynamic Input/Dynamisms/BPn100sH0.1_1.txt result GA-joanA100-BPn100sH0.1f10_1.csv stop 1000 pop 100 algorithm ri freq 100
		#
		# This particular example denotes:
		# Solve a DKP (constructed by the static kanpsack problem instance "joanA100", changing every 100 generations and with severity of Hamming distance of 0.1) by the adaptive (random immigrants based) usage of the GA.
```
	Note that the algorithm versions used in the paper are adaptive and restarting usage of the algorithms. The adaptive version of the algorithms is represented as "ri" and the restarting version as "restart".

	`python RIGA.py instance <Static instance> dynamic <Dynamism file> result <Output name> stop <Maximum generations> pop <Population size> algorithm <Algorithm version> freq <Period>`.
	

# Results
This folder contains ALL the results of the work "On the Elusivity of Dynamic Combinatorial Optimisation Problems". On the one hand, the folder contains the expected performance of the algorithms under a given set of periods and magnitudes of changes on the subfolder called "Mean performance".

On the other hand, the plots are displayed in the subfolder called "Plots", where the user can find four types of plots: the median and iterquartile range plots, the elusivity heatmaps, the simplex plots and the statistical (Bayesian) heatmaps. The names of the files represent the initial static instance name, the algorithm used and the performance measure.

The following specifications are showed to facilitate the understanding of the results.
	- `Median and interquartile/<Instance>-f<Period>-m[H|C|U|K]<Rotation degree>-a[ACO|GA|PBIL].pdf`
	- `Heatmaps/<Instance>-ri[ACO|GA|PBIL]-[Fbog|Eacc].pdf`
	- `Bayesian analysis/Simplex plot/<Bayesian range value>/Simplex-[ACO|GA|PBIL]-<Instance>-f<Period>-m[H|C|U|K]<Rotation degree>-[Fbog|Eacc].pdf`
	- `Bayesian analysis/Bayesian Heatmap/<Bayesian range value>/<Instance>-[H|C|U|K]-ri[ACO|GA|PBIL]-Bayes_<Bayesian range value>_[Fbog|Eacc].pdf`