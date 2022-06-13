# Dirichlet Mixture Model for Probabilistic Reversal Learning
## Description
An application of the model from ["A generative framework for the study of delusions"](https://www.sciencedirect.com/science/article/pii/S0920996420306277) developed by Tore Erdmann and Christoph Mathys. The model is used for simulating probabilistic reversal learning tasks performed by delusional and non-delusional agents. The implemented probabilistic reversal learning task is based on ["Defining the neural mechanisms of probabilistic reversal learning using event-related functional magnetic resonance imaging"](https://pubmed.ncbi.nlm.nih.gov/12040063/).

## Requirements
The code was developed and tested using [Julia](https://julialang.org/) Version 1.7.0-rc1 and requires the following packages:
* DataFrames
* Distributions
* FileIO
* JLD2
* Plots
* Random
* StatsBase
* StatsFuns
* StatsPlots

For convenience, we include a Julia environment with the above packages in the repository. \
Additionally, we use Make to generate the directories for the output of the code, however, it is not required as the directories can be manualy created.

## Setting up the project
To setup the required directories run the following command in the commandline
```
make init
```
The directories can also be setup manually in the root directory of the repository. The following directories should be created:
* io
* io/plots
* io/results

Afterwards, start the Julia REPL and run the following command to go into pkg mode
```
]
```
followed by
```
activate .
```
to activate the Julia envrionment.

## Usage
To generate simulation results run the following command in Julia REPL:
```
include("main.jl")
```
To run simulations with different parameters, change the parameters in the main.jl file. The following parameters can be changed:
* `method`: Which experiment to run. Can be either "3p_10t" for 3 phases with 10 trials or "cools" for 3 phases with at most 50 trials incremented in steps of 10
* `mu_tau_c`: Expected precision of the control agent
* `mu_tau_p`: Expected precision of the patient agent
* `n_runs`: How many runs to simulate per agent
* `n_history`: Models the memory of an agent. The agent will remember the last `n_history` trials.
* `belief_strength`: How many samples to use for reinforcing the current hypothesis.

## Support
For questions concerning the project, please contact us at: \
Wiona Glänzer, wglaenzer@ethz.ch \
Katarzyna Kransapolska, krasnopk@ethz.ch \
Timofey Shpakov, tshpakov@ethz.ch

## Authors and acknowledgment
We want to thank Tore Erdmann, Christoph Mathys and Alexander Hess for their support and guidance during this projects.
