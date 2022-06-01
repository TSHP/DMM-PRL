include("./src/sim.jl")
include("./src/eval.jl")
include("./src/plot.jl")

method = "cools" # Possible methods: 3p_10t (3 phases à 10 trials), cools (3 phases, at most 50 trials, incremented in steps of 10)

mu_tau_c = 1000
mu_tau_p = 1 / 1000

n_runs = 10
n_history = 5

# Init model parameters
M1 = Dict([("name", "patient"), ("mm", 0), ("pm", 0.01), ("mp", mu_tau_c), ("pp", 10), ("alpha", 1), ("m", 1.5)])
M2 = Dict([("name", "control"), ("mm", 0), ("pm", 0.01), ("mp", mu_tau_p), ("pp", 10), ("alpha", 1), ("m", 1.5)])

models = [M1, M2]

# Run simulation
Simulations.run_prl_sim(models, n_runs, method)

# Run evaluation
Evaluation.evaluate_prl(method)

# Plot results
DMM_Plots.generate_plots(method)