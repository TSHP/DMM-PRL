include("./src/sim.jl")
include("./src/eval.jl")

# init model parameters
M1 = Dict([("name", "patient"), ("mm", 0), ("pm", 0.01), ("mp", 1000), ("pp", 10), ("alpha", 1), ("m", 1.5)])
M2 = Dict([("name", "control"), ("mm", 0), ("pm", 0.01), ("mp", 1/1000), ("pp", 10), ("alpha", 1), ("m", 1.5)])

models = [M1, M2]
n_runs = 10
method = "cools" # possible methods: 3p_10t (3 phases à 10 trials), cools (3 phases, at most 50 trials, incremented in steps of 10)

# run simulation
Simulations.run_prl_sim(models, n_runs, method)

# run evaluation
Eval.evaluate_prl(method)

print("Made it to the end")