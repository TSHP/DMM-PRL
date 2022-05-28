include("./src/sim.jl")
include("./src/eval.jl")

# init model parameters
M1 = Dict([("name", "patient"), ("mm", 0), ("pm", 0.01), ("mp", 1000), ("pp", 10), ("alpha", 1), ("m", 1.5)])
M2 = Dict([("name", "control"), ("mm", 0), ("pm", 0.01), ("mp", 1000), ("pp", 10), ("alpha", 1), ("m", 1.5)])

model_params = [M1, M2]

# run experiment
# Simulations.run_prl_sim(model_params, 10)

# run evaluation
Eval.evaluate_prl()