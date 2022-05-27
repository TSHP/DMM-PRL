include("./src/prl_experiment.jl")

params = PRL.ModelParams(0, 0.01, 1000, 10, 1, 1.5) # mm, pm, mp, pp, alpha, m

results = PRL.run_experiment(params, "urn_probs_delusional")
