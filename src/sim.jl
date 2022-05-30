module Simulations
    include("prl_sim.jl")
    using DataFrames
    using JLD2
    # run simulation of the prl experiment specified in "method"
    function run_prl_sim(models, n_iter, method, output=true)
        for model in models
            model_params_df = DataFrame()

            draws_dict = Dict()
            probs_dict = Dict()
            std_devs_dict = Dict()
            learning_results = Dict()

            filename = "prl_urn_probs_" * model["name"]

            # run n_iter iterations of experiment with model
            for it in range(1,n_iter)
                params = PRL.ModelParams(model["mm"], model["pm"], model["mp"], model["pp"], model["alpha"], model["m"])
                # save model parameters
                if it == 1
                    model_params_df[!, :it] = [it]
                    for field in fieldnames(typeof(params))
                        model_params_df[!, string(field)] = [getfield(params, field)]
                    end
                else
                    values = [getfield(params, field) for field in fieldnames(typeof(params))]
                    values = [it;values]
                    push!(model_params_df, values)
                end

                # run simulation
                results = PRL.run_experiment(params, filename, it, method, output)

                draws_dict[string(it)] = getfield(results, :draws)[3:length(getfield(results, :draws))]
                probs_dict[string(it)] = getfield(results, :probabilities)
                std_devs_dict[string(it)] = getfield(results, :std_deviations)

                learning_results["phases_learned_" * string(it)] = getfield(results, :phases_learned)
                learning_results["iterations_needed_" * string(it)] = getfield(results, :iterations_needed)                
            end

            # save results
            print("Saving results...")
            save("./io/results/" * filename * "_model_params.jld2", "data", model_params_df)
            save("./io/results/" * filename * "_draws.jld2", "data", draws_dict)
            save("./io/results/" * filename * "_probs.jld2", "data", probs_dict)
            save("./io/results/" * filename * "_std_devs.jld2", "data", std_devs_dict)
            save("./io/results/" * filename * "_learning_results.jld2", "data", learning_results)
        end
    end
end