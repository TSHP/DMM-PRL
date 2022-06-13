module Simulations
    using DataFrames
    using JLD2
    include("prl_sim.jl")
    include("constants.jl")

    # Run simulation of the probabilistic reversal learning experiment specified in "method"
    function run_prl_sim(models, n_history, belief_strength, n_iter, method, output = false)
        for model in models

            # Initialize empty dataframe and dictionaries to store results
            model_params_df = DataFrame()

            draws_dict = Dict()
            probs_dict = Dict()
            std_devs_dict = Dict()
            learning_results = Dict()
            no_clusters_dict = Dict()
            cluster_switches = Dict()

            # Run n_iter iterations of experiment with model
            for it in range(1, n_iter)
                params = PRL.ModelParams(model["mm"], model["pm"], model["mp"], model["pp"], model["alpha"], model["m"])
                # Save model parameters
                if it == 1
                    model_params_df[!, :it] = [it]
                    for field in fieldnames(typeof(params))
                        model_params_df[!, string(field)] = [getfield(params, field)]
                    end
                else
                    values = [getfield(params, field) for field in fieldnames(typeof(params))]
                    values = [it; values]
                    push!(model_params_df, values)
                end

                # Run simulation
                results = PRL.run_experiment(params, n_history, belief_strength, it, method, output)

                draws_dict[string(it)] = getfield(results, :draws)[3:length(getfield(results, :draws))]
                probs_dict[string(it)] = getfield(results, :probabilities)
                std_devs_dict[string(it)] = getfield(results, :std_deviations)
                no_clusters_dict[string(it)] = getfield(results, :no_clusters)
                cluster_switches[string(it)] = getfield(results, :cluster_switches)

                learning_results["phases_learned_" * string(it)] = getfield(results, :phases_learned)
                learning_results["iterations_needed_" * string(it)] = getfield(results, :iterations_needed)                
            end

            # Save results
            filename = "prl_" * model["name"]
            save(results_folder * filename * "_model_params.jld2", "data", model_params_df)
            save(results_folder * filename * "_draws.jld2", "data", draws_dict)
            save(results_folder * filename * "_probs.jld2", "data", probs_dict)
            save(results_folder * filename * "_std_devs.jld2", "data", std_devs_dict)
            save(results_folder * filename * "_no_clusters.jld2", "data", no_clusters_dict)
            save(results_folder * filename * "_cluster_switches.jld2", "data", cluster_switches)
            save(results_folder * filename * "_learning_results.jld2", "data", learning_results)
        end
    end
end