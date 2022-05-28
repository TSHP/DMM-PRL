module Simulations
    include("prl_sim.jl")
    using DataFrames, CSV
    
    function run_prl_sim(models, n_iter, output=true)

        for model in models

            model_params_df = DataFrame()
            draws_df = DataFrame()
            probs_df = DataFrame()
            std_devs_df = DataFrame()

            filename = "prl_urn_probs_"*model["name"]

            for it in range(1,n_iter)
                params = PRL.ModelParams(model["mm"], model["pm"], model["mp"], model["pp"], model["alpha"], model["m"])
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

                results = PRL.run_experiment(params, filename, it, output)
                
                draws_df[!, string(it)] = getfield(results, :draws)[3:length(getfield(results, :draws))] # save without inital 2 values
                probs_df[!, string(it)] = getfield(results, :probabilities)
                std_devs_df[!, string(it)] = getfield(results, :std_deviations)
                
                # header = [string(result) for result in fieldnames(typeof(results))]
                
            end

            # save results
            CSV.write("./io/results/"*filename*"_model_params.csv", model_params_df)
            CSV.write("./io/results/"*filename*"_draws.csv", draws_df)
            CSV.write("./io/results/"*filename*"_probs.csv", probs_df)
            CSV.write("./io/results/"*filename*"_std_devs.csv", std_devs_df)
        end
    end
end