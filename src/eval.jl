module Eval
    using CSV, DataFrames
    using StatsPlots

    struct EvalParams
        correct_decisions::UInt16
        valid_lose_shift::UInt16
        valid_lose_stay::UInt16
        valid_win_shift::UInt16
        invalid_lose_shift::UInt16
        invalid_win_shift::UInt16
    end

    function make_bar_plot()
        results_folder = "./io/results/"

        data_files = [file for file in readdir(results_folder) if occursin("eval", file)]

        results_df = DataFrame()
        for file in data_files
            tmp = DataFrame(CSV.File(results_folder*file))
            results_df = vcat(results_df, tmp)
        end

        categories = names(results_df)[2:7]
        model_names = results_df[!,"model_name"]
        model_names = unique!(deepcopy(model_names))

        mean = []
        median = []

        for name in model_names
            tmp = results_df[findall(in([name]), results_df.model_name), :]
            push!(mean, describe(tmp)[!, "mean"][2:7])
            push!(median, describe(tmp)[!, "median"][2:7])
        end

        if length(model_names) == 2
            ticklabel = string.(categories)
            label1 = model_names[1]
            label2 = model_names[2]
            p_mean = groupedbar([mean[2] mean[1]], bar_position = :dodge, bar_width=0.7, xticks=(1:6, ticklabel), xrotation=20, labels = [label2 label1])
            title!("Mean")
            savefig(p_mean, "./io/plots/prl_urn_probs_eval_mean.png")
            p_median = groupedbar([median[2] median[1]], bar_position = :dodge, bar_width=0.7, xticks=(1:6, ticklabel), xrotation=20, labels = [label2 label1])
            title!("Median")
            savefig(p_median, "./io/plots/prl_urn_probs_eval_median.png")
        else
            throw("Bar plot comparing more than 2 models not implemented")
        end

    end

    function evaluate_prl(experiment="0", decision_bnd=[0.5,0.5])

        results_folder = "./io/results"

        data_files = [file for file in readdir(results_folder) if startswith(file, "prl")]

        model_names = unique!([split(file, '_')[4] for file in data_files])

        for model_name in model_names
            filename = "prl_urn_probs_"*model_name
            draws_df = DataFrame(CSV.File("./io/results/"*filename*"_draws.csv"))
            probs_df = DataFrame(CSV.File("./io/results/"*filename*"_probs.csv"))
            eval_prl_df = DataFrame()

            for it in names(draws_df)
                draws = draws_df[!, it]
                probs = probs_df[!, it]

                valid_lose_shift = 0
                valid_lose_stay = 0
                valid_win_shift = 0
                invalid_lose_shift = 0
                invalid_win_shift = 0
                decisions = deepcopy(probs)
                decisions[decisions.<decision_bnd[1]] .= 0
                decisions[decisions.>=decision_bnd[2]] .= 1

                if experiment=="0"
                    tmp = hcat(ones((1,10)), zeros((1,10)), ones((1,10)))
                    correct = reshape(tmp, size(decisions))
                elseif experiment == "test"
                    tmp = hcat(ones((1,1)), zeros((1,1)), ones((1,1)))
                    correct = reshape(tmp, size(decisions))
                end

                @assert(length(correct)==length(decisions))

                for (idx, d) in enumerate(decisions)
                    if idx == length(decisions) - 1
                        break
                    elseif correct[idx] == draws[idx] && correct[idx] != d && decisions[idx+1] != d # draw was the correct color, model made incorrect guess and changed in next step
                        valid_lose_shift += 1
                    elseif correct[idx] == draws[idx] && correct[idx] != d && decisions[idx+1] == d # draw was the correct color, model made incorrect guess and did not change in next step
                        valid_lose_stay += 1
                    elseif correct[idx] == draws[idx] && correct[idx] == d && decisions[idx+1] != d # draw was the correct color, model made correct guess and changed in next step
                        valid_win_shift += 1
                    elseif correct[idx] != draws[idx] && correct[idx] != d && decisions[idx+1] != d # draw was not the correct color, model made incorrect guess and changed in next step
                        invalid_lose_shift += 1
                    elseif correct[idx] != draws[idx] && correct[idx] == d && decisions[idx+1] != d # draw was not the correct color, model made correct guess and changed in next step
                        invalid_win_shift += 1
                    end

                end

                correct_decisions = sum(decisions.==correct)

                eval_params = EvalParams(correct_decisions, valid_lose_shift, valid_lose_stay, valid_win_shift, invalid_lose_shift, invalid_win_shift)

                if it == string(1)
                    eval_prl_df[!, it] = [it]
                    for field in fieldnames(typeof(eval_params))
                        eval_prl_df[!, string(field)] = [getfield(eval_params, field)]
                    end
                    eval_prl_df[!, :model_name] = [model_name]
                else
                    values = [getfield(eval_params, field) for field in fieldnames(typeof(eval_params))]
                    values = [it;values;model_name]
                    push!(eval_prl_df, values)
                    
                end

                CSV.write("./io/results/"*filename*"_eval.csv", eval_prl_df)
            end
        end
        make_bar_plot()
    end
end