module Eval
    using CSV, DataFrames

    struct EvalParams
        correct_decisions
        valid_lose_shift
        valid_lose_stay
        valid_win_shift
        invalid_lose_shift
        invalid_win_shift
    end

    function evaluate_prl(experiment="0", decision_bnd=[0.5,0.5])

        results_folder = "./io/results"

        data_files = [file for file in readdir(results_folder) if startswith(file, "prl")]

        model_names = unique!([split(file, '_')[4] for file in data_files])

        for model_name in model_names:
            filename = "prl_urn_probs_"
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
                else
                    values = [getfield(eval_params, field) for field in fieldnames(typeof(eval_params))]
                    values = [it;values]
                    push!(eval_prl_df, values)
                end

                CSV.write("./io/results/"*filename*"_eval.csv", eval_prl_df)
            end
        end
    end
end