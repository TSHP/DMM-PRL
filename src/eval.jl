module Evaluation
    using DataFrames
    using JLD2, FileIO
    include("constants.jl")

    function evaluate_prl(method, decision_bnd = [0.5, 0.5])
        data_files = [file for file in readdir(results_folder) if startswith(file, "prl")]

        model_names = unique!([split(file, '_')[2] for file in data_files]) # TODO: Not so nice

        for model_name in model_names
            filename = "prl_" * model_name

            learning_results = Dict(load(results_folder * filename * "_learning_results.jld2")["data"])
            draws_dict = Dict(load(results_folder * filename * "_draws.jld2")["data"])
            probs_dict = Dict(load(results_folder * filename * "_probs.jld2")["data"])
            
            eval_prl_df = DataFrame(it = Any[], correct_decisions = Any[], valid_lose_shift = Any[], valid_lose_stay = Any[], 
                                    valid_win_shift = Any[], invalid_lose_shift = Any[], invalid_win_shift = Any[], model_name = Any[])

            for it in sort([key for key in keys(draws_dict)])
                draws = draws_dict[it]
                probs = probs_dict[it]

                valid_lose_shift = 0
                valid_lose_stay = 0
                valid_win_shift = 0
                invalid_lose_shift = 0
                invalid_win_shift = 0
                decisions = deepcopy(probs)
                decisions[decisions .< decision_bnd[1]] .= 0
                decisions[decisions .>= decision_bnd[2]] .= 1

                # get "correct" state sequence for each method
                if method == "3p_10t"
                    tmp = hcat(ones((1, 10)), zeros((1, 10)), ones((1, 10)))
                    correct = reshape(tmp, size(decisions))
                elseif method == "cools"
                    ph_learned = learning_results["phases_learned_" * string(it)]
                    it_needed = learning_results["iterations_needed_" * string(it)]
                    l_p1 = length(ph_learned) > 0 ? it_needed[1] : 0
                    l_p2 = length(ph_learned) > 1 ? it_needed[2] : 0
                    l_p3 = length(ph_learned) > 2 ? it_needed[3] : 0
                    
                    if (l_p1 + l_p2 + l_p3) != 0
                        l = length(probs) / (l_p1 + l_p2 + l_p3)
                        tmp = hcat(ones((1, Int(l_p1 * l))), zeros((1, Int(l_p2 * l))), ones((1, Int(l_p3 * l))))
                        correct = reshape(tmp, size(decisions))
                    else
                        correct = repeat([1], length(decisions))
                    end
                else
                    throw("Method " * string(method) * " not implemented")
                end

                @assert(length(correct) == length(decisions))
                @assert(length(correct) == length(draws))

                for (idx, d) in enumerate(decisions)
                    if idx == length(decisions) - 1
                        break
                    elseif correct[idx] == draws[idx] && correct[idx] != d && decisions[idx+1] != d # draw was the correct color, model made incorrect guess and changed in next step
                        valid_lose_shift += 1
                    elseif correct[idx] == draws[idx] && correct[idx] != d && decisions[idx+1] == d # draw was the correct color, model made incorrect guess and did not change in next step
                        valid_lose_stay += 1
                    elseif correct[idx] == draws[idx] && correct[idx] == d && decisions[idx+1] != d # draw was the correct color, model made correct guess and changed in next step
                        valid_win_shift += 1
                    elseif correct[idx] != draws[idx] && correct[idx] != d && decisions[idx + 1] != d # draw was not the correct color, model made incorrect guess and changed in next step
                        invalid_lose_shift += 1
                    elseif correct[idx] != draws[idx] && correct[idx] == d && decisions[idx + 1] != d # draw was not the correct color, model made correct guess and changed in next step
                        invalid_win_shift += 1
                    end

                end

                correct_decisions = sum(decisions .== correct)

                push!(eval_prl_df, [it, correct_decisions, valid_lose_shift, valid_lose_stay, valid_win_shift, invalid_lose_shift, invalid_win_shift, model_name]) 

                save(results_folder * filename * "_eval.jld2", "data", eval_prl_df)
            end
        end
    end
end