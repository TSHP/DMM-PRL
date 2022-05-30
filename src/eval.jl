module Eval
    using DataFrames
    using StatsPlots, StatsBase, Statistics
    using JLD2, FileIO
    include("./utils.jl")

    struct EvalParams
        correct_decisions::UInt16
        valid_lose_shift::UInt16
        valid_lose_stay::UInt16
        valid_win_shift::UInt16
        invalid_lose_shift::UInt16
        invalid_win_shift::UInt16
    end

    function make_bar_plots(method)
        results_folder = "./io/results/"

        data_files = [file for file in readdir(results_folder) if occursin("eval", file)]
        data_files_cools = [file for file in readdir(results_folder) if occursin("learning_results", file)]

        # read results
        results_df = DataFrame()
        for file in data_files
            tmp = DataFrame(load(results_folder * file)["data"])
            results_df = vcat(results_df, tmp)
        end

        categories = names(results_df)[2:7]
        model_names = results_df[!, "model_name"]
        model_names = unique!(deepcopy(model_names))

        # read and restructure number of phases reached and number of iterations needed
        results_learning = Dict()
        for name in model_names
            for file in data_files_cools
                if occursin(name, file)
                    tmp = Dict(load(results_folder * file)["data"])
                    ph_tmp = []
                    it_tmp = []
                    for it in range(1, Int(length(keys(tmp)) / 2))
                        ph_learned = tmp["phases_learned_" * string(it)]
                        push!(ph_tmp, length(ph_learned))
                        it_needed = tmp["iterations_needed_" * string(it)]
                        push!(it_tmp, it_needed)
                    end
                    phases_reached_tmp = [count(>=(element),ph_tmp) for element in range(1,3) if element > 0] 
                    phases_reached = [sum(phases_reached_tmp[idx:length(phases_reached_tmp)]) for (idx, element) in enumerate(phases_reached_tmp)] # how often each phase was reached
                    
                    # restructure number of iterations needed
                    iterations_tmp = []
                    for i in range(1, maximum(ph_tmp))
                        vals = []
                        for element in it_tmp
                            if length(element) >= i
                                append!(vals, element[i])
                            end
                        end
                        push!(iterations_tmp, vals)
                    end

                    mean_its = [StatsBase.mean(element) for element in iterations_tmp]
                    median_its = [StatsBase.median(element) for element in iterations_tmp]

                    results_learning[name * "_reached_phase"] = phases_reached
                    results_learning[name * "_mean_iterations"] = mean_its
                    results_learning[name * "_median_iterations"] = median_its
                end
            end
        end

        mean = []
        median = []

        # mean and median values for transition counts observed
        for name in model_names
            tmp = results_df[findall(in([name]), results_df.model_name), :]
            push!(mean, describe(tmp)[!, "mean"][2:7])
            push!(median, describe(tmp)[!, "median"][2:7])
        end

        # make bar plots
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
            
            if method == "cools"
                ticklabel = range(1, length(results_learning[label2*"_reached_phase"]))
                p_learned_phase = groupedbar([results_learning[label2*"_reached_phase"] results_learning[label1*"_reached_phase"]], bar_position = :dodge, bar_width=0.7, xticks=(1:6, ticklabel), labels = [label2 label1])
                title!("Correctly learned phase")
                savefig(p_learned_phase, "./io/plots/correctly_learned_phase.png")
                p_mean_tries = groupedbar([results_learning[label2*"_mean_iterations"] results_learning[label1*"_mean_iterations"]], bar_position = :dodge, bar_width=0.7, xticks=(1:6, ticklabel), labels = [label2 label1])
                title!("Mean tries until phase learned")
                savefig(p_mean_tries, "./io/plots/mean_tries.png")
                p_median_tries = groupedbar([results_learning[label2*"_median_iterations"] results_learning[label1*"_median_iterations"]], bar_position = :dodge, bar_width=0.7, xticks=(1:6, ticklabel), labels = [label2 label1])
                title!("Median tries until phase learned")
                savefig(p_median_tries, "./io/plots/median_tries.png")
            end
        else
            throw("Bar plot comparing more than 2 models not implemented")
        end

    end

    function evaluate_prl(method, decision_bnd=[0.5,0.5])

        results_folder = "./io/results"

        data_files = [file for file in readdir(results_folder) if startswith(file, "prl")]

        model_names = unique!([split(file, '_')[4] for file in data_files]) # not so nice

        for model_name in model_names
            filename = "prl_urn_probs_" * model_name

            learning_results = Dict(load("./io/results/" * filename * "_learning_results.jld2")["data"])
            draws_dict = Dict(load("./io/results/" * filename * "_draws.jld2")["data"])
            probs_dict = Dict(load("./io/results/" * filename * "_probs.jld2")["data"])
            
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
                        l = length(probs)/(l_p1+l_p2+l_p3)
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

                #eval_params = EvalParams(correct_decisions, valid_lose_shift, valid_lose_stay, valid_win_shift, invalid_lose_shift, invalid_win_shift)
                eval_params = [correct_decisions, valid_lose_shift, valid_lose_stay, valid_win_shift, invalid_lose_shift, invalid_win_shift]

                print(eval_prl_df)
                print([it; eval_params; model_name])
                push!(eval_prl_df, [it; eval_params; model_name]) 
                #if it == 1
                #    eval_prl_df[!, :it] = [it]
                #    for field in fieldnames(typeof(eval_params))
                #        eval_prl_df[!, string(field)] = [getfield(eval_params, field)]
                #    end
                #    eval_prl_df[!, :model_name] = [model_name]
                #else
                #    values = [getfield(eval_params, field) for field in fieldnames(typeof(eval_params))]
                #print(eval_prl_df)
                #    values = [it; values; model_name]
                #    print(values)
                #    push!(eval_prl_df, values)
                    
                #end

                save("./io/results/" * filename * "_eval.jld2", "data", eval_prl_df)
            end
        end

        make_bar_plots(method)
    end
end