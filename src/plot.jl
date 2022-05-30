module DMM_Plots
    using DataFrames
    using StatsPlots, StatsBase
    using JLD2, FileIO
    include("constants.jl")

    function generate_plots(method)
        make_bar_plots(method)
    end

    function make_bar_plots(method)
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
            p_mean = groupedbar([mean[2] mean[1]], bar_position = :dodge, bar_width = 0.7, xticks = (1:6, ticklabel), xrotation = 20, labels = [label2 label1])
            title!("Mean")
            savefig(p_mean, plots_folder * "prl_urn_probs_eval_mean.png")
            p_median = groupedbar([median[2] median[1]], bar_position = :dodge, bar_width = 0.7, xticks = (1:6, ticklabel), xrotation = 20, labels = [label2 label1])
            title!("Median")
            savefig(p_median, plots_folder * "prl_urn_probs_eval_median.png")
            
            if method == "cools"
                ticklabel = range(1, length(results_learning[label2 * "_reached_phase"]))
                p_learned_phase = groupedbar([results_learning[label2 * "_reached_phase"] results_learning[label1 * "_reached_phase"]], 
                                                bar_position = :dodge, bar_width = 0.7, xticks = (1:6, ticklabel), labels = [label2 label1])
                title!("Correctly learned phase")
                savefig(p_learned_phase, plots_folder * "correctly_learned_phase.png")
                p_mean_tries = groupedbar([results_learning[label2 * "_mean_iterations"] results_learning[label1 * "_mean_iterations"]], 
                                            bar_position = :dodge, bar_width = 0.7, xticks = (1:6, ticklabel), labels = [label2 label1])
                title!("Mean tries until phase learned")
                savefig(p_mean_tries, plots_folder * "mean_tries.png")
                p_median_tries = groupedbar([results_learning[label2 * "_median_iterations"] results_learning[label1 * "_median_iterations"]], 
                                            bar_position = :dodge, bar_width = 0.7, xticks = (1:6, ticklabel), labels = [label2 label1])
                title!("Median tries until phase learned")
                savefig(p_median_tries, plots_folder * "median_tries.png")
            end
        else
            throw("Bar plot comparing more than 2 models not implemented")
        end

    end
end