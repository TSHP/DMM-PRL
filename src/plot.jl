module DMM_Plots
    using DataFrames
    using StatsPlots, StatsBase
    using JLD2, FileIO
    include("constants.jl")

    my_cols = [(red = 0, green = 8, blue = 20), # dark blue
           (red = 0, green = 29, blue = 61), # slightly lighter dark blue
           (red = 0, green = 53, blue = 102), # medium dark blue
           (red = 255,green =  195, blue = 0), # yellow
           (red = 255,green =  214, blue = 10)] # slightly lighter yellow
    my_cols = [RGB([ai / 255 for ai in a]...) for a in my_cols]
    my_cols = [my_cols; Colors.distinguishable_colors(6)]

    function generate_plots(method)
        make_bar_plots(method)
        make_probability_plots()
    end

    function make_probability_plots()
        prob_control_file = [file for file in readdir(results_folder) if occursin("probs", file) && occursin("control", file)][1]
        draw_control_file = [file for file in readdir(results_folder) if occursin("draws", file) && occursin("control", file)][1]

        probs_control = load(results_folder * prob_control_file)["data"]
        draws_control = load(results_folder * draw_control_file)["data"]

        for key in keys(probs_control)
            prob_plot = plot(probs_control[key], xlabel = "Number of drawn beads", ylabel = "Estimated probability", linewidth = 2, xlabelfontsize = 10, ylabelfontsize = 10, color=my_cols[3])
            draws_plot = scatter(draws_control[key], xlabel = "Number of drawn beads", ylabel = "Bead drawn", linewidth = 2, xlabelfontsize = 10, ylabelfontsize = 10,  color=my_cols[3])
            yticks!([0, 1])
            plot(prob_plot, draws_plot, layout = (2, 1), plot_title = "Estimated probability of beads coming from urn 1 (control)", plot_titlefontsize=12, color=my_cols[3], fontfamily="serif-roman", legend=false)
            png(plots_folder * "probs_control_" * key * ".png")
        end

        prob_patient_file = [file for file in readdir(results_folder) if occursin("probs", file) && occursin("patient", file)][1]
        draw_patient_file = [file for file in readdir(results_folder) if occursin("draws", file) && occursin("patient", file)][1]

        probs_patient = load(results_folder * prob_patient_file)["data"]
        draws_patient = load(results_folder * draw_patient_file)["data"]

        for key in keys(probs_patient)
            prob_plot = plot(probs_patient[key], xlabel="Number of drawn beads", ylabel="Estimated probability", 
                            linewidth = 2, xlabelfontsize = 10, ylabelfontsize = 10, color=my_cols[3])
            draws_plot = scatter(draws_patient[key], xlabel="Number of drawn beads", ylabel="Bead drawn", 
                            linewidth = 2, xlabelfontsize = 10, ylabelfontsize = 10, color=my_cols[3])
            yticks!([0, 1])
            plot(prob_plot, draws_plot, layout = (2, 1), plot_title = "Estimated probability of beads coming from urn 1 (patient)", 
                plot_titlefontsize=12, color=my_cols[3], fontfamily="serif-roman", legend=false)
            png(plots_folder * "probs_patient_" * key * ".png")
        end
        
    end

    function make_bar_plots(method)
        data_files = [file for file in readdir(results_folder) if occursin("eval", file)]
        data_files_cools = [file for file in readdir(results_folder) if occursin("learning_results", file)]

        # Read results
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
                    phases_reached = [sum(phases_reached_tmp[idx:length(phases_reached_tmp)]) for (idx, _) in enumerate(phases_reached_tmp)]# how often each phase was reached
                    
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

            ticklabel[1] = "correct decisions/ \n #trials"

            for (it,label) in enumerate(ticklabel[2:length(ticklabel)]) # make nicer label
                ticklabel[it+1] = replace(label, "_" => " ")
            end

            p_mean = groupedbar([mean[2] mean[1]], bar_position = :dodge, bar_width = 0.7, xticks = (1:6, ticklabel), xrotation = 20, 
                                labels = [label2 label1], color=[my_cols[2] my_cols[4]], title = "Mean", fontfamily="serif-roman", size=(720, 440))
            savefig(p_mean, plots_folder * "prl_urn_probs_eval_mean.png")

            p_median = groupedbar([median[2] median[1]], bar_position = :dodge, bar_width = 0.7, xticks = (1:6, ticklabel), xrotation = 20, 
                                labels = [label2 label1], color=[my_cols[2] my_cols[4]], title="Median", fontfamily="serif-roman", size=(720, 440))
            savefig(p_median, plots_folder * "prl_urn_probs_eval_median.png")
            
            if method == "cools"
                ticklabel = range(1, length(results_learning[label2 * "_reached_phase"]))
                p_learned_phase = groupedbar([results_learning[label2 * "_reached_phase"] results_learning[label1 * "_reached_phase"]], 
                                                bar_position = :dodge, bar_width = 0.7, xticks = (1:6, ticklabel), labels = [label2 label1], 
                                                color=[my_cols[2] my_cols[4]], title="Correctly learned phase", fontfamily="serif-roman")
                savefig(p_learned_phase, plots_folder * "correctly_learned_phase.png")

                p_mean_tries = groupedbar([results_learning[label2 * "_mean_iterations"] results_learning[label1 * "_mean_iterations"]], 
                                            bar_position = :dodge, bar_width = 0.7, xticks = (1:6, ticklabel), labels = [label2 label1], 
                                            color=[my_cols[2] my_cols[4]], title="Mean tries until phase learned", fontfamily="serif-roman")
                title!("Mean tries until phase learned")
                savefig(p_mean_tries, plots_folder * "mean_tries.png")

                p_median_tries = groupedbar([results_learning[label2 * "_median_iterations"] results_learning[label1 * "_median_iterations"]], 
                                            bar_position = :dodge, bar_width = 0.7, xticks = (1:6, ticklabel), labels = [label2 label1], 
                                            color=[my_cols[2] my_cols[4]], title="Median tries until phase learned", fontfamily="serif-roman")
                title!("Median tries until phase learned")
                savefig(p_median_tries, plots_folder * "median_tries.png")
            end
        else
            throw("Bar plot comparing more than 2 models not implemented")
        end

    end
end