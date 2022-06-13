module DMM_Plots
    using DataFrames
    using StatsPlots, StatsBase
    using Plots.PlotMeasures
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
        make_probability_plots(method)
    end

    function make_probability_plots(method)
        prob_control_file = [file for file in readdir(results_folder) if occursin("probs", file) && occursin("control", file)][1]
        draw_control_file = [file for file in readdir(results_folder) if occursin("draws", file) && occursin("control", file)][1]
        clusters_control_file = [file for file in readdir(results_folder) if occursin("clusters", file) && occursin("control", file)][1]
        cluster_switches_control_file = [file for file in readdir(results_folder) if occursin("cluster_switches", file) && occursin("control", file)][1]

        probs_control = load(results_folder * prob_control_file)["data"]
        draws_control = load(results_folder * draw_control_file)["data"]
        clusters_control = load(results_folder * clusters_control_file)["data"]
        cluster_switches_control = load(results_folder * cluster_switches_control_file)["data"]

        yticks_prob = [0, 0.2, 0.4, 0.6, 0.8, 1]
        ylims_prob=[0, 1]
        yticks_cluster = range(2, 6)
        ylims_cluster = [2, 6]

        for key in keys(probs_control)
            prob_plot = plot(probs_control[key], 
                            ylabel = "Estimated \n probability",
                            color = my_cols[3],
                            yticks = yticks_prob,
                            ylims = ylims_prob)
            cluster_plot = plot(clusters_control[key], 
                            ylabel = "#Clusters",
                            color = my_cols[3],
                            # yticks = yticks_cluster,
                            # ylims = ylims_cluster
                            )
            cluster_switches_plot = scatter(cluster_switches_control[key], 
                            ylabel = "Assigned \n new cluster", 
                            yticks = [],
                            color = my_cols[4],
                            markerstrokecolor = my_cols[4],
                            markersize = 0)
            vline!(findall(x->x, cluster_switches_control[key]), color = my_cols[4])
            draws_plot = scatter(draws_control[key], 
                            xlabel = "Number of drawn beads",
                            ylabel = "Bead drawn", 
                            xlabelfontsize = 12, 
                            yticks = [0, 1],
                            color = my_cols[3])
            plot(prob_plot, cluster_switches_plot, cluster_plot, draws_plot, 
                            layout = (4, 1), 
                            plot_title = "PRL Task (control)", 
                            plot_titlefontsize = 16, 
                            ylabelfontsize = 12, 
                            linewidth = 2,
                            fontfamily = "serif-roman", 
                            legend = false, 
                            size = (700, 650))
            png(plots_folder * "probs_control_" * key * ".png")
        end

        prob_patient_file = [file for file in readdir(results_folder) if occursin("probs", file) && occursin("patient", file)][1]
        draw_patient_file = [file for file in readdir(results_folder) if occursin("draws", file) && occursin("patient", file)][1]
        clusters_patient_file = [file for file in readdir(results_folder) if occursin("clusters", file) && occursin("patient", file)][1]
        cluster_switches_patient_file = [file for file in readdir(results_folder) if occursin("cluster_switches", file) && occursin("patient", file)][1]

        probs_patient = load(results_folder * prob_patient_file)["data"]
        draws_patient = load(results_folder * draw_patient_file)["data"]
        cluster_patient = load(results_folder * clusters_patient_file)["data"]
        cluster_switches_patient = load(results_folder * cluster_switches_patient_file)["data"]

        for key in keys(probs_patient)
            prob_plot = plot(probs_patient[key], 
                            ylabel = "Estimated \n probability", 
                            color = my_cols[3],
                            yticks = yticks_prob,
                            ylims = ylims_prob)
            cluster_plot = plot(cluster_patient[key], 
                            ylabel = "#Clusters", 
                            color = my_cols[3],
                            # yticks = yticks_cluster,
                            # ylims = ylims_cluster
                            )
            cluster_switches_plot = scatter(cluster_switches_patient[key],
                            ylabel = "Assigned \n new cluster", 
                            yticks = [], 
                            color = my_cols[4],
                            markerstrokecolor = my_cols[4],
                            markersize = 0)
            vline!(findall(x->x, cluster_switches_patient[key]), color = my_cols[4])
            draws_plot = scatter(draws_patient[key],
                            xlabel = "Number of drawn beads",
                            ylabel = "Bead drawn", 
                            xlabelfontsize = 12, 
                            color = my_cols[3], 
                            yticks = [0, 1])
            plot(prob_plot, cluster_switches_plot, cluster_plot, draws_plot, 
                            layout = (4, 1), 
                            plot_title = "PRL Task (patient)", 
                            plot_titlefontsize = 16, 
                            linewidth = 2,
                            fontfamily = "serif-roman", 
                            legend = false, 
                            size = (700, 650))
            png(plots_folder * "probs_patient_" * key * ".png")
        end
        
        if method == "3p_10t"
            sum_switches_control = sum([cluster_switches_control[key] for key in keys(probs_control)])
            sum_switches_patient = sum([cluster_switches_patient[key] for key in keys(probs_patient)])
            draws_plot = scatter(draws_control["1"], 
                            xlabel = "Number of drawn beads",
                            ylabel = "Bead drawn", 
                            xlabelfontsize = 12, 
                            yticks = [0, 1],
                            color = my_cols[3],
                            legend = false,
                            bottom_margin = 5mm)
            sum_switches_plot = groupedbar([sum_switches_control sum_switches_patient],
                                            bar_position = :dodge, bar_width = 1, labels = ["control" "patient"], left_margin = 5mm, 
                                                color = [my_cols[2] my_cols[4]], fontfamily = "serif-roman", xlims = [1, 30], ylabel = "Sum of cluster \n switches")
            plot(sum_switches_plot, draws_plot, 
                layout = (2, 1), 
                plot_title = "Number of cluster switches at drawn bead", 
                plot_titlefontsize = 16, 
                ylabelfontsize = 12, 
                linewidth = 2,
                fontfamily = "serif-roman", 
                size = (800, 450))
            savefig(plots_folder * "sum_switches.png")

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
                    phases_reached = [count(>=(element), ph_tmp) for element in range(1, 3) if element > 0] 
                    
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

            ticklabel[1] = "correct decisions/ \n attempt"

            for (it,label) in enumerate(ticklabel[2:length(ticklabel)])
                ticklabel[it+1] = replace(label, "_" => " ")
            end

            p_mean = groupedbar([mean[2] mean[1]], bar_position = :dodge, bar_width = 0.7, xticks = (1:6, ticklabel), xrotation = 25, 
                                labels = [label2 label1], color = [my_cols[2] my_cols[4]], title = "Mean", fontfamily = "serif-roman", size = (1200, 800),
                                titlefontsize = 24, tickfontsize = 20, legendfontsize = 20, bottommargin = 6mm, leftmargin = 22mm)
            savefig(p_mean, plots_folder * "prl_urn_probs_eval_mean.png")

            p_median = groupedbar([median[2] median[1]], bar_position = :dodge, bar_width = 0.7, xticks = (1:6, ticklabel), xrotation = 25, 
                                labels = [label2 label1], color = [my_cols[2] my_cols[4]], title = "Median", fontfamily = "serif-roman", size = (1200, 800),
                                titlefontsize = 24, tickfontsize = 20, legendfontsize = 20, bottommargin = 6mm, leftmargin = 22mm)
            savefig(p_median, plots_folder * "prl_urn_probs_eval_median.png")
            
            if method == "cools"
                ticklabel = range(1, length(results_learning[label2 * "_reached_phase"]))
                p_learned_phase = groupedbar([results_learning[label2 * "_reached_phase"] results_learning[label1 * "_reached_phase"]], 
                                                bar_position = :dodge, bar_width = 0.7, xticks = (1:6, ticklabel), labels = [label2 label1], 
                                                color = [my_cols[2] my_cols[4]], title = "Correctly learned phase", fontfamily = "serif-roman", 
                                                plot_titlefontsize = 16, xlabelfontsize = 12, ylabelfontsize = 12)
                savefig(p_learned_phase, plots_folder * "correctly_learned_phase.png")

                p_mean_tries = groupedbar([results_learning[label2 * "_mean_iterations"] results_learning[label1 * "_mean_iterations"]], 
                                            bar_position = :dodge, bar_width = 0.7, xticks = (1:6, ticklabel), labels = [label2 label1], 
                                            color = [my_cols[2] my_cols[4]], title = "Mean tries until phase learned", fontfamily = "serif-roman",
                                            plot_titlefontsize = 16, xlabelfontsize = 12, ylabelfontsize = 12)
                savefig(p_mean_tries, plots_folder * "mean_tries.png")

                p_median_tries = groupedbar([results_learning[label2 * "_median_iterations"] results_learning[label1 * "_median_iterations"]], 
                                            bar_position = :dodge, bar_width = 0.7, xticks = (1:6, ticklabel), labels = [label2 label1], 
                                            color = [my_cols[2] my_cols[4]], title = "Median tries until phase learned", fontfamily = "serif-roman",
                                            plot_titlefontsize = 16, xlabelfontsize = 12, ylabelfontsize = 12)
                savefig(p_median_tries, plots_folder * "median_tries.png")
            end
        else
            throw("Bar plot comparing more than 2 models not implemented")
        end

    end
end