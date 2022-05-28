module PRL
    using Distributions, StatsFuns, Random
    using Plots, Colors
    include("./gmm.jl")
    include("./utils.jl")

    struct ModelParams
        mm
        pm
        mp
        pp
        alpha
        m
    end

    struct Results
        draws
        probabilities
        std_deviations
    end
   
    function generate_bead_seq(method="0")
        seq = []
        urn_0 = Binomial(1, 0.15)
        urn_1 = Binomial(1, 0.85)

        if method == "0"
            n_draws=10
            for it in 1:3
                draws = it%2 == 0 ? rand(urn_0, n_draws) : rand(urn_1, n_draws)
                seq = [seq; draws]
            end
        elseif method == "test"
            n_draws=1
            for it in 1:3
                draws = it%2 == 0 ? rand(urn_0, n_draws) : rand(urn_1, n_draws)
                seq = [seq; draws]
            end
        end
        return seq
    end

    function get_urn_probs(draws, n_history)
        urn_probs = []
        for draw in 2:length(draws)
            lower = draw > n_history ? draw - n_history : 1
            push!(urn_probs, mean(draws[lower:draw]))
        end
        return urn_probs
    end

    function clean_log_odds!(log_odds)
        replace!(log_odds, Inf=>NaN)
        replace!(log_odds, NaN=>maximum(filter(!isnan, log_odds)))

        replace!(log_odds, -Inf=>NaN)
        replace!(log_odds, NaN=>minimum(filter(!isnan, log_odds)))
    end

    function save_plots(draws, plots, filename)
        beads_plot = scatter(draws[3:length(draws)], linewidth = 2, xlabelfontsize = 7, ylabelfontsize = 7, legendfontsize = 6, legend = false)
        yticks!([0, 1])
        plot(plots[1], plots[3], beads_plot, layout = (3, 1), plot_title = "Estimated probability of the beads coming from urn 1", titlefontsize = 9)
        png("./io/plots/"*filename*".png")
    end

    function save_results(filename, eval, params)

    end

    function run_experiment(params, filename, seed, output=false, test=false)
        Random.seed!(seed)
        n_steps=10
        n_history=5

        # Generate beads, probabilities and log odds from sequence
        M = GMM.Model(pm = params.pm, mp = params.mp, pp = params.pp, alpha = params.alpha)

        if test
            draws = generate_bead_seq("test")
        else
            draws = generate_bead_seq()
        end
        pushfirst!(draws, 0)
        pushfirst!(draws, 1)

        urn_probs = get_urn_probs(draws, n_history)
        urn_log_odds = [log(urn_probs[i] / (1 - urn_probs[i])) for i in 1:length(urn_probs)]
        clean_log_odds!(urn_log_odds)
        
        probs = []
        std_devs = []
        nof_cluster_centers = []
        plots = []

        # start simulation
        if output
            print("Starting simulation...\n")
        end
        for draw in 2:length(urn_log_odds)
            comps_init = []

            # Draw from urn with a history of n_history
            lower = draw > n_history ? draw - n_history : 1
            xinit = urn_log_odds[lower:(draw - 1)]
            nk = length(xinit)
            zinit = repeat([1], nk)

            # Do MCMC
            n_iter = 1000
            chn = GMM.do_mcmc(xinit, ones(2), target(GMM.Model()), proposal, n = n_iter)
            theta = mean(chn.theta[round(Int, n_iter/2):end, :], dims = 1)

            push!(comps_init, (n = nk, theta = theta))

            # Draw new bead from urn
            xnew = [urn_log_odds[draw]]

            if output
                @show draw
            end

            znew, comps = GMM.init_mixture(xnew, xinit, zinit, deepcopy(comps_init), M)
            x = [xinit; xnew]; z = [zinit; znew]
            # consolidate
            z, comps = GMM.update_mixture(x, z, comps, M; n_steps = n_steps)
            push!(nof_cluster_centers, length(comps))

            # Get cluster mean to compute probability
            pred_log_odd = comps[last(z)].theta[1]
            pred_log_odd_std = comps[last(z)].theta[2]
            prob = exp(pred_log_odd) / (1 + exp(pred_log_odd))
            std = exp(pred_log_odd_std) / (1 + exp(pred_log_odd_std))
            push!(probs, prob)
            push!(std_devs, std)
        end

        label = "mu = $(params.mp)"
        p1 = plot(probs, labels = label, xlabel = "Number of drawn beads", ylabel = "Estimated probability", linewidth = 2, xlabelfontsize = 7, ylabelfontsize = 7, legendfontsize = 6, legend = false)
        push!(plots, p1)
        p2 = plot(std_devs, labels = label, xlabel = "Number of drawn beads", ylabel = "Estimated standard deviation", linewidth = 2, xlabelfontsize = 7, ylabelfontsize = 7, legendfontsize = 6, legend = false)
        push!(plots, p2)
        p3 = plot(nof_cluster_centers, labels = label, xlabel = "Number of drawn beads", ylabel = "Number of clusters", linewidth = 2, xlabelfontsize = 7, ylabelfontsize = 7, legendfontsize = 6, legend = false)
        push!(plots, p3)

        save_plots(draws, plots, filename)

        return Results(draws, probs, std_devs)
    end
end
