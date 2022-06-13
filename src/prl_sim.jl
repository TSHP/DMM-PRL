module PRL
    using Distributions, Random
    include("dmm.jl")
    include("constants.jl")

    struct ModelParams
        mm::Float16
        pm::Float16
        mp::Float16
        pp::Float16
        alpha::Float16
        m::Float16
    end

    struct Results
        draws::Array
        probabilities::Array
        std_deviations::Array
        no_clusters::Array
        cluster_switches::Array
        phases_learned::Array
        iterations_needed::Array
    end

    # Anonymous functions for model, maybe move somewhere else
    prior(m::Real, p::Real, M::NamedTuple) = logpdf(M.prior.prior_m, m) + logpdf(truncated(M.prior.prior_p, 0, Inf), p)
    target(M) = (x, theta) -> sum(logpdf.(Normal(theta[1], 1 / sqrt(theta[2])), x)) + prior(theta..., M)
    proposal(theta) = [rand(Normal(theta[1], 1)), rand(truncated(Normal(theta[2], 1), 0, Inf))]

    function generate_bead_seq(method, phase = 1)
        seq = []
        urn_0 = Binomial(1, 0.2)
        urn_1 = Binomial(1, 0.8)
        n_draws = 10

        if method == "3p_10t"
            # 3 phases, draw from different urn each phase
            for it in 1:3
                draws = it % 2 == 0 ? rand(urn_0, n_draws) : rand(urn_1, n_draws)
                seq = [seq; draws]
            end
        elseif method=="cools"
            # 3 phases, draw from different urn each phase
            seq = phase % 2 == 0 ? rand(urn_0, n_draws) : rand(urn_1, n_draws) 
        else
            throw("Method " * string(method) * " not implemented")
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
        replace!(log_odds, Inf => NaN)
        replace!(log_odds, NaN => maximum(filter(!isnan, log_odds)))

        replace!(log_odds, -Inf => NaN)
        replace!(log_odds, NaN => minimum(filter(!isnan, log_odds)))
    end

    # Set number of phases and iterations according to experiment
    function setup_experiment(method)
        if method == "3p_10t"
            n_phases = 1
            max_iter = 1
        elseif method == "cools"
            n_phases = 3
            max_iter = 5
        else
            throw("Method " * string(method) * " not implemented")
        end

        return (n_phases, max_iter)
    end

    # Return true if more then the specified percentage of "guesses" are correct
    function check_learned(phase, probs, correct_perc = 0.9, decision_bnd = [0.5, 0.5])
        seq_length = length(probs)
        decisions = deepcopy(probs)
        decisions[decisions .< decision_bnd[1]] .= 0
        decisions[decisions .>= decision_bnd[2]] .= 1

        correct_seq = phase % 2 == 0 ? zeros(seq_length) : ones(seq_length)

        return sum(decisions .== correct_seq) >= correct_perc*seq_length
    end

    # Run a phase of the experiment
    function run_phase(M, urn_log_odds, n_history, belief_strength, output)
        n_steps = 10
        probs = []
        std_devs = []
        nof_cluster_centers = []
        cluster_switches = []

        # Initialize with just one cluster
        comps_prev = []

        # Draw from urn with a history of n_history
        xprev = [urn_log_odds[1]]
        nk = length(xprev)
        zprev = repeat([1], nk)

        # Initial theta
        theta = [0.0, 2.5]

        push!(comps_prev, (n = nk, theta = theta))

        for draw in 2:length(urn_log_odds)
            # Draw new bead from urn
            xnew = [urn_log_odds[draw]]

            if output @show draw end

            znew, comps = DMM.init_mixture(xnew, xprev, zprev, deepcopy(comps_prev), M, target, proposal)
            x = [xprev; xnew]; z = [zprev; znew]
            
            # Consolidate
            z, comps = DMM.update_mixture(x, z, comps, M; n_steps = n_steps, target, proposal)
            push!(nof_cluster_centers, length(comps))

            # Check if observation is assigned to a new cluster or existing one
            new_cluster_check = comps[last(z)].n == 1
            push!(cluster_switches, new_cluster_check)

            # Get cluster mean to compute probability
            pred_log_odd = comps[last(z)].theta[1]
            pred_log_odd_std = comps[last(z)].theta[2]
            prob = exp(pred_log_odd) / (1 + exp(pred_log_odd))
            std = exp(pred_log_odd_std) / (1 + exp(pred_log_odd_std))
            push!(probs, prob)
            push!(std_devs, std)

            # Update observations and clusters for next iteration
            # lower = draw + 1 > n_history ? draw + 1 - n_history : 1
            # xprev = urn_log_odds[lower:draw]

            # Strengthen most recent hypothesis by sampling n_history new observations from the hypothesis
            # d = Normal(comps[last(z)].theta[1], comps[last(z)].theta[2])
            # xprev = rand(d, belief_strength)

            # Use previous hypothesis for next iteration
            # comps_prev = []
            # push!(comps_prev, (n = length(xprev), theta = comps[last(z)].theta))
            # zprev = repeat([1], length(xprev))

            xprev_tmp = []
            zprev_tmp = []
            comps_prev = []
            n = sum([comps_n.n for comps_n in comps])

            distributions = [Normal(comps_n.theta[1], comps_n.theta[2]) for comps_n in comps]
            for (i, distr) in enumerate(distributions)
                nof_samples = Int(round(comps[i].n/n))*belief_strength > 0 ? Int(round(comps[i].n/n))*belief_strength : 10
                x = rand(distr, nof_samples)
                push!(xprev_tmp, x)
                push!(comps_prev, (n = length(x), theta = comps[i].theta))
                push!(zprev_tmp, repeat([i], length(x)))
            end

            xprev = vcat(xprev_tmp...)
            zprev = vcat(zprev_tmp...)
        end

        return (probs, std_devs, nof_cluster_centers, cluster_switches)
    end

    function run_experiment(params, n_history, belief_strength, seed, method, output = false)
        Random.seed!(seed)

        # Init model
        M = DMM.Model(pm = params.pm, mp = params.mp, pp = params.pp, alpha = params.alpha)
        
        all_probs = []
        all_std_devs = []
        all_nof_cluster_centers = []
        all_draws = []
        all_cluster_switches = []

        # Get experiment settings
        n_phases, max_iter = setup_experiment(method)

        phases_learned = []
        iterations_needed = []

        # Start simulation
        if output print("Starting simulation...\n") end

        for phase in range(1, n_phases)
            if output @show phase end
            phase_learned = false
            on_iteration = 1
            iteration = 1

            # Add draws while maximum iterations not exceeded and phase not learned
            while iteration in range(1, max_iter) && !phase_learned
                if output @show iteration end

                # Generate appropriate draw sequence
                if method == "3p_10t"
                    draws = generate_bead_seq(method, phase)
                elseif method == "cools"
                    draws = generate_bead_seq(method, phase)
                else
                    throw("Method " * string(method) * " not implemented")
                end

                # At first pass add initial 0, 1
                if phase == 1 && iteration == 1
                    pushfirst!(draws, 0)
                    pushfirst!(draws, 1)
                end

                # Update draw sequence
                append!(all_draws, draws)
                
                # Calculate probabilities and log odds
                urn_probs = get_urn_probs(all_draws, n_history)
                urn_log_odds = [log(urn_probs[i] / (1 - urn_probs[i])) for i in 1:length(urn_probs)]
                clean_log_odds!(urn_log_odds)

                # Run experiment
                probs, std_devs, nof_cluster_centers, cluster_switches = run_phase(M, urn_log_odds, n_history, belief_strength, output)

                # Update results
                segment_length = phase == 1 && iteration == 1 ? length(draws) - 2 : length(draws)
                append!(all_probs, probs[(length(probs) - segment_length + 1): length(probs)])
                append!(all_std_devs, std_devs[(length(std_devs)- segment_length + 1): length(std_devs)])
                append!(all_nof_cluster_centers, nof_cluster_centers[length(nof_cluster_centers) - segment_length + 1: length(nof_cluster_centers)])
                append!(all_cluster_switches, cluster_switches[length(cluster_switches) - segment_length + 1: length(cluster_switches)])

                # Check if model learned state with newly added sequence
                if method == "cools"
                    start_current = length(probs) - segment_length + 1  # start at 1, 11, 21, ...
                    end_current = length(probs)
                    phase_learned = check_learned(phase, probs[start_current:end_current])
                    if phase_learned on_iteration = iteration end
                end
                
                iteration += 1
            end

            if method == "cools"
                if !phase_learned 
                    append!(iterations_needed, iteration)
                    break
                end
                # Update learned phases and number of iterations needed
                append!(phases_learned, phase)
                append!(iterations_needed, on_iteration)
            end
        end

        return Results(all_draws, all_probs, all_std_devs, all_nof_cluster_centers, all_cluster_switches, phases_learned, iterations_needed)
    end
end
