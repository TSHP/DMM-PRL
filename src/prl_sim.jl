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
        phases_learned::Array
        iterations_needed::Array
    end

    # TODO: Anonymous functions for model, maybe move somewhere else
    prior(m::Real, p::Real, M::NamedTuple) = logpdf(M.prior.prior_m, m) + logpdf(truncated(M.prior.prior_p, 0, Inf), p)
    target(M) = (x, theta) -> sum(logpdf.(Normal(theta[1], 1 / sqrt(theta[2])), x)) + prior(theta..., M)
    proposal(theta) = [rand(Normal(theta[1], 1)), rand(truncated(Normal(theta[2], 1), 0, Inf))]

    function generate_bead_seq(method, phase = 1)
        seq = []
        urn_0 = Binomial(1, 0.15)
        urn_1 = Binomial(1, 0.85)
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
    function check_learned(phase, probs, correct_perc = 0.8, decision_bnd = [0.5, 0.5])
        seq_length = length(probs)
        decisions = deepcopy(probs)
        decisions[decisions .< decision_bnd[1]] .= 0
        decisions[decisions .>= decision_bnd[2]] .= 1

        correct_seq = phase % 2 == 0 ? zeros(seq_length) : ones(seq_length)

        return sum(decisions .== correct_seq) >= correct_perc*seq_length
    end

    # Run a phase of the experiment
    function run_phase(M, urn_log_odds, n_history, output)
        n_steps = 10
        probs = []
        std_devs = []
        nof_cluster_centers = []

        for draw in 2:length(urn_log_odds)
            comps_init = []

            # Draw from urn with a history of n_history
            lower = draw > n_history ? draw - n_history : 1
            xinit = urn_log_odds[lower:(draw - 1)]
            nk = length(xinit)
            zinit = repeat([1], nk)

            # Do MCMC
            n_iter = 1000
            chn = DMM.do_mcmc(xinit, ones(2), target(DMM.Model()), proposal, n = n_iter)
            theta = mean(chn.theta[round(Int, n_iter / 2):end, :], dims = 1)

            push!(comps_init, (n = nk, theta = theta))

            # Draw new bead from urn
            xnew = [urn_log_odds[draw]]

            if output
                @show draw
            end

            znew, comps = DMM.init_mixture(xnew, xinit, zinit, deepcopy(comps_init), M, target, proposal)
            x = [xinit; xnew]; z = [zinit; znew]

            # consolidate
            z, comps = DMM.update_mixture(x, z, comps, M; n_steps = n_steps, target, proposal)
            push!(nof_cluster_centers, length(comps))

            # Get cluster mean to compute probability
            pred_log_odd = comps[last(z)].theta[1]
            pred_log_odd_std = comps[last(z)].theta[2]
            prob = exp(pred_log_odd) / (1 + exp(pred_log_odd))
            std = exp(pred_log_odd_std) / (1 + exp(pred_log_odd_std))
            push!(probs, prob)
            push!(std_devs, std)
        end

        return (probs, std_devs, nof_cluster_centers)
    end

    function run_experiment(params, seed, method, output = false)
        Random.seed!(seed)

        # Average over n_history past draws for log odds
        n_history = 5

        # Init model
        M = DMM.Model(pm = params.pm, mp = params.mp, pp = params.pp, alpha = params.alpha)
        
        all_probs = []
        all_std_devs = []
        all_nof_cluster_centers = []
        all_draws = []

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
                probs, std_devs, nof_cluster_centers = run_phase(M, urn_log_odds, n_history, output)

                # Update results
                segment_length = phase == 1 && iteration == 1 ? length(draws) - 2 : length(draws)
                append!(all_probs, probs[(length(probs) - segment_length + 1): length(probs)])
                append!(all_std_devs, std_devs[(length(std_devs)- segment_length + 1): length(std_devs)])
                append!(all_nof_cluster_centers, nof_cluster_centers[length(nof_cluster_centers)-segment_length+1: length(nof_cluster_centers)])

                # Check if model learned state with newly added sequence
                if method == "cools"
                    start_current = (iteration - 1) * segment_length + 1  # start at 1, 11, 21, ...
                    end_current = length(probs)
                    phase_learned = check_learned(phase, probs[start_current:end_current])
                    if phase_learned on_iteration = iteration end
                    if !phase_learned && iteration == max_iter 
                        all_draws = all_draws[1:(length(all_draws) - 50)] # yes this is very hardcoded. no i'm not fixing it -KK
                        all_probs = all_probs[1:(length(all_probs) - 50)]
                        all_std_devs = all_std_devs[1:(length(all_std_devs) - 50)]
                        all_nof_cluster_centers = all_nof_cluster_centers[1:(length(all_nof_cluster_centers) - 50)]
                    end
                end
                
                iteration += 1
            end

            if method == "cools"
                if !phase_learned 
                    break
                end
                # Update learned phases and number of iterations needed
                append!(phases_learned, phase)
                append!(iterations_needed, on_iteration)
            end
        end

        return Results(all_draws, all_probs, all_std_devs, phases_learned, iterations_needed)
    end
end
