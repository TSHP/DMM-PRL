module DMM
    using Distributions, StatsFuns, Random

    # MCMC inference for updating of parameters of component specific models
    function do_mcmc(x, theta, target::Function, proposal::Function; n=1000) 
        vec = Array{typeof(theta), 1}(undef, n)
        ll = Array{Real, 1}(undef, n)
        vec[1] = theta
        ll[1]  = target(x, theta)  
        for i in 2:n
            mcmc_step!(i, vec, ll, x, target, proposal)
        end
        (theta = Array(hcat(vec...)'), ll = ll)
    end

    function mcmc_step!(i, vec, ll, x, target::Function, proposal::Function)
        can = proposal(vec[i - 1])
        lik = target(x, can)
        if (rand() < exp(lik - ll[i - 1])) 
            vec[i] = can
            ll[i]  = lik
        else
            vec[i] = vec[i-1]
            ll[i]  = ll[i-1]
        end
        return nothing
    end

    # Compute predictive probabilities for label assingment
    function predictive_prob(ynew, phi, j, comps, M)
        K = length(comps)
        n = sum([c.n for c in comps])
        # Compute assignment probabilities
        if j <= K
            log(comps[j].n / (n - 1 + M.alpha)) + logpdf(M.likelihood(phi[j]), ynew)
        else
            log((M.alpha / M.m) / (n - 1 + M.alpha)) + logpdf(M.likelihood(phi[j]), ynew)
        end
    end


    # Do a first pass to initialize znew
    function init_mixture(xnew, x, z, comps, M::NamedTuple, target::Function, proposal::Function)
        @assert length(x) == length(z)
        # First pass to initialize znew
        nnew = length(xnew)
        znew = zeros(Int, nnew)
        alph = M.alpha[1]
        m = M.m
        n = nnew + sum([c.n for c in comps])
        for i in 1:nnew
            # Create candidates
            phi = [[c.theta for c in comps];
                [[rand(p) for p in M.prior] for rep in 1:m]]
            ps = [predictive_prob(xnew[i], phi, j, comps, M) for j in 1:length(phi)]
            ps .= exp.(ps .- logsumexp(ps))
            znew[i] = rand(Categorical(ps))
            if znew[i] > length(comps)
                # Create new components with the sampled parameters phi[z]
                push!(comps, (n = 1, theta = phi[znew[i]]))
                # Set to K + 1
                znew[i] = length(comps) 
            else 
                n_iter = 10
                chn = do_mcmc([x[z .== znew[i]]; xnew[znew .== znew[i]]],
                            [comps[znew[i]].theta...], target(M), proposal, n = n_iter)
                theta_new = mean(chn.theta[round(Int, n_iter / 2):end, :], dims = 1)
                comps[znew[i]] = (n = comps[znew[i]].n + 1, theta = theta_new)
            end
        end
        znew, comps
    end

    # Do n_steps steps of Gibbs sampling
    function update_mixture(x, z, comps, M; n_steps = 1, target::Function, proposal::Function)
        @assert length(x) == length(z)
        @assert length(x) == sum([c.n for c in comps])
        n = length(x)
        m = M.m
        alph = M.alpha[1]
        for iter in 1:n_steps
            # Iterate over observations
            inds = shuffle(1:length(x))
            for i in inds
                # Create candidates
                ni = comps[z[i]].n
                if ni == 1
                    # If last element of a cluster,
                    # Remove that cluster, but use phi
                    # In proposal (at position K^{-}+1)
                    j = z[i]; oldcomp = comps[j]
                    comps = comps[setdiff(1:length(comps), j)]
                    # Fill gap in indices
                    z[z .>= j] .= z[z .>= j] .- 1
                    phi = [[comps[j].theta for j in 1:length(comps)];
                        [oldcomp.theta'];
                        [[rand(p) for p in M.prior] for rep in 1:m-1]]
                else
                    comps[z[i]] = (n = comps[z[i]].n - 1,
                                theta = comps[z[i]].theta)
                    phi = [[comps[j].theta for j in 1:length(comps)];
                        [[rand(p) for p in M.prior] for rep in 1:m]]
                end
                @assert length(phi) == length(comps) + m
                ps = [predictive_prob(x[i], phi, j, comps, M) for j in 1:length(phi)]
                ps .= exp.(ps .- logsumexp(ps))
                z[i] = rand(Categorical(ps))
                if z[i] >= length(comps)
                    # Create new components with the sampled parameters phi[z]
                    push!(comps, (n = 1, theta = phi[z[i]]))
                    # Set to K + 1
                    z[i] = length(comps) 
                else 
                    n_iter = 10
                    chn = do_mcmc(x[z.== z[i]], [comps[z[i]].theta...], target(M),
                                proposal, n = n_iter)
                    theta_new = mean(chn.theta[round(Int, n_iter/2):end, :], dims = 1)
                    comps[z[i]] = (n = comps[z[i]].n + 1, theta = theta_new)
                end
                @assert maximum(z) <= length(comps)
            end
        end
        z, comps
    end

    # Setup model M
    function Model(;mm = 0.0, pm = .1, mp = 1, pp = .1, alpha = 1, m = 2)
        (likelihood = (theta) -> Normal(theta[1], 1/sqrt(theta[2])),
        prior = (prior_m = Normal(mm, 1/sqrt(pm)),
                prior_p = truncated(Normal(mp, 1/sqrt(pp)), 0, Inf)),
        alpha = alpha,
        m = m)
    end
    
end