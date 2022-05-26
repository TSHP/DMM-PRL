include("./src/gmm.jl")

prior(m::Real, p::Real, M::NamedTuple) = logpdf(M.prior.prior_m, m) +
    logpdf(truncated(M.prior.prior_p, 0, Inf), p)
target(M) = (x, theta) -> sum(logpdf.(Normal(theta[1], 1/sqrt(theta[2])), x)) + prior(theta..., M)
proposal(theta) = [rand(Normal(theta[1], 1)), rand(truncated(Normal(theta[2], 1), 0, Inf))]
## setup models
mps = [1/1000, 1000]
M1 = Model(pm = 0.01, mp = mps[1], pp = 10, alpha = 1)
M2 = Model(pm = 0.01, mp = mps[2], pp = 10, alpha = 1)

# Do urn simulation
n_steps = 10
ylim = (-0.2, 3.0)
nnew = 1
ndraws = 50
probs = []
results = []
plots = []

# Generate beads from same seed
seed = 9369
Random.seed!(seed)

urn_1 = Binomial(1, 0.85)
urn_0 = Binomial(1, 0.15)

draws_1 = rand(urn_1, ndraws)
pushfirst!(draws_1, 1)
pushfirst!(draws_1, 0)

draws_0 = rand(urn_0, ndraws)

draws = [draws_1; draws_0]

n_history = 10

urn_probs = []
for draw in 2:length(draws)
    lower = draw > n_history ? draw - n_history : 1
    push!(urn_probs, mean(draws[lower:draw]))
end

urn_log_odds = [log(urn_probs[i] / (1 - urn_probs[i])) for i in 1:length(urn_probs)]

replace!(urn_log_odds, Inf=>NaN)
replace!(urn_log_odds, NaN=>maximum(filter(!isnan, urn_log_odds)))

replace!(urn_log_odds, -Inf=>NaN)
replace!(urn_log_odds, NaN=>minimum(filter(!isnan, urn_log_odds)))

# start simulations from same seed
seed = 9369
for  (index, M) in enumerate([M1, M2])
    seed = 42
    Random.seed!(seed)
    probs = []
    standard_deviations = []
    num_cluster_centers = []

    # TODO: Add more initial 0s and 1s
    for draw in 2:length(urn_log_odds)
        ## Draw from urn with a history of n_history
        lower = draw > n_history ? draw - n_history : 1
        xinit = urn_log_odds[lower:(draw - 1)]
        nk = length(xinit)
        zinit = repeat([1], nk)

        ## Do MCMC
        n_iter = 1000
        chn = do_mcmc(xinit, ones(2), target(Model()), proposal, n = n_iter)
        theta = mean(chn.theta[round(Int, n_iter/2):end, :], dims = 1)

        comps_init = []
        push!(comps_init, (n = nk, theta = theta))

        ## Draw new bead from urn
        xnew = [urn_log_odds[draw]]

        znew, comps = init_mixture(xnew, xinit, zinit, deepcopy(comps_init), M)
        x = [xinit; xnew]; z = [zinit; znew]
        ## consolidate
        z, comps = update_mixture(x, z, comps, M; n_steps = n_steps)
        push!(num_cluster_centers, length(comps))

        # Get cluster mean to compute probability
        pred_log_odd = comps[last(z)].theta[1]
        pred_log_odd_std = comps[last(z)].theta[2]
        prob = exp(pred_log_odd) / (1 + exp(pred_log_odd))
        std = exp(pred_log_odd_std) / (1 + exp(pred_log_odd_std))
        push!(probs, prob)
        push!(standard_deviations, std)
    end
    label = "mu = $(mps[index])"
    p1 = plot(probs, labels = label, xlabel = "Number of drawn beads", ylabel = "Estimated probability", linewidth = 2, xlabelfontsize = 7, ylabelfontsize = 7, legendfontsize = 6, legend = false)
    push!(plots, p1)
    p2 = plot(standard_deviations, labels = label, xlabel = "Number of drawn beads", ylabel = "Estimated standard deviation", linewidth = 2, xlabelfontsize = 7, ylabelfontsize = 7, legendfontsize = 6, legend = false)
    push!(plots, p2)
    p3 = plot(num_cluster_centers, labels = label, xlabel = "Number of drawn beads", ylabel = "Number of clusters", linewidth = 2, xlabelfontsize = 7, ylabelfontsize = 7, legendfontsize = 6, legend = false)
    push!(plots, p3)
end

beads_plot = scatter(draws[3:length(draws)], linewidth = 2, xlabelfontsize = 7, ylabelfontsize = 7, legendfontsize = 6, legend = false)
yticks!([0, 1])
plot(plots[1], plots[3], beads_plot, layout = (3, 1), plot_title = "Estimated probability of the beads coming from urn 1", titlefontsize = 9)
png("./io/plots/urn_probs_healthy.png")
plot(plots[4], plots[6], beads_plot, layout = (3, 1), plot_title = "Estimated probability of the beads coming from urn 1", titlefontsize = 9)
png("./io/plots/urn_probs_delusional.png")