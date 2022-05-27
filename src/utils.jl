prior(m::Real, p::Real, M::NamedTuple) = logpdf(M.prior.prior_m, m) +
    logpdf(truncated(M.prior.prior_p, 0, Inf), p)
target(M) = (x, theta) -> sum(logpdf.(Normal(theta[1], 1/sqrt(theta[2])), x)) + prior(theta..., M)
proposal(theta) = [rand(Normal(theta[1], 1)), rand(truncated(Normal(theta[2], 1), 0, Inf))]