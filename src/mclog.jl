#=
Online statistics

m   # mean(x[1:n])
m2  #  sum of squares of differences from the current mean, ``\textstyle\sum_{i=1}^n (x_i - \bar x_n)^2``
n   # number of iterations

Implemented as "dependent iterator" where next has an additional argument
=#

_diag(x) = diag(x)
_diag(x::Number) = x

"""
    mcstart(x) -> state

Create state for random chain online statitics. The entries/value of `x` are ignored
"""
mcstart(yy::Array{T}) where {T} = (zeros(T, size(yy))/1, zeros(typeof(outer(zero(T))), size(yy))/1, 0)
mcstart(y::T) where {T<:Number} = (zero(T)/one(T), zero(T)/one(T), 0)

"""
    mcnext(state, x) -> state

Update random chain online statistics when new chain value `x` was
observed. Return new `state`.
"""
function mcnext(mc, x)  
    m, m2, n = mc
    delta = x - m
    m = m + delta/(n + 1)
    delta2 = x - m
    m2 = m2 + delta .* delta2
    m, m2, n + 1
end 

function mcnext(mc,  x::Vector{<:AbstractArray}) # fix me: use covariance
    m, m2, n = mc
    delta = x - m
    m = m + delta/(n+1)
    m2 = m2 + map(outer, delta, (x - m))
    m, m2, n + 1
end

function mcnext!(mc,  x::Vector{<:AbstractArray}) 
    m, m2, n = mc
    for i in 1:length(m2)
        delta = x[i] - m[i]
        m[i]  += (delta)/(n+1)
        m2[i] += outer(delta, x[i]-m[i])
    end
    m, m2, n + 1
end

"""
    mcmeanband(mc)

Compute marginal confidence interval for the chain *mean* using normal approximation
"""
function mcbandmean(mc)
    m, m2, k = mc
    Q = sqrt(2.)*erfinv(0.95)
    
    ste = eltype(m)[sqrt.(_diag(v)*(1/(k - 1))) for v in m2 ]*sqrt(1/k)
    m - Q*ste, m + Q*ste
end

"""
    mcband(mc)

Compute marginal 95% coverage interval for the chain from normal approximation.
"""

function mcband(mc) 
    m, m2, k = mc
    Q = sqrt(2.)*erfinv(0.95)
    
    std = eltype(m)[sqrt.(_diag(v) * (1/(k - 1))) for v in m2]
    m-Q*std, m+Q*std
end

"""
    mcstats(mc)

Compute mean and covariance estimates.
"""
function mcstats(mc) 
    m, m2, k = mc
    cov = m2/(k - 1)
    m, cov
end