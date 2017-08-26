#=
Online statistics

m   # mean(x[1:n])
m2  #  sum of squares of differences from the current mean, ``\textstyle\sum_{i=1}^n (x_i - \bar x_n)^2``
n   # number of iterations

Implemented as "dependent iterator" where next has an additional argument
=#

"""
    mcstart(x) -> state

Create state for random chain online statitics. The entries/value of `x` are ignored
"""
mcstart(yy::Array{T}) where {T} = (zeros(T, size(yy))/1, zeros(T, size(yy))/1, 0)
mcstart(y::T) where {T<:Number} = (zero(T)/one(T), zero(T)/one(T), 0)

"""
    mcnext(state, x) -> state

Update random chain online statistics when new chain value `x` was
observed. Return new `state`.
"""
function mcnext(mc, x)  
    m, m2, n = mc
    delta = x - m
    n = n + 1
    m = m + delta/n
    delta2 = x - m
    m2 = m2 + delta .* delta2
    m, m2, n
end 

function mcnext(mc,  x::Vector{<:AbstractArray}) # fix me: use covariance
    m, m2, n = mc
    delta = x - m
    n = n + 1
    m = m + delta*(1/n)
    m2 = m2 + map((x,y)->x.*y, delta, (x - m))
    m, m2, n
end

"""
    mcmeanband(mc)

Compute marginal confidence interval for the chain mean using normal approximation
"""
function mcbandmean(mc)
    m, m2, k = mc
    Q = sqrt(2.)*erfinv(0.95)
    
    ste = eltype(m)[sqrt(v) for v in m2 * (1/(k - 1))]*sqrt(1/k)
    m - Q*ste, m + Q*ste
end

"""
    mcband(mc)

Compute marginal 95% coverage interval for the chain from normal approximation.
"""
function mcband(mc) 
    m, m2, k = mc
    Q = sqrt(2.)*erfinv(0.95)
    
    std = eltype(m)[sqrt(v) for v in m2 * (1/(k - 1))]
    m-Q*std, m+Q*std
end
