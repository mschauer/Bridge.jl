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


"""
    mcmarginalstats(mcstates) -> mean, std

Compute `mean`` and marginal standard deviations `std` for 2d plots. 
"""
function mcmarginalstats(states) 
    xx, vv = Bridge.mcstats(states[1])
    Xmean = copy(xx)
    Xstd = map(x->sqrt.(diag(x)), vv)
    for i in 2:length(states)
        pop!(Xmean); pop!(Xstd)
        xx, vv = Bridge.mcstats(states[i])
        append!(Xmean, xx)
        append!(Xstd, map(x->sqrt.(diag(x)), vv))
    end 
    Xmean, Xstd
end

"""
    MeanCov(itr)

Iterator interface for online mean and covariance
Iterates are triples `mean, λ, cov/λ`  
   
"""
struct MeanCov{T}
    iter::T
end
import Base: length, IteratorSize
length(x::MeanCov) = length(x.iter)
IteratorSize(::Type{MeanCov{T}}) where {T} = IteratorSize(T) isa Base.HasShape ?  Base.HasLength() : IteratorSize(T)


@inline function iterate(mc::MeanCov) 
    u = iterate(mc.iter)
    if u === nothing
        return nothing
    end
    x, state = u
    n = 0
    delta = copy(x)
    m = delta/(n+1)
    
    m2 = outer(delta, x - m)
    (m, 1/n, m2), (m, m2, delta, n + 1, state)
end


iterate(mc::MeanCov, mcstate) = iterate_(mc, ismutable(typeof(mcstate[1])), mcstate...)    

function iterate_(mc::MeanCov, ::Val{false}, m, m2, delta, n, state ) 
    u = iterate(mc.iter, state)
    if u === nothing
        return nothing
    end
    x, state = u
    delta = x - m
    m  += delta/(n+1)
    m2 += outer(delta, x - m)
    (m, 1/n, m2), (m, m2, delta, n + 1, state)
end


function iterate_(mc::MeanCov, ::Val{true}, m, m2, delta, n, state ) 
    u = iterate(mc.iter, state)
    if u === nothing
        return nothing
    end
    x, state = u
        
    for i in eachindex(m)
        delta[i] = x[i] - m[i]
        m[i] += (delta[i])/(n+1)
    end
    for i in eachindex(m)
        for j in eachindex(m)
            m2[i,j] += outer(delta[i], x[j]-m[j])
        end
    end
    (m, 1/n, m2), (m, m2, delta, n + 1, state)
end

