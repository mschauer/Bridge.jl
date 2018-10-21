export MeanCov, MeanVar
import Base.iterate

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

Iterator interface for online mean and covariance estimates.
Iterates are triples `mean, λ, cov/λ`

Example:

```
c = Channel{Vector}(1)
m = Bridge.MeanCov(c)
put!(c, rand(5))
u = iterate(m)
put!(c, rand(5))
u = iterate(m, u[2])
close(c)
u[1][1]
```

```
m = Bridge.MeanCov(Channel{Vector{Float64}}(1))
u = register!(m, rand(5))
u = register!(m, rand(5), u)
close(m)
u[1][1]
```

"""
struct MeanCov{T}
    iter::T
end
import Base: length, IteratorSize
length(x::MeanCov) = length(x.iter)
IteratorSize(::Type{MeanCov{T}}) where {T} = IteratorSize(T) isa Base.HasShape ?  Base.HasLength() : IteratorSize(T)

function register!(m::MeanCov{<:Channel}, x)
    put!(m.iter, x)
    iterate(m)
end
function register!(m::MeanCov{<:Channel}, x, u)
    put!(m.iter, x)
    iterate(m, u[2])
end
Base.close(m::MeanCov{<:Channel}) = close(m.iter)


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


struct MeanVar{T}
    iter::T
end
import Base: length, IteratorSize
length(x::MeanVar) = length(x.iter)
IteratorSize(::Type{MeanVar{T}}) where {T} = IteratorSize(T) isa Base.HasShape ?  Base.HasLength() : IteratorSize(T)

function register!(m::MeanVar{<:Channel}, x)
    put!(m.iter, x)
    iterate(m)
end
function register!(m::MeanVar{<:Channel}, x, u)
    put!(m.iter, x)
    iterate(m, u[2])
end
Base.close(m::MeanVar{<:Channel}) = close(m.iter)


@inline function iterate(mc::MeanVar)
    u = iterate(mc.iter)
    if u === nothing
        return nothing
    end
    x, state = u
    n = 0
    delta = copy(x)
    m = delta/(n+1)

    m2 =(delta).*(x - m)
    (m, 1/n, m2), (m, m2, delta, n + 1, state)
end

iterate(mc::MeanVar, mcstate) = iterate_(mc, ismutable(typeof(mcstate[1])), mcstate...)


function iterate_(mc::MeanVar, ::Val{false}, m, m2, delta, n, state )
    u = iterate(mc.iter, state)
    if u === nothing
        return nothing
    end
    x, state = u
    delta = x - m
    m  += delta/(n+1)
    m2 += delta .* (x - m)
    (m, 1/n, m2), (m, m2, delta, n + 1, state)
end


function iterate_(mc::MeanVar, ::Val{true}, m, m2, delta, n, state )
    u = iterate(mc.iter, state)
    if u === nothing
        return nothing
    end
    x, state = u

    for i in eachindex(m)
        delta[i] = x[i] - m[i]
        m[i] += (delta[i])/(n+1)
        m2[i] += outer(delta[i], x[i] - m[i])
    end
    (m, 1/n, m2), (m, m2, delta, n + 1, state)
end

export OnlineStat
import Base: push!
import Statistics: mean, cov, std

"""

```
stats = map(OnlineStat, (x, θ, η))

map(push!, stats, (x, θ, η))

mean.(stats)
cov.(stats)
```
"""
mutable struct OnlineStat{S}
    state::S
    function OnlineStat(x)
        m = Bridge.MeanVar(Channel{typeof(x)}(1))
        u = Bridge.register!(m, x)
        new{typeof((m,u...))}((m,u...))
    end
end
function push!(o::OnlineStat, x)
    @assert o.state != nothing # cannot happen for now
    m, val, s = o.state
    u = Bridge.register!(m, x, (val, s))
    o.state = (o.state[1], u[1], u[2])
end
mean(o::OnlineStat) = o.state[2][1]
std(o::OnlineStat) = sqrt.(o.state[2][2]*o.state[2][3])
#=
S = OnlineStat(ones(5))
for i in 2:10
    push!(S, i*ones(5))
end
mean(S)
cov(S)
cov(repeat(collect(1:10)', outer=5)') == cov(S)
=#
