
struct State{P}
    q::Vector{P}
    p::Vector{P}
end
q(x::State) = x.q
p(x::State) = x.p
q(x::State, i) = x.q[i]
p(x::State, i) = x.p[i]



import Base:iterate, eltype, copy, copyto!, zero, eachindex, getindex, setindex!, size, vec
iterate(x::State) = (x.q, true)
iterate(x::State, s) = s ? (x.p, false) : nothing
copy(x::State) = State(copy(x.q), copy(x.p))
function copyto!(x::State, y::State)
    copyto!(x.q, y.q)
    copyto!(x.p, y.p)
    x
end
function copyto!(x::State, y)
    for i in eachindex(x)
        x[i] = y[i]
    end
    x
end
#eltype(::State{P}) = P


Base.broadcastable(x::State) = x
Broadcast.BroadcastStyle(::Type{<:State}) = Broadcast.Style{State}()



size(s::State) = (2, length(s.p))

zero(x::State) = State(zero(x.p), zero(x.q))
zero!(v) = fill!(v, zero(eltype(v)))
function zero!(x::State)
    zero!(x.p)
    zero!(x.q)
    x
end



function getindex(x::State, I::CartesianIndex)
    if I[1] == 1
        x.q[I[2]]
    elseif I[1] == 2
        x.p[I[2]]
    else
        throw(BoundsError())
    end
end
function setindex!(x::State, val, I::CartesianIndex)
    if I[1] == 1
        x.q[I[2]] = val
    elseif I[1] == 2
        x.p[I[2]] = val
    else
        throw(BoundsError())
    end
end

eachindex(x::State) = CartesianIndices((Base.OneTo(2), eachindex(x.p)))

import Base: *, +, /, -
import LinearAlgebra: norm
import Bridge: outer
*(c::Number, x::State) = State(c*x.q, c*x.p)
*(x::State,c::Number) = State(x.q*c, x.p*c)
+(x::State, y::State) = State(x.q + y.q, x.p + y.p)
-(x::State, y::State) = State(x.q - y.q, x.p - y.p)
function outer(x::State, y::State)
    [outer(x[i],y[j]) for i in eachindex(x), j in eachindex(y)]
end
# unc(a::Array) = State([sqrt(a[1, i, 1, i]) for i in 1:size(a, 2)],[sqrt(a[2, i, 2, i]) for i in 1:size(a, 2)])

norm(x::State) = norm(x.q) + norm(x.q)

/(x::State, y) = State(x.q/y, x.p/y)
vec(x::State) = vec([x[J] for J in eachindex(x)])
