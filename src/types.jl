ismutable(el) = ismutable(typeof(el))
ismutable(::Type) = Val(false)
ismutable(::Type{<:Array}) = Val(true)



import Base: getindex, setindex!, length, copy, vcat, keys, values, iterate
import Base: zero

import Base: valtype

const IndexedTime = Tuple{Int64,Float64}


"""
    ContinuousTimeProcess{T}

Types inheriting from the abstract type `ContinuousTimeProcess{T}` characterize
the properties of a `T`-valued stochastic process, play
a similar role as distribution types like `Exponential` in the package
`Distributions`.
"""
abstract type ContinuousTimeProcess{T} end
const ProcessOrCoefficients = Union{ContinuousTimeProcess,Tuple{Function,Function}}


"""
    a(t, x, P::ProcessOrCoefficients)

Fallback for `a(t, x, P)` calling `σ(t, x, P)*σ(t, x, P)'`.
"""
a(t, x, P::ProcessOrCoefficients) = outer(σ(t, x, P))
Γ(t, x, P::ProcessOrCoefficients) = inv(a(t, x, P))

abstract type AbstractPath{T} end

"""
    valtype(::ContinuousTimeProcess) -> T

Returns statespace (type) of a `ContinuousTimeProcess{T]`.
"""
valtype(::ContinuousTimeProcess{T}) where {T} = T
valtype(::AbstractPath{T}) where {T} = T

"""
    outertype(P::ContinuousTimeProcess) -> T

Returns the type of `outer(x)`, where `x` is a state of `P`
"""
outertype(::ContinuousTimeProcess{Float64}) = Float64
outertype(P::ContinuousTimeProcess{<:StaticArray}) = typeof(outer(zero(valtype(P))))


"""
    SamplePath{T} <: AbstractPath{T}

The struct
```
struct SamplePath{T}
    tt::Vector{Float64}
    yy::Vector{T}
    SamplePath{T}(tt, yy) where {T} = new(tt, yy)
end
```
serves as container for discretely observed `ContinuousTimeProcess`es and for the sample path returned
by direct and approximate samplers. `tt` is the vector of the grid points of the observation/simulation
and `yy` is the corresponding vector of states.

It supports `getindex, setindex!, length, copy, vcat`.
"""
struct SamplePath{T} <: AbstractPath{T}
    tt::Vector{Float64}
    yy::Vector{T}
    SamplePath{T}(tt, yy) where {T} = new(tt, yy)
end
SamplePath(tt, yy::Vector{T}) where {T} = SamplePath{T}(tt, yy)

samplepath(tt, v) = samplepath(tt, v, ismutable(v))

samplepath(tt, v, ::Val{false}) = SamplePath(tt, fill(v, length(tt)))
samplepath(tt, v, ::Val{true}) = SamplePath(tt, [copy(v) for t in tt])


copy(X::SamplePath{T}) where {T} = SamplePath{T}(copy(X.tt), copy(X.yy))
length(X::SamplePath) = length(X.tt)

getindex(V::SamplePath, I::AbstractArray) = SamplePath(V.tt[I], V.yy[I])
getindex(V::SamplePath, i::Integer) = V.tt[i] => V.yy[i]
vcat(Ys::SamplePath{T}...) where {T} = SamplePath{T}(vcat(map(Y -> Y.tt, Ys)...), vcat(map(Y -> Y.yy, Ys)...))

# iterator
# start(Y::SamplePath) = 1
# next(Y::SamplePath, state) = Y[state], state + 1
# done(V::SamplePath, state) = state > endof(V)

# lastindex(V::SamplePath) = lastindex(V.tt)

function setindex!(V::SamplePath, y, I)
    V.tt[I], V.yy[I] = y
end

"""
    endpoint!(X::SamplePath, v)

Convenience functions setting the endpoint of `X to `v`.
"""
function endpoint!(X::SamplePath, v)
    X.yy[end] = v
    X
end

keys(X::SamplePath) = X.tt
values(X::SamplePath) = X.yy
SamplePath(X::Vector{Pair{Float64,T}}) where {T} = SamplePath{T}(map(first, X), map(last, X))

zero(X::SamplePath) = SamplePath(X.tt, X.yy)

import Base.broadcast
broadcast(f, X::SamplePath, Y::SamplePath) = (X.tt != Y.tt) ? throw(ArgumentError("differing sample times")) : SamplePath(X.tt, f.(X.yy, Y.yy))


struct VSamplePath{T} <: Bridge.AbstractPath{T}
    tt::Vector{Float64}
    yy::Matrix{T}
    function VSamplePath(tt, yy::Matrix{T}) where {T}
        length(tt) != size(yy, 2) && throw(DimensionMismatch("length(tt) != size(yy, 2)"))
        new{T}(tt, yy)
    end
end

length(X::VSamplePath) = length(X.tt)

allsametime(xx) = all(x -> x.tt == xx[1].tt, xx)
function stack(args::SamplePath...)
    assert(allsametime(args))
    VSamplePath(args[1].tt, vcat((X.yy' for X in args)...))
end

# separate a zip of two Vectors
if VERSION < v"1.1-"
sep(Z::Base.Iterators.Zip2{Vector{T1},Vector{T2}}) where {T1,T2} =
    T1[z[1] for z in Z], T2[z[2] for z in Z] # takes into account the minimum of length
else
sep(Z::Base.Iterators.Zip{Tuple{Vector{T1},Vector{T2}}}) where {T1,T2} =
    T1[z[1] for z in Z], T2[z[2] for z in Z] # takes into account the minimum of length
end


copy(X::VSamplePath) = VSamplePath(copy(X.tt), copy(X.yy))

"""
Like `VSamplePath`, but with assumptions on `tt` and dimensionality.
Planned replacement for `VSamplePath`
"""
struct GSamplePath{S,T,P} <: Bridge.AbstractPath{T}
    tt::S
    yy::P
    GSamplePath(tt::S, yy::P) where {S, P<:AbstractArray{T}} where {T} = new{S,T,P}(tt,yy)
end

import Base: length
length(X::GSamplePath) = length(X.tt)


"""
    Increments{S<:AbstractPath{T}}

Iterator over the increments of an AbstractPath.
Iterates over `(i, tt[i], tt[i+1]-tt[i], yy[i+1]-y[i])`.
"""
mutable struct Increments{S<:AbstractPath}
    X::S
end

iterate(dX::Increments, i = 1) = i + 1 > length(dX.X.tt) ? nothing : ((i, dX.X.tt[i], dX.X.tt[i+1]-dX.X.tt[i], dX.X.yy[.., i+1]-dX.X.yy[.., i]), i + 1)

increments(X::AbstractPath) = Increments(X)

# Interoperatibility SDEs

b!(t, u, du, fg::Tuple{Function,Function}) = fg[1](t, u, du)
σ!(t, u, dw, dm, fg::Tuple{Function,Function}) = fg[2](t, u, dw, dm)
b(t, x, fg::Tuple{Function,Function}) = fg[1](t, x)
σ(t, x, fg::Tuple{Function,Function}) = fg[2](t, x)

_b!((i,t)::IndexedTime, u, du, fg::Tuple{Function,Function}) = fg[1](t, u, du)
_b((i,t)::IndexedTime, x, fg::Tuple{Function,Function}) = fg[1](t, x)

# Interoperatibility ODEs

F(t, x, f::Tuple{Function}) = f[1](t, x)
#@inline _F(t, x, F::Function) = F(t, x)
#@inline _F(t, x, F::Tuple{Function,Function}) = F[1](t)*x + F[2](t)
