import Base: getindex, setindex!, length, copy, vcat, start, next, done, endof, keys, values
import Base: zero

import Base: valtype
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

It supports `getindex, setindex!, length, copy, vcat, start, next, done, endof`.
"""
struct SamplePath{T} <: AbstractPath{T}
    tt::Vector{Float64}
    yy::Vector{T}
    SamplePath{T}(tt, yy) where {T} = new(tt, yy)
end
SamplePath(tt, yy::Vector{T}) where {T} = SamplePath{T}(tt, yy)

samplepath(tt, v) = SamplePath(tt, fill(v, length(tt))) 


copy(X::SamplePath{T}) where {T} = SamplePath{T}(copy(X.tt), copy(X.yy))
length(X::SamplePath) = length(X.tt)

getindex(V::SamplePath, I::AbstractArray) = SamplePath(V.tt[I], V.yy[I])
getindex(V::SamplePath, i::Integer) = V.tt[i] => V.yy[i]
vcat(Ys::SamplePath{T}...) where {T} = SamplePath{T}(vcat(map(Y -> Y.tt, Ys)...), vcat(map(Y -> Y.yy, Ys)...))

# iterator
start(Y::SamplePath) = 1
next(Y::SamplePath, state) = Y[state], state + 1
done(V::SamplePath, state) = state > endof(V)
endof(V::SamplePath) = endof(V.tt)

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


struct VSamplePath{T} <: Bridge.AbstractPath{T}
    tt::Vector{Float64}
    yy::Matrix{T}
    function VSamplePath(tt, yy::Matrix{T}) where {T} 
        length(tt) != size(yy, 2) && throw(DimensionMismatch("length(tt) != size(yy, 2)"))
        new{T}(tt, yy) 
    end    
end

length(X::VSamplePath) = length(X.tt)

# separate a Zip2
sep(Z::Base.Iterators.Zip2{Vector{T1},Vector{T2}}) where {T1,T2} = 
    T1[z[1] for z in Z], T2[z[2] for z in Z] # takes into account the minimum of length


"""
    Increments{S<:AbstractPath{T}}
    
Iterator over the increments of an AbstractPath. 
Iterates over `(i, tt[i], tt[i+1]-tt[i], yy[i+1]-y[i])`.
"""
type Increments{S<:AbstractPath}
    X::S
end
start(dX::Increments) = 1
next(dX::Increments, i) = (i, dX.X.tt[i], dX.X.tt[i+1]-dX.X.tt[i], dX.X.yy[.., i+1]-dX.X.yy[.., i]), i + 1
done(dX::Increments, i) = i + 1 > length(dX.X.tt)
increments(X::AbstractPath) = Increments(X)

# Interoperatibility SDEs

b!(t, u, du, fg::Tuple{Function,Function}) = fg[1](t, u, du)
σ!(t, u, dw, dm, fg::Tuple{Function,Function}) = fg[2](t, u, dw, dm)
b(t, x, fg::Tuple{Function,Function}) = fg[1](t, x)
σ(t, x, fg::Tuple{Function,Function}) = fg[2](t, x)

# Interoperatibility ODEs

@inline _F(t, x, F::Function) = F(t, x)
@inline _F(t, x, F::Tuple{Function,Function}) = F[1](t)*x + F[2](t)