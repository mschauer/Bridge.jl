import Base: getindex, setindex!, length, copy, vcat, start, next, done, endof

import Base: valtype
"""
    ContinuousTimeProcess{T}

Types inheriting from the abstract type ContinuousTimeProcess{T} characterize 
the properties of a `T`-valued stochastic process, play
a similar role as distribution types like ``Exponential` in the package
`Distributions`.
"""
abstract type ContinuousTimeProcess{T} end
abstract type AbstractPath{T} end

"""
    valtype(::ContinuousTimeProcess) -> T

Returns statespace (type) of a `ContinuousTimeProcess{T]`.
"""
valtype(::ContinuousTimeProcess{T}) where {T} = T
valtype(::AbstractPath{T}) where {T} = T

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
serves as container for discretely observed ``ContinuousTimeProcess`es and for the sample path returned
by direct and approximate samplers. `tt`` is the vector of the grid points of the observation/simulation 
and `yy` is the corresponding vector of states.

It supports `getindex, setindex!, length, copy, vcat, start, next, done, endof`.
"""
struct SamplePath{T} <: AbstractPath{T}
    tt::Vector{Float64}
    yy::Vector{T}
    SamplePath{T}(tt, yy) where {T} = new(tt, yy)
end
SamplePath(tt, yy::Vector{T}) where {T} = SamplePath{T}(tt, yy)
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

function endpoint!(X::SamplePath, v)
    X.yy[end] = v
    X
end


# reinterpret to array

mat(X::SamplePath{SVector{d,T}}) where {d,T} = reshape(reinterpret(T, X.yy), d, length(X.yy))
mat(yy::Vector{SVector{d,T}}) where {d,T} = reshape(reinterpret(T, yy), d, length(yy))

unmat(A::Matrix{T}) where {T} = reinterpret(SVector{size(A, 1),T}, A[:])
unmat(::Type{SVector{d,T}}, A::Matrix{T}) where {d,T} = reinterpret(SVector{d,T}, A[:])

# separate a Zip2
sep(Z::Base.Iterators.Zip2{Vector{T1},Vector{T2}}) where {T1,T2} = 
    T1[z[1] for z in Z], T2[z[2] for z in Z] # takes into account the minimum of length


