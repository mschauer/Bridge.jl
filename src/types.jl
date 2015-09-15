using FixedSizeArrays

import Base: getindex, setindex!, length, copy
abstract CTPro{T} 


typealias BiPro{T} CTPro{Vec{2,T}}

immutable CTPath{T}
    tt :: Vector{Float64}
    yy :: Vector{T}
    CTPath(tt, yy) = new(tt, yy)
end
copy{T}(X::CTPath{T}) = CTPath{T}(copy(X.tt), copy(X.yy))

typealias BiPath{T} CTPath{Vec{2,T}}


length(X::CTPath) = length(X.tt)


getindex(V::CTPath, I) = (V.tt[I], V.yy[I])
#endof(V::CTPath) = endof(V.tt)

function setindex!(V::CTPath,y, I)
    V.tt[I],V.yy[I] = y
end

function setv!(X::CTPath, v)
    X.yy[end] = v
    X
end



mat{d,T}(X::CTPath{Vec{d,T}}) = reshape(reinterpret(T,X.yy), d, length(X.yy))
mat{d,T}(yy::Vector{Vec{d,T}}) = reshape(reinterpret(T,yy), d, length(yy))

unmat{T}(A::Matrix{T}) = reinterpret(Vec{size(A,1),T},A[:])
unmat{d,T}(_::Type{Vec{d,T}}, A::Matrix{T}) = reinterpret(Vec{d,T},A[:])

#include("wiener.jl")