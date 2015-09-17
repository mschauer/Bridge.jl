import Base: getindex, setindex!, length, copy, vcat, start, next, done, endof
abstract CTPro{T} 

immutable CTPath{T}
    tt :: Vector{Float64}
    yy :: Vector{T}
    CTPath(tt, yy) = new(tt, yy)
end
CTPath{T}(tt, yy::Vector{T}) = CTPath{T}(tt, yy)
copy{T}(X::CTPath{T}) = CTPath{T}(copy(X.tt), copy(X.yy))
length(X::CTPath) = length(X.tt)
getindex(V::CTPath, I::AbstractArray) = CTPath(V.tt[I], V.yy[I])
getindex(V::CTPath, i::Integer) = (V.tt[i], V.yy[i])
vcat{T}(Ys::CTPath{T}...) = CTPath{T}(vcat(map(Y->Y.tt, Ys)...),vcat(map(Y->Y.yy, Ys)...))


start(Y::CTPath) = 1
next(Y::CTPath, state) = Y[state], state+1
done(V::CTPath, state) = state>endof(V)
endof(V::CTPath) = endof(V.tt)

function setindex!(V::CTPath,y, I)
    V.tt[I],V.yy[I] = y
end

function setv!(X::CTPath, v)
    X.yy[end] = v
    X
end


#

mat{d,T}(X::CTPath{Vec{d,T}}) = reshape(reinterpret(T,X.yy), d, length(X.yy))
mat{d,T}(yy::Vector{Vec{d,T}}) = reshape(reinterpret(T,yy), d, length(yy))

unmat{T}(A::Matrix{T}) = reinterpret(Vec{size(A,1),T},A[:])
unmat{d,T}(_::Type{Vec{d,T}}, A::Matrix{T}) = reinterpret(Vec{d,T},A[:])

# seperate a Zip2
sep{T1,T2}(Z::Base.Zip2{Vector{T1},Vector{T2}}) = T1[z[1] for z in Z], T2[z[2] for z in Z]


