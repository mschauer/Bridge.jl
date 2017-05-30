import Base: getindex, setindex!, length, copy, vcat, start, next, done, endof
abstract ContinuousTimeProcess{T} 

immutable SamplePath{T}
    tt :: Vector{Float64}
    yy :: Vector{T}
    SamplePath(tt, yy) = new(tt, yy)
end
SamplePath{T}(tt, yy::Vector{T}) = SamplePath{T}(tt, yy)
copy{T}(X::SamplePath{T}) = SamplePath{T}(copy(X.tt), copy(X.yy))
length(X::SamplePath) = length(X.tt)
getindex(V::SamplePath, I::AbstractArray) = SamplePath(V.tt[I], V.yy[I])
getindex(V::SamplePath, i::Integer) = (V.tt[i], V.yy[i])
vcat{T}(Ys::SamplePath{T}...) = SamplePath{T}(vcat(map(Y->Y.tt, Ys)...),vcat(map(Y->Y.yy, Ys)...))


start(Y::SamplePath) = 1
next(Y::SamplePath, state) = Y[state], state+1
done(V::SamplePath, state) = state>endof(V)
endof(V::SamplePath) = endof(V.tt)

function setindex!(V::SamplePath,y, I)
    V.tt[I],V.yy[I] = y
end

function setv!(X::SamplePath, v)
    X.yy[end] = v
    X
end


#

mat{d,T}(X::SamplePath{Vec{d,T}}) = reshape(reinterpret(T,X.yy), d, length(X.yy))
mat{d,T}(yy::Vector{Vec{d,T}}) = reshape(reinterpret(T,yy), d, length(yy))

unmat{T}(A::Matrix{T}) = reinterpret(Vec{size(A,1),T},A[:])
unmat{d,T}(_::Type{Vec{d,T}}, A::Matrix{T}) = reinterpret(Vec{d,T},A[:])

# separate a Zip2
sep{T1,T2}(Z::Base.Iterators.Zip2{Vector{T1},Vector{T2}}) = T1[z[1] for z in Z], T2[z[2] for z in Z] # takes into account the minimum of length


