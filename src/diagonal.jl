#using FixedSizeArrays
import Base: getindex,setindex!,==,-,+,*,/,\,transpose,ctranspose,convert, size, abs, real, imag, conj, eye, inv
import Base.LinAlg: ishermitian, issymmetric, isposdef, factorize, diag, trace, det, logdet, expm, logm, sqrtm

@generated function scale{T, M, N}(a::SMatrix{M, N, T}, b::SVector{N,T})
    expr = [:((SVector(column(a, $i)) * b[$i])._)for i=1:N]
    :( SMatrix($(expr...)) )
end
@generated function scale{T, M, N}(b::SVector{M,T}, a::SMatrix{M, N, T})
    expr = [:((SVector(row(a, $i)) * b[$i])._)for i=1:M]
    :( transpose(SMatrix($(expr...))) )
end
#function scale{T, M}(a::Vec{M,T}, b::Vec{M, T})
#    a.*b
#end

if !isdefined(:FixedDiagonal) 
immutable FixedDiagonal{N,T}
    diag::SVector{N,T}
end    
end

function \{T,M}(D::FixedDiagonal, b::SVector{M,T} )
    D.diag .* b
end


FixedDiagonal(A::SMatrix) = FixedDiagonal(diag(A))


convert{N,T}(::Type{FixedDiagonal{N,T}}, D::FixedDiagonal{N,T}) = D
convert{N,T}(::Type{FixedDiagonal{N,T}}, D::FixedDiagonal) = FixedDiagonal{N,T}(convert(SVector{N,T}, D.diag))


size(D::FixedDiagonal) = (length(D.diag),length(D.diag))

function size(D::FixedDiagonal,d::Integer)
    if d<1
        throw(ArgumentError("dimension must be ≥ 1, got $d"))
    end
    return d<=2 ? length(D.diag) : 1
end

function getindex{T}(D::FixedDiagonal{T}, i::Int, j::Int)  
    if i == j
        D.diag[i]
    else
        zero(T)
    end
end
function setindex!(D::FixedDiagonal, v, i::Int, j::Int)
    if i == j
        unsafe_setindex!(D.diag, v, i)
    elseif v != 0
        throw(ArgumentError("cannot set an off-diagonal index ($i, $j) to a nonzero value ($v)"))
    end
    D
end

ishermitian{T<:Real}(D::FixedDiagonal{T}) = true
ishermitian(D::FixedDiagonal) = all(D.diag .== real(D.diag))
issym(D::FixedDiagonal) = true
isposdef(D::FixedDiagonal) = all(D.diag .> 0)

factorize(D::FixedDiagonal) = D

abs(D::FixedDiagonal) = FixedDiagonal(abs(D.diag))
real(D::FixedDiagonal) = FixedDiagonal(real(D.diag))
imag(D::FixedDiagonal) = FixedDiagonal(imag(D.diag))

==(Da::FixedDiagonal, Db::FixedDiagonal) = Da.diag == Db.diag
-(A::FixedDiagonal) = FixedDiagonal(-A.diag)
+(Da::FixedDiagonal, Db::FixedDiagonal) = FixedDiagonal(Da.diag + Db.diag)
-(Da::FixedDiagonal, Db::FixedDiagonal) = FixedDiagonal(Da.diag - Db.diag)
-(A::FixedDiagonal, B::SMatrix) = eye(typeof(B))*A - B


*{T<:Number}(x::T, D::FixedDiagonal) = FixedDiagonal(x * D.diag)
*{T<:Number}(D::FixedDiagonal, x::T) = FixedDiagonal(D.diag * x)
/{T<:Number}(D::FixedDiagonal, x::T) = FixedDiagonal(D.diag / x)
*(Da::FixedDiagonal, Db::FixedDiagonal) = FixedDiagonal(Da.diag .* Db.diag)
*(D::FixedDiagonal, V::SVector) = D.diag .* V
*(V::SVector, D::FixedDiagonal) = D.diag .* V
*(A::SMatrix, D::FixedDiagonal) = scale(A,D.diag)
*(D::FixedDiagonal, A::SMatrix) = scale(D.diag,A)

/(Da::FixedDiagonal, Db::FixedDiagonal) = FixedDiagonal(Da.diag ./ Db.diag )

conj(D::FixedDiagonal) = FixedDiagonal(conj(D.diag))
transpose(D::FixedDiagonal) = D
ctranspose(D::FixedDiagonal) = conj(D)

diag(D::FixedDiagonal) = D.diag
trace(D::FixedDiagonal) = sum(D.diag)
det(D::FixedDiagonal) = prod(D.diag)
logdet{N,T<:Real}(D::FixedDiagonal{N,T}) = sum(log(D.diag))
function logdet{N,T<:Complex}(D::FixedDiagonal{N,T}) #Make sure branch cut is correct
    x = sum(log(D.diag))
    -pi<imag(x)<pi ? x : real(x)+(mod2pi(imag(x)+pi)-pi)*im
end


eye{N,T}(::Type{FixedDiagonal{N,T}}) = FixedDiagonal(one(SVector{n,Int}))

expm(D::FixedDiagonal) = FixedDiagonal(exp(D.diag))
logm(D::FixedDiagonal) = FixedDiagonal(log(D.diag))
sqrtm(D::FixedDiagonal) = FixedDiagonal(sqrt(D.diag))

\(D::FixedDiagonal, B::SMatrix) = scale(1 ./ D.diag, B)
/(B::SMatrix, D::FixedDiagonal) = scale(1 ./ D.diag, B)
\(Da::FixedDiagonal, Db::FixedDiagonal) = FixedDiagonal(Db.diag ./ Da.diag)

function inv{N,T}(D::FixedDiagonal{N,T})
    for i = 1:length(D.diag)
        if D.diag[i] == zero(T)
            throw(SingularException(i))
        end
    end
    FixedDiagonal(one(T)./D.diag)
end

