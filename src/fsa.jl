using StaticArrays
import Base: *, +, -, /, \, ctranspose, zero, chol, trace, logdet, lyap

logdet(m::SMatrix) = log(det(m))
logdet(x::Real) = log(x)
 
function sumlogdiag{m,T}(A::SMatrix{m,m,T}, d=m) 
    t = zero(T)
    for i in 1:m
        t += log(A[i,i])
    end
    t
end   
 
zero{T, NDim, SIZE}(prototype::SMatrix{T,NDim,SIZE}) = zero(typeof(prototype))

 
