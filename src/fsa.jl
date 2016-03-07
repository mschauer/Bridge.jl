using FixedSizeArrays
import Base: *, +, -, /, \, ctranspose, zero, chol, trace, logdet, lyap

logdet(m::FixedSizeArrays.Mat) = log(det(m))
logdet(x::Real) = log(x)
 
function sumlogdiag{m,T}(A::Mat{m,m,T}, d=m) 
    t = zero(T)
    for i in 1:m
        t += log(A[i,i])
    end
    t
end   
 
zero{T, NDim, SIZE}(_::FixedSizeArrays.FixedArray{T,NDim,SIZE}) = zero(typeof(_))

 
