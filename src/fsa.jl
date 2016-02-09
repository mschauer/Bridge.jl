using FixedSizeArrays
import Base: *, +, -, /, \, ctranspose, zero, chol, trace, logdet, lyap

function chol{T}(m::Mat{2,2,T})
    m[1,2]==m[2,1]' || error("Matrix not symmetric")
    l11 = chol(m[1,1])
    @fsa    [l11        inv(l11)*m[1,2]
            zero(T) chol(m[2,2] - (m[2,1]*inv(m[1,1])*m[1,2]), Val{:L})]
end

function chol{T}(m::Mat{2,2,T}, ::Type{Val{:L}})
    m[1,2]==m[2,1]' || error("Matrix not symmetric")
    l11 = chol(m[1,1])'
    @fsa    [l11   zero(T) 
            m[2,1]*inv(l11) chol(m[2,2] - (m[2,1]*inv(m[1,1])*m[1,2]), Val{:L})]
end


logdet(m::FixedSizeArrays.Mat) = log(det(m))
logdet(x::Real) = log(x)

lyap{m,T}(A::Mat{m,m,T},C::Mat{m,m,T}) = Mat(lyap(Matrix(A),Matrix(C)))

function trace{m,T}(A::Mat{m,m,T}) 
    t = zero(T)
    for i in 1:m
        t += A[i,i]
    end
    t
end    


function sumlogdiag{m,T}(A::Mat{m,m,T}, d=m) 
    t = zero(T)
    for i in 1:m
        t += log(A[i,i])
    end
    t
end   

*(J::Base.LinAlg.UniformScaling, A::FixedSizeArrays.FixedArray) = J.λ*A
*(A::FixedSizeArrays.FixedArray, J::Base.LinAlg.UniformScaling) = A*J.λ
/(A::FixedSizeArrays.FixedArray, J::Base.LinAlg.UniformScaling) = A/J.λ
\(J::Base.LinAlg.UniformScaling,v::FixedSizeArrays.FixedVector) = v/J.λ

#chol(J::Base.LinAlg.UniformScaling, ::Type{Val{:U}}) = sqrt(J.λ)*I
#chol(J::Base.LinAlg.UniformScaling, ::Type{Val{:L}}) = sqrt(J.λ)*I
# #chol(z::Float64, ::Type{Val{:L}}) = sqrt(z)

+{m, n, T}(A::Mat{m,n, T}, J::Base.LinAlg.UniformScaling) = A + J.λ*eye(Mat{m,n,T})
+{m, n, T}(J::Base.LinAlg.UniformScaling, A::Mat{m,n, T}) = J.λ*eye(Mat{m,n,T}) + A
-{m, n, T}(A::Mat{m,n, T}, J::Base.LinAlg.UniformScaling) = A - J.λ*eye(Mat{m,n,T})
-{m, n, T}(J::Base.LinAlg.UniformScaling, A::Mat{m,n, T}) = J.λ*eye(Mat{m,n,T}) - A

zero{T, NDim, SIZE}(_::FixedSizeArrays.FixedArray{T,NDim,SIZE}) = zero(typeof(_))


\{m,n,T}(mat::Mat{m,n,T}, v::Vec{n, T}) = inv(mat)*v

import Base.randn
immutable RandnFunctor{T} <: FixedSizeArrays.Func{1} end
@inline call{T}(rf::Type{RandnFunctor{T}}, i...) = randn(T)
@inline randn{FSA <: FixedArray}(x::Type{FSA}) = map(RandnFunctor{eltype(FSA)}, FSA)
