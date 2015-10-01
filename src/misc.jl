import Base: *, +, -, /, ctranspose, zero, dot, chol
*(J::Base.LinAlg.UniformScaling, A::FixedSizeArrays.FixedArray) = J.λ*A
*(A::FixedSizeArrays.FixedArray, J::Base.LinAlg.UniformScaling) = A*J.λ
/(J::Base.LinAlg.UniformScaling, A::FixedSizeArrays.FixedArray) = J.λ/A
/(A::FixedSizeArrays.FixedArray, J::Base.LinAlg.UniformScaling) = A/J.λ
chol(J::Base.LinAlg.UniformScaling, _) = chol(J.λ)*I


+{m, n, T}(A::Mat{m,n, T}, J::Base.LinAlg.UniformScaling) = A + J.λ*eye(Mat{m,n,T})
+{m, n, T}(J::Base.LinAlg.UniformScaling, A::Mat{m,n, T}) = J.λ*eye(Mat{m,n,T}) + A
-{m, n, T}(A::Mat{m,n, T}, J::Base.LinAlg.UniformScaling) = A - J.λ*eye(Mat{m,n,T})
-{m, n, T}(J::Base.LinAlg.UniformScaling, A::Mat{m,n, T}) = J.λ*eye(Mat{m,n,T}) - A

zero{T, NDim, SIZE}(_::FixedSizeArrays.FixedArray{T,NDim,SIZE}) = zero(typeof(_))

dot(J::Base.LinAlg.UniformScaling{Float64}, b::Float64) = J.λ*b


function cumsum0(dx::Vector)
        n = length(dx) + 1
        x = similar(dx, n)
        x[1] = 0.0      
        for i in 2:n
                x[i] = x[i-1] + dx[i-1] 
        end
        x
end

import Base: randn, rand


randn(::Type{Float64}) = randn()
randn{T}(::Type{Complex{T}}) = Complex(randn(T), randn(T))
immutable RandnFunctor{T} <: FixedSizeArrays.Func{1} end
@inline call{T}(rf::Type{RandnFunctor{T}}, i...) = randn(T)
@inline randn{FSA <: FixedArray}(x::Type{FSA}) = map(RandnFunctor{eltype(FSA)}, FSA)

function chol{n,T}(m::Mat{n,n,T},::Type{Val{:L}})
    m[1,2]==m[2,1] || throw(Error("Matrix not symmetric"))
    l11 = sqrt(m[1, 1])
    @fsa    [l11        m[1,2]/l11
            zero(T) sqrt(m[2, 2] - m[2, 1]^2 / m[1, 1])]
end

function chol{n,T}(m::Mat{n,n,T},::Type{Val{:R}})
    m[1,2]==m[2,1] || throw(Error("Matrix not symmetric"))
    l11 = sqrt(m[1, 1])
    @fsa    [l11   zero(T) 
            m[1,2]/l11 sqrt(m[2, 2] - m[2, 1]^2 / m[1, 1])]
end


sumlogdiag(A,d) = sum(log(diag(A)))
sumlogdiag{T}(J::UniformScaling{T},d) = log(J.λ)*d
function logpdfnormal(x, A) 
    S = chol(A, Val{:L})
    d = length(x)
     -((norm(S\x))^2 + 2sumlogdiag(S,d) + d*log(2pi))/2
end

